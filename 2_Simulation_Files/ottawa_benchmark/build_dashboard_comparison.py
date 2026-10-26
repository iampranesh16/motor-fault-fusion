"""Build a recording-level Ottawa sync ablation for the Streamlit dashboard.

The Ottawa recordings provide acoustic data and a speed reference. Hall channels
are synthesized by the existing preprocessing model, so this benchmark must not
be described as validation on measured Hall data.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET_DIR = ROOT / "dataset_ottawa"
FEATURES_CSV = HERE / "training_features_ottawa_synced.csv"
FEATURE_COLUMNS = [
    "ac_bpfo", "ac_bpfo_2x", "ac_1x", "ac_2x", "ac_rms", "ac_kurt",
    "mag_array_rms", "mag_spatial_mode1", "mag_sideband_ratio",
    "mag_spatial_quad", "consistency_gate",
]
ACOUSTIC_COLUMNS = FEATURE_COLUMNS[:6]
OUT_DIR = HERE / "dashboard_artifacts"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "2_Simulation_Files"))
from src.features import extract_features  # noqa: E402


def build_unsynced_features(sync_df: pd.DataFrame) -> pd.DataFrame:
    """Extract baseline (raw-time acoustic + fixed 50 Hz reference) features."""
    files = {p.name: p for p in DATASET_DIR.rglob("*.mat")}
    missing = sorted(set(sync_df["file_name"]) - set(files))
    if missing:
        raise FileNotFoundError(f"Missing Ottawa recordings: {missing[:5]}")
    output = []
    fs = 16000
    total_samples = 10 * fs
    window_samples = fs
    for file_name, rows in sync_df.groupby("file_name", sort=False):
        contents = sio.loadmat(files[file_name])
        raw = contents["data"]
        acoustic = raw[:, 1] if raw.ndim == 2 else raw.flatten()
        downsampled = signal.resample(acoustic, total_samples)
        rows = rows.sort_values("window_idx")
        for _, row in rows.iterrows():
            idx = int(row["window_idx"])
            win = downsampled[idx * window_samples:(idx + 1) * window_samples]
            features = extract_features(
                {"acoustic": win, "hall": np.zeros((6, 1000), dtype=np.float64),
                 "meta": {"fs_acoustic": fs, "fs_hall": 1000}},
                mode="naive_fusion", nominal_fe=50.0, nominal_slip=0.02,
            )
            record = {k: row[k] for k in ["file_name", "fault_class", "speed_index", "speed_hz_mean", "window_idx"]}
            record.update(dict(zip(FEATURE_COLUMNS, map(float, features))))
            output.append(record)
    return pd.DataFrame(output)


def fit_metrics(train: pd.DataFrame, test: pd.DataFrame, mode: str, classes: list[str]):
    model = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=42, n_jobs=-1)
    model.fit(train[ACOUSTIC_COLUMNS], train["fault_class"])
    pred = model.predict(test[ACOUSTIC_COLUMNS])
    precision, recall, f1, support = precision_recall_fscore_support(
        test["fault_class"], pred, labels=classes, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(test["fault_class"], pred)),
        "macro_f1": float(f1_score(test["fault_class"], pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test["fault_class"], pred, average="weighted", zero_division=0)),
        "classes": classes,
        "confusion_matrix": confusion_matrix(test["fault_class"], pred, labels=classes).tolist(),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(classes)},
    }
    return model, pred, metrics


def main():
    sync = pd.read_csv(FEATURES_CSV)
    unsync = build_unsynced_features(sync)
    paired = sync[["file_name", "fault_class", "speed_index", "speed_hz_mean", "window_idx"]].copy()
    for column in FEATURE_COLUMNS:
        paired[f"sync__{column}"] = sync[column].astype(float)
        paired[f"unsync__{column}"] = unsync[column].astype(float)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(paired, paired["fault_class"], groups=paired["file_name"]))
    train, test = paired.iloc[train_idx].copy(), paired.iloc[test_idx].copy()
    classes = sorted(paired["fault_class"].unique().tolist())
    results = {"split": "GroupShuffleSplit by file_name, test_size=0.20, random_state=42",
               "n_train_windows": int(len(train)), "n_test_windows": int(len(test)),
               "n_train_recordings": int(train.file_name.nunique()),
               "n_test_recordings": int(test.file_name.nunique()),
               "test_recordings": sorted(test.file_name.unique().tolist()),
               "note": "Acoustic-only comparison. Ottawa acoustic recordings and speed metadata are real; magnetic feature columns are excluded because preprocessing synthesizes Hall channels from the known class label."}

    sync_train = train[["fault_class"] + [f"sync__{c}" for c in FEATURE_COLUMNS]].rename(
        columns={f"sync__{c}": c for c in FEATURE_COLUMNS})
    sync_test = test[["fault_class"] + [f"sync__{c}" for c in FEATURE_COLUMNS]].rename(
        columns={f"sync__{c}": c for c in FEATURE_COLUMNS})
    unsync_train = train[["fault_class"] + [f"unsync__{c}" for c in FEATURE_COLUMNS]].rename(
        columns={f"unsync__{c}": c for c in FEATURE_COLUMNS})
    unsync_test = test[["fault_class"] + [f"unsync__{c}" for c in FEATURE_COLUMNS]].rename(
        columns={f"unsync__{c}": c for c in FEATURE_COLUMNS})
    sync_model, sync_pred, sync_metrics = fit_metrics(sync_train, sync_test, "sync", classes)
    unsync_model, unsync_pred, unsync_metrics = fit_metrics(unsync_train, unsync_test, "unsync", classes)

    # Save the held-out predictions alongside metadata for the dashboard's matched sample view.
    test_view = test[["file_name", "fault_class", "speed_index", "speed_hz_mean", "window_idx"]].copy()
    for c in FEATURE_COLUMNS:
        test_view[f"sync__{c}"] = test[f"sync__{c}"].to_numpy()
        test_view[f"unsync__{c}"] = test[f"unsync__{c}"].to_numpy()
    test_view["sync_prediction"] = sync_pred
    test_view["sync_confidence"] = sync_model.predict_proba(sync_test[ACOUSTIC_COLUMNS]).max(axis=1)
    test_view["unsync_prediction"] = unsync_pred
    test_view["unsync_confidence"] = unsync_model.predict_proba(unsync_test[ACOUSTIC_COLUMNS]).max(axis=1)
    test_view.to_csv(OUT_DIR / "ottawa_grouped_test_predictions.csv", index=False)
    joblib.dump(sync_model, OUT_DIR / "ottawa_synced_grouped_model.pkl")
    joblib.dump(unsync_model, OUT_DIR / "ottawa_unsynced_grouped_model.pkl")
    results["synced"] = sync_metrics
    results["unsynced"] = unsync_metrics
    (OUT_DIR / "ottawa_grouped_benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"synced": sync_metrics, "unsynced": unsync_metrics,
                      "test_recordings": results["test_recordings"]}, indent=2))


if __name__ == "__main__":
    main()
