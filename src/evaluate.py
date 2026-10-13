import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from src.generator import generate_signals
from src.features import extract_features

CLASSES = ["healthy", "bearing", "imbalance"]

def build_held_out_datasets(n_train_per_class: int = 100, n_test_per_class: int = 80, seed: int = 42):
    """
    Strictly disjoint held-out operating conditions:
    - Train: fe in [47, 50, 53] Hz, theta0 constrained near 0 deg ([-0.25, 0.25] rad), SNR [8, 18] dB
    - Test (Held-Out): unseen speeds fe in [41, 44, 56, 59] Hz, full 360-deg random rotation theta0 in [0.4, 2*pi-0.4], SNR [3, 14] dB
    """
    rng = np.random.default_rng(seed)
    train_records, test_records = [], []
    
    for label_idx, cls in enumerate(CLASSES):
        for i in range(n_train_per_class):
            fe = rng.choice([47.0, 50.0, 53.0]) + rng.uniform(-0.4, 0.4)
            slip = rng.uniform(0.015, 0.025)
            theta0 = rng.uniform(-0.25, 0.25)  # Fixed lab calibration angle during training
            snr = rng.uniform(8.0, 18.0)
            sev = 0.0 if cls == "healthy" else rng.uniform(0.35, 0.95)
            sig = generate_signals(
                cls, fe=fe, slip=slip, severity=sev, theta0=theta0,
                snr_db=snr, duration=1.0, realistic_interference=True, seed=seed+i*10+label_idx
            )
            train_records.append((sig, label_idx))
            
        for j in range(n_test_per_class):
            fe = rng.choice([41.0, 44.0, 56.0, 59.0]) + rng.uniform(-0.6, 0.6)
            slip = rng.uniform(0.01, 0.035)
            # In held-out deployment, array is clipped on at arbitrary angles including quadrature blind spots
            theta0 = rng.choice([rng.uniform(0.5, 1.5), rng.uniform(1.8, 2.8), rng.uniform(3.5, 5.5)])
            snr = rng.uniform(3.0, 14.0)
            sev = 0.0 if cls == "healthy" else rng.uniform(0.30, 0.90)
            sig = generate_signals(
                cls, fe=fe, slip=slip, severity=sev, theta0=theta0,
                snr_db=snr, duration=1.0, realistic_interference=True, seed=seed+5000+j*10+label_idx
            )
            test_records.append((sig, label_idx))
            
    return train_records, test_records

def bootstrap_macro_f1_ci(y_true, y_pred, n_boot=300, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def compute_false_alarms_per_hour(y_true, y_pred, persistence_k=3, window_sec=1.0):
    healthy_mask = (y_true == 0)
    false_pos_stream = (y_pred[healthy_mask] != 0).astype(int)
    raw_fa_rate = np.mean(false_pos_stream)
    
    persisted_alarms = 0
    streak = 0
    for fp in false_pos_stream:
        if fp == 1:
            streak += 1
            if streak == persistence_k:
                persisted_alarms += 1
        else:
            streak = 0
    hours_simulated = (len(false_pos_stream) * window_sec) / 3600.0
    fa_per_hour = persisted_alarms / max(hours_simulated, 1e-6)
    return float(raw_fa_rate * 100.0), float(fa_per_hour)

def run_ablation_study(save_dir="results", return_details=False):
    os.makedirs(save_dir, exist_ok=True)
    train_data, test_data = build_held_out_datasets()
    
    modes = ["acoustic_only", "hall_only", "naive_fusion", "fusion_sync", "full_proposed"]
    rows = []
    details = {}
    
    for mode in modes:
        t0 = time.perf_counter()
        X_train = np.array([extract_features(rec[0], mode=mode) for rec in train_data])
        y_train = np.array([rec[1] for rec in train_data])
        
        X_test = np.array([extract_features(rec[0], mode=mode) for rec in test_data])
        y_test = np.array([rec[1] for rec in test_data])
        elapsed_ms_per_win = ((time.perf_counter() - t0) / (len(train_data) + len(test_data))) * 1000.0
        
        clf = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        ci_low, ci_high = bootstrap_macro_f1_ci(y_test, y_pred)
        raw_fp_pct, fa_per_hr = compute_false_alarms_per_hour(y_test, y_pred, persistence_k=3)
        
        rows.append({
            "mode": mode,
            "macro_f1": round(macro_f1, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "raw_false_alarm_pct": round(raw_fp_pct, 2),
            "persisted_fa_per_hour": round(fa_per_hr, 2),
            "host_latency_ms": round(elapsed_ms_per_win, 2)
        })
        details[mode] = {"clf": clf, "y_true": y_test, "y_pred": y_pred, "cm": confusion_matrix(y_test, y_pred)}
        
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "ablation.csv")
    df.to_csv(csv_path, index=False)
    print("\n--- Held-Out Condition Ablation Table ---")
    print(df.to_string(index=False))
    if return_details:
        return df, details
    return df
