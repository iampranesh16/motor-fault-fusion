#!/usr/bin/env python3
"""
train_edge_model.py

Trains an embedded-constrained Random Forest classifier on preprocessed University of Ottawa
motor features:
1. Loads training_features_ottawa.csv and splits into train and test sets.
2. Trains a Random Forest classifier strictly constrained to <= 80 trees and max depth <= 7
   (matching the STM32F4/G4 and ESP32-S3 microcontroller hardware budget).
3. Evaluates and prints the macro-F1 score, weighted-F1 score, and classification report.
4. Generates and displays a publication-quality confusion matrix (saved for research paper use).
5. Exports model weights (both .joblib and structured .json node architecture) ready for MCU
   quantization and C-struct generation.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import joblib
import pickle

# The standard 11 features matching the microcontroller DSP extraction pipeline
FEATURE_COLUMNS = [
    "ac_bpfo",
    "ac_bpfo_2x",
    "ac_1x",
    "ac_2x",
    "ac_rms",
    "ac_kurt",
    "mag_array_rms",
    "mag_spatial_mode1",
    "mag_sideband_ratio",
    "mag_spatial_quad",
    "consistency_gate"
]

def load_and_preprocess_data(
    csv_path: str = "training_features_ottawa_synced.csv",
    classes_mode: str = "all",
    test_size: float = 0.2,
    group_split: bool = False,
    random_seed: int = 42
):
    """
    Loads dataset and splits into training and testing partitions.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Feature CSV '{csv_path}' not found. "
            f"Please run 'preprocess_ottawa.py' first to generate this file."
        )

    df = pd.read_csv(csv_path)
    print(f"[Data] Loaded {len(df)} samples from '{csv_path}'")

    # Validate feature columns exist
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns in CSV: {missing_cols}")

    # Class filtering
    if classes_mode == "core":
        # 3 core classes of the motor fault fusion project
        core_classes = ["healthy", "bearing", "imbalance"]
        df = df[df["fault_class"].isin(core_classes)].copy()
        print(f"[Data] Filtered to 3 core project classes: {core_classes} ({len(df)} samples)")
    elif classes_mode == "all":
        print(f"[Data] Using all {df['fault_class'].nunique()} classes present in dataset")
    else:
        raise ValueError(f"Unknown classes_mode '{classes_mode}'. Choose 'all' or 'core'.")

    X = df[FEATURE_COLUMNS].copy()
    y = df["fault_class"].copy()

    # Split into train/test
    if group_split:
        print("[Data] Splitting by recording file (GroupShuffleSplit) to test unseen motor runs...")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_idx, test_idx = next(gss.split(X, y, groups=df["file_name"]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        print(f"[Data] Performing stratified train/test split (test_size={test_size}, seed={random_seed})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

    print(f"[Data] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test, sorted(y.unique().tolist())

def train_constrained_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 80,
    max_depth: int = 7,
    random_seed: int = 42
) -> RandomForestClassifier:
    """
    Trains a Random Forest strictly constrained by microcontroller hardware budgets:
    - n_estimators <= 80 trees
    - max_depth <= 7
    """
    # Enforce MCU budget limits
    if n_estimators > 80:
        print(f"[Warning] Constraining n_estimators from {n_estimators} down to 80 (MCU hardware limit).")
        n_estimators = 80
    if max_depth > 7:
        print(f"[Warning] Constraining max_depth from {max_depth} down to 7 (MCU hardware limit).")
        max_depth = 7

    print(f"\n[Training] Fitting Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion="gini",
        random_state=random_seed,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("[Training] Model training complete.")
    return clf

def evaluate_model(clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series, classes: list):
    """
    Evaluates model predictions and prints comprehensive performance metrics.
    """
    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print(f"             TEST SET PERFORMANCE EVALUATION")
    print("=" * 65)
    print(f"  Macro-F1 Score   : {macro_f1:.4f}  ({macro_f1 * 100.0:.2f}%)")
    print(f"  Weighted-F1 Score: {weighted_f1:.4f}  ({weighted_f1 * 100.0:.2f}%)")
    print(f"  Test Accuracy    : {accuracy:.4f}  ({accuracy * 100.0:.2f}%)")
    print("-" * 65)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    print("=" * 65)

    return y_pred, macro_f1

def plot_and_display_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    classes: list,
    output_path: str = "results/confusion_matrix_ottawa_rf.png"
):
    """
    Generates, prints to terminal, and saves a publication-quality confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    # 1. Print formatted ASCII table to terminal for immediate display
    print("\nConfusion Matrix (Rows: True, Columns: Predicted):")
    header_labels = [c[:8] for c in classes]
    print(f"{'True \\ Pred':<18} " + " ".join([f"{h:>8}" for h in header_labels]))
    for idx, true_cls in enumerate(classes):
        row_str = " ".join([f"{cm[idx, j]:>8d}" for j in range(len(classes))])
        print(f"{true_cls:<18} {row_str}")

    # 2. Plot publication-grade figure
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 9.5})

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized True Positive Rate", rotation=270, labelpad=15)

    tick_marks = np.arange(len(classes))
    clean_labels = [c.replace("_", " ").title() for c in classes]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(clean_labels, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticklabels(clean_labels, fontsize=9.5)

    # Annotate cells with both count and percentage
    thresh = 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            color = "white" if cm_norm[i, j] > thresh else "black"
            text = f"{val}\n({pct:.1f}%)" if val > 0 else "0"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8.5, weight="normal")

    ax.set_ylabel("True Health State", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Health State", fontsize=11, fontweight="bold")
    ax.set_title("Edge-Constrained Random Forest Confusion Matrix\n(80 Trees, Max Depth 7 | Ottawa Dataset)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n[Figure] Confusion matrix successfully saved to: {os.path.abspath(output_path)}")

    # Show figure if running in interactive graphical environment
    try:
        if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps"]:
            plt.show()
    except Exception:
        pass
    plt.close()

def export_model_for_mcu_quantization(
    clf: RandomForestClassifier,
    classes: list,
    output_dir: str = "results",
    model_name: str = "model_ottawa_rf"
):
    """
    Exports model weights into:
    1. Standard serialized joblib file (.joblib)
    2. Structured JSON tree architecture (.json) detailing node thresholds, split features,
       and class vote leaves for embedded quantization and C-struct code generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib_path = os.path.join(output_dir, f"{model_name}.joblib")
    json_path = os.path.join(output_dir, f"{model_name}_weights.json")
    pkl_path = "ottawa_synced_model.pkl"
    output_pkl_path = os.path.join(output_dir, "ottawa_synced_model.pkl")

    # 1. Save standard pickle model
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n[Export] Saved trained model pickle to: {os.path.abspath(pkl_path)}")

    # 2. Save standard joblib
    joblib.dump(clf, joblib_path)
    print(f"[Export] Saved serializable model to: {os.path.abspath(joblib_path)}")

    # 3. Extract full node-level architecture for all trees
    total_nodes = 0
    total_leaves = 0
    trees_structure = []

    for tree_idx, dt in enumerate(clf.estimators_):
        t = dt.tree_
        node_count = int(t.node_count)
        total_nodes += node_count
        
        # Leaf detection: feature == -2
        is_leaf = (t.feature == -2)
        total_leaves += int(np.sum(is_leaf))

        # Calculate dominant class per node
        class_votes = t.value.squeeze(axis=1) # shape (node_count, n_classes)
        leaf_classes = np.argmax(class_votes, axis=1).tolist()

        tree_dict = {
            "tree_id": tree_idx,
            "node_count": node_count,
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "split_feature": t.feature.tolist(),
            "threshold": [float(round(val, 6)) for val in t.threshold],
            "leaf_prediction": leaf_classes
        }
        trees_structure.append(tree_dict)

    export_payload = {
        "model_type": "RandomForestClassifier",
        "n_estimators": len(clf.estimators_),
        "max_depth": clf.max_depth,
        "n_features": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "classes": classes,
        "total_nodes_all_trees": total_nodes,
        "total_leaves_all_trees": total_leaves,
        "mcu_flash_estimate_bytes": total_nodes * 12,  # 12 bytes per struct node (int8 feat, int8/16 child, float16/int16 thresh)
        "trees": trees_structure
    }

    with open(json_path, "w") as f:
        json.dump(export_payload, f, indent=2)

    est_flash_kb = (total_nodes * 12) / 1024.0
    print(f"[Export] Saved MCU quantization weights JSON to: {os.path.abspath(json_path)}")
    print(f"[Embedded Stats] Total Nodes across 80 trees: {total_nodes:,} ({total_leaves:,} leaves)")
    print(f"[Embedded Stats] Analytical Flash Footprint: ~{est_flash_kb:.2f} KB (Target Limit: <= 100 KB)")

def main():
    parser = argparse.ArgumentParser(
        description="Train embedded-constrained Random Forest on University of Ottawa preprocessed features."
    )
    parser.add_argument(
        "--data-path",
        default="training_features_ottawa_synced.csv",
        help="Path to preprocessed features CSV (default: training_features_ottawa_synced.csv)"
    )
    parser.add_argument(
        "--classes",
        choices=["all", "core"],
        default="all",
        help="Class selection mode: 'all' (all 8 Ottawa classes) or 'core' (healthy, bearing, imbalance; default: all)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--group-split",
        action="store_true",
        help="Split by recording file (GroupShuffleSplit) instead of random stratified window split"
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=80,
        help="Number of Random Forest trees (maximum <= 80; default: 80)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=7,
        help="Maximum tree depth (maximum <= 7; default: 7)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save confusion matrix and model weights (default: results)"
    )
    args = parser.parse_args()

    # Step 1: Load and split dataset
    X_train, X_test, y_train, y_test, classes = load_and_preprocess_data(
        csv_path=args.data_path,
        classes_mode=args.classes,
        test_size=args.test_size,
        group_split=args.group_split,
        random_seed=args.seed
    )

    # Step 2: Train constrained Random Forest
    clf = train_constrained_random_forest(
        X_train=X_train,
        y_train=y_train,
        n_estimators=args.trees,
        max_depth=args.max_depth,
        random_seed=args.seed
    )

    # Step 3: Calculate and print metrics (Macro-F1 score)
    y_pred, macro_f1 = evaluate_model(clf, X_test, y_test, classes)

    # Step 4: Generate, display, and export confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix_ottawa_rf.png")
    plot_and_display_confusion_matrix(y_test, y_pred, classes, output_path=cm_path)

    # Step 5: Export model weights for MCU quantization
    export_model_for_mcu_quantization(clf, classes, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
