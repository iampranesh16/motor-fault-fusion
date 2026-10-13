import os
import pandas as pd
from src.evaluate import run_ablation_study

def test_gate3_ablation_and_heldout_metrics():
    df = run_ablation_study(save_dir="results")
    
    # 1. Verify results/ablation.csv exists
    assert os.path.exists("results/ablation.csv"), "results/ablation.csv was not generated"
    
    # 2. Verify confidence intervals and required columns exist
    required_cols = {"mode", "macro_f1", "ci_95_low", "ci_95_high", "persisted_fa_per_hour"}
    assert required_cols.issubset(set(df.columns)), f"Missing columns in {df.columns}"
    
    # 3. Verify full_proposed hits >= 90% Macro-F1 on held-out conditions and beats uncalibrated baselines
    full_row = df[df["mode"] == "full_proposed"].iloc[0]
    naive_row = df[df["mode"] == "naive_fusion"].iloc[0]
    
    assert full_row["macro_f1"] >= 0.90, f"Proposed Macro-F1 {full_row['macro_f1']:.3f} < 0.90 target"
    assert full_row["macro_f1"] > naive_row["macro_f1"], "Proposed method did not outperform naive unsynced fusion"

def test_gate3_no_fault_healthy_case():
    """Verifies that a healthy motor is accurately classified as HEALTHY (class 0) with no false alarm."""
    from src.generator import generate_signals
    from src.features import extract_features
    from src.evaluate import build_held_out_datasets
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    train_data, _ = build_held_out_datasets(n_train_per_class=80, n_test_per_class=10, seed=42)
    X_tr = np.array([extract_features(r[0], mode="full_proposed") for r in train_data])
    y_tr = np.array([r[1] for r in train_data])
    clf = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=42).fit(X_tr, y_tr)

    # Test healthy motor across 3 unseen speeds and orientations
    for fe_test, theta_test in [(42.5, 1.1), (50.0, 2.7), (57.5, 4.8)]:
        healthy_sig = generate_signals(
            fault_class="healthy", fe=fe_test, slip=0.02, severity=0.0,
            theta0=theta_test, snr_db=10.0, realistic_interference=True, seed=999
        )
        feats = extract_features(healthy_sig, mode="full_proposed").reshape(1, -1)
        pred_class = int(clf.predict(feats)[0])
        assert pred_class == 0, f"False alarm triggered on healthy motor at fe={fe_test}Hz!"

def test_gate3_no_fault_healthy_case():
    """Verifies that a healthy motor is accurately classified as HEALTHY (class 0) with no false alarm."""
    from src.generator import generate_signals
    from src.features import extract_features
    from src.evaluate import build_held_out_datasets
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    train_data, _ = build_held_out_datasets(n_train_per_class=80, n_test_per_class=10, seed=42)
    X_tr = np.array([extract_features(r[0], mode="full_proposed") for r in train_data])
    y_tr = np.array([r[1] for r in train_data])
    clf = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=42).fit(X_tr, y_tr)

    # Test healthy motor across 3 unseen speeds and orientations
    for fe_test, theta_test in [(42.5, 1.1), (50.0, 2.7), (57.5, 4.8)]:
        healthy_sig = generate_signals(
            fault_class="healthy", fe=fe_test, slip=0.02, severity=0.0,
            theta0=theta_test, snr_db=10.0, realistic_interference=True, seed=999
        )
        feats = extract_features(healthy_sig, mode="full_proposed").reshape(1, -1)
        pred_class = int(clf.predict(feats)[0])
        assert pred_class == 0, f"False alarm triggered on healthy motor at fe={fe_test}Hz!"
