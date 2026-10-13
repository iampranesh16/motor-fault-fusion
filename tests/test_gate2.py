import numpy as np
from src.generator import generate_signals
from src.dsp import estimate_fe, compute_acoustic_order_spectrum, compute_spatial_invariants

def test_gate2_fe_estimation_accuracy():
    for true_fe in [45.3, 50.0, 54.7]:
        data = generate_signals(fault_class="bearing", fe=true_fe, snr_db=10.0, duration=1.0, seed=101)
        est_fe = estimate_fe(data["hall"], data["meta"]["fs_hall"])
        err = abs(est_fe - true_fe)
        assert err < 0.05, f"fe error {err:.4f} Hz >= 0.05 Hz"

def test_gate2_order_axis_speed_invariance():
    bpfo_order_true = 3.585
    for fe in [40.0, 45.0, 50.0, 55.0, 60.0]:
        data = generate_signals(fault_class="bearing", fe=fe, slip=0.02, severity=0.9, snr_db=15.0, duration=1.0)
        est_fe = estimate_fe(data["hall"], data["meta"]["fs_hall"])
        fr_est = est_fe * (1.0 - 0.02) / 2.0
        
        orders, order_mag = compute_acoustic_order_spectrum(data["acoustic"], data["meta"]["fs_acoustic"], fr_est)
        mask = (orders >= 2.5) & (orders <= 4.5)
        peak_order = orders[mask][np.argmax(order_mag[mask])]
        
        order_err = abs(peak_order - bpfo_order_true)
        assert order_err < 0.02, f"At fe={fe}Hz, peak order {peak_order:.3f} off from {bpfo_order_true}"

def test_gate2_spatial_invariance_100_thetas():
    rng = np.random.default_rng(42)
    thetas = rng.uniform(0, 2 * np.pi, 100)
    
    clean_vals, noisy_vals = [], []
    for idx, th in enumerate(thetas):
        d_clean = generate_signals(fault_class="imbalance", fe=50.0, severity=0.8, theta0=th, snr_db=80.0, duration=1.0, seed=idx)
        inv_clean = compute_spatial_invariants(d_clean["hall"], d_clean["meta"]["fs_hall"], 50.0, d_clean["meta"]["fr"])
        clean_vals.append(inv_clean["array_rms_invariant"])
        
        d_noisy = generate_signals(fault_class="imbalance", fe=50.0, severity=0.8, theta0=th, snr_db=10.0, duration=1.0, seed=idx)
        inv_noisy = compute_spatial_invariants(d_noisy["hall"], d_noisy["meta"]["fs_hall"], 50.0, d_noisy["meta"]["fr"])
        noisy_vals.append(inv_noisy["array_rms_invariant"])
        
    clean_cv = (np.std(clean_vals) / np.mean(clean_vals)) * 100.0
    noisy_cv = (np.std(noisy_vals) / np.mean(noisy_vals)) * 100.0
    print(f"\n[Spatial Invariance] Noise-free CV: {clean_cv:.4f}%, 10dB Noisy CV: {noisy_cv:.2f}%")
    assert clean_cv < 1.0, f"Noise-free spatial variation {clean_cv:.3f}% >= 1.0%"

def test_gate2_no_sync_baseline_fails_on_speed_change():
    bpfo_order_true = 3.585
    data = generate_signals(fault_class="bearing", fe=42.0, slip=0.02, severity=0.9, snr_db=20.0, duration=1.0)
    fr_assumed_fixed = 50.0 * (1.0 - 0.02) / 2.0
    
    orders_nosync, mag_nosync = compute_acoustic_order_spectrum(
        data["acoustic"], data["meta"]["fs_acoustic"], fr_assumed_fixed
    )
    mask = (orders_nosync >= 2.5) & (orders_nosync <= 4.5)
    peak_order_nosync = orders_nosync[mask][np.argmax(mag_nosync[mask])]
    
    err_nosync = abs(peak_order_nosync - bpfo_order_true)
    assert err_nosync > 0.4, f"No-sync baseline unexpectedly close to BPFO: err={err_nosync:.3f}"
