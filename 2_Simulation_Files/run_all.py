import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.generator import generate_signals
from src.dsp import estimate_fe, compute_acoustic_order_spectrum, compute_spatial_invariants
from src.features import extract_features
from src.evaluate import run_ablation_study, build_held_out_datasets, CLASSES

def generate_all_figures_and_reports():
    os.makedirs("results", exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    print("[1/5] Generating Gate 1 Figure (results/fig1_gate1_generator.png)...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fe, col in zip([42.0, 50.0, 58.0], ["#1f77b4", "#2ca02c", "#d62728"]):
        d = generate_signals(fault_class="bearing", fe=fe, severity=0.9, snr_db=20.0, duration=1.0)
        fs = d["meta"]["fs_acoustic"]
        b, a = butter(4, [2000/(fs/2), 3600/(fs/2)], btype="band")
        env = np.abs(hilbert(filtfilt(b, a, d["acoustic"])))
        env -= np.mean(env)
        freqs = np.fft.rfftfreq(len(env), 1/fs)
        mag = np.abs(np.fft.rfft(env)) / len(env)
        mask = (freqs >= 40) & (freqs <= 130)
        axes[0].plot(freqs[mask], mag[mask], color=col, label=f"fe={fe:.0f}Hz (BPFO={d['meta']['bpfo']:.1f}Hz)")
        axes[0].axvline(d["meta"]["bpfo"], color=col, linestyle="--", alpha=0.5)
    axes[0].set_title("Raw Acoustic Envelope Spectrum (Speed-Dependent Shift)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Envelope Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    d_imb = generate_signals(fault_class="imbalance", fe=50.0, severity=0.85, snr_db=25.0, duration=2.0)
    fs_h = d_imb["meta"]["fs_hall"]
    freqs_h = np.fft.rfftfreq(len(d_imb["hall"][0]), 1/fs_h)
    mag_h = 20 * np.log10(np.abs(np.fft.rfft(d_imb["hall"][0] * np.hanning(len(d_imb["hall"][0])))) / len(d_imb["hall"][0]) + 1e-12)
    mask_h = (freqs_h >= 15) & (freqs_h <= 85)
    axes[1].plot(freqs_h[mask_h], mag_h[mask_h], color="#ff7f0e")
    for f_m, lbl in [(50.0, "fe (50Hz)"), (50.0 - d_imb["meta"]["fr"], "fe - fr"), (50.0 + d_imb["meta"]["fr"], "fe + fr")]:
        axes[1].axvline(f_m, color="crimson", linestyle="--", alpha=0.7, label=lbl)
    axes[1].set_title("Hall Stray-Flux Spectrum (Fundamental & Rotor Sidebands)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/fig1_gate1_generator.png", dpi=180)
    plt.close()

    print("[2/5] Generating Gate 2 Figure (results/fig2_gate2_sync_and_spatial.png)...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fe, col in zip([42.0, 50.0, 58.0], ["#1f77b4", "#2ca02c", "#d62728"]):
        d = generate_signals(fault_class="bearing", fe=fe, severity=0.9, snr_db=15.0, duration=1.0)
        est_fe = estimate_fe(d["hall"], d["meta"]["fs_hall"])
        fr_est = est_fe * (1.0 - 0.02) / 2.0
        orders, o_mag = compute_acoustic_order_spectrum(d["acoustic"], d["meta"]["fs_acoustic"], fr_est)
        mask_o = (orders >= 2.5) & (orders <= 4.5)
        axes[0].plot(orders[mask_o], o_mag[mask_o], color=col, label=f"Synced fe={fe:.0f}Hz")
    # Also plot No-Sync failure at 42 Hz
    d_42 = generate_signals(fault_class="bearing", fe=42.0, severity=0.9, snr_db=15.0, duration=1.0)
    orders_ns, o_mag_ns = compute_acoustic_order_spectrum(d_42["acoustic"], d_42["meta"]["fs_acoustic"], 50.0 * 0.98 / 2.0)
    mask_ns = (orders_ns >= 2.5) & (orders_ns <= 4.5)
    axes[0].plot(orders_ns[mask_ns], o_mag_ns[mask_ns], color="gray", linestyle=":", linewidth=2, label="No-Sync Baseline (42Hz)")
    axes[0].axvline(3.585, color="black", linestyle="--", label="True BPFO Order (3.585x)")
    axes[0].set_title("Order-Normalized Acoustic Spectrum (Sync ON vs OFF)")
    axes[0].set_xlabel("Harmonic Order (multiples of fr)")
    axes[0].set_ylabel("Normalized Magnitude")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, alpha=0.3)

    angles_deg = np.linspace(0, 360, 72)
    ch0_raw, spatial_inv = [], []
    for deg in angles_deg:
        th = np.deg2rad(deg)
        d_rot = generate_signals(fault_class="imbalance", fe=50.0, severity=0.8, theta0=th, snr_db=15.0, duration=1.0, seed=int(deg)+1)
        ch0_raw.append(np.sqrt(np.mean(d_rot["hall"][0] ** 2)))
        inv = compute_spatial_invariants(d_rot["hall"], d_rot["meta"]["fs_hall"], 50.0, d_rot["meta"]["fr"])
        spatial_inv.append(inv["array_rms_invariant"])
    axes[1].plot(angles_deg, ch0_raw / np.max(ch0_raw), "r--", label="Uncalibrated Single Sensor (Ch0 RMS)")
    axes[1].plot(angles_deg, spatial_inv / np.max(spatial_inv), "b-", linewidth=2.2, label="Proposed 6-Sensor Spatial-DFT Invariant")
    axes[1].set_title("Orientation Sensitivity Sweep across Array Rotation (theta0)")
    axes[1].set_xlabel("Global Array Rotation Angle theta0 (Degrees)")
    axes[1].set_ylabel("Normalized Feature Magnitude")
    axes[1].set_ylim(0, 1.15)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/fig2_gate2_sync_and_spatial.png", dpi=180)
    plt.close()

    print("[3/5] Running Held-Out Ablation & Generating Figure 3 (results/fig3_ablation_and_snr_sweep.png)...")
    df_abl, details = run_ablation_study(save_dir="results", return_details=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    modes_pretty = ["Acoustic\nOnly", "Hall\nOnly", "Naive\nFusion", "Fusion +\nSync", "Full Proposed\n(Sync+Spatial)"]
    y_f1 = df_abl["macro_f1"].values
    err_low = y_f1 - df_abl["ci_95_low"].values
    err_high = df_abl["ci_95_high"].values - y_f1
    bars = axes[0].bar(modes_pretty, y_f1, yerr=[err_low, err_high], capsize=5, color=["#8c564b", "#e377c2", "#7f7f7f", "#1f77b4", "#2ca02c"])
    axes[0].axhline(0.90, color="red", linestyle="--", alpha=0.7, label="Target (0.90 Macro-F1)")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Held-Out Macro-F1 Score")
    axes[0].set_title("Ablation Study on Held-Out Speeds & Orientations")
    for bar, val in zip(bars, y_f1):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.2f}", ha="center", fontweight="bold", fontsize=9)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, axis="y", alpha=0.3)

    # SNR Sweep (-5 dB to +20 dB)
    snr_levels = [-5, 0, 5, 10, 15, 20]
    train_recs, _ = build_held_out_datasets(n_train_per_class=80, n_test_per_class=20, seed=99)
    for mode, lbl, col, mk in [
        ("acoustic_only", "Acoustic Only (No Sync)", "#8c564b", "s"),
        ("naive_fusion", "Naive Fusion", "#7f7f7f", "^"),
        ("fusion_sync", "Fusion + Order Sync", "#1f77b4", "d"),
        ("full_proposed", "Full Proposed (Sync + Spatial)", "#2ca02c", "o")
    ]:
        X_tr = np.array([extract_features(r[0], mode=mode) for r in train_recs])
        y_tr = np.array([r[1] for r in train_recs])
        clf_m = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=42).fit(X_tr, y_tr)
        snr_f1s = []
        for snr_v in snr_levels:
            rng = np.random.default_rng(100 + snr_v)
            X_te, y_te = [], []
            for c_idx, cls in enumerate(CLASSES):
                for k in range(25):
                    fe_v = rng.choice([42.0, 57.0])
                    th_v = rng.uniform(0.5, 5.5)
                    sev_v = 0.0 if cls == "healthy" else 0.65
                    sig = generate_signals(cls, fe=fe_v, severity=sev_v, theta0=th_v, snr_db=float(snr_v), realistic_interference=True, seed=k*17+c_idx)
                    X_te.append(extract_features(sig, mode=mode))
                    y_te.append(c_idx)
            snr_f1s.append(f1_score(y_te, clf_m.predict(X_te), average="macro"))
        axes[1].plot(snr_levels, snr_f1s, marker=mk, color=col, linewidth=2, label=lbl)
    axes[1].set_title("Noise Robustness Sweep (SNR -5 dB to +20 dB)")
    axes[1].set_xlabel("Ambient Signal-to-Noise Ratio (dB)")
    axes[1].set_ylabel("Macro-F1 on Unseen Conditions")
    axes[1].set_ylim(0.2, 1.05)
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/fig3_ablation_and_snr_sweep.png", dpi=180)
    plt.close()

    print("[4/5] Generating Figure 4 Confusion Matrices (results/fig4_confusion_matrices.png)...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mode, title in zip(axes, ["naive_fusion", "full_proposed"], ["Naive Fusion (No Sync / Uncalibrated)", "Full Proposed (Order Sync + Spatial-DFT)"]):
        cm = details[mode]["cm"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), [c.capitalize() for c in CLASSES])
        ax.set_yticks(range(3), [c.capitalize() for c in CLASSES])
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > 40 else "black", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/fig4_confusion_matrices.png", dpi=180)
    plt.close()

    print("[5/5] Writing Embedded Resource Budget (results/embedded_budget.md)...")
    host_lat = df_abl[df_abl["mode"] == "full_proposed"]["host_latency_ms"].iloc[0]
    with open("results/embedded_budget.md", "w") as f:
        f.write(f"""# Embedded-Readiness Resource Budget (Estimated)

> **Disclosure:** All figures below are analytical embedded resource estimates for deploying the 1.0 s fast-path pipeline on an **STM32F407 / STM32G4 (Cortex-M4F @ 168 MHz)** or **ESP32-S3 (@ 240 MHz)** microcontroller. Host Python execution time per 1-second window was measured at **{host_lat:.2f} ms** by `run_all.py`.

## 1. Memory Footprint (SRAM & Flash)

| Buffer / Component | Calculation | Size (KB) | Target Limit |
|---|---|---|---|
| **Acoustic Ring Buffer** (16 kS/s, 1.0 s, `int16`) | `16,000 * 2 B` | 32.0 KB | |
| **Hall 6-Channel Buffer** (10 kS/s, 1.0 s, decimated 10x to 1 kS/s for fe & sidebands, `int16`) | `6 * 1,000 * 2 B` | 12.0 KB | |
| **FFT / Hilbert Scratchpad** (4096-pt real FFT in-place `float32`) | `4,096 * 4 B * 2` | 32.0 KB | |
| **Random Forest / Quantized Classifier** (80 trees, depth 7, `int8`/struct nodes) | `~1,800 nodes * 12 B` | 21.6 KB (Flash) | <= 100 KB |
| **Total Active SRAM** | Buffers + Scratchpad + Stack | **~82.0 KB** | **<= 256 KB** |
| **Total Flash (Firmware + CMSIS-DSP + Classifier)** | Code + Constants + Model | **~96.0 KB** | **<= 1,024 KB** |

## 2. Compute Latency per 1.0 s Window (Cortex-M4F @ 168 MHz)

| DSP Stage | Operations | Est. Cycles | Est. Time (ms) |
|---|---|---|---|
| **Hall Phasor Projection & Spatial-DFT** | 6 ch x 1,000 MACs + 6-pt DFT | ~45,000 | 0.27 ms |
| **Acoustic Band-Pass + Envelope + 16k FFT** | 4th-order IIR + Radix-2 RFFT | ~2,900,000 | 17.26 ms |
| **Order Axis Cubic Interpolation** | 2,000 order bins | ~120,000 | 0.71 ms |
| **Feature Extraction + Classifier Inference** | 11 features + 80 trees (depth 7) | ~15,000 | 0.09 ms |
| **Total Compute Latency per Window** | | **~3.08M cycles** | **~18.33 ms (< 2% CPU duty)** |
""")
    print("\n[SUCCESS] Master pipeline complete! All tables, plots, and budgets written to results/.")

if __name__ == "__main__":
    generate_all_figures_and_reports()
