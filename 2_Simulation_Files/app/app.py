import os
import sys
# Add the root directory to Python's path so Streamlit Cloud can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
import scipy.io as sio
from scipy import signal as scipy_signal
from scipy.interpolate import CubicSpline
from src.generator import generate_signals
from src.dsp import estimate_fe, compute_acoustic_order_spectrum, compute_spatial_invariants
from src.features import extract_features
from src.evaluate import build_held_out_datasets, CLASSES
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Hall + Acoustic Motor Fault Fusion", layout="wide")

@st.cache_resource
def load_trained_models():
    train_data, _ = build_held_out_datasets(n_train_per_class=80, n_test_per_class=10, seed=42)
    models = {}
    for mode in ["naive_fusion", "full_proposed"]:
        X_tr = np.array([extract_features(r[0], mode=mode) for r in train_data])
        y_tr = np.array([r[1] for r in train_data])
        models[mode] = RandomForestClassifier(n_estimators=80, max_depth=7, random_state=42).fit(X_tr, y_tr)
    return models

st.title("Non-Invasive Motor Fault Diagnosis: Hall Stray-Flux + Acoustic Order Fusion")
st.caption("Ottawa real-audio benchmark with a simulated six-sensor Hall overlay, plus the original simulation sandbox.")

FEATURE_COLUMNS = [
    "ac_bpfo", "ac_bpfo_2x", "ac_1x", "ac_2x", "ac_rms", "ac_kurt",
    "mag_array_rms", "mag_spatial_mode1", "mag_sideband_ratio",
    "mag_spatial_quad", "consistency_gate",
]
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OTTAWA_DIR = os.path.join(PROJECT_ROOT, "2_Simulation_Files", "ottawa_benchmark")

@st.cache_data
def load_ottawa_features():
    return pd.read_csv(os.path.join(OTTAWA_DIR, "dashboard_artifacts", "ottawa_grouped_test_predictions.csv"))

@st.cache_resource
def load_ottawa_models():
    folder = os.path.join(OTTAWA_DIR, "dashboard_artifacts")
    return (joblib.load(os.path.join(folder, "ottawa_synced_grouped_model.pkl")),
            joblib.load(os.path.join(folder, "ottawa_unsynced_grouped_model.pkl")))

@st.cache_data
def load_benchmark():
    with open(os.path.join(OTTAWA_DIR, "dashboard_artifacts", "ottawa_grouped_benchmark.json"), encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_ottawa_audio_window(file_name, window_idx):
    data_root = Path(PROJECT_ROOT) / "dataset_ottawa"
    matches = list(data_root.rglob(file_name))
    if not matches:
        raise FileNotFoundError(f"Ottawa recording not found: {file_name}")
    raw = sio.loadmat(matches[0])["data"]
    acoustic_full = raw[:, 1] if raw.ndim == 2 else raw.flatten()
    fs = 16000
    acoustic_full = scipy_signal.resample(acoustic_full, 10 * fs)
    start = int(window_idx) * fs
    return acoustic_full[start:start + fs]

@st.cache_data
def ottawa_order_spectra(file_name, window_idx, speed_index):
    """Compute fixed-reference and speed-synchronized envelope spectra for one raw Ottawa window."""
    acoustic = load_ottawa_audio_window(file_name, window_idx)
    fs = 16000
    sos = scipy_signal.butter(4, [2000, 3600], btype="bandpass", fs=fs, output="sos")
    envelope = np.abs(scipy_signal.hilbert(scipy_signal.sosfiltfilt(sos, acoustic)))
    envelope -= envelope.mean()
    window = np.hanning(len(envelope))
    magnitude = np.abs(np.fft.rfft(envelope * window)) / (window.sum() + 1e-12)
    freq = np.fft.rfftfreq(len(envelope), 1 / fs)
    unsync_order = freq / 24.5

    idx = int(window_idx)
    speed_index = int(speed_index)
    if speed_index == 1:
        f_start, alpha = 15.0, 0.0
    elif speed_index == 2:
        f_start, alpha = 30.0, 0.0
    elif speed_index == 3:
        f_start, alpha = 45.0, 0.0
    elif speed_index == 4:
        f_start, alpha = 60.0, 0.0
    elif speed_index == 5:
        f_start, alpha = 15.0 + 3.0 * idx, 3.0
    elif speed_index == 6:
        f_start, alpha = 30.0 + 3.0 * idx, 3.0
    elif speed_index == 7:
        f_start, alpha = 45.0 - 3.0 * idx, -3.0
    else:
        f_start, alpha = 60.0 - 3.0 * idx, -3.0
    t_orig = np.arange(len(envelope)) / fs
    theta = 2.0 * np.pi * (f_start * t_orig + 0.5 * alpha * t_orig ** 2)
    theta_uniform = np.linspace(theta[0], theta[-1], len(envelope), endpoint=False)
    delta_turns = (theta_uniform - theta[0]) / (2.0 * np.pi)
    if abs(alpha) > 1e-6:
        t_target = (-f_start + np.sqrt(np.maximum(0.0, f_start ** 2 + 2.0 * alpha * delta_turns))) / alpha
    else:
        t_target = delta_turns / f_start
    t_target = np.clip(t_target, 0.0, 1.0 - 1.0 / fs)
    synced_env = CubicSpline(t_orig, envelope, extrapolate=True)(t_target)
    synced_window = np.hanning(len(synced_env))
    synced_mag = np.abs(np.fft.rfft((synced_env - synced_env.mean()) * synced_window)) / (synced_window.sum() + 1e-12)
    rotations = f_start + 0.5 * alpha
    synced_order = np.fft.rfftfreq(len(synced_env), d=1.0 / len(synced_env)) / rotations
    return unsync_order, magnitude, synced_order, synced_mag

def simulate_ottawa_hall_overlay(fault_class, fr_reference, slip, severity, theta_deg, snr_db, seed=42):
    """Create a clearly synthetic six-channel Hall illustration at the Ottawa window's mean speed."""
    fs_hall = 10000
    n_samples = fs_hall
    t = np.arange(n_samples) / fs_hall
    fe = 2.0 * fr_reference / (1.0 - slip)
    theta0 = np.deg2rad(theta_deg)
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    hall = np.zeros((6, n_samples), dtype=float)

    # These stylized amplitudes are for an illustrative overlay only; Ottawa has no Hall measurements.
    fault_sideband = {
        "healthy": 0.0,
        "bearing": 0.05,
        "imbalance": 0.25,
        "bowed_rotor": 0.14,
        "broken_rotor_bar": 0.18,
        "misalignment": 0.20,
        "stator_winding": 0.16,
        "voltage_unbalance": 0.12,
    }.get(str(fault_class), 0.12) * float(severity)

    for i, angle in enumerate(angles):
        spatial_angle = angle + theta0
        flux = np.cos(spatial_angle) * np.sin(2.0 * np.pi * fe * t)
        if fault_sideband:
            modulation = np.cos(2.0 * spatial_angle)
            flux += fault_sideband * modulation * (
                np.sin(2.0 * np.pi * (fe - fr_reference) * t)
                + np.sin(2.0 * np.pi * (fe + fr_reference) * t)
            )
        hall[i] = flux

    rng = np.random.default_rng(seed)
    noise_std = np.sqrt(np.mean(hall ** 2) / (10.0 ** (float(snr_db) / 10.0)) + 1e-12)
    hall += rng.normal(0.0, noise_std, hall.shape)
    return {"hall": hall, "fs_hall": fs_hall, "fe_overlay": fe}

with st.sidebar:
    real_data = st.toggle("Data Source: Simulation vs. Ottawa Hybrid", value=True)
    if real_data:
        ottawa_df = load_ottawa_features()
        if "ottawa_sample" not in st.session_state or st.session_state.ottawa_sample >= len(ottawa_df):
            st.session_state.ottawa_sample = int(np.random.randint(len(ottawa_df)))
        if st.button("Sample another held-out Ottawa window"):
            st.session_state.ottawa_sample = int(np.random.randint(len(ottawa_df)))
        sample = ottawa_df.iloc[st.session_state.ottawa_sample]
        st.header("1. Real Ottawa recording")
        st.caption(f"{sample['file_name']} · window {int(sample['window_idx'])} · label: {str(sample['fault_class']).replace('_', ' ')} · mean speed: {float(sample['speed_hz_mean']):.2f} Hz")
        st.header("2. Synthetic Hall overlay")
        st.caption("These controls affect only the six simulated Hall channels. Ottawa audio and its label stay fixed.")
        slip_slider = st.slider("Motor Slip (s)", 0.005, 0.050, 0.020, 0.005)
        severity_slider = st.slider("Fault Severity", 0.1, 1.0, 0.85, 0.05)
        theta_deg = st.slider("Hall Array Mounting Angle theta0 (deg)", 0, 360, 78, 5)
        snr_slider = st.slider("Ambient SNR (dB)", -5.0, 25.0, 10.0, 1.0)
        sync_enabled = st.toggle("Enable Electrical Frequency Sync & Spatial-DFT", value=True)
        st.caption("Supply electrical frequency is derived for the overlay from Ottawa's mean shaft speed and the slip assumption.")
    else:
        st.header("1. Motor & Fault Parameters")
        fault_class = st.selectbox("Injected Condition", ["bearing", "imbalance", "healthy"], index=0)
        fe_slider = st.slider("Supply Electrical Freq f_e (Hz)", 38.0, 62.0, 43.0, 0.5)
        slip_slider = st.slider("Motor Slip (s)", 0.005, 0.050, 0.020, 0.005)
        severity_slider = st.slider("Fault Severity", 0.1, 1.0, 0.85, 0.05) if fault_class != "healthy" else 0.0
        st.header("2. Perturbations (Orientation & Noise)")
        theta_deg = st.slider("Hall Array Mounting Angle theta0 (deg)", 0, 360, 78, 5)
        snr_slider = st.slider("Ambient SNR (dB)", -5.0, 25.0, 10.0, 1.0)
        sync_enabled = st.toggle("Enable Electrical Frequency Sync & Spatial-DFT", value=True)

if real_data:
    sync_model, unsync_model = load_ottawa_models()
    acoustic_columns = FEATURE_COLUMNS[:6]
    sync_row = sample[[f"sync__{c}" for c in acoustic_columns]].astype(float).to_frame().T
    sync_row.columns = acoustic_columns
    unsync_row = sample[[f"unsync__{c}" for c in acoustic_columns]].astype(float).to_frame().T
    unsync_row.columns = acoustic_columns
    sync_probs = sync_model.predict_proba(sync_row)[0]
    unsync_probs = unsync_model.predict_proba(unsync_row)[0]
    sync_prediction = str(sync_model.classes_[int(np.argmax(sync_probs))]).upper()
    unsync_prediction = str(unsync_model.classes_[int(np.argmax(unsync_probs))]).upper()
    pred_label = sync_prediction
    conf_pct = float(np.max(sync_probs)) * 100.0
    true_label = str(sample["fault_class"]).replace("_", " ").upper()
    fr_ottawa = float(sample["speed_hz_mean"])
    hall_overlay = simulate_ottawa_hall_overlay(
        str(sample["fault_class"]), fr_ottawa, slip_slider,
        severity_slider, theta_deg, snr_slider,
        seed=1000 + int(sample["window_idx"]),
    )
    data = {
        "acoustic": load_ottawa_audio_window(sample["file_name"], int(sample["window_idx"])),
        "hall": hall_overlay["hall"],
        "meta": {"fs_acoustic": 16000, "fs_hall": hall_overlay["fs_hall"]},
    }
    est_fe = estimate_fe(data["hall"], data["meta"]["fs_hall"], f_min=10.0, f_max=150.0)
    active_fe = est_fe if sync_enabled else 50.0
    active_slip = slip_slider if sync_enabled else 0.02
    fr_ref = active_fe * (1.0 - active_slip) / 2.0
    sp = compute_spatial_invariants(data["hall"], data["meta"]["fs_hall"], active_fe, fr_ref)
    sp_value = sp["array_rms_invariant"]
    fe_display = hall_overlay["fe_overlay"]
    fr_display = fr_ref
    theta_display = f"{theta_deg}° (simulated)"
    benchmark = load_benchmark()
    sync_metrics, unsync_metrics = benchmark["synced"], benchmark["unsynced"]
    st.info("Hybrid demo: the microphone window and fault label are real Ottawa data; the six Hall channels are simulated from the known label, Ottawa mean speed, and the overlay sliders. The Hall overlay is illustrative and is excluded from the Ottawa predictions and benchmark scores because using a label-conditioned simulation as a model input would leak the answer. Ottawa has no measured stray-flux Hall channels.")
    left, right = st.columns(2)
    left.metric("Ottawa acoustic · speed synchronized", sync_prediction, f"{np.max(sync_probs)*100:.1f}% uncalibrated score · {'Correct' if sync_prediction.replace('_', ' ') == true_label else 'Incorrect'}")
    right.metric("Ottawa acoustic · fixed reference", unsync_prediction, f"{np.max(unsync_probs)*100:.1f}% uncalibrated score · {'Correct' if unsync_prediction.replace('_', ' ') == true_label else 'Incorrect'}")
    st.caption(f"Same held-out window: {sample['file_name']} · window {int(sample['window_idx'])} · true label: {true_label}")
    m1, m2 = st.columns(2)
    m1.metric("Sync accuracy / macro-F1", f"{sync_metrics['accuracy']*100:.1f}%", f"Macro-F1 {sync_metrics['macro_f1']:.3f}")
    m2.metric("Baseline accuracy / macro-F1", f"{unsync_metrics['accuracy']*100:.1f}%", f"Macro-F1 {unsync_metrics['macro_f1']:.3f}")
    st.caption(f"Acoustic-only matched recording-level holdout: {benchmark['n_test_windows']} windows from {benchmark['n_test_recordings']} unseen recordings; both use 80 trees, max depth 7, the same six acoustic features, and the same split. These metrics do not measure Hall fusion.")
    with st.expander("Held-out confusion matrices"):
        cm_left, cm_right = st.columns(2)
        cm_left.write("Synchronized")
        cm_left.dataframe(pd.DataFrame(sync_metrics["confusion_matrix"], index=sync_metrics["classes"], columns=sync_metrics["classes"]), use_container_width=True)
        cm_right.write("Fixed-reference baseline")
        cm_right.dataframe(pd.DataFrame(unsync_metrics["confusion_matrix"], index=unsync_metrics["classes"], columns=unsync_metrics["classes"]), use_container_width=True)
else:
    models = load_trained_models()
    data = generate_signals(
        fault_class=fault_class, fe=fe_slider, slip=slip_slider,
        severity=severity_slider, theta0=np.deg2rad(theta_deg), snr_db=snr_slider,
        duration=1.0, realistic_interference=True, seed=123
    )
    est_fe = estimate_fe(data["hall"], data["meta"]["fs_hall"])
    active_fe = est_fe if sync_enabled else 50.0
    active_slip = slip_slider if sync_enabled else 0.02
    fr_ref = active_fe * (1.0 - active_slip) / 2.0
    sp = compute_spatial_invariants(data["hall"], data["meta"]["fs_hall"], active_fe, fr_ref)
    active_mode = "full_proposed" if sync_enabled else "naive_fusion"
    feats = extract_features(data, mode=active_mode, nominal_slip=active_slip).reshape(1, -1)
    probs = models[active_mode].predict_proba(feats)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx].upper()
    conf_pct = probs[pred_idx] * 100.0
    true_label = fault_class.upper()
    fe_display, fr_display, sp_value, theta_display = fe_slider, fr_ref, sp["array_rms_invariant"], f"{theta_deg}°"

# Big visual alert banner for the plant technician
if real_data:
    st.info(f"Ottawa acoustic-only diagnosis: **{pred_label}** · uncalibrated model score **{conf_pct:.1f}%** · recording label **{true_label}**. The simulated Hall overlay is not used for this prediction.")
elif pred_label == "HEALTHY":
    if true_label == "HEALTHY":
        st.success(f"✅ **SYSTEM STATUS: NORMAL (HEALTHY)** — No motor faults detected. | Confidence: **{conf_pct:.1f}%** | Ground Truth: **{true_label}**")
    else:
        st.warning(f"⚠️ **MISSED FAULT (FALSE NEGATIVE): Predicted HEALTHY, but actual motor has {true_label} FAULT!** — Turn ON Sync & Spatial-DFT to fix! | Confidence: **{conf_pct:.1f}%**")
else:
    if pred_label == true_label:
        st.error(f"🚨 **FAULT DETECTED: {pred_label} FAULT** — Immediate maintenance required! | Severity: **{int(severity_slider*100)}%** | Confidence: **{conf_pct:.1f}%**")
    else:
        st.warning(f"⚠️ **MISCLASSIFICATION: Predicted {pred_label} FAULT (Actual: {true_label})** | Confidence: **{conf_pct:.1f}%**")

if real_data:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ottawa shaft-speed reference", f"{fr_ottawa:.2f} Hz", "Real dataset metadata")
    col2.metric("Hall overlay supply frequency", f"{fe_display:.2f} Hz", f"Hall estimate: {est_fe:.2f} Hz")
    col3.metric("Active order-tracking speed", f"{fr_display:.2f} Hz", "Hall synchronized" if sync_enabled else "Fixed 50 Hz assumption")
    col4.metric("Simulated Spatial-DFT invariant", f"{sp_value:.3f}", f"Array angle: {theta_display}")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True / Estimated f_e", f"{fe_slider:.2f} Hz / {est_fe:.2f} Hz", f"Err: {abs(est_fe - fe_slider):.3f} Hz")
    col2.metric("Derived Rotor Speed f_r", f"{fr_display:.2f} Hz", "Synced to Hall" if sync_enabled else "Fixed 50Hz Assumed")
    col3.metric("Predicted Diagnosis", pred_label, f"Confidence: {conf_pct:.1f}%")
    col4.metric("Spatial-DFT Invariant", f"{sp_value:.3f}", f"Array Angle: {theta_display}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if real_data:
    unsync_order, unsync_mag, sync_order, sync_mag = ottawa_order_spectra(sample["file_name"], int(sample["window_idx"]), int(sample["speed_index"]))
    umask = (unsync_order >= 0.5) & (unsync_order <= 10.0)
    smask = (sync_order >= 0.5) & (sync_order <= 10.0)
    axes[0].plot(unsync_order[umask], unsync_mag[umask], color="#d62728", alpha=0.8, label="Fixed 24.5 Hz reference")
    axes[0].plot(sync_order[smask], sync_mag[smask], color="#2ca02c", linewidth=2, label="Speed-synchronized angular resampling")
    axes[0].set_xlim(0.5, 10.0)
else:
    orders, o_mag = compute_acoustic_order_spectrum(data["acoustic"], data["meta"]["fs_acoustic"], fr_ref)
    mask = (orders >= 2.0) & (orders <= 5.0)
    axes[0].plot(orders[mask], o_mag[mask], color="#2ca02c" if sync_enabled else "#d62728", linewidth=2)
axes[0].axvline(3.585, color="black", linestyle="--", label="Expected BPFO (3.585x)")
axes[0].set_title("Ottawa Acoustic Envelope Order Spectrum" if real_data else f"Acoustic Order Spectrum ({'SYNC ON' if sync_enabled else 'SYNC OFF - Peak Shifted!'})")
axes[0].set_xlabel("Shaft Order (multiples of f_r)")
axes[0].set_ylabel("Envelope Order Magnitude")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

if real_data:
    sensor_rms = np.sqrt(np.mean(data["hall"] ** 2, axis=1))
    axes[1].bar([f"S{i+1}" for i in range(6)], sensor_rms, color="#1f77b4", alpha=0.7, label="Simulated Hall channel RMS")
    axes[1].axhline(sp["array_rms_invariant"] * np.sqrt(2), color="crimson", linestyle="--", linewidth=2, label="Simulated Spatial-DFT reference")
    axes[1].set_title(f"Simulated 6-Sensor Hall Profile (theta0 = {theta_deg}°)")
else:
    sensor_rms = np.sqrt(np.mean(data["hall"] ** 2, axis=1))
    axes[1].bar([f"S{i+1}" for i in range(6)], sensor_rms, color="#1f77b4", alpha=0.7, label="Individual Sensor RMS (Varies with theta0)")
    axes[1].axhline(sp["array_rms_invariant"] * np.sqrt(2), color="crimson", linestyle="--", linewidth=2, label="Spatial-DFT Invariant Reference")
    axes[1].set_title(f"6-Sensor Circumferential Hall Array Profile (theta0 = {theta_deg}°)")
axes[1].set_ylabel("RMS Flux Amplitude")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
if real_data:
    st.caption("Right-hand Hall bars and Spatial-DFT reference are simulated, using Ottawa's known label and mean-speed metadata. The six Hall channels are not present in the Ottawa recording.")

if os.path.exists("results/ablation.csv"):
    st.subheader("Held-Out Condition Ablation Summary (`results/ablation.csv`)")
    st.dataframe(pd.read_csv("results/ablation.csv"), use_container_width=True)
