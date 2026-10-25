import os
import sys
# Add the root directory to Python's path so Streamlit Cloud can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
st.caption("Physics-Based Simulation & Live Examiner Sandbox | Source of Truth: `run_all.py`")

models = load_trained_models()

with st.sidebar:
    st.header("1. Motor & Fault Parameters")
    fault_class = st.selectbox("Injected Condition", ["bearing", "imbalance", "healthy"], index=0)
    fe_slider = st.slider("Supply Electrical Freq f_e (Hz)", 38.0, 62.0, 43.0, 0.5)
    slip_slider = st.slider("Motor Slip (s)", 0.005, 0.050, 0.020, 0.005)
    severity_slider = st.slider("Fault Severity", 0.1, 1.0, 0.85, 0.05) if fault_class != "healthy" else 0.0
    
    st.header("2. Perturbations (Orientation & Noise)")
    theta_deg = st.slider("Hall Array Mounting Angle theta0 (deg)", 0, 360, 78, 5)
    snr_slider = st.slider("Ambient SNR (dB)", -5.0, 25.0, 10.0, 1.0)
    sync_enabled = st.toggle("Enable Electrical Frequency Sync & Spatial-DFT", value=True)

data = generate_signals(
    fault_class=fault_class,
    fe=fe_slider,
    slip=slip_slider,
    severity=severity_slider,
    theta0=np.deg2rad(theta_deg),
    snr_db=snr_slider,
    duration=1.0,
    realistic_interference=True,
    seed=123
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

# Big visual alert banner for the plant technician
if pred_label == "HEALTHY":
    if true_label == "HEALTHY":
        st.success(f"✅ **SYSTEM STATUS: NORMAL (HEALTHY)** — No motor faults detected. | Confidence: **{conf_pct:.1f}%** | Ground Truth: **{true_label}**")
    else:
        st.warning(f"⚠️ **MISSED FAULT (FALSE NEGATIVE): Predicted HEALTHY, but actual motor has {true_label} FAULT!** — Turn ON Sync & Spatial-DFT to fix! | Confidence: **{conf_pct:.1f}%**")
else:
    if pred_label == true_label:
        st.error(f"🚨 **FAULT DETECTED: {pred_label} FAULT** — Immediate maintenance required! | Severity: **{int(severity_slider*100)}%** | Confidence: **{conf_pct:.1f}%**")
    else:
        st.warning(f"⚠️ **MISCLASSIFICATION: Predicted {pred_label} FAULT (Actual: {true_label})** | Confidence: **{conf_pct:.1f}%**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("True / Estimated f_e", f"{fe_slider:.2f} Hz / {est_fe:.2f} Hz", f"Err: {abs(est_fe - fe_slider):.3f} Hz")
col2.metric("Derived Rotor Speed f_r", f"{fr_ref:.2f} Hz", "Synced to Hall" if sync_enabled else "Fixed 50Hz Assumed")
col3.metric("Predicted Diagnosis", pred_label, f"Confidence: {conf_pct:.1f}%")
col4.metric("Spatial-DFT Invariant", f"{sp['array_rms_invariant']:.3f}", f"Array Angle: {theta_deg}°")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
orders, o_mag = compute_acoustic_order_spectrum(data["acoustic"], data["meta"]["fs_acoustic"], fr_ref)
mask = (orders >= 2.0) & (orders <= 5.0)
axes[0].plot(orders[mask], o_mag[mask], color="#2ca02c" if sync_enabled else "#d62728", linewidth=2)
axes[0].axvline(3.585, color="black", linestyle="--", label="Expected BPFO (3.585x)")
axes[0].set_title(f"Acoustic Order Spectrum ({'SYNC ON' if sync_enabled else 'SYNC OFF - Peak Shifted!'})")
axes[0].set_xlabel("Shaft Order (multiples of f_r)")
axes[0].set_ylabel("Envelope Order Magnitude")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

sensor_rms = np.sqrt(np.mean(data["hall"] ** 2, axis=1))
axes[1].bar([f"S{i+1}" for i in range(6)], sensor_rms, color="#1f77b4", alpha=0.7, label="Individual Sensor RMS (Varies with theta0)")
axes[1].axhline(sp["array_rms_invariant"] * np.sqrt(2), color="crimson", linestyle="--", linewidth=2, label="Spatial-DFT Invariant Reference")
axes[1].set_title(f"6-Sensor Circumferential Hall Array Profile (theta0 = {theta_deg}°)")
axes[1].set_ylabel("RMS Flux Amplitude")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

if os.path.exists("results/ablation.csv"):
    st.subheader("Held-Out Condition Ablation Summary (`results/ablation.csv`)")
    st.dataframe(pd.read_csv("results/ablation.csv"), use_container_width=True)
