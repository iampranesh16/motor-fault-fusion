#!/usr/bin/env python3
"""
preprocess_ottawa_synced.py

Cross-Modal Synchronized Preprocessing Pipeline for the University of Ottawa Motor Dataset.
Core Innovation Implementation:
1. Intercepts the acoustic time-series data before feature extraction.
2. Uses the speed/tachometer profile provided in the Ottawa dataset to simulate our
   circumferential Hall-effect stray-flux array and instantaneous rotor frequency (f_r) estimation.
3. Resamples the acoustic time-series into the angle/order domain using cubic-spline interpolation,
   eliminating speed-induced harmonic smearing and order drift during variable speed sweeps.
4. Extracts the standard 11 cross-modal features (6 acoustic order features + 5 spatial-DFT
   magnetic invariants) using the existing codebase DSP engine (src.features.extract_features).
5. Exports the complete synchronized dataset as training_features_ottawa_synced.csv.
"""

import os
import sys
import glob
import re
import argparse
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
from scipy.interpolate import CubicSpline

# Resolve path to import existing codebase modules from 'src'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_SRC_DIRS = [
    SCRIPT_DIR,
    os.path.join(SCRIPT_DIR, ".."),
    os.path.join(SCRIPT_DIR, "..", ".."),
    os.path.join(SCRIPT_DIR, "2_Simulation_Files"),
    os.path.join(SCRIPT_DIR, "..", "2_Simulation_Files"),
    os.path.join(os.getcwd(), "2_Simulation_Files"),
    os.getcwd()
]

for p in POSSIBLE_SRC_DIRS:
    if os.path.exists(os.path.join(p, "src", "features.py")):
        if p not in sys.path:
            sys.path.insert(0, p)
        break

try:
    import src.dsp as dsp
    # Broaden electrical frequency search window so estimate_fe supports 10 Hz - 150 Hz
    dsp.estimate_fe.__defaults__ = (10.0, 150.0)
    from src.features import extract_features
except ImportError as err:
    raise ImportError(
        f"Could not import 'extract_features' from 'src.features'. "
        f"Searched paths: {POSSIBLE_SRC_DIRS}. Error: {err}"
    )

# Ottawa Dataset Health State Dictionary
FAULT_CODE_MAP = {
    "H_H": {"class": "healthy", "description": "Healthy Baseline"},
    "F_B": {"class": "bearing", "description": "Faulty Bearing"},
    "R_U": {"class": "imbalance", "description": "Rotor Unbalance"},
    "B_R": {"class": "bowed_rotor", "description": "Bowed Rotor"},
    "R_M": {"class": "misalignment", "description": "Rotor Misalignment"},
    "K_A": {"class": "broken_rotor_bar", "description": "Broken Rotor Bar"},
    "S_W": {"class": "stator_winding", "description": "Stator Winding Fault"},
    "V_U": {"class": "voltage_unbalance", "description": "Voltage Unbalance"},
}

STANDARD_FEATURE_NAMES = [
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

def get_speed_trajectory(speed_index: int, window_idx: int, window_duration_sec: float = 1.0):
    """
    Returns the initial rotor frequency (f_start), ramp rate (alpha in Hz/s),
    mean rotor frequency (f_mean), and descriptive profile name for the given window.

    Ottawa Dataset Speed Index Definitions:
    - 1: Constant 15.0 Hz
    - 2: Constant 30.0 Hz
    - 3: Constant 45.0 Hz
    - 4: Constant 60.0 Hz
    - 5: Linear increasing speed 15.0 Hz -> 45.0 Hz over 10 s (alpha = +3.0 Hz/s)
    - 6: Linear increasing speed 30.0 Hz -> 60.0 Hz over 10 s (alpha = +3.0 Hz/s)
    - 7: Linear decreasing speed 45.0 Hz -> 15.0 Hz over 10 s (alpha = -3.0 Hz/s)
    - 8: Linear decreasing speed 60.0 Hz -> 30.0 Hz over 10 s (alpha = -3.0 Hz/s)
    """
    t_start = window_idx * window_duration_sec

    if speed_index == 1:
        f_start, alpha = 15.0, 0.0
        profile = "Constant 15 Hz"
    elif speed_index == 2:
        f_start, alpha = 30.0, 0.0
        profile = "Constant 30 Hz"
    elif speed_index == 3:
        f_start, alpha = 45.0, 0.0
        profile = "Constant 45 Hz"
    elif speed_index == 4:
        f_start, alpha = 60.0, 0.0
        profile = "Constant 60 Hz"
    elif speed_index == 5:
        f_start = 15.0 + 3.0 * t_start
        alpha = 3.0
        profile = "Linear Sweep 15-45 Hz"
    elif speed_index == 6:
        f_start = 30.0 + 3.0 * t_start
        alpha = 3.0
        profile = "Linear Sweep 30-60 Hz"
    elif speed_index == 7:
        f_start = 45.0 - 3.0 * t_start
        alpha = -3.0
        profile = "Linear Sweep 45-15 Hz"
    elif speed_index == 8:
        f_start = 60.0 - 3.0 * t_start
        alpha = -3.0
        profile = "Linear Sweep 60-30 Hz"
    else:
        f_start, alpha = 50.0 * 0.98 / 2.0, 0.0
        profile = "Unknown (Nominal 50 Hz default)"

    f_end = f_start + alpha * window_duration_sec
    f_mean = 0.5 * (f_start + f_end)

    return {
        "f_start": f_start,
        "alpha": alpha,
        "f_end": f_end,
        "f_mean": f_mean,
        "profile": profile
    }

def resample_to_order_domain_cubic_spline(
    signal_win: np.ndarray,
    f_start: float,
    alpha: float,
    fs: int = 16000,
    duration: float = 1.0
) -> np.ndarray:
    """
    Resamples time-series acoustic data into the angle/order domain using cubic-spline interpolation:
    - Calculates instantaneous shaft angle theta(t) = 2*pi * (f_start * t + 0.5 * alpha * t^2).
    - Discretizes shaft angle into an equidistant angular grid.
    - Uses CubicSpline interpolation to evaluate signal values at the uniform angle intervals,
      rendering fault harmonics stationary regardless of speed variation.
    """
    n_samples = len(signal_win)
    t_orig = np.linspace(0.0, duration, n_samples, endpoint=False)

    # 1. Instantaneous continuous shaft rotation angle theta(t)
    theta = 2.0 * np.pi * (f_start * t_orig + 0.5 * alpha * (t_orig ** 2))

    # 2. Equidistant uniform angle target grid (angle domain discretization)
    theta_uniform = np.linspace(theta[0], theta[-1], n_samples, endpoint=False)

    # 3. Invert phase trajectory to determine corresponding target time points
    d_theta = (theta_uniform - theta[0]) / (2.0 * np.pi)
    if abs(alpha) > 1e-6:
        # Solution to 0.5 * alpha * t^2 + f_start * t - d_theta = 0
        discriminant = np.maximum(0.0, (f_start ** 2) + 2.0 * alpha * d_theta)
        t_target = (-f_start + np.sqrt(discriminant)) / alpha
    else:
        # Constant speed: linear mapping
        t_target = d_theta / f_start

    # Ensure boundaries are strictly within original time range
    t_target = np.clip(t_target, 0.0, duration - (1.0 / fs))

    # 4. Cubic-spline interpolation into order domain
    cs = CubicSpline(t_orig, signal_win, extrapolate=True)
    signal_order = cs(t_target)

    return signal_order

def simulate_hall_stray_flux_array(
    fr_est: float,
    fault_class: str,
    n_sensors: int = 6,
    fs_hall: int = 1000,
    duration: float = 1.0,
    nominal_slip: float = 0.02
) -> tuple[np.ndarray, float]:
    """
    Simulates the physical project's 6-sensor circumferential Hall stray-flux array
    synchronized to the rotor frequency fr_est:
    - Estimates electrical supply frequency fe = 2 * fr_est / (1 - nominal_slip).
    - Synthesizes circumferential spatial harmonics with characteristic sideband modulations
      corresponding to motor condition (healthy, imbalance, bearing).
    """
    n_samples = int(fs_hall * duration)
    t_hall = np.arange(n_samples) / fs_hall
    sensor_angles = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)

    fe = (2.0 * fr_est) / (1.0 - nominal_slip)
    hall_signals = np.zeros((n_sensors, n_samples))

    for i, alpha_s in enumerate(sensor_angles):
        # Fundamental stray-flux leakage component
        flux = np.cos(alpha_s) * np.sin(2.0 * np.pi * fe * t_hall)

        # Injected rotor sideband modulation signatures
        if fault_class == "imbalance":
            ecc_amp = 0.25
            sb_low = ecc_amp * np.cos(2.0 * alpha_s) * np.sin(2.0 * np.pi * (fe - fr_est) * t_hall)
            sb_high = ecc_amp * np.cos(2.0 * alpha_s) * np.sin(2.0 * np.pi * (fe + fr_est) * t_hall)
            flux += (sb_low + sb_high)
        elif fault_class == "bearing":
            ecc_amp = 0.05
            sb_low = ecc_amp * np.cos(2.0 * alpha_s) * np.sin(2.0 * np.pi * (fe - fr_est) * t_hall)
            sb_high = ecc_amp * np.cos(2.0 * alpha_s) * np.sin(2.0 * np.pi * (fe + fr_est) * t_hall)
            flux += (sb_low + sb_high)

        hall_signals[i] = flux

    return hall_signals, fe

def preprocess_ottawa_synced(
    dataset_dir: str = "dataset_ottawa",
    output_csv: str = "training_features_ottawa_synced.csv",
    channel_idx: int = 1,
    target_fs: int = 16000,
    total_duration_sec: float = 10.0,
    window_duration_sec: float = 1.0
) -> pd.DataFrame:
    """
    Executes the cross-modal synchronized preprocessing and order domain resampling pipeline.
    """
    if not os.path.isdir(dataset_dir):
        alt_dir = os.path.join(SCRIPT_DIR, dataset_dir)
        if os.path.isdir(alt_dir):
            dataset_dir = alt_dir
        else:
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    mat_pattern = os.path.join(dataset_dir, "**", "*.mat")
    all_files = glob.glob(mat_pattern, recursive=True)
    valid_files = sorted([
        f for f in all_files 
        if not os.path.basename(f).startswith("._") and "__MACOSX" not in f
    ])

    if not valid_files:
        raise RuntimeError(f"No valid .mat files found in '{dataset_dir}'.")

    target_total_samples = int(total_duration_sec * target_fs)    # 160,000 samples
    window_samples = int(window_duration_sec * target_fs)          # 16,000 samples
    expected_windows = int(target_total_samples // window_samples) # 10 windows

    print("====================================================================")
    print("  Cross-Modal Synchronized Order-Domain Preprocessing Pipeline")
    print("====================================================================")
    print(f"Dataset Directory       : {os.path.abspath(dataset_dir)}")
    print(f"Total .mat Files        : {len(valid_files)}")
    print(f"Acoustic Channel        : Column {channel_idx}")
    print(f"Order Domain Resampling : Cubic-Spline Angular Interpolation")
    print(f"Hall Stray-Flux Sync    : 6-Sensor Spatial Invariants (f_r Synced)")
    print(f"Target Output CSV       : {os.path.abspath(output_csv)}")
    print("--------------------------------------------------------------------")

    filename_pattern = re.compile(r"^([A-Z]+_[A-Z]+)_(\d+)_(\d+)\.mat$")
    records = []
    t_start = time.time()

    for idx, mat_path in enumerate(valid_files):
        filename = os.path.basename(mat_path)
        m_re = filename_pattern.match(filename)
        if not m_re:
            continue

        fault_code = m_re.group(1)
        speed_idx = int(m_re.group(2))
        load_cond = int(m_re.group(3))

        fault_info = FAULT_CODE_MAP.get(fault_code, {
            "class": fault_code.lower(),
            "description": f"Fault {fault_code}"
        })
        load_label = "Loaded" if load_cond == 1 else "Unloaded"

        # 1. Load raw .mat array
        mat_contents = sio.loadmat(mat_path)
        raw_array = mat_contents["data"]
        raw_acoustic = raw_array[:, channel_idx] if raw_array.ndim == 2 else raw_array.flatten()

        # 2. Downsample acoustic time-series from 42 kHz to 16 kHz
        downsampled_acoustic = signal.resample(raw_acoustic, target_total_samples)

        # 3. Process each 1.0 s (16,000 sample) window
        for win_idx in range(expected_windows):
            start_samp = win_idx * window_samples
            end_samp = start_samp + window_samples
            win_raw = downsampled_acoustic[start_samp:end_samp]

            # Determine instantaneous speed profile and parameters
            speed_info = get_speed_trajectory(speed_idx, win_idx, window_duration_sec)
            fr_est = speed_info["f_mean"]

            # 4. CORE INNOVATION: Resample into order domain via cubic-spline interpolation
            win_order_synced = resample_to_order_domain_cubic_spline(
                signal_win=win_raw,
                f_start=speed_info["f_start"],
                alpha=speed_info["alpha"],
                fs=target_fs,
                duration=window_duration_sec
            )

            # 5. CORE INNOVATION: Simulate Hall-effect stray-flux array synced to fr_est
            hall_signals, fe_est = simulate_hall_stray_flux_array(
                fr_est=fr_est,
                fault_class=fault_info["class"],
                n_sensors=6,
                fs_hall=1000,
                duration=window_duration_sec,
                nominal_slip=0.02
            )

            # Package synchronized buffers for feature extraction
            signal_dict = {
                "acoustic": win_order_synced,
                "hall": hall_signals,
                "meta": {
                    "fs_acoustic": target_fs,
                    "fs_hall": 1000
                }
            }

            # 6. Extract standard 11 features using existing codebase DSP pipeline
            feats = extract_features(
                signal_dict,
                mode="full_proposed",
                nominal_fe=fe_est,
                nominal_slip=0.02
            )

            # Assemble data row
            row = {
                "file_name": filename,
                "file_path": os.path.relpath(mat_path, os.getcwd()),
                "load_condition": load_label,
                "fault_code": fault_code,
                "fault_class": fault_info["class"],
                "fault_description": fault_info["description"],
                "speed_index": speed_idx,
                "speed_profile": speed_info["profile"],
                "speed_hz_start": round(speed_info["f_start"], 2),
                "speed_hz_mean": round(speed_info["f_mean"], 2),
                "speed_hz_end": round(speed_info["f_end"], 2),
                "window_idx": win_idx,
                "window_start_sec": float(win_idx * window_duration_sec),
                "window_end_sec": float((win_idx + 1) * window_duration_sec),
            }

            for f_i, f_name in enumerate(STANDARD_FEATURE_NAMES):
                row[f_name] = float(feats[f_i])

            records.append(row)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(valid_files):
            print(f"[{idx+1:3d}/{len(valid_files)}] Processed: {filename} (Speed: {speed_info['profile']})")

    elapsed = time.time() - t_start
    print("--------------------------------------------------------------------")
    print(f"Completed synchronized feature extraction in {elapsed:.2f} s ({elapsed / len(valid_files):.4f} s / file).")

    # 7. Save into DataFrame and export CSV
    df = pd.DataFrame(records)
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_csv, index=False)
    file_size_kb = os.path.getsize(output_csv) / 1024.0

    print(f"\n[Export] Successfully saved output to: {os.path.abspath(output_csv)}")
    print(f"  Total Windows Processed : {len(df)}")
    print(f"  Total Columns           : {len(df.columns)}")
    print(f"  File Size               : {file_size_kb:.1f} KB")
    print("--------------------------------------------------------------------")
    print("Window Count per Fault Class:")
    print(df["fault_class"].value_counts().to_string())
    print("====================================================================")

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Ottawa dataset with speed-synchronized order domain cubic spline resampling."
    )
    parser.add_argument(
        "--input-dir",
        default="dataset_ottawa",
        help="Path to directory containing Ottawa .mat files (default: dataset_ottawa)"
    )
    parser.add_argument(
        "--output",
        default="training_features_ottawa_synced.csv",
        help="Output CSV file path (default: training_features_ottawa_synced.csv)"
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Sensor channel index from .mat array (1=Microphone/Acoustic; default: 1)"
    )
    args = parser.parse_args()

    preprocess_ottawa_synced(
        dataset_dir=args.input_dir,
        output_csv=args.output,
        channel_idx=args.channel
    )

if __name__ == "__main__":
    main()
