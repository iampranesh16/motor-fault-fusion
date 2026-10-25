#!/usr/bin/env python3
"""
preprocess_ottawa.py

Preprocesses the University of Ottawa Motor Dataset to match embedded microcontroller constraints:
1. Iterates through .mat files in dataset_ottawa/ using scipy.io.loadmat.
2. Downsamples the extracted acoustic arrays from 42 kHz down to 16 kHz using scipy.signal.resample
   (10 s file = 420,000 samples -> 160,000 samples to match acoustic ring buffer specs).
3. Slices the 160,000-sample array into 10 separate 1.0-second windows (16,000 samples each).
4. Imports the DSP feature extraction function from our existing codebase (src.features.extract_features)
   to process each 16,000-sample window into the standard 11 features.
5. Saves the final output into a single lightweight pandas DataFrame and exports training_features_ottawa.csv.
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
    from src.features import extract_features
except ImportError as err:
    raise ImportError(
        f"Could not import 'extract_features' from 'src.features'. "
        f"Searched paths: {POSSIBLE_SRC_DIRS}. Error: {err}"
    )

# Ottawa Dataset Fault Mapping
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

SPEED_INDEX_MAP = {
    1: 15.0,  # 15 Hz rotor rotational speed
    2: 30.0,  # 30 Hz rotor rotational speed
    3: 45.0,  # 45 Hz rotor rotational speed
    4: 60.0,  # 60 Hz rotor rotational speed
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

def parse_filename_metadata(filename: str) -> dict:
    """
    Parses Ottawa filename convention: {FaultCode}_{SpeedIndex}_{LoadCondition}.mat
    Example: B_R_1_1.mat -> Fault: B_R, Speed: 1, Load: 1 (Loaded)
    """
    basename = os.path.basename(filename)
    match = re.match(r"^([A-Z]+_[A-Z]+)_(\d+)_(\d+)\.mat$", basename)
    if not match:
        return {
            "fault_code": "UNKNOWN",
            "fault_class": "unknown",
            "fault_description": "Unknown",
            "speed_index": -1,
            "speed_hz": np.nan,
            "load_condition": -1,
            "load_label": "Unknown"
        }
    
    fault_code = match.group(1)
    speed_idx = int(match.group(2))
    load_cond = int(match.group(3))
    
    fault_info = FAULT_CODE_MAP.get(fault_code, {
        "class": fault_code.lower(),
        "description": f"Fault {fault_code}"
    })
    
    speed_hz = SPEED_INDEX_MAP.get(speed_idx, np.nan)
    load_label = "Loaded" if load_cond == 1 else "Unloaded"
    
    return {
        "fault_code": fault_code,
        "fault_class": fault_info["class"],
        "fault_description": fault_info["description"],
        "speed_index": speed_idx,
        "speed_hz": speed_hz,
        "load_condition": load_cond,
        "load_label": load_label
    }

def preprocess_ottawa_dataset(
    dataset_dir: str = "dataset_ottawa",
    output_csv: str = "training_features_ottawa.csv",
    channel_idx: int = 1,
    target_fs: int = 16000,
    raw_fs: int = 42000,
    window_duration_sec: float = 1.0,
    total_duration_sec: float = 10.0,
    extraction_mode: str = "acoustic_only"
) -> pd.DataFrame:
    """
    Processes all Ottawa .mat files into downsampled 1.0 s windows and extracts 11 DSP features.
    """
    if not os.path.isdir(dataset_dir):
        # Check if dataset_dir is relative to SCRIPT_DIR or parent dirs
        found_dir = None
        for candidate in [
            os.path.join(SCRIPT_DIR, dataset_dir),
            os.path.join(SCRIPT_DIR, "..", dataset_dir),
            os.path.join(SCRIPT_DIR, "..", "..", dataset_dir),
            os.path.join(os.getcwd(), dataset_dir)
        ]:
            if os.path.isdir(candidate):
                found_dir = candidate
                break
        if found_dir:
            dataset_dir = found_dir
        else:
            raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    # 1. Discover all .mat files (excluding macOS metadata ._* files and __MACOSX folders)
    mat_pattern = os.path.join(dataset_dir, "**", "*.mat")
    all_files = glob.glob(mat_pattern, recursive=True)
    valid_files = sorted([
        f for f in all_files 
        if not os.path.basename(f).startswith("._") and "__MACOSX" not in f
    ])

    if not valid_files:
        raise RuntimeError(f"No valid .mat files found in '{dataset_dir}'.")

    print(f"============================================================")
    print(f"  University of Ottawa Motor Dataset Preprocessing Pipeline")
    print(f"============================================================")
    print(f"Dataset Directory : {os.path.abspath(dataset_dir)}")
    print(f"Found .mat Files  : {len(valid_files)}")
    print(f"Target Sample Rate: {target_fs} Hz (Downsampling from {raw_fs} Hz)")
    print(f"Window Specs      : {window_duration_sec} s ({int(target_fs * window_duration_sec)} samples / window)")
    print(f"Channel Extracted : Column {channel_idx} (Acoustic Sound Pressure)")
    print(f"DSP Feature Mode  : {extraction_mode} (Standard 11 features)")
    print(f"Output File       : {os.path.abspath(output_csv)}")
    print(f"------------------------------------------------------------")

    target_total_samples = int(total_duration_sec * target_fs)    # 160,000 samples
    window_samples = int(window_duration_sec * target_fs)          # 16,000 samples
    expected_windows = int(target_total_samples // window_samples) # 10 windows

    records = []
    t_start = time.time()

    for idx, mat_path in enumerate(valid_files):
        filename = os.path.basename(mat_path)
        meta = parse_filename_metadata(filename)
        
        # Step 1: Load .mat file
        try:
            mat_contents = sio.loadmat(mat_path)
        except Exception as e:
            print(f"[{idx+1}/{len(valid_files)}] Error reading {filename}: {e}")
            continue

        if "data" not in mat_contents:
            print(f"[{idx+1}/{len(valid_files)}] Warning: 'data' key not found in {filename}. Skipping.")
            continue

        raw_array = mat_contents["data"]
        
        # Extract desired channel
        if raw_array.ndim == 2:
            if channel_idx < raw_array.shape[1]:
                raw_signal = raw_array[:, channel_idx]
            else:
                raw_signal = raw_array[:, 0]
        else:
            raw_signal = raw_array.flatten()

        # Step 2: Downsample from 42 kHz to 16 kHz (420,000 -> 160,000 samples)
        downsampled_signal = signal.resample(raw_signal, target_total_samples)

        # Reference speed for order tracking
        nominal_fe = 50.0
        nominal_slip = 0.02
        if not np.isnan(meta["speed_hz"]):
            # Synchronize reference electrical frequency to the constant rotor speed:
            # fr = fe * (1 - slip) / 2  =>  fe = 2 * fr / (1 - slip)
            nominal_fe = (2.0 * meta["speed_hz"]) / (1.0 - nominal_slip)

        # Step 3: Slice into ten separate 1.0-second windows (16,000 samples each)
        for win_idx in range(expected_windows):
            start_samp = win_idx * window_samples
            end_samp = start_samp + window_samples
            win_data = downsampled_signal[start_samp:end_samp]

            # Construct input structure expected by extract_features
            signal_dict = {
                "acoustic": win_data,
                "hall": np.zeros((6, 1000), dtype=np.float64),  # Standard 6-ch decimated Hall buffer
                "meta": {
                    "fs_acoustic": target_fs,
                    "fs_hall": 1000
                }
            }

            # Step 4: Extract standard 11 features
            feats = extract_features(
                signal_dict,
                mode=extraction_mode,
                nominal_fe=nominal_fe,
                nominal_slip=nominal_slip
            )

            # Assemble record with metadata + 11 standard features
            row = {
                "file_name": filename,
                "file_path": os.path.relpath(mat_path, os.getcwd()),
                "load_condition": meta["load_label"],
                "fault_code": meta["fault_code"],
                "fault_class": meta["fault_class"],
                "fault_description": meta["fault_description"],
                "speed_index": meta["speed_index"],
                "speed_hz": meta["speed_hz"],
                "window_idx": win_idx,
                "window_start_sec": float(win_idx * window_duration_sec),
                "window_end_sec": float((win_idx + 1) * window_duration_sec),
            }

            # Add named 11 features
            for feat_idx, feat_name in enumerate(STANDARD_FEATURE_NAMES):
                row[feat_name] = float(feats[feat_idx])

            records.append(row)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(valid_files):
            print(f"[{idx+1:3d}/{len(valid_files)}] Processed: {filename} -> {expected_windows} windows extracted.")

    elapsed = time.time() - t_start
    print(f"------------------------------------------------------------")
    print(f"Completed feature extraction in {elapsed:.2f} s ({elapsed / len(valid_files):.4f} s / file).")

    # Step 5: Save into single lightweight pandas DataFrame & export CSV
    df = pd.DataFrame(records)
    
    # Save CSV
    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    df.to_csv(output_csv, index=False)
    file_size_kb = os.path.getsize(output_csv) / 1024.0

    print(f"Successfully saved output to '{output_csv}'")
    print(f"  Total Rows (Windows) : {len(df)}")
    print(f"  Total Columns        : {len(df.columns)}")
    print(f"  File Size            : {file_size_kb:.1f} KB")
    print(f"------------------------------------------------------------")
    print(f"Window Count per Fault Class:")
    print(df["fault_class"].value_counts().to_string())
    print(f"============================================================")

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess University of Ottawa Motor Dataset into 16 kHz 1.0 s windows and extract 11 features."
    )
    parser.add_argument(
        "--input-dir",
        default="dataset_ottawa",
        help="Path to directory containing Ottawa .mat files (default: dataset_ottawa)"
    )
    parser.add_argument(
        "--output",
        default="training_features_ottawa.csv",
        help="Output CSV file path (default: training_features_ottawa.csv)"
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Sensor channel index from .mat data array (0=Accelerometer, 1=Microphone/Acoustic; default: 1)"
    )
    parser.add_argument(
        "--mode",
        choices=["acoustic_only", "full_proposed", "naive_fusion"],
        default="acoustic_only",
        help="Feature extraction mode (default: acoustic_only)"
    )
    args = parser.parse_args()

    preprocess_ottawa_dataset(
        dataset_dir=args.input_dir,
        output_csv=args.output,
        channel_idx=args.channel,
        extraction_mode=args.mode
    )

if __name__ == "__main__":
    main()
