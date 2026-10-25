import numpy as np
from src.dsp import estimate_fe, compute_acoustic_order_spectrum, compute_spatial_invariants

def extract_features(data: dict, mode: str = "full_proposed", nominal_fe: float = 50.0, nominal_slip: float = 0.02) -> np.ndarray:
    hall = data["hall"]
    acoustic = data["acoustic"]
    fs_h = data["meta"]["fs_hall"]
    fs_ac = data["meta"]["fs_acoustic"]
    
    # 1. Electrical & Rotor Reference Frequency (Synced vs Unsynced fixed 50Hz)
    if mode in ["fusion_sync", "full_proposed"]:
        fe_ref = estimate_fe(hall, fs_h)
    else:
        fe_ref = nominal_fe
        
    fr_ref = fe_ref * (1.0 - nominal_slip) / 2.0
    
    # 2. Acoustic Order Spectrum Features (Normalized to background order floor)
    orders, order_mag = compute_acoustic_order_spectrum(acoustic, fs_ac, fr_ref, use_envelope=True)
    _, raw_order_mag = compute_acoustic_order_spectrum(acoustic, fs_ac, fr_ref, use_envelope=False)
    
    env_floor = float(np.median(order_mag[(orders >= 1.5) & (orders <= 8.0)]) + 1e-9)
    raw_floor = float(np.median(raw_order_mag[(orders >= 0.5) & (orders <= 5.0)]) + 1e-9)
    
    def band_snr(ord_axis, spec, center, floor_val, half_width=0.06):
        mask = (ord_axis >= center - half_width) & (ord_axis <= center + half_width)
        return float(np.max(spec[mask]) / floor_val) if np.any(mask) else 0.0

    ac_bpfo = band_snr(orders, order_mag, 3.585, env_floor, 0.06)
    ac_bpfo_2x = band_snr(orders, order_mag, 7.170, env_floor, 0.10)
    ac_1x = band_snr(orders, raw_order_mag, 1.0, raw_floor, 0.06)
    ac_2x = band_snr(orders, raw_order_mag, 2.0, raw_floor, 0.06)
    ac_rms = float(np.sqrt(np.mean(acoustic ** 2)))
    ac_kurt = float(np.mean((acoustic - np.mean(acoustic)) ** 4) / (np.std(acoustic) ** 4 + 1e-12))
    
    acoustic_feats = [ac_bpfo, ac_bpfo_2x, ac_1x, ac_2x, ac_rms, ac_kurt]
    
    # 3. Magnetic Features (Spatial-DFT Invariant vs Single Uncalibrated Sensor Ch0)
    if mode == "full_proposed":
        sp = compute_spatial_invariants(hall, fs_h, fe_ref, fr_ref)
        mag_feats = [
            sp["array_rms_invariant"],
            sp["spatial_mode1_mag"],
            sp["sideband_ratio"],
            float(sp["spatial_dft_profile"][2] + sp["spatial_dft_profile"][-2])
        ]
        consistency_gate = sp["sideband_ratio"] * ac_1x + (ac_bpfo * sp["array_rms_invariant"])
        mag_feats.append(float(consistency_gate))
    else:
        # Uncalibrated single sensor Ch0 (sensitive to mounting orientation theta0 & unsynced frequency drift)
        ch0 = hall[0]
        t = np.arange(len(ch0)) / fs_h
        fe_amp = float(np.abs(np.mean(ch0 * np.exp(-1j * 2 * np.pi * fe_ref * t))))
        sb_low = float(np.abs(np.mean(ch0 * np.exp(-1j * 2 * np.pi * (fe_ref - fr_ref) * t))))
        sb_high = float(np.abs(np.mean(ch0 * np.exp(-1j * 2 * np.pi * (fe_ref + fr_ref) * t))))
        ch0_rms = float(np.sqrt(np.mean(ch0 ** 2)))
        # Without spatial normalization, raw sideband amplitude varies wildly with cos(2*theta0)
        mag_feats = [ch0_rms, fe_amp, (sb_low + sb_high), 0.0, 0.0]

    if mode == "acoustic_only":
        mag_feats = [0.0] * len(mag_feats)
    elif mode == "hall_only":
        acoustic_feats = [0.0] * len(acoustic_feats)
        
    return np.array(acoustic_feats + mag_feats, dtype=np.float64)
