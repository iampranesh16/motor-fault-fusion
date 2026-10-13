import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d

def estimate_fe(hall_signals: np.ndarray, fs_hall: int, f_min: float = 35.0, f_max: float = 65.0) -> float:
    n_sensors, n_samples = hall_signals.shape
    window = np.hanning(n_samples)
    
    avg_mag_sq = np.zeros(n_samples // 2 + 1)
    for i in range(n_sensors):
        sig = (hall_signals[i] - np.mean(hall_signals[i])) * window
        spec = np.abs(np.fft.rfft(sig))
        avg_mag_sq += spec ** 2
    
    avg_mag = np.sqrt(avg_mag_sq / n_sensors)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs_hall)
    df = freqs[1] - freqs[0]
    
    mask = (freqs >= f_min) & (freqs <= f_max)
    valid_indices = np.where(mask)[0]
    k = valid_indices[np.argmax(avg_mag[valid_indices])]
    
    if 0 < k < len(avg_mag) - 1:
        y1, y2, y3 = np.log(avg_mag[k - 1] + 1e-12), np.log(avg_mag[k] + 1e-12), np.log(avg_mag[k + 1] + 1e-12)
        denom = 2.0 * (2.0 * y2 - y1 - y3)
        delta = (y3 - y1) / denom if abs(denom) > 1e-12 else 0.0
        delta = np.clip(delta, -0.5, 0.5)
    else:
        delta = 0.0
        
    return float(freqs[k] + delta * df)

def compute_acoustic_order_spectrum(
    acoustic: np.ndarray,
    fs_acoustic: int,
    fr_ref: float,
    max_order: float = 10.0,
    order_res: float = 0.005,
    use_envelope: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    if use_envelope:
        b, a = butter(4, [2000.0 / (fs_acoustic / 2.0), 3600.0 / (fs_acoustic / 2.0)], btype='band')
        filtered = filtfilt(b, a, acoustic)
        sig = np.abs(hilbert(filtered))
        sig -= np.mean(sig)
    else:
        sig = acoustic - np.mean(acoustic)
        
    n = len(sig)
    window = np.hanning(n)
    fft_mag = np.abs(np.fft.rfft(sig * window)) / np.sum(window)
    freqs = np.fft.rfftfreq(n, 1.0 / fs_acoustic)
    
    raw_orders = freqs / fr_ref
    target_orders = np.arange(0.0, max_order + order_res, order_res)
    
    interpolator = interp1d(raw_orders, fft_mag, kind='cubic', bounds_error=False, fill_value=0.0)
    order_mag = interpolator(target_orders)
    return target_orders, order_mag

def compute_spatial_invariants(hall_signals: np.ndarray, fs_hall: int, fe_est: float, fr_est: float) -> dict:
    n_sensors, n_samples = hall_signals.shape
    t = np.arange(n_samples) / fs_hall
    
    exp_fe = np.exp(-1j * 2.0 * np.pi * fe_est * t)
    phasors_fe = (hall_signals @ exp_fe) / n_samples
    
    spatial_dft = np.abs(np.fft.fft(phasors_fe)) / n_sensors
    array_rms_invariant = float(np.sqrt(np.mean(np.abs(phasors_fe) ** 2)))
    
    exp_sb_low = np.exp(-1j * 2.0 * np.pi * (fe_est - fr_est) * t)
    exp_sb_high = np.exp(-1j * 2.0 * np.pi * (fe_est + fr_est) * t)
    
    sb_low_energy = np.sum(np.abs((hall_signals @ exp_sb_low) / n_samples) ** 2)
    sb_high_energy = np.sum(np.abs((hall_signals @ exp_sb_high) / n_samples) ** 2)
    fe_total_energy = np.sum(np.abs(phasors_fe) ** 2) + 1e-12
    
    sideband_ratio = float(np.sqrt((sb_low_energy + sb_high_energy) / fe_total_energy))
    
    return {
        "array_rms_invariant": array_rms_invariant,
        "spatial_mode1_mag": float(spatial_dft[1] + spatial_dft[-1]),
        "sideband_ratio": sideband_ratio,
        "spatial_dft_profile": spatial_dft
    }
