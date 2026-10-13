import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from src.generator import generate_signals

def _extract_envelope_peak(acoustic, fs, f_min=40.0, f_max=150.0):
    b, a = butter(4, [2000 / (fs / 2), 3600 / (fs / 2)], btype='band')
    filtered = filtfilt(b, a, acoustic)
    envelope = np.abs(hilbert(filtered))
    envelope -= np.mean(envelope)
    
    fft_mag = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), 1.0 / fs)
    
    mask = (freqs >= f_min) & (freqs <= f_max)
    peak_freq = freqs[mask][np.argmax(fft_mag[mask])]
    return peak_freq

def test_gate1_bearing_bpfo_peak():
    data = generate_signals(fault_class="bearing", fe=50.0, slip=0.02, severity=0.9, snr_db=25.0, duration=2.0)
    expected_bpfo = data["meta"]["bpfo"]
    measured_peak = _extract_envelope_peak(data["acoustic"], data["meta"]["fs_acoustic"])
    rel_error = abs(measured_peak - expected_bpfo) / expected_bpfo
    assert rel_error < 0.01, f"BPFO peak {measured_peak:.2f} Hz off from {expected_bpfo:.2f} Hz"

def test_gate1_speed_scaling_proportionality():
    for fe_test in [40.0, 50.0, 60.0]:
        data = generate_signals(fault_class="bearing", fe=fe_test, slip=0.02, severity=0.9, snr_db=25.0, duration=2.0)
        expected_bpfo = data["meta"]["bpfo"]
        measured_peak = _extract_envelope_peak(data["acoustic"], data["meta"]["fs_acoustic"], f_min=30.0, f_max=180.0)
        rel_error = abs(measured_peak - expected_bpfo) / expected_bpfo
        assert rel_error < 0.01, f"At fe={fe_test}Hz, measured {measured_peak:.2f}Hz vs {expected_bpfo:.2f}Hz"

def test_gate1_hall_sidebands():
    data = generate_signals(fault_class="imbalance", fe=50.0, slip=0.02, severity=0.8, snr_db=30.0, duration=2.0)
    hall_ch0 = data["hall"][0]
    fs_h = data["meta"]["fs_hall"]
    fe = data["meta"]["fe"]
    fr = data["meta"]["fr"]
    
    fft_mag = np.abs(np.fft.rfft(hall_ch0 * np.hanning(len(hall_ch0))))
    freqs = np.fft.rfftfreq(len(hall_ch0), 1.0 / fs_h)
    
    def get_bin_mag(target_f):
        idx = np.argmin(np.abs(freqs - target_f))
        return np.max(fft_mag[max(0, idx-1):idx+2])
    
    mag_fe = get_bin_mag(fe)
    mag_sb_low = get_bin_mag(fe - fr)
    mag_sb_high = get_bin_mag(fe + fr)
    noise_floor = np.median(fft_mag)
    
    assert mag_fe > 50 * noise_floor
    assert mag_sb_low > 5 * noise_floor
    assert mag_sb_high > 5 * noise_floor
