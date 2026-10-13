import numpy as np
from dataclasses import dataclass

@dataclass
class MotorConfig:
    p: int = 2                      # Pole pairs (4-pole motor -> p = 2)
    fs_acoustic: int = 16000        # 16 kS/s acoustic sampling
    fs_hall: int = 10000            # 10 kS/s Hall array sampling
    n_sensors: int = 6              # 6 circumferential Hall sensors
    bpfo_order: float = 3.585       # 6205 bearing: Nb=9, d/D~0.203, beta=0
    bpfi_order: float = 5.415       # 6205 inner race order
    f_res: float = 2800.0           # Structural resonance carrier (Hz)

def generate_signals(
    fault_class: str = "healthy",
    fe: float = 50.0,
    slip: float = 0.02,
    severity: float = 0.8,
    theta0: float = 0.0,
    snr_db: float = 20.0,
    duration: float = 1.0,
    realistic_interference: bool = False,
    seed: int = 42,
    cfg: MotorConfig = MotorConfig()
) -> dict:
    rng = np.random.default_rng(seed)
    fr = fe * (1.0 - slip) / cfg.p
    bpfo = cfg.bpfo_order * fr

    # --- 1. HALL STRAY-FLUX ARRAY (6 sensors @ 10 kS/s) ---
    n_hall = int(cfg.fs_hall * duration)
    t_hall = np.arange(n_hall) / cfg.fs_hall
    sensor_angles = np.linspace(0, 2 * np.pi, cfg.n_sensors, endpoint=False)
    
    # Per-sensor gain tolerance (+/-10% when realistic_interference=True)
    sensor_gains = rng.uniform(0.90, 1.10, cfg.n_sensors) if realistic_interference else np.ones(cfg.n_sensors)
    
    hall_clean = np.zeros((cfg.n_sensors, n_hall))
    for i, alpha in enumerate(sensor_angles):
        spatial_angle = alpha + theta0
        flux = np.cos(spatial_angle) * np.sin(2 * np.pi * fe * t_hall)
        
        if fault_class == "imbalance" and severity > 0:
            ecc_amp = 0.25 * severity
            sb_lower = ecc_amp * np.cos(2 * spatial_angle) * np.sin(2 * np.pi * (fe - fr) * t_hall)
            sb_upper = ecc_amp * np.cos(2 * spatial_angle) * np.sin(2 * np.pi * (fe + fr) * t_hall)
            flux += (sb_lower + sb_upper)
        elif fault_class == "bearing" and severity > 0:
            ecc_amp = 0.05 * severity
            flux += ecc_amp * np.cos(2 * spatial_angle) * (
                np.sin(2 * np.pi * (fe - fr) * t_hall) + np.sin(2 * np.pi * (fe + fr) * t_hall)
            )

        hall_clean[i] = sensor_gains[i] * flux

    hall_power = np.mean(hall_clean ** 2)
    hall_noise_std = np.sqrt(hall_power / (10 ** (snr_db / 10.0)))
    hall_signals = hall_clean + rng.normal(0, hall_noise_std, hall_clean.shape)

    # --- 2. NON-CONTACT ACOUSTIC CHANNEL (1 mic @ 16 kS/s) ---
    n_ac = int(cfg.fs_acoustic * duration)
    t_ac = np.arange(n_ac) / cfg.fs_acoustic
    
    # Random microphone distance/coupling attenuation when realistic_interference=True
    mic_coupling = rng.uniform(0.45, 1.35) if realistic_interference else 1.0
    
    ac_clean = mic_coupling * (0.15 * np.sin(2 * np.pi * fr * t_ac) + 0.08 * np.sin(2 * np.pi * 2 * fr * t_ac))
    
    if fault_class == "imbalance" and severity > 0:
        # In non-contact acoustics, low-frequency rotor imbalance is partially attenuated by air coupling
        ac_imb_scale = 0.22 if realistic_interference else 0.60
        ac_clean += mic_coupling * (ac_imb_scale * severity) * np.sin(2 * np.pi * fr * t_ac)
        ac_clean += mic_coupling * (0.12 * severity) * np.sin(2 * np.pi * 2 * fr * t_ac)

    if fault_class == "bearing" and severity > 0:
        mod_env = (
            1.0
            + 0.85 * np.cos(2 * np.pi * bpfo * t_ac)
            + 0.30 * np.cos(2 * np.pi * 2 * bpfo * t_ac)
            + 0.10 * np.cos(2 * np.pi * 3 * bpfo * t_ac)
        )
        carrier = np.sin(2 * np.pi * cfg.f_res * t_ac)
        ac_clean += mic_coupling * (0.8 * severity) * mod_env * carrier

    if realistic_interference:
        # Add asynchronous neighbouring machine hum (random 20-32 Hz tone) + occasional factory clanks
        f_async = rng.uniform(19.0, 31.0)
        ac_clean += rng.uniform(0.05, 0.22) * np.sin(2 * np.pi * f_async * t_ac)

    ac_power = np.mean(ac_clean ** 2) + 1e-12
    ac_noise_std = np.sqrt(ac_power / (10 ** (snr_db / 10.0)))
    acoustic_signal = ac_clean + rng.normal(0, ac_noise_std, ac_clean.shape)

    return {
        "t_hall": t_hall,
        "hall": hall_signals,
        "t_acoustic": t_ac,
        "acoustic": acoustic_signal,
        "meta": {
            "fault_class": fault_class,
            "fe": fe,
            "fr": fr,
            "slip": slip,
            "bpfo": bpfo,
            "theta0": theta0,
            "severity": severity,
            "snr_db": snr_db,
            "fs_hall": cfg.fs_hall,
            "fs_acoustic": cfg.fs_acoustic
        }
    }
