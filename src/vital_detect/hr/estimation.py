import numpy as np
from scipy.signal import periodogram

from vital_detect.signals.processing import pulse_trace, resample_rgb


def estimate(timestamps, rgb, config):
    grid, samples, fps, fraction = resample_rgb(timestamps, rgb, config)
    raw, filtered = pulse_trace(samples, fps, config)
    n = len(grid)

    nfft = max(n, 8 * 2 ** int(np.ceil(np.log2(n))))
    f, raw_power = periodogram(raw, fs=fps, window="hann", nfft=nfft, detrend=False)
    _, power = periodogram(filtered, fs=fps, window="hann", nfft=nfft, detrend=False)
    band = (f >= config.low_hz) & (f <= config.high_hz)
    total = raw_power[1:].sum()
    band_fraction = float(raw_power[band].sum() / total) if total > 0 else 0.0
    if band_fraction < config.min_band_power_fraction:
        raise ValueError("Insufficient physiological-band power; likely out-of-band artifact")
    indices = np.flatnonzero(band)
    if not len(indices) or power[band].sum() <= 1e-12:
        raise ValueError("No usable spectral power in physiological band")
    peak = indices[np.argmax(power[band])]

    neighborhood = band & (np.abs(f - f[peak]) <= max(0.1, 1.5 * fps / n))
    concentration = float(power[neighborhood].sum() / power[band].sum())
    if concentration < config.min_peak_power_fraction:
        raise ValueError("Diffuse spectrum; insufficient peak concentration")
    if peak == indices[0] or peak == indices[-1]:
        raise ValueError("Spectral peak touches HR band boundary; unsupported estimate")
    return {
        "estimated_hr_bpm": float(f[peak] * 60),
        "start_time_s": float(grid[0]),
        "end_time_s": float(grid[-1]),
        "effective_fps": fps,
        "valid_fraction": fraction,
        "band_power_fraction": band_fraction,
        "peak_power_fraction": concentration,
        "frequency_resolution_bpm": float(60 * fps / n),
    }
