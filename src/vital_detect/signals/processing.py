import numpy as np
from scipy import signal


def validate_fps(fps, high_hz):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")
    if not np.isfinite(high_hz) or high_hz <= 0 or high_hz >= fps / 2:
        raise ValueError("HR upper cutoff must be strictly below the Nyquist frequency")


def resample_rgb(timestamps, rgb, config):
    t, rgb = np.asarray(timestamps, float), np.asarray(rgb, float)
    if t.ndim != 1 or rgb.shape != (len(t), 3) or len(t) < 2:
        raise ValueError("Expected N timestamps and N x 3 RGB samples")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Video timestamps must be finite and strictly increasing")
    valid = np.isfinite(rgb).all(axis=1)
    fraction = float(valid.mean())
    if fraction < config.min_valid_fraction or valid.sum() < 2:
        raise ValueError("Insufficient valid ROI samples")
    tv, xv = t[valid], rgb[valid]

    fps = float(1 / np.median(np.diff(t)))
    validate_fps(fps, config.high_hz)
    if np.max(np.diff(tv)) > config.max_gap_s + 1e-9:
        raise ValueError("Signal gap exceeds max_gap_s; long gaps are never filled")
    if tv[-1] - tv[0] < config.min_duration_s - 1e-8:
        raise ValueError("Insufficient signal duration")
    count = int(np.floor((tv[-1] - tv[0]) * fps + 1e-7)) + 1

    grid = np.linspace(tv[0], tv[-1], count)
    fps = float((count - 1) / (tv[-1] - tv[0]))
    validate_fps(fps, config.high_hz)
    samples = np.column_stack([np.interp(grid, tv, xv[:, i]) for i in range(3)])
    return grid, samples, fps, fraction


def crop_interval(t, rgb, start, end, max_gap_s):
    if start < t[0] - 1e-9 or end > t[-1] + 1e-9 or end <= start:
        raise ValueError("Interval outside video coverage")
    middle = (t > start + 1e-9) & (t < end - 1e-9)

    def boundary(value):
        closest = int(np.argmin(abs(t - value)))
        if abs(t[closest] - value) <= 1e-9:
            return rgb[closest]
        right = np.searchsorted(t, value)
        left = right - 1
        if t[right] - t[left] > max_gap_s or not np.isfinite(rgb[[left, right]]).all():
            return np.full(3, np.nan)
        weight = (value - t[left]) / (t[right] - t[left])
        return (1 - weight) * rgb[left] + weight * rgb[right]

    return np.r_[start, t[middle], end], np.vstack([boundary(start), rgb[middle], boundary(end)])


def pos(rgb, fps, window_s=1.6):
    rgb = np.asarray(rgb, float)
    length = int(round(window_s * fps))
    if rgb.ndim != 2 or rgb.shape[1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("POS expects finite N x 3 RGB")
    if length < 3 or len(rgb) < length:
        raise ValueError("Insufficient samples for POS normalization window")
    h, weights = np.zeros(len(rgb)), np.zeros(len(rgb))
    projection = np.array([[0, 1, -1], [-2, 1, 1]])
    for start in range(len(rgb) - length + 1):
        clip = rgb[start : start + length]
        means = clip.mean(axis=0)
        if np.any(means <= 1e-8):
            raise ValueError("POS channel mean is zero or negative")
        s = projection @ (clip / means).T
        sd = np.std(s, axis=1)
        if sd[1] < 1e-12:
            raise ValueError("Degenerate POS projection; insufficient chromatic variation")
        pulse = s[0] + sd[0] / sd[1] * s[1]
        h[start : start + length] += pulse - pulse.mean()
        weights[start : start + length] += 1
    return h / weights


def pulse_trace(rgb, fps, config):
    validate_fps(fps, config.high_hz)
    raw = rgb[:, 1].copy() if config.method == "green" else pos(rgb, fps, config.pos_window_s)
    detrended = signal.detrend(raw, type="linear")
    sd = np.std(detrended)
    if not np.isfinite(sd) or sd < 1e-10:
        raise ValueError("Flat or invalid pulse signal")
    normalized = detrended / sd
    sos = signal.butter(
        config.filter_order, [config.low_hz, config.high_hz], btype="bandpass", fs=fps, output="sos"
    )
    try:
        filtered = signal.sosfiltfilt(sos, normalized)
    except ValueError as exc:
        raise ValueError("Signal too short for zero-phase filtering") from exc
    return normalized, filtered
