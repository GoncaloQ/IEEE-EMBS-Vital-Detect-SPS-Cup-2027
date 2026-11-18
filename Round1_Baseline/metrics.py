import numpy as np


def calculate_metrics(estimated, reference):
    e, r = np.asarray(estimated, float), np.asarray(reference, float)
    if e.ndim != 1 or e.shape != r.shape:
        raise ValueError("Paired HR arrays must be one-dimensional and equal length")
    if not np.isfinite(e).all() or not np.isfinite(r).all():
        raise ValueError("All pairs must be finite; exclude failures explicitly before metrics")
    n = len(e)
    with np.errstate(over="ignore", invalid="ignore"):
        d = e - r
    if not np.isfinite(d).all():
        raise ValueError("Paired differences exceed the finite numeric range")
    scale = float(np.max(abs(d))) if n else 0
    normalized = d / scale if scale else d
    reasons = {}
    mae = float(np.mean(abs(normalized)) * scale) if n else None
    rmse = float(np.sqrt(np.mean(normalized**2)) * scale) if n else None
    bias = float(np.mean(normalized) * scale) if n else None
    sd = float(np.std(normalized, ddof=1) * scale) if n >= 2 else None
    correlation = None
    if n < 2:
        reasons["pearson_r"] = "Fewer than two pairs"
        reasons["bland_altman_limits"] = "Fewer than two pairs; sample SD undefined"
    elif np.all(e == e[0]) or np.all(r == r[0]):
        reasons["pearson_r"] = "Constant predicted or reference HR"
    else:
        x = e / np.max(abs(e))
        y = r / np.max(abs(r))
        x, y = x - x.mean(), y - y.mean()
        x, y = x / np.max(abs(x)), y / np.max(abs(y))
        correlation = float(np.clip(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1))
    if not n:
        for name in ("mae_bpm", "rmse_bpm", "median_absolute_error_bpm", "bland_altman_bias"):
            reasons[name] = "No valid paired observations"
    lower = bias - 1.96 * sd if sd is not None else None
    upper = bias + 1.96 * sd if sd is not None else None
    if sd is not None and not np.isfinite([sd, lower, upper]).all():
        raise ValueError("Bland–Altman statistics exceed the finite numeric range")
    return {
        "n_pairs": n,
        "mae_bpm": mae,
        "rmse_bpm": rmse,
        "pearson_r": correlation,
        "median_absolute_error_bpm": float(np.median(abs(normalized)) * scale) if n else None,
        "bland_altman": {
            "bias_bpm": bias,
            "sample_sd_bpm": sd,
            "lower_loa_bpm": lower,
            "upper_loa_bpm": upper,
            "interpretation": "Descriptive normal-theory limits; not confidence intervals",
            "small_sample_warning": "Unstable limits; fewer than 30 pairs" if n < 30 else None,
        },
        "undefined_reasons": reasons,
    }
