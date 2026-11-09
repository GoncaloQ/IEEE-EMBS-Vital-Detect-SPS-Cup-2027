from dataclasses import replace

import numpy as np
import pytest
from conftest import synthetic_rgb

from vital_detect.config import Config
from vital_detect.hr.estimation import estimate
from vital_detect.signals.processing import pos, resample_rgb, validate_fps


@pytest.mark.parametrize("fps", [10, 15, 24, 29.97, 30, 60])
@pytest.mark.parametrize("bpm", [54, 72, 113, 162])
@pytest.mark.parametrize("method", ["green", "pos"])
def test_known_heart_rates(fps, bpm, method):
    t, rgb = synthetic_rgb(fps, 30, bpm)
    result = estimate(t, rgb, Config(method=method))
    assert abs(result["estimated_hr_bpm"] - bpm) < 1
    assert result["effective_fps"] == pytest.approx(fps)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps(fps):
    with pytest.raises(ValueError, match="FPS"):
        validate_fps(fps, 3)


@pytest.mark.parametrize("fps", [4, 6])
def test_nyquist(fps):
    t, rgb = synthetic_rgb(fps)
    with pytest.raises(ValueError, match="Nyquist"):
        estimate(t, rgb, Config())


@pytest.mark.parametrize("bpm", [12, 240])
def test_out_of_band_not_reported_as_plausible_hr(bpm):
    t, rgb = synthetic_rgb(bpm=bpm)
    with pytest.raises(ValueError, match="band"):
        estimate(t, rgb, Config())


def test_short_flat_and_missing_signals():
    t, rgb = synthetic_rgb(seconds=3)
    with pytest.raises(ValueError, match="duration"):
        estimate(t, rgb, Config())
    t, rgb = synthetic_rgb()
    with pytest.raises(ValueError, match="Flat"):
        estimate(t, np.full_like(rgb, 100), Config())
    for bad in (np.nan, np.inf):
        missing = rgb.copy()
        missing[100:150] = bad
        with pytest.raises(ValueError):
            estimate(t, missing, Config())


def test_small_gaps_interpolate_no_zeros_and_edges_trim():
    t, rgb = synthetic_rgb()
    rgb[100:102] = np.nan
    rgb[:2] = np.nan
    result = estimate(t, rgb, Config())
    assert result["start_time_s"] == pytest.approx(t[2])
    assert abs(result["estimated_hr_bpm"] - 72) < 1
    grid, values, _, _ = resample_rgb(t, rgb, Config())
    assert grid[0] == t[2]
    assert values.min() > 50


def test_long_gap_and_low_coverage():
    t, rgb = synthetic_rgb()
    rgb[100:115] = np.nan
    with pytest.raises(ValueError, match="gap"):
        estimate(t, rgb, Config())
    rgb[:300] = np.nan
    with pytest.raises(ValueError, match="valid ROI"):
        estimate(t, rgb, Config())


def test_irregular_timestamps_use_actual_rate():
    t, _ = synthetic_rgb(fps=25)
    t += 0.001 * np.sin(np.arange(len(t)))
    pulse = np.sin(2 * np.pi * 1.2 * t)
    rgb = 100 + pulse[:, None] * [1, 3, 1]
    assert abs(estimate(t, rgb, Config())["estimated_hr_bpm"] - 72) < 1
    for invalid in (t[::-1], np.zeros_like(t), np.full_like(t, np.nan)):
        with pytest.raises(ValueError, match="timestamps"):
            estimate(invalid, rgb, Config())


def test_pos_matches_independent_projection_implementation():
    _, rgb = synthetic_rgb(fps=20, seconds=3)
    n, length = len(rgb), 32
    expected, counts = np.zeros(n), np.zeros(n)
    for end in range(length, n + 1):
        c = rgb[end - length : end].T
        c = c / c.mean(axis=1, keepdims=True)
        a, b = c[1] - c[2], -2 * c[0] + c[1] + c[2]
        h = a + np.std(a) / np.std(b) * b
        expected[end - length : end] += h - h.mean()
        counts[end - length : end] += 1
    np.testing.assert_allclose(pos(rgb, 20), expected / counts, atol=1e-14)
    with pytest.raises(ValueError, match="Degenerate"):
        pos(np.ones((60, 3)), 20)
    with pytest.raises(ValueError, match="mean"):
        pos(np.zeros((60, 3)), 20)


def test_diffuse_noise_is_rejected():
    t, rgb = synthetic_rgb()
    rgb[:, 1] = 100 + np.random.default_rng(42).normal(size=len(t))
    with pytest.raises(ValueError):
        estimate(t, rgb, Config())


def test_minimum_duration_means_observed_interval():
    t, rgb = synthetic_rgb(fps=30, seconds=10)
    with pytest.raises(ValueError, match="duration"):
        estimate(t, rgb, Config(min_duration_s=10))


def test_disabled_windows_do_not_restrict_recording_duration():
    config = Config(window_enabled=False, min_duration_s=20)
    t, rgb = synthetic_rgb(seconds=30)
    assert abs(estimate(t, rgb, config)["estimated_hr_bpm"] - 72) < 1


@pytest.mark.parametrize(
    "changes",
    [
        {"low_hz": 4},
        {"filter_order": 0},
        {"stride_s": 0},
        {"window_s": 5},
        {"min_valid_fraction": -1},
        {"sync_policy": "guess"},
        {"reference_offset_s": 2},
        {"smoothing_alpha": float("nan")},
        {"method": "oracle"},
    ],
)
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes)
