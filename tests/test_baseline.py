"""Exercise the team's real numerical and ROI functions, with controlled landmarks."""

import importlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import write_video

from Round1_Baseline import channels_cheeks, channels_forehead, hr_extraction
from Round1_Baseline.evaluate import Options, evaluate, extract_signal
from Round1_Baseline.filters import band_pass_butter_filtering, smoothness_priors_detrend


def test_team_numerical_path_and_existing_fft_sign_behavior():
    t = np.arange(750) / 25
    raw = 120 + 3 * np.sin(2 * np.pi * 1.2 * t)
    detrended = smoothness_priors_detrend(raw, (25 / (2 * np.pi * 0.1)) ** 2)
    filtered = band_pass_butter_filtering(detrended, 25, [1, 2], 3)
    spectrum, frequency = hr_extraction.ffourier_transform(filtered, 25)
    expected = frequency[np.abs(spectrum).argmax()] * 60
    assert hr_extraction.get_hr_bpm(filtered, fps=25) == expected
    assert hr_extraction.extract_hr(raw, 25, [1, 2]) == expected
    assert expected == pytest.approx(72, abs=2)


def test_fft_sign_behavior_is_preserved(monkeypatch):
    # This limitation is documented, not silently corrected by consolidation.
    monkeypatch.setattr(
        hr_extraction, "ffourier_transform",
        lambda signal, fps: (np.array([0, 1, 4]), np.array([0, 0.5, -0.5])),
    )
    assert hr_extraction.get_hr_bpm(np.ones(3)) == -30


@pytest.mark.parametrize("fps", [25, 30, 29.97])
def test_sliding_windows_preserve_samples_and_center_times(fps):
    signal = np.arange(round(13 * fps))[::-1]
    windows, times = hr_extraction.sliding_window(signal, fps=fps)
    width, step = round(10 * fps), round(fps)
    expected = [signal[start:start + width] for start in range(0, len(signal) - width + 1, step)]
    np.testing.assert_array_equal(windows, expected)
    np.testing.assert_allclose(times, (np.arange(len(expected)) * step + width / 2) / fps)
    assert not windows.flags.writeable


@pytest.mark.parametrize("signal", [[], [1, 2, 3]])
def test_short_hr_trace_has_consistent_empty_results(signal):
    windows, times = hr_extraction.sliding_window(signal)
    assert windows.shape == (0, 300) and times.shape == (0,)
    time, hr = hr_extraction.extract_hr_signal(signal)
    assert time.size == 0 and hr == []


@pytest.mark.parametrize("kwargs", [
    {"fps": 0}, {"fps": float("nan")}, {"fps": float("inf")},
    {"window_size_seconds": -1}, {"step_size_seconds": 0},
    {"step_size_seconds": 0.001},
])
def test_invalid_window_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        hr_extraction.sliding_window(np.arange(600), **kwargs)


def test_hr_trace_uses_the_actual_frame_rate():
    fps = 25
    signal = np.sin(2 * np.pi * 1.2 * np.arange(325) / fps)
    time, hr = hr_extraction.extract_hr_signal(signal, fps=fps)
    np.testing.assert_array_equal(time, [5, 6, 7, 8])
    np.testing.assert_allclose(hr, 72)


def test_single_video_demo_writes_hr_plots_without_fake_reference(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fps = 25
    signal = 120 + 3 * np.sin(2 * np.pi * 1.2 * np.arange(325) / fps)
    for module, name in (
        (channels_forehead, "signal_extraction_main_forehead"),
        (channels_cheeks, "signal_extraction_main_cheeks"),
    ):
        monkeypatch.setattr(module, name, lambda *args: (signal, fps))
    hr_extraction.main()
    plots = list((tmp_path / "Results").glob("*.png"))
    assert len(plots) == 8
    assert all(path.stat().st_size > 0 for path in plots)
    assert "Round1_Baseline.evaluate" in capsys.readouterr().out


@pytest.mark.parametrize("roi", ["cheeks", "forehead"])
def test_real_team_roi_to_hr_evaluation(dataset, tmp_path, monkeypatch, roi):
    points = [SimpleNamespace(x=0.5, y=0.4) for _ in range(478)]
    for indices, cx in (
        ([117, 118, 101, 36, 205, 207, 187, 147, 123], 0.35),
        ([346, 347, 330, 266, 425, 427, 411, 376, 352], 0.65),
    ):
        for idx, angle in zip(indices, np.linspace(0, 2 * np.pi, len(indices), endpoint=False)):
            points[idx] = SimpleNamespace(x=cx + 0.1 * np.cos(angle), y=0.6 + 0.1 * np.sin(angle))
    for i, idx in enumerate([70, 63, 105, 66, 107, 336, 296, 334, 293, 300]):
        points[idx] = SimpleNamespace(x=0.3 + i * 0.04, y=0.35)
    for i, idx in enumerate([103, 67, 109, 10, 338, 297, 332]):
        points[idx] = SimpleNamespace(x=0.3 + i * 0.06, y=0.25)
    points[9], points[1] = SimpleNamespace(x=0.5, y=0.35), SimpleNamespace(x=0.5, y=0.5)
    detector_times = []

    class Detector:
        def detect_for_video(self, image, timestamp):
            detector_times.append(timestamp)
            return SimpleNamespace(face_landmarks=[points])

        def close(self):
            pass

    monkeypatch.setattr(
        channels_cheeks.vision.FaceLandmarker, "create_from_options", lambda _: Detector()
    )
    monkeypatch.setattr(channels_cheeks, "download_model_if_missing", lambda _: None)
    monkeypatch.setattr(channels_forehead, "download_model_if_missing", lambda _: None)
    summary = evaluate(
        dataset, tmp_path / roi, Options(roi=roi, window_enabled=False), ["subject1"]
    )
    assert summary["counts"]["recordings_successful"] == 1
    assert summary["metrics"]["mae_bpm"] < 3
    assert len(detector_times) == 550
    assert np.all(np.diff(detector_times) > 0)
    assert summary["signal_extractor"] == "team ROI"


def test_evaluation_adapter_uses_team_roi_and_rejects_count_mismatch(tmp_path, monkeypatch):
    video = tmp_path / "trace.avi"
    write_video(video, seconds=12)
    values = np.full((300, 3), [100, 120, 80.0])
    values[10] = 0

    def extractor(path, model):
        return *values.T, 25

    monkeypatch.setattr(channels_cheeks, "extract_upper_cheeks_rgb", extractor)
    t, signal, metadata = extract_signal(video, Options())
    assert len(t) == 300 and np.isnan(signal[10])
    assert metadata["invalid_frame_count"] == 1
    assert signal[11] == 120
    values = values[:-1]
    with pytest.raises(ValueError, match="frame count"):
        extract_signal(video, Options())


def test_missing_model_download_is_explicit_call_only(tmp_path, monkeypatch):
    from Round1_Baseline import utils

    calls = []
    monkeypatch.setattr(
        utils.urllib.request, "urlretrieve", lambda url, path: calls.append((url, path))
    )
    path = tmp_path / "models/face.task"
    utils.download_model_if_missing(path)
    assert path.parent.is_dir() and calls[0][1] == path
    path.write_bytes(b"already installed")
    utils.download_model_if_missing(path)
    assert len(calls) == 1


def test_imports_do_not_open_video_download_or_plot(tmp_path):
    code = """
import importlib
import cv2
import urllib.request
from matplotlib.figure import Figure

def fail(*args, **kwargs):
    raise AssertionError("Unexpected import-time operation")
cv2.VideoCapture = fail
urllib.request.urlretrieve = fail
Figure.savefig = fail
for name in ("channels_cheeks", "channels_forehead", "hr_extraction", "filters", "metrics", "utils", "evm", "evaluate", "dataset", "video", "reporting", "robustness"):
    importlib.import_module("Round1_Baseline." + name)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name in ("filters", "metrics", "utils", "evm"):
        assert importlib.import_module("Round1_Baseline." + name)
