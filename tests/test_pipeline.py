import csv
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest
from conftest import controlled_signal

from Round1_Baseline import evaluate as module
from Round1_Baseline.dataset import discover
from Round1_Baseline.evaluate import Options, evaluate, evaluate_recording, main, prepare_interval


def test_outputs_failures_and_reproducibility(dataset, tmp_path):
    output = tmp_path / "out"
    summary = evaluate(dataset, output, signal_extractor=controlled_signal)
    assert summary["counts"] == dict(
        recordings_attempted=3,
        recordings_successful=2,
        recordings_failed=1,
        windows_attempted=4,
        windows_successful=4,
    )
    assert summary["metrics"]["n_pairs"] == 2
    assert summary["metrics"]["mae_bpm"] < 3
    assert {p.name for p in output.iterdir()} == {
        "per_video_results.csv",
        "per_window_results.csv",
        "summary_metrics.json",
        "config.json",
        "evaluation_failures.csv",
        "estimated_vs_reference.png",
        "bland_altman.png",
        "error_distribution.png",
        "evaluation_report.md",
    }
    with (output / "per_video_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[2]["status"] == "failed"
    assert "Missing file" in rows[2]["reason"]
    assert len(rows[0]["video_sha256"]) == 64
    assert (
        json.loads((output / "summary_metrics.json").read_text())["metrics"] == summary["metrics"]
    )
    assert json.loads((output / "config.json").read_text()) == Options().to_dict()
    for filename in ("estimated_vs_reference.png", "bland_altman.png", "error_distribution.png"):
        assert cv2.imread(str(output / filename)) is not None
    second = evaluate(dataset, tmp_path / "second", signal_extractor=controlled_signal)
    assert second["metrics"] == summary["metrics"]


def test_actual_estimator_is_called_and_labels_cannot_change_prediction(dataset, monkeypatch):
    recording = discover(dataset)[0]
    real = module.hr_extraction.extract_hr
    calls = []

    def spy(signal, fps, fc, N):
        calls.append((signal.copy(), fps, fc, N))
        return real(signal, fps, fc, N=N)

    monkeypatch.setattr(module.hr_extraction, "extract_hr", spy)
    options = Options(window_enabled=False)
    first, _ = evaluate_recording(recording, options, controlled_signal)
    lines = recording.reference_path.read_text().splitlines()
    lines[1] = " ".join("140" for _ in lines[1].split())
    recording.reference_path.write_text("\n".join(lines))
    altered, _ = evaluate_recording(recording, options, controlled_signal)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    assert calls[0][2:] == ([1.0, 2.0], 3)
    assert first["estimated_hr_bpm"] == altered["estimated_hr_bpm"]
    assert altered["reference_hr_bpm"] == pytest.approx(140)


def test_clock_assumption_and_exact_window_reference(dataset):
    recording = discover(dataset)[0]
    lines = recording.reference_path.read_text().splitlines()
    t = np.array([float(v) for v in lines[2].split()])
    lines[1] = " ".join(str(v) for v in 60 + t)
    lines[2] = " ".join(str(v) for v in t + 1000)
    recording.reference_path.write_text("\n".join(lines))
    row, _ = evaluate_recording(recording, Options(), controlled_signal)
    assert row["status"] == "failed" and "do not overlap" in row["reason"]
    row, windows = evaluate_recording(
        recording, Options(sync_policy="recording-start", window_s=10), controlled_signal
    )
    assert row["status"] == "paired"
    assert len(windows) >= 2
    for window in windows:
        assert window["status"] == "paired"
        assert window["reference_hr_bpm"] == pytest.approx(
            60 + (window["start_time_s"] + window["end_time_s"]) / 2
        )


def test_missing_samples_and_window_failures_remain_visible(dataset):
    def missing(path, options):
        t, x, meta = controlled_signal(path, options)
        x[(t > 5) & (t < 8)] = np.nan
        return t, x, meta

    row, windows = evaluate_recording(discover(dataset)[0], Options(window_s=10), missing)
    assert row["status"] == "failed"
    assert any(w["status"] == "paired" for w in windows)
    assert any(w["status"] == "failed" for w in windows)


def test_all_failed_reports_and_raw_negative_hr(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(module.hr_extraction, "extract_hr", lambda *a, **kw: -72)
    out = tmp_path / "failed"
    summary = evaluate(dataset, out, signal_extractor=controlled_signal)
    assert summary["counts"]["recordings_failed"] == 3
    assert summary["metrics"]["mae_bpm"] is None
    assert summary["metrics"]["bland_altman"]["lower_loa_bpm"] is None
    with (out / "per_video_results.csv").open() as handle:
        first = next(csv.DictReader(handle))
    assert first["raw_estimated_hr_bpm"] == "-72.0"
    assert "FFT sign" in first["reason"]
    assert "nonpositive" in (out / "evaluation_failures.csv").read_text()
    assert cv2.imread(str(out / "bland_altman.png")) is not None


def test_extractor_failure_does_not_abort_batch(dataset, tmp_path):
    def broken(*args):
        raise RuntimeError("detector unavailable")

    summary = evaluate(dataset, tmp_path / "broken", signal_extractor=broken)
    assert summary["counts"]["recordings_failed"] == 3
    assert "detector unavailable" in (tmp_path / "broken/evaluation_failures.csv").read_text()


def test_interval_interpolation_no_extrapolation_or_long_gaps():
    t = np.arange(600) / 25
    x = 100 + np.sin(2 * np.pi * 1.2 * t)
    x[100:102] = np.nan
    values, fps = prepare_interval(t, x, 0.02, 20.02, Options())
    assert np.isfinite(values).all() and fps == pytest.approx(25)
    assert values[0] == pytest.approx((x[0] + x[1]) / 2)
    x[100:120] = np.nan
    with pytest.raises(ValueError, match="gap"):
        prepare_interval(t, x, 0.02, 20.02, Options(min_valid_fraction=0.8))
    with pytest.raises(ValueError, match="coverage"):
        prepare_interval(t, x, -1, 20, Options())
    with pytest.raises(ValueError, match="timestamps"):
        prepare_interval(t[::-1], x, 0, 20, Options())
    with pytest.raises(ValueError, match="duration"):
        prepare_interval(t, np.ones_like(t), 0, 2, Options())
    with pytest.raises(ValueError, match="Flat"):
        prepare_interval(t, np.ones_like(t), 0, 20, Options())
    with pytest.raises(ValueError, match="Nyquist"):
        prepare_interval(np.arange(60) / 4, np.arange(60), 0, 12, Options())


def test_cli_saved_configuration_reproduction_and_exit_codes(dataset, tmp_path, monkeypatch):
    import re
    import shlex

    original = module.evaluate
    monkeypatch.setattr(
        module, "evaluate", lambda *args: original(*args, signal_extractor=controlled_signal)
    )
    output = tmp_path / "with spaces"
    args = [
        "--data-dir",
        str(dataset),
        "--output-dir",
        str(output),
        "--subjects",
        "subject2",
        "--no-windows",
    ]
    assert main(args) == 0
    assert main(args) == 2
    first = json.loads((output / "summary_metrics.json").read_text())
    assert first["counts"]["windows_attempted"] == 0
    command = shlex.split(
        re.search(
            r"```bash\n(.*?)\n```", (output / "evaluation_report.md").read_text(), re.S
        ).group(1)
    )
    assert main(command[3:]) == 0
    second = json.loads(
        (output.with_name(output.name + "_repeated") / "summary_metrics.json").read_text()
    )
    assert first["metrics"] == second["metrics"]
    assert first["selection"] == second["selection"]
    assert main(["--data-dir", str(dataset), "--output-dir", str(tmp_path / "all")]) == 1
    bad = tmp_path / "bad.json"
    bad.write_text('{"method": "pos"}')
    assert main([*args[:2], "--output-dir", str(tmp_path / "bad"), "--config", str(bad)]) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"low_hz": 3},
        {"filter_order": 0},
        {"stride_s": 0},
        {"window_s": 5},
        {"min_valid_fraction": -1},
        {"sync_policy": "guess"},
        {"reference_offset_s": 2},
        {"roi": "left_cheek"},
        {"channel": "x"},
        {"window_enabled": 1},
        {"max_gap_s": float("nan")},
    ],
)
def test_invalid_options(changes):
    with pytest.raises(ValueError):
        replace(Options(), **changes)
