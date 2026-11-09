import csv
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest
from conftest import ControlledROI, write_video

from vital_detect.cli import main
from vital_detect.config import Config
from vital_detect.datasets.ubfc import discover
from vital_detect.pipeline import evaluate, evaluate_recording, extract_trace, infer
from vital_detect.roi.face import FaceROI, landmark_mask
from vital_detect.video.reader import VideoReader


def test_incremental_video_rgb_timestamps_and_fps(tmp_path):
    path = tmp_path / "periodic.avi"
    write_video(path, fps=24, seconds=12)
    with VideoReader(path) as reader:
        t, frames = zip(*reader)
        assert reader.metadata["fps"] == pytest.approx(24)
        assert reader.metadata["processed_frame_count"] == 288
        assert reader.metadata["duration_s"] == pytest.approx(12)
    assert np.all(np.diff(t) > 0)

    assert frames[0][:, :, 0].mean() > frames[0][:, :, 2].mean()
    result = infer(path, Config(), ControlledROI)
    assert abs(result["estimated_hr_bpm"] - 72) < 2


def test_all_outputs_failures_and_reproducibility(dataset, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = evaluate(dataset, a, Config(), roi_factory=ControlledROI)
    second = evaluate(dataset, b, Config(), roi_factory=ControlledROI)
    assert first["metrics"] == second["metrics"]
    assert first["counts"]["recordings_attempted"] == 3
    assert first["counts"]["recordings_successful"] == 2
    assert first["counts"]["recordings_failed"] == 1
    assert first["metrics"]["mae_bpm"] < 2
    assert {p.name for p in a.iterdir()} == {
        "per_video_results.csv",
        "per_window_results.csv",
        "summary_metrics.json",
        "evaluation_failures.csv",
        "estimated_vs_reference.png",
        "bland_altman.png",
        "error_distribution.png",
        "evaluation_report.md",
        "config.json",
    }
    with (a / "per_video_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["subject_id"] == "subject1"
    assert rows[2]["status"] == "failed"
    assert len(rows[0]["video_sha256"]) == 64
    summary = json.loads((a / "summary_metrics.json").read_text())
    assert summary["window_metrics_descriptive_only"]["n_pairs"] > 0
    assert summary["metrics"]["n_pairs"] == 2
    assert json.loads((a / "config.json").read_text()) == Config().to_dict()
    for filename in ("estimated_vs_reference.png", "bland_altman.png", "error_distribution.png"):
        assert cv2.imread(str(a / filename)) is not None


def test_reference_never_influences_prediction(dataset):
    recording = discover(dataset)[0]
    row, _ = evaluate_recording(recording, Config(), ControlledROI)
    lines = recording.reference_path.read_text().splitlines()
    lines[1] = " ".join("140" for _ in lines[1].split())
    recording.reference_path.write_text("\n".join(lines) + "\n")
    altered, _ = evaluate_recording(recording, Config(), ControlledROI)
    assert row["estimated_hr_bpm"] == altered["estimated_hr_bpm"]
    assert altered["reference_hr_bpm"] == pytest.approx(140)


def test_unknown_overlap_fails_until_explicit_origin_assumption(dataset):
    recording = discover(dataset)[0]
    lines = recording.reference_path.read_text().splitlines()
    lines[2] = " ".join(str(float(value) + 1000) for value in lines[2].split())
    recording.reference_path.write_text("\n".join(lines) + "\n")
    row, _ = evaluate_recording(recording, Config(), ControlledROI)
    assert row["status"] == "failed"
    assert "do not overlap" in row["reason"]
    row, _ = evaluate_recording(recording, Config(sync_policy="recording-start"), ControlledROI)
    assert row["status"] == "paired"
    assert abs(row["estimated_hr_bpm"] - 72) < 2


def test_windows_reference_changes_with_time(dataset):
    recording = discover(dataset)[0]
    lines = recording.reference_path.read_text().splitlines()
    times = np.array([float(v) for v in lines[2].split()])
    lines[1] = " ".join(str(v) for v in (60 + times))
    recording.reference_path.write_text("\n".join(lines) + "\n")
    _, windows = evaluate_recording(recording, Config(window_s=10, stride_s=5), ControlledROI)
    assert len(windows) >= 2
    for window in windows:
        assert window["status"] == "paired"
        expected = 60 + (window["start_time_s"] + window["end_time_s"]) / 2
        assert window["reference_hr_bpm"] == pytest.approx(expected)
    assert windows[1]["reference_hr_bpm"] > windows[0]["reference_hr_bpm"]


def test_failed_face_frames_stay_nan(dataset):
    class MissingROI(ControlledROI):
        def sample(self, rgb, timestamp):
            return (
                (np.full(3, np.nan), "no_face")
                if 5 < timestamp < 8
                else super().sample(rgb, timestamp)
            )

    path = discover(dataset)[0].video_path
    trace = extract_trace(path, Config(), MissingROI)
    assert trace.metadata["invalid_frame_count"] > 0
    assert trace.metadata["valid_frame_count"] + trace.metadata["invalid_frame_count"] == len(
        trace.rgb
    )
    assert not np.any(trace.rgb == 0)
    row, windows = evaluate_recording(discover(dataset)[0], Config(window_s=10), MissingROI)
    assert row["status"] == "failed"
    assert any(w["status"] == "paired" for w in windows)


def test_all_failed_still_generates_null_metrics(dataset, tmp_path):
    class BrokenROI(ControlledROI):
        def sample(self, rgb, timestamp):
            raise RuntimeError("Controlled detector failure")

    out = tmp_path / "failed"
    summary = evaluate(dataset, out, Config(), roi_factory=BrokenROI)
    assert summary["metrics"]["mae_bpm"] is None
    assert summary["metrics"]["bland_altman"]["lower_loa_bpm"] is None
    assert (out / "bland_altman.png").exists()
    assert "RuntimeError" in (out / "evaluation_failures.csv").read_text()
    with (out / "per_video_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["fps"] == "25.0"
    assert rows[0]["invalid_frame_count"] == "1"


def test_corrupt_empty_and_missing_video(tmp_path):
    path = tmp_path / "invalid.avi"
    for value in (b"", b"corrupted-video"):
        path.write_bytes(value)
        with pytest.raises(ValueError):
            VideoReader(path)
    with pytest.raises(ValueError):
        VideoReader(tmp_path / "missing.mp4")


def test_masks_valid_bounds_and_empty():
    from vital_detect.roi.face import LEFT_CHEEK, RIGHT_CHEEK

    points = np.full((478, 2), 0.5)
    angle = np.linspace(0, 2 * np.pi, len(LEFT_CHEEK), endpoint=False)
    points[LEFT_CHEEK] = np.c_[0.7 + 0.1 * np.cos(angle), 0.5 + 0.12 * np.sin(angle)]
    points[RIGHT_CHEEK] = np.c_[0.3 + 0.1 * np.cos(angle), 0.5 + 0.12 * np.sin(angle)]
    both = landmark_mask(points, (100, 200, 3), "cheeks")
    left = landmark_mask(points, (100, 200, 3), "left_cheek")
    right = landmark_mask(points, (100, 200, 3), "right_cheek")
    np.testing.assert_array_equal(both, left | right)
    assert both[0, 0] == 0
    with pytest.raises(ValueError, match="small"):
        landmark_mask(np.full((478, 2), 0.5), (100, 200), "cheeks")
    points[LEFT_CHEEK] = 2
    with pytest.raises(ValueError, match="boundaries"):
        landmark_mask(points, (100, 200), "left_cheek")


def test_cli_bad_config_missing_model_and_missing_data(tmp_path, capsys):
    assert main(["infer", "--video", str(tmp_path / "none.avi")]) == 2
    assert (
        main(["evaluate", "--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")]) == 2
    )
    config = tmp_path / "bad.json"
    config.write_text('{"made_up": 1}')
    assert main(["infer", "--video", "none.avi", "--config", str(config)]) == 2
    assert "Unknown configuration" in capsys.readouterr().err
    with pytest.raises(ValueError, match="Missing Face"):
        FaceROI(replace(Config(), model_path=str(tmp_path / "absent.task")))


def test_cli_full_dataset_command_and_config_override(dataset, tmp_path, monkeypatch):
    from vital_detect import pipeline

    original = pipeline.evaluate
    monkeypatch.setattr(
        pipeline, "evaluate", lambda *args: original(*args, roi_factory=ControlledROI)
    )
    output = tmp_path / "cli"
    args = [
        "evaluate",
        "--dataset",
        "ubfc",
        "--data-dir",
        str(dataset),
        "--subjects",
        "subject2",
        "subject1",
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0
    assert main(args) == 2
    summary = json.loads((output / "summary_metrics.json").read_text())
    assert summary["counts"]["subjects_evaluated"] == 2
    config = output / "config.json"
    assert main([*args[:-1], str(tmp_path / "rerun"), "--config", str(config)]) == 0


def test_report_reproduction_command_and_disabled_windows(dataset, tmp_path, monkeypatch):
    import re
    import shlex

    from vital_detect import pipeline

    original = pipeline.evaluate
    monkeypatch.setattr(
        pipeline, "evaluate", lambda *args: original(*args, roi_factory=ControlledROI)
    )
    output = tmp_path / "results with spaces"
    assert (
        main(
            [
                "evaluate",
                "--data-dir",
                str(dataset),
                "--subjects",
                "subject2",
                "--output-dir",
                str(output),
                "--no-windows",
            ]
        )
        == 0
    )
    first = json.loads((output / "summary_metrics.json").read_text())
    assert first["counts"]["windows_attempted"] == 0
    assert first["configuration"]["window_enabled"] is False
    command = shlex.split(
        re.search(
            r"```bash\n(.*?)\n```", (output / "evaluation_report.md").read_text(), re.S
        ).group(1)
    )
    assert main(command[1:]) == 0
    second = json.loads(
        (output.with_name(output.name + "_repeated") / "summary_metrics.json").read_text()
    )
    assert second["metrics"] == first["metrics"]
    assert second["selection"] == first["selection"]
