import shutil

import pytest
from conftest import ControlledROI, write_video

from vital_detect.config import Config
from vital_detect.robustness.benchmark import benchmark, probe, transform

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg/ffprobe optional external tools unavailable")
@pytest.mark.parametrize("condition", ["C4_VGA", "C4_QVGA", "C5_10fps_500kbps"])
def test_real_encoding_meets_measured_conditions(tmp_path, condition):
    source = tmp_path / "source.avi"
    write_video(source, fps=25, seconds=12, size=(640, 480))
    target = tmp_path / ("small.mp4" if condition.startswith("C5") else "small.avi")
    evidence = transform(source, target, condition)
    measured = evidence["measured"]
    assert measured == probe(target)
    if condition == "C4_QVGA":
        assert (measured["width"], measured["height"]) == (320, 240)
    elif condition == "C5_10fps_500kbps":
        assert measured["fps"] == pytest.approx(10)
        assert measured["bitrate_bps"] <= 500_000
    else:
        assert (measured["width"], measured["height"]) == (640, 480)


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg/ffprobe optional external tools unavailable")
def test_full_benchmark_outputs_and_native_not_upscaled(dataset, tmp_path):
    result = benchmark(
        dataset, tmp_path / "bench", Config(), ["subject1"], roi_factory=ControlledROI
    )
    assert len(result["conditions"]) == 4
    for name, summary in result["conditions"].items():
        assert summary["counts"]["recordings_successful"] == 1, name
    assert (tmp_path / "bench" / "condition_comparison.csv").exists()
    assert result["transformations"][0]["source"]["width"] == 96
    assert result["transformations"][0]["measured"]["width"] == 96
    import csv

    with (tmp_path / "bench" / "condition_comparison.csv").open() as handle:
        assert all(row["same_interval"] == "True" for row in csv.DictReader(handle))


def test_missing_ffmpeg_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="ffprobe"):
        probe(tmp_path / "none.avi")
    with pytest.raises(ValueError, match="ffmpeg"):
        benchmark(tmp_path, tmp_path / "out", Config())


@pytest.mark.parametrize("bitrate", [None, 501_000])
def test_unverified_or_excess_bitrate_cannot_pass_c5(monkeypatch, tmp_path, bitrate):
    import importlib

    module = importlib.import_module("vital_detect.robustness.benchmark")
    source = {
        "width": 640,
        "height": 480,
        "fps": 30,
        "bitrate_bps": 1_000_000,
        "container_bitrate_bps": 1_000_000,
        "duration_s": 30,
        "start_time_s": 0,
    }
    encoded = {**source, "fps": 10, "bitrate_bps": bitrate, "container_bitrate_bps": bitrate}
    metadata = iter([source, encoded])
    monkeypatch.setattr(module, "probe", lambda path: next(metadata))
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/ffmpeg")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="bitrate"):
        transform(tmp_path / "source.avi", tmp_path / "encoded.mp4", "C5_10fps_500kbps")


@pytest.mark.parametrize("condition", ["C4_QVGA", "C5_10fps_500kbps"])
def test_wrong_transformation_fps_is_rejected(monkeypatch, tmp_path, condition):
    import importlib

    module = importlib.import_module("vital_detect.robustness.benchmark")
    source = {
        "width": 640,
        "height": 480,
        "fps": 30,
        "bitrate_bps": 100_000,
        "container_bitrate_bps": 100_000,
        "duration_s": 30,
        "start_time_s": 0,
    }
    encoded = {**source, "width": 320, "height": 240, "fps": 8}
    metadata = iter([source, encoded])
    monkeypatch.setattr(module, "probe", lambda path: next(metadata))
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/ffmpeg")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="FPS"):
        transform(tmp_path / "source.avi", tmp_path / "encoded.avi", condition)


def test_probe_uses_container_when_stream_fields_are_unavailable(monkeypatch, tmp_path):
    import importlib
    import json
    from types import SimpleNamespace

    module = importlib.import_module("vital_detect.robustness.benchmark")
    payload = {
        "streams": [
            {
                "width": 640,
                "height": 480,
                "avg_frame_rate": "30/1",
                "duration": "N/A",
                "start_time": "N/A",
                "bit_rate": "N/A",
            }
        ],
        "format": {"duration": "12", "start_time": "0", "bit_rate": "400000"},
    }
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/ffprobe")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )
    result = probe(tmp_path / "source.avi")
    assert result["duration_s"] == 12
    assert result["bitrate_bps"] == 400_000


def test_wrong_requested_resolution_and_unknown_origin_are_rejected(monkeypatch, tmp_path):
    import importlib

    module = importlib.import_module("vital_detect.robustness.benchmark")
    source = {
        "width": 640,
        "height": 480,
        "fps": 30,
        "bitrate_bps": 100_000,
        "container_bitrate_bps": 100_000,
        "duration_s": 30,
        "start_time_s": 0,
    }
    for encoded, message in [
        ({**source, "width": 160, "height": 120}, "resolution"),
        ({**source, "start_time_s": None}, "unknown"),
    ]:
        metadata = iter([source, encoded])
        monkeypatch.setattr(module, "probe", lambda path: next(metadata))
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/ffmpeg")
        monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
        with pytest.raises(ValueError, match=message):
            transform(tmp_path / "source.avi", tmp_path / "encoded.avi", "C4_QVGA")


def test_failed_source_probe_cannot_silently_use_full_interval(dataset, tmp_path, monkeypatch):
    import csv
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("vital_detect.robustness.benchmark")
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/tool")

    def invalid_probe(path):
        raise ValueError("unknown duration")

    monkeypatch.setattr(module, "probe", invalid_probe)
    monkeypatch.setattr(module, "_run", lambda command: SimpleNamespace(stdout="ffmpeg test"))
    summary = benchmark(
        dataset, tmp_path / "out", Config(), ["subject1"], constraints=(), roi_factory=ControlledROI
    )
    assert summary["conditions"]["native"]["counts"]["recordings_failed"] == 1
    with (tmp_path / "out/native/per_video_results.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert "common benchmark interval" in row["reason"]
    assert row["method"] == "green"


def test_ffmpeg_diagnostic_is_preserved(monkeypatch):
    import importlib
    import subprocess

    module = importlib.import_module("vital_detect.robustness.benchmark")

    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="unsupported codec")

    monkeypatch.setattr(module.subprocess, "run", failed)
    with pytest.raises(ValueError, match="unsupported codec"):
        module._run(["ffmpeg", "-version"])
