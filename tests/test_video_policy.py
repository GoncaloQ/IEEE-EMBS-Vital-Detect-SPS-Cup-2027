import cv2
import numpy as np
import pytest

from vital_detect.video.reader import VideoReader


class FakeCapture:
    def __init__(self, times, fps=24, reported_count=None):
        self.times = times
        self.fps = fps
        self.reported_count = len(times) if reported_count is None else reported_count
        self.index = 0
        self.closed = False

    def isOpened(self):
        return True

    def get(self, key):
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_WIDTH: 32,
            cv2.CAP_PROP_FRAME_HEIGHT: 24,
            cv2.CAP_PROP_FRAME_COUNT: self.reported_count,
            cv2.CAP_PROP_POS_MSEC: self.times[max(0, self.index - 1)] * 1000,
        }[key]

    def read(self):
        if self.index == len(self.times):
            return False, None
        self.index += 1
        return True, np.zeros((24, 32, 3), np.uint8)

    def release(self):
        self.closed = True


def reader_with_backend(tmp_path, monkeypatch, times, **kwargs):
    path = tmp_path / "fake.avi"
    path.write_bytes(b"test-backend")
    backend = FakeCapture(times, **kwargs)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: backend)
    return path, backend


def test_missing_pts_falls_back_consistently(tmp_path, monkeypatch):
    path, backend = reader_with_backend(tmp_path, monkeypatch, [0, 0, 0, 0], fps=24)
    with VideoReader(path) as reader:
        samples = list(reader)
        np.testing.assert_allclose([t for t, _ in samples], np.arange(4) / 24)
        assert "CFR assumption" in reader.metadata["timestamp_source"]
    assert backend.closed


def test_irregular_pts_preserved_not_replaced_by_nominal_rate(tmp_path, monkeypatch):
    times = [2, 2.04, 2.09, 2.15]
    path, _ = reader_with_backend(tmp_path, monkeypatch, times)
    with VideoReader(path) as reader:
        assert [t for t, _ in reader] == pytest.approx(times)
        assert reader.metadata["timestamp_source"] == "opencv_pos_msec"


def test_regressing_pts_and_truncated_known_frame_count_fail(tmp_path, monkeypatch):
    path, backend = reader_with_backend(tmp_path, monkeypatch, [0, 0.04, 0.03])
    with pytest.raises(ValueError, match="Nonmonotonic"):
        with VideoReader(path) as reader:
            list(reader)
    assert backend.closed
    path, _ = reader_with_backend(tmp_path, monkeypatch, [0, 0.04], reported_count=4)
    with pytest.raises(ValueError, match="Premature"):
        with VideoReader(path) as reader:
            list(reader)


def test_first_timestamp_regression_cannot_be_hidden_by_fps_fallback(tmp_path, monkeypatch):
    path, _ = reader_with_backend(tmp_path, monkeypatch, [2, 1.9, 2.1])
    with pytest.raises(ValueError, match="timestamp"):
        with VideoReader(path) as reader:
            list(reader)


@pytest.mark.parametrize("fps", [0, -1, float("nan")])
def test_bad_metadata_fps_fails_and_releases(tmp_path, monkeypatch, fps):
    path, backend = reader_with_backend(tmp_path, monkeypatch, [0, 0.04], fps=fps)
    with pytest.raises(ValueError, match="FPS"):
        VideoReader(path)
    assert backend.closed
