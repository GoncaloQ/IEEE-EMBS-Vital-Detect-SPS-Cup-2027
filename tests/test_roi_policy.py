from types import SimpleNamespace

import numpy as np
import pytest

from vital_detect.config import Config
from vital_detect.roi.face import FaceROI


def fake_face_roi(faces):
    roi = FaceROI.__new__(FaceROI)
    roi.config = Config()
    roi.mp = SimpleNamespace(Image=lambda **kwargs: None, ImageFormat=SimpleNamespace(SRGB=0))
    roi.detector = SimpleNamespace(
        detect_for_video=lambda *args: SimpleNamespace(face_landmarks=faces)
    )
    roi.previous = None
    roi.last_center = None
    roi.last_size = None
    roi.last_ms = -1
    return roi


def test_two_faces_cannot_contribute_to_one_trace():
    roi = fake_face_roi([[], []])
    with pytest.raises(ValueError, match="Multiple faces"):
        roi.sample(np.zeros((20, 20, 3), np.uint8), 0)


def test_no_detection_returns_nan_and_clears_smoothing():
    roi = fake_face_roi([])
    roi.previous = np.ones((478, 2))
    sample, status = roi.sample(np.zeros((20, 20, 3), np.uint8), 0)
    assert np.isnan(sample).all()
    assert status == "no_face"
    assert roi.previous is None


def test_abrupt_geometry_cannot_silently_reassign_target():
    face = [SimpleNamespace(x=0.8, y=0.8) for _ in range(478)]
    roi = fake_face_roi([face])
    roi.last_center = np.array([0.2, 0.2])
    roi.last_size = 0.2
    with pytest.raises(ValueError, match="continuity"):
        roi.sample(np.zeros((20, 20, 3), np.uint8), 0)


def test_forehead_mask_rgb_values_and_nonfinite_landmarks():
    from vital_detect.roi.face import FOREHEAD

    points = np.full((478, 2), 0.5)
    angle = np.linspace(0, 2 * np.pi, len(FOREHEAD), endpoint=False)
    points[FOREHEAD] = np.c_[0.5 + 0.2 * np.cos(angle), 0.25 + 0.15 * np.sin(angle)]
    face = [SimpleNamespace(x=x, y=y) for x, y in points]
    roi = fake_face_roi([face])
    roi.config = Config(roi="forehead")
    rgb = np.full((100, 120, 3), [200, 100, 50], np.uint8)
    values, status = roi.sample(rgb, 0)
    assert status == "valid"
    np.testing.assert_allclose(values, [200, 100, 50])
    face[0].x = np.nan
    values, status = roi.sample(rgb, 0.04)
    assert status == "invalid_landmarks"
    assert np.isnan(values).all()
