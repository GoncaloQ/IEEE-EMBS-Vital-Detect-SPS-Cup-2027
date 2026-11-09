from pathlib import Path

import cv2
import numpy as np

RIGHT_CHEEK = [117, 118, 101, 36, 205, 207, 187, 147, 123]
LEFT_CHEEK = [346, 347, 330, 266, 425, 427, 411, 376, 352]

FOREHEAD = [103, 67, 109, 10, 338, 297, 332, 336, 296, 9, 66, 107]


def landmark_mask(landmarks, shape, roi):
    height, width = shape[:2]
    points = np.asarray(landmarks, float)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2:
        raise ValueError("Expected at least 468 normalized 2D facial landmarks")
    groups = {
        "forehead": [FOREHEAD],
        "left_cheek": [LEFT_CHEEK],
        "right_cheek": [RIGHT_CHEEK],
        "cheeks": [LEFT_CHEEK, RIGHT_CHEEK],
    }
    if roi not in groups:
        raise ValueError("Unknown ROI")
    mask = np.zeros((height, width), np.uint8)
    for group in groups[roi]:
        polygon = points[group]
        if not np.isfinite(polygon).all() or np.any((polygon < 0) | (polygon > 1)):
            raise ValueError("ROI landmarks outside image boundaries")
        polygon = polygon * [width - 1, height - 1]

        polygon = polygon.mean(axis=0) + 0.8 * (polygon - polygon.mean(axis=0))
        hull = cv2.convexHull(np.round(polygon).astype(np.int32))
        region = np.zeros_like(mask)
        cv2.fillConvexPoly(region, hull, 255)
        if np.count_nonzero(region) < 16:
            raise ValueError("ROI mask empty or too small")
        mask |= region
    return mask


class FaceROI:
    def __init__(self, config):
        if not Path(config.model_path).is_file():
            raise ValueError("Missing Face Landmarker model; run vital-detect download-model")
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp, self.config = mp, config
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=config.model_path, delegate=python.BaseOptions.Delegate.CPU
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.previous = None
        self.last_center = None
        self.last_size = None
        self.last_ms = -1

    def close(self):
        self.detector.close()

    def sample(self, rgb, timestamp):
        ms = int(round(timestamp * 1000))
        if ms <= self.last_ms:
            raise ValueError("Frame timestamps too close for MediaPipe millisecond timebase")
        self.last_ms = ms
        result = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ms
        )
        if len(result.face_landmarks) > 1:
            raise ValueError(
                "Multiple faces detected; single-subject baseline cannot select identity"
            )
        if not result.face_landmarks:
            self.previous = None
            return np.full(3, np.nan), "no_face"
        points = np.array([[p.x, p.y] for p in result.face_landmarks[0]])
        if points.ndim != 2 or points.shape[0] < 468 or not np.isfinite(points).all():
            self.previous = None
            return np.full(3, np.nan), "invalid_landmarks"
        center = points.mean(axis=0)
        size = np.linalg.norm(np.ptp(points, axis=0))
        if self.last_center is not None:
            if (
                np.linalg.norm(center - self.last_center) > 0.5 * self.last_size
                or not 0.5 < size / self.last_size < 2
            ):
                raise ValueError("Abrupt face geometry change; target continuity is uncertain")
        self.last_center, self.last_size = center, max(size, 1e-8)
        alpha = self.config.smoothing_alpha
        smoothed = points if self.previous is None else alpha * points + (1 - alpha) * self.previous
        self.previous = smoothed
        try:
            mask = landmark_mask(smoothed, rgb.shape, self.config.roi)
        except ValueError:
            return np.full(3, np.nan), "invalid_mask"
        return np.array(cv2.mean(rgb, mask=mask)[:3]), "valid"
