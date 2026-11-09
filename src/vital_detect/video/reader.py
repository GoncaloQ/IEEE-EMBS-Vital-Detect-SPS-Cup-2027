from pathlib import Path

import cv2
import numpy as np

from vital_detect.signals.processing import validate_fps


class VideoReader:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise ValueError("Video is missing or empty")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            self.close()
            raise ValueError("Video decoder could not open file")
        try:
            fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            validate_fps(fps, 0.01)
            self.metadata = {
                "fps": fps,
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "reported_frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "processed_frame_count": 0,
                "duration_s": None,
                "timestamp_source": None,
            }
            if min(self.metadata["width"], self.metadata["height"]) <= 0:
                raise ValueError("Invalid video dimensions")
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.cap.release()

    def __iter__(self):
        initial = []
        for _ in range(2):
            ok, frame = self.cap.read()
            if not ok:
                break
            initial.append((frame, float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000))
        if not initial:
            raise ValueError("Video contains no decodable frames")
        use_pts = all(np.isfinite(t) and t >= 0 for _, t in initial)
        if len(initial) == 2:
            use_pts = use_pts and initial[1][1] > initial[0][1]
        if not use_pts and any(np.isfinite(t) and t != 0 for _, t in initial):
            raise ValueError(
                "Invalid initial decoder timestamps; fallback would hide a timebase error"
            )
        self.metadata["timestamp_source"] = (
            "opencv_pos_msec" if use_pts else "frame_index / metadata_fps (CFR assumption)"
        )
        index, previous, first = 0, None, None
        while True:
            if index < len(initial):
                frame, pts = initial[index]
            else:
                ok, frame = self.cap.read()
                if not ok:
                    break
                pts = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            timestamp = pts if use_pts else index / self.metadata["fps"]
            if not np.isfinite(timestamp) or (previous is not None and timestamp <= previous):
                raise ValueError("Nonmonotonic decoder timestamps; cannot mix timebases")
            if frame is None or frame.shape[:2] != (
                self.metadata["height"],
                self.metadata["width"],
            ):
                raise ValueError("Invalid decoded frame or changing resolution")
            if first is None:
                first = timestamp
            previous = timestamp
            index += 1
            self.metadata["processed_frame_count"] = index
            self.metadata["duration_s"] = timestamp - first + 1 / self.metadata["fps"]
            yield timestamp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        expected = self.metadata["reported_frame_count"]
        if expected > 0 and index < expected:
            raise ValueError(f"Premature decoding failure: decoded {index}/{expected} frames")
