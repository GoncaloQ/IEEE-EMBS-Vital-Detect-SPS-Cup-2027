from pathlib import Path

import cv2
import numpy as np
import pytest


class ControlledROI:
    def __init__(self, config):
        pass

    def sample(self, rgb, timestamp):
        return rgb.mean(axis=(0, 1)), "valid"

    def close(self):
        pass


def synthetic_rgb(fps=30, seconds=30, bpm=72):
    t = np.arange(int(fps * seconds)) / fps
    pulse = np.sin(2 * np.pi * bpm / 60 * t)

    rgb = np.array([160, 120, 90]) * (1 + pulse[:, None] * np.array([0.002, 0.012, 0.004]))
    return t, rgb


def write_video(path, fps=25, seconds=22, bpm=72, size=(96, 72)):
    path = Path(path)
    t, rgb = synthetic_rgb(fps, seconds, bpm)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("Synthetic MJPG encoder unavailable")
    try:
        for color in rgb:
            frame = np.empty((size[1], size[0], 3), np.uint8)

            frame[:] = np.round([90, 120 + 10 * (color[1] / 120 - 1) / 0.012, 160])
            writer.write(frame)
    finally:
        writer.release()
    return t


def write_gt(path, fps=25, seconds=22, bpm=72):
    t = np.arange(int(fps * seconds)) / fps
    rows = (np.sin(2 * np.pi * bpm / 60 * t), np.full(len(t), bpm), t)
    Path(path).write_text("\n".join(" ".join(f"{v:.9g}" for v in row) for row in rows) + "\n")


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "synthetic_dataset"
    for subject, bpm in (("subject1", 72), ("subject2", 96)):
        folder = root / subject
        folder.mkdir(parents=True)
        write_video(folder / "vid.avi", bpm=bpm)
        write_gt(folder / "ground_truth.txt", bpm=bpm)
    missing = root / "subject3"
    missing.mkdir()
    return root
