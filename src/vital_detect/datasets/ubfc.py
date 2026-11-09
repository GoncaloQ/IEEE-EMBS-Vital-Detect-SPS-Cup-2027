import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Recording:
    subject_id: str
    recording_id: str
    video_path: Path
    reference_path: Path

    def validate(self):
        folder = self.video_path.parent
        if not re.fullmatch(r"subject[0-9]+", self.subject_id):
            raise ValueError("Subject ID must exactly match subject<number>")
        if folder.name != self.subject_id or self.reference_path.parent != folder:
            raise ValueError("Incorrect recording/reference association")
        if self.recording_id != f"{self.subject_id}/vid.avi":
            raise ValueError("Recording ID does not match the subject/video association")
        if self.video_path.name != "vid.avi" or self.reference_path.name != "ground_truth.txt":
            raise ValueError("Expected Dataset 2 vid.avi and ground_truth.txt")
        if folder.is_symlink():
            raise ValueError("Subject directory aliases are not accepted")
        for path in (self.video_path, self.reference_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing file: {self.subject_id}/{path.name}")
            if path.resolve().parent != folder.resolve():
                raise ValueError("Recording file resolves outside its subject folder")


def discover(root, subjects=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Dataset directory does not exist")
    names = (
        list(subjects)
        if subjects is not None
        else [
            p.name
            for p in root.iterdir()
            if re.fullmatch(r"subject[0-9]+", p.name) and (p.is_dir() or p.is_symlink())
        ]
    )
    if len(set(names)) != len(names):
        raise ValueError("Duplicate subject selections would duplicate observations")
    if not names:
        raise ValueError("No subject<number> directories found; select the DATASET_2 root")
    if any(not re.fullmatch(r"subject[0-9]+", name) for name in names):
        raise ValueError("Subject selections must exactly match subject<number>")
    return [
        Recording(
            name, f"{name}/vid.avi", root / name / "vid.avi", root / name / "ground_truth.txt"
        )
        for name in sorted(names, key=lambda s: (int(s[7:]), s))
    ]


@dataclass(frozen=True)
class GroundTruth:
    ppg: np.ndarray
    hr_bpm: np.ndarray
    timestamps_s: np.ndarray
    source_path: Path

    @property
    def duplicate_timestamp_count(self):
        return int(np.sum(np.diff(self.timestamps_s) == 0))

    def aligned_times(self, video_start_s, config):
        shift = config.reference_offset_s
        if config.sync_policy == "recording-start":
            shift += video_start_s - self.timestamps_s[0]
        return self.timestamps_s + shift

    def mean_interval(self, start, end, *, times=None, max_gap_s=1.0):
        t = self.timestamps_s if times is None else np.asarray(times, dtype=float)
        if t.shape != self.timestamps_s.shape or not np.isfinite(t).all():
            raise ValueError("Invalid aligned reference timestamps")
        if np.any(np.diff(t) < 0):
            raise ValueError("Aligned reference timestamps must not decrease")

        t, first, counts = np.unique(t, return_index=True, return_counts=True)
        hr = np.add.reduceat(self.hr_bpm, first) / counts
        if not np.isfinite([start, end, max_gap_s]).all() or end <= start or max_gap_s <= 0:
            raise ValueError("Invalid reference interval or gap limit")
        if start < t[0] - 1e-9 or end > t[-1] + 1e-9:
            raise ValueError("Insufficient reference coverage; no extrapolation allowed")
        left = max(0, np.searchsorted(t, start, side="right") - 1)
        right = min(len(t) - 1, np.searchsorted(t, end, side="left"))
        if np.any(np.diff(t[left : right + 1]) > max_gap_s):
            raise ValueError("Reference interval contains an unsupported timestamp gap")
        inside = (t > start) & (t < end)
        knots = np.r_[start, t[inside], end]
        values = np.interp(knots, t, hr)
        integral = np.sum(np.diff(knots) * (values[1:] + values[:-1]) / 2)
        return float(integral / (end - start)), int(right - left + 1)


def read_ground_truth(path):
    path = Path(path)
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) != 3:
        raise ValueError("Expected exactly three rows: PPG, HR (BPM), timestamps (seconds)")
    try:
        rows = [np.array([float(token) for token in line.split()]) for line in lines]
    except ValueError as exc:
        raise ValueError("Nonnumeric ground truth") from exc
    ppg, hr, t = rows
    if len(t) < 2 or not len(ppg) == len(hr) == len(t):
        raise ValueError("Ground truth rows must have equal lengths with at least two samples")
    if not all(np.isfinite(row).all() for row in rows):
        raise ValueError("Nonfinite ground truth values")
    if np.any((hr < 20) | (hr > 300)):
        raise ValueError("HR outside 20..300 BPM; check units/format (PPG is not HR)")
    if t[0] < 0 or np.any(np.diff(t) < 0) or t[-1] <= t[0]:
        raise ValueError("Timestamps must be nonnegative, nondecreasing, and span positive time")
    if np.median(np.diff(t)) > 1 or t[-1] - t[0] > 3600:
        raise ValueError("Unsupported timestamp scale; expected Dataset 2 seconds")
    return GroundTruth(ppg, hr, t, path)
