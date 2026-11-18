"""Evaluate the team's existing ROI and HR functions against UBFC sensor HR."""

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from . import hr_extraction
from .dataset import discover, read_ground_truth
from .metrics import calculate_metrics
from .video import VideoReader, validate_fps

METHOD = "team-extract_hr"


@dataclass(frozen=True)
class Options:
    roi: str = "cheeks"
    channel: str = "g"
    model_path: str = "models/face_landmarker.task"
    low_hz: float = 1.0
    high_hz: float = 2.0
    filter_order: int = 3
    min_duration_s: float = 10.0
    window_s: float = 15.0
    stride_s: float = 5.0
    window_enabled: bool = True
    max_gap_s: float = 0.25
    min_valid_fraction: float = 0.9
    reference_max_gap_s: float = 1.0
    sync_policy: str = "shared-clock"
    reference_offset_s: float = 0.0
    offset_provenance: str = ""

    def __post_init__(self):
        if self.roi not in ("cheeks", "forehead") or self.channel not in ("r", "g", "b"):
            raise ValueError("Choose team cheeks/forehead ROI and r/g/b channel")
        positive = (
            self.low_hz,
            self.high_hz,
            self.min_duration_s,
            self.window_s,
            self.stride_s,
            self.max_gap_s,
            self.reference_max_gap_s,
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Durations, gaps and cutoffs must be finite and positive")
        if self.low_hz >= self.high_hz:
            raise ValueError("low_hz must be below high_hz")
        if type(self.filter_order) is not int or not 1 <= self.filter_order <= 10:
            raise ValueError("filter_order must be an integer in 1..10")
        if type(self.window_enabled) is not bool:
            raise ValueError("window_enabled must be boolean")
        if self.window_enabled and self.window_s < self.min_duration_s:
            raise ValueError("window_s must be at least min_duration_s")
        if not np.isfinite(self.min_valid_fraction) or not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if self.sync_policy not in ("shared-clock", "recording-start"):
            raise ValueError("Unknown synchronization policy")
        if not np.isfinite(self.reference_offset_s):
            raise ValueError("reference_offset_s must be finite")
        if self.reference_offset_s and not self.offset_provenance.strip():
            raise ValueError("Nonzero offset requires independent offset_provenance")

    def to_dict(self):
        return asdict(self)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_signal(video_path, options):
    """Use team masks unchanged; independently verify frame times and decoder count.

    Two decode passes deliberately preserve the existing team ROI implementation.
    No frames are retained. Team zero-filled missed detections become NaN only at
    this evaluation boundary, so they cannot masquerade as measured color values.
    """
    from .channels_cheeks import extract_upper_cheeks_rgb
    from .channels_forehead import extract_forehead_rgb_signals

    with VideoReader(video_path) as reader:
        times = np.array([timestamp for timestamp, _ in reader])
        metadata = dict(reader.metadata)
    extractor = (
        extract_upper_cheeks_rgb if options.roi == "cheeks" else extract_forehead_rgb_signals
    )
    r, g, b, team_fps = extractor(str(video_path), options.model_path)
    rgb = np.column_stack((r, g, b)).astype(float)
    if rgb.shape != (len(times), 3):
        raise ValueError("Team extractor/decoder frame count mismatch")
    if team_fps != int(metadata["fps"]):
        raise ValueError("Team extractor/decoder FPS mismatch")
    missing = ~np.isfinite(rgb).all(axis=1) | np.all(rgb == 0, axis=1)
    signal = rgb[:, "rgb".index(options.channel)].copy()
    signal[missing] = np.nan
    metadata.update(valid_frame_count=int((~missing).sum()), invalid_frame_count=int(missing.sum()))
    return times, signal, metadata


def prepare_interval(timestamps, signal, start, end, options):
    """Bounded timestamp interpolation only; all scientific filtering stays in the team code."""
    t, x = np.asarray(timestamps, float), np.asarray(signal, float)
    if t.ndim != 1 or x.shape != t.shape or len(t) < 2:
        raise ValueError("Expected matching one-dimensional timestamps and signal")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Video timestamps must be finite and strictly increasing")
    if (
        not np.isfinite([start, end]).all()
        or start < t[0] - 1e-9
        or end > t[-1] + 1e-9
        or end <= start
    ):
        raise ValueError("Interval outside video coverage")
    selected = (t >= start - 1e-9) & (t <= end + 1e-9)
    valid = np.isfinite(x)
    if not selected.any() or valid[selected].mean() < options.min_valid_fraction:
        raise ValueError("Insufficient valid ROI samples")
    tv, xv = t[valid], x[valid]
    if len(tv) < 2 or start < tv[0] - 1e-9 or end > tv[-1] + 1e-9:
        raise ValueError("Missing ROI support at requested interval boundaries")
    left = max(0, np.searchsorted(tv, start, side="right") - 1)
    right = min(len(tv) - 1, np.searchsorted(tv, end, side="left"))
    if np.any(np.diff(tv[left : right + 1]) > options.max_gap_s + 1e-9):
        raise ValueError("Signal gap exceeds max_gap_s; long gaps are never filled")
    if end - start < options.min_duration_s - 1e-8:
        raise ValueError("Insufficient signal duration")
    fps = float(1 / np.median(np.diff(t)))
    count = int(np.floor((end - start) * fps + 1e-7)) + 1
    if count < 3:
        raise ValueError("Insufficient samples")
    grid = np.linspace(start, end, count)
    fps = float((count - 1) / (end - start))
    validate_fps(fps, options.high_hz)
    samples = np.interp(grid, tv, xv)
    if np.ptp(samples) <= 1e-10:
        raise ValueError("Flat or invalid pulse signal")
    return samples, fps


def evaluate_recording(
    recording, options, signal_extractor=None, video_override=None, interval_bounds=None
):
    began = time.perf_counter()
    row = dict(
        subject_id=recording.subject_id,
        recording_id=recording.recording_id,
        method=METHOD,
        roi=options.roi,
        channel=options.channel,
        status="failed",
        reason="",
    )
    windows = []
    try:
        recording.validate()
        reference = read_ground_truth(recording.reference_path)
        path = video_override if video_override is not None else recording.video_path
        video_hash = sha256_file(path)
        row.update(
            video_sha256=video_hash,
            reference_sha256=sha256_file(recording.reference_path),
            source_video_sha256=video_hash
            if Path(path) == recording.video_path
            else sha256_file(recording.video_path),
            reference_duplicate_timestamps=reference.duplicate_timestamp_count,
        )
        extractor = extract_signal if signal_extractor is None else signal_extractor
        t, signal, metadata = extractor(path, options)
        row.update(metadata)
        t, signal = np.asarray(t, float), np.asarray(signal, float)
        # Validate before using endpoints, even for injected test/external extractors.
        if t.ndim != 1 or signal.shape != t.shape or len(t) < 2:
            raise ValueError("Expected matching timestamps and signal")
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Video timestamps must be finite and strictly increasing")
        aligned = reference.aligned_times(t[0], options)
        start, end = max(t[0], aligned[0]), min(t[-1], aligned[-1])
        if interval_bounds is not None:
            start, end = max(start, interval_bounds[0]), min(end, interval_bounds[1])
        if end <= start:
            raise ValueError("Video and reference intervals do not overlap; check synchronization")
        row.update(sync_policy=options.sync_policy, reference_offset_s=options.reference_offset_s)

        def pair(a, b, base):
            item = dict(
                base,
                requested_start_s=float(a),
                requested_end_s=float(b),
                status="failed",
                reason="",
            )
            try:
                samples, fps = prepare_interval(t, signal, a, b, options)
                # The sole HR estimator: no POS, periodogram or replacement filtering.
                hr = float(
                    hr_extraction.extract_hr(
                        samples, fps, [options.low_hz, options.high_hz], N=options.filter_order
                    )
                )
                item.update(
                    raw_estimated_hr_bpm=hr if np.isfinite(hr) else None,
                    start_time_s=float(a),
                    end_time_s=float(b),
                    effective_fps=fps,
                )
                if not np.isfinite(hr) or hr <= 0:
                    raise ValueError(
                        "Team extract_hr returned nonpositive/nonfinite HR; FFT sign requires team review"
                    )
                ref, count = reference.mean_interval(
                    a, b, times=aligned, max_gap_s=options.reference_max_gap_s
                )
                item.update(
                    estimated_hr_bpm=hr,
                    reference_hr_bpm=ref,
                    reference_sample_count=count,
                    signed_error_bpm=hr - ref,
                    absolute_error_bpm=abs(hr - ref),
                    status="paired",
                )
            except Exception as exc:
                item["reason"] = f"{type(exc).__name__}: {exc}"
            return item

        row = pair(start, end, row)
        index = 0
        while (
            options.window_enabled
            and start + index * options.stride_s + options.window_s <= end + 1e-8
        ):
            a = start + index * options.stride_s
            windows.append(
                pair(
                    a,
                    a + options.window_s,
                    dict(
                        subject_id=recording.subject_id,
                        recording_id=recording.recording_id,
                        window_index=index,
                        method=METHOD,
                        roi=options.roi,
                        channel=options.channel,
                    ),
                )
            )
            index += 1
    except Exception as exc:
        row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    row["processing_time_s"] = time.perf_counter() - began
    return row, windows


def summarize(rows, windows, options, dataset_name="UBFC-rPPG Dataset 2"):
    valid = [r for r in rows if r["status"] == "paired"]
    valid_windows = [r for r in windows if r["status"] == "paired"]

    def metrics(items):
        return calculate_metrics(
            [r["estimated_hr_bpm"] for r in items], [r["reference_hr_bpm"] for r in items]
        )

    packages = {}
    for name in (
        "vital-detect",
        "numpy",
        "scipy",
        "opencv-contrib-python",
        "mediapipe",
        "matplotlib",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    return dict(
        run_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        dataset=dataset_name,
        configuration=options.to_dict(),
        method=METHOD,
        selection={"subjects": [r["subject_id"] for r in rows]},
        model_sha256=sha256_file(options.model_path)
        if Path(options.model_path).is_file()
        else None,
        software={"python": platform.python_version(), **packages},
        metrics=metrics(valid),
        window_metrics_descriptive_only=metrics(valid_windows),
        counts=dict(
            recordings_attempted=len(rows),
            recordings_successful=len(valid),
            recordings_failed=len(rows) - len(valid),
            windows_attempted=len(windows),
            windows_successful=len(valid_windows),
        ),
    )


def evaluate(
    data_dir,
    output_dir,
    options=None,
    subjects=None,
    signal_extractor=None,
    dataset_name="UBFC-rPPG Dataset 2",
):
    from .reporting import write_outputs

    options = Options() if options is None else options
    rows, windows = [], []
    for recording in discover(data_dir, subjects):
        row, wins = evaluate_recording(recording, options, signal_extractor)
        rows.append(row)
        windows.extend(wins)
        print(f"{recording.subject_id}: {row['status']} {row['reason']}", flush=True)
    summary = summarize(rows, windows, options, dataset_name)
    summary["dataset_root"] = str(Path(data_dir).resolve())
    summary["signal_extractor"] = (
        "team ROI" if signal_extractor is None else "custom/test extractor"
    )
    write_outputs(output_dir, rows, windows, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument(
        "--config", type=Path, help="Saved evaluation options JSON; flags override it"
    )
    parser.add_argument("--benchmark", nargs="+", choices=("C4", "C5"))
    parser.add_argument("--roi", choices=("cheeks", "forehead"))
    parser.add_argument("--channel", choices=("r", "g", "b"))
    parser.add_argument("--model-path")
    parser.add_argument("--sync-policy", choices=("shared-clock", "recording-start"))
    parser.add_argument("--offset-provenance")
    parser.add_argument(
        "--windows", dest="window_enabled", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--filter-order", type=int)
    for key in (
        "low-hz",
        "high-hz",
        "min-duration-s",
        "window-s",
        "stride-s",
        "max-gap-s",
        "min-valid-fraction",
        "reference-max-gap-s",
        "reference-offset-s",
    ):
        parser.add_argument(f"--{key}", type=float)
    args = parser.parse_args(argv)
    try:
        values = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(values, dict) or set(values) - Options.__dataclass_fields__.keys():
            raise ValueError("Unknown configuration keys or invalid JSON object")
        values.update(
            {
                key: getattr(args, key)
                for key in Options.__dataclass_fields__
                if getattr(args, key, None) is not None
            }
        )
        options = Options(**values)
        if args.output_dir.exists() and (
            not args.output_dir.is_dir() or any(args.output_dir.iterdir())
        ):
            raise ValueError("Output exists and is nonempty; choose a new directory")
        if args.benchmark:
            from .robustness import benchmark

            result = benchmark(
                args.data_dir, args.output_dir, options, args.subjects, args.benchmark
            )
            return int(any(s["counts"]["recordings_failed"] for s in result["conditions"].values()))
        result = evaluate(args.data_dir, args.output_dir, options, args.subjects)
        return int(result["counts"]["recordings_failed"] > 0)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
