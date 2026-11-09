import hashlib
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from vital_detect import __version__
from vital_detect.datasets.ubfc import discover, read_ground_truth
from vital_detect.evaluation.metrics import calculate_metrics
from vital_detect.hr.estimation import estimate
from vital_detect.roi.face import FaceROI
from vital_detect.signals.processing import crop_interval
from vital_detect.video.reader import VideoReader


@dataclass
class Trace:
    timestamps: np.ndarray
    rgb: np.ndarray
    metadata: dict


class TraceExtractionError(ValueError):
    def __init__(self, cause, metadata):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.metadata = metadata


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_trace(video_path, config, roi_factory=FaceROI):
    times, samples, statuses = [], [], Counter()
    with VideoReader(video_path) as reader:
        roi = None

        def metadata_snapshot():
            valid_count = sum(bool(np.isfinite(sample).all()) for sample in samples)
            return {
                **reader.metadata,
                "valid_frame_count": valid_count,
                "invalid_frame_count": reader.metadata["processed_frame_count"] - valid_count,
                "frame_status_counts": dict(statuses),
            }

        try:
            roi = roi_factory(config)
            for timestamp, rgb in reader:
                sample, status = roi.sample(rgb, timestamp)
                sample = np.asarray(sample, float)
                if sample.shape != (3,):
                    raise ValueError("ROI provider must return exactly three RGB means")
                times.append(timestamp)
                samples.append(sample)
                statuses[status] += 1
        except Exception as exc:
            raise TraceExtractionError(exc, metadata_snapshot()) from exc
        finally:
            if roi is not None:
                roi.close()
        metadata = metadata_snapshot()
    values = np.asarray(samples)
    return Trace(np.asarray(times), values, metadata)


def infer(video_path, config, roi_factory=FaceROI):
    start = time.perf_counter()
    trace = extract_trace(video_path, config, roi_factory)
    prediction = estimate(trace.timestamps, trace.rgb, config)
    return {
        **trace.metadata,
        **prediction,
        "method": config.method,
        "roi": config.roi,
        "processing_time_s": time.perf_counter() - start,
        "status": "inferred",
        "configuration": config.to_dict(),
    }


def evaluate_recording(
    recording, config, roi_factory=FaceROI, video_override=None, interval_bounds=None
):
    started = time.perf_counter()
    row = {
        "subject_id": recording.subject_id,
        "recording_id": recording.recording_id,
        "video_path": recording.recording_id,
        "method": config.method,
        "roi": config.roi,
        "status": "failed",
        "reason": "",
        "estimated_hr_bpm": None,
        "reference_hr_bpm": None,
    }
    windows = []
    try:
        recording.validate()
        reference = read_ground_truth(recording.reference_path)
        row["reference_duplicate_timestamps"] = reference.duplicate_timestamp_count
        path = video_override if video_override is not None else recording.video_path
        row["video_sha256"] = sha256_file(path)
        row["source_video_sha256"] = (
            row["video_sha256"]
            if Path(path) == recording.video_path
            else sha256_file(recording.video_path)
        )
        row["reference_sha256"] = sha256_file(recording.reference_path)
        trace = extract_trace(path, config, roi_factory)
        row.update(trace.metadata)
        t = trace.timestamps
        aligned = reference.aligned_times(t[0], config)

        start, end = max(t[0], aligned[0]), min(t[-1], aligned[-1])
        if interval_bounds is not None:
            start, end = max(start, interval_bounds[0]), min(end, interval_bounds[1])
        if end <= start:
            raise ValueError("Video and reference intervals do not overlap; check synchronization")
        row.update(
            sync_policy=config.sync_policy,
            reference_offset_s=config.reference_offset_s,
            available_start_s=float(start),
            available_end_s=float(end),
        )

        def paired_interval(a, b, base):
            item = {
                **base,
                "requested_start_s": float(a),
                "requested_end_s": float(b),
                "status": "failed",
                "reason": "",
            }
            try:
                segment_t, segment_rgb = crop_interval(t, trace.rgb, a, b, config.max_gap_s)
                prediction = estimate(segment_t, segment_rgb, config)
                item.update(prediction)
                ref, count = reference.mean_interval(
                    prediction["start_time_s"],
                    prediction["end_time_s"],
                    times=aligned,
                    max_gap_s=config.reference_max_gap_s,
                )
                error = prediction["estimated_hr_bpm"] - ref
                item.update(
                    reference_hr_bpm=ref,
                    reference_sample_count=count,
                    signed_error_bpm=error,
                    absolute_error_bpm=abs(error),
                    status="paired",
                )
            except (ValueError, OSError) as exc:
                item.update(status="failed", reason=str(exc))
            return item

        row = paired_interval(start, end, row)
        index = 0
        while (
            config.window_enabled
            and start + index * config.stride_s + config.window_s <= end + 1e-8
        ):
            a = start + index * config.stride_s
            windows.append(
                paired_interval(
                    a,
                    a + config.window_s,
                    {
                        "subject_id": recording.subject_id,
                        "recording_id": recording.recording_id,
                        "window_index": index,
                        "method": config.method,
                        "roi": config.roi,
                    },
                )
            )
            index += 1
    except Exception as exc:
        if isinstance(exc, TraceExtractionError):
            row.update(exc.metadata)
        row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    row["processing_time_s"] = time.perf_counter() - started
    return row, windows


def summarize(rows, windows, config, dataset_name="UBFC-rPPG Dataset 2"):
    valid = [r for r in rows if r["status"] == "paired"]
    valid_windows = [r for r in windows if r["status"] == "paired"]
    packages = {}
    for name in ("numpy", "scipy", "opencv-contrib-python", "mediapipe", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    import platform

    metrics = calculate_metrics(
        [r["estimated_hr_bpm"] for r in valid], [r["reference_hr_bpm"] for r in valid]
    )
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": __version__,
        "dataset": dataset_name,
        "selection": {
            "subjects": [r["subject_id"] for r in rows],
            "recording_ids": [r["recording_id"] for r in rows],
        },
        "configuration": config.to_dict(),
        "model_sha256": sha256_file(config.model_path)
        if Path(config.model_path).is_file()
        else None,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **packages,
        },
        "metrics": metrics,
        "window_metrics_descriptive_only": calculate_metrics(
            [r["estimated_hr_bpm"] for r in valid_windows],
            [r["reference_hr_bpm"] for r in valid_windows],
        ),
        "counts": {
            "subjects_attempted": len({r["subject_id"] for r in rows}),
            "subjects_evaluated": len({r["subject_id"] for r in valid}),
            "recordings_attempted": len(rows),
            "recordings_successful": len(valid),
            "recordings_failed": len(rows) - len(valid),
            "valid_pairs": len(valid),
            "success_percentage": 100 * len(valid) / len(rows) if rows else 0,
            "mean_processing_time_s": float(np.mean([r["processing_time_s"] for r in rows]))
            if rows
            else None,
            "windows_attempted": len(windows),
            "windows_successful": len(valid_windows),
        },
        "protocol": {
            "primary_unit": "one pair per recording (one recording per UBFC subject)",
            "recording_estimate": "dominant spectral HR over timestamp intersection, trimmed invalid ends",
            "reference": "time-weighted piecewise-linear sensor HR (BPM), exact prediction interval",
            "difference": "estimated minus reference (BPM)",
            "sync_policy": config.sync_policy,
            "synchronization_status": "dataset/common-clock convention assumed; hardware offset unverified",
            "offset_provenance": config.offset_provenance,
            "windows": "correlated repeated observations; pooled limits are descriptive, not subject-level inference",
            "ba_assumptions": "independent recording differences, approximate normality, constant variance; no CI",
            "limitations": [
                "No clinical validation or C3 multi-person identity tracking",
                "UBFC native VGA is not the specified HD competition baseline",
                "Spectral quality gates cannot distinguish every motion/illumination artifact from pulse",
                "Dominant recording frequency may differ from mean reference when HR varies",
            ],
        },
    }


def evaluate(
    data_dir,
    output_dir,
    config,
    subjects=None,
    roi_factory=FaceROI,
    dataset_name="UBFC-rPPG Dataset 2",
):
    from vital_detect.reporting.outputs import write_outputs

    recordings = discover(data_dir, subjects)
    rows, windows = [], []
    for recording in recordings:
        row, win = evaluate_recording(recording, config, roi_factory)
        rows.append(row)
        windows.extend(win)
        print(f"{recording.subject_id}: {row['status']} {row.get('reason', '')}", flush=True)
    summary = summarize(rows, windows, config, dataset_name)
    summary["roi_provider"] = f"{roi_factory.__module__}.{roi_factory.__qualname__}"
    summary["dataset_root"] = str(Path(data_dir).resolve())
    write_outputs(output_dir, rows, windows, summary)
    return summary
