import json
import math
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

from vital_detect.datasets.ubfc import discover
from vital_detect.pipeline import evaluate_recording, summarize
from vital_detect.reporting.outputs import write_csv, write_json, write_outputs


def _run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{command[0]} failed: {(exc.stderr or str(exc)).strip()}") from exc


def probe(path):
    if not shutil.which("ffprobe"):
        raise ValueError("ffprobe is required for verified robustness experiments")
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,bit_rate,duration,start_time:format=duration,bit_rate,start_time",
            "-of",
            "json",
            str(path),
        ],
    )
    value = json.loads(result.stdout)
    if not value.get("streams"):
        raise ValueError("ffprobe found no video stream")
    stream = value["streams"][0]
    container = value.get("format", {})

    def numeric(key, *sources):
        for source in sources:
            try:
                result = float(source[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(result):
                return result
        return None

    bitrate = numeric("bit_rate", stream, container)
    container_bitrate = numeric("bit_rate", container)
    try:
        fps = float(Fraction(stream["avg_frame_rate"]))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("ffprobe could not determine average FPS") from exc
    metadata = {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "bitrate_bps": int(bitrate) if bitrate is not None else None,
        "container_bitrate_bps": int(container_bitrate) if container_bitrate is not None else None,
        "duration_s": numeric("duration", stream, container),
        "start_time_s": numeric("start_time", stream, container),
    }
    if min(metadata["width"], metadata["height"]) <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("ffprobe returned invalid dimensions or FPS")
    if metadata["duration_s"] is None or metadata["duration_s"] <= 0:
        raise ValueError("ffprobe could not determine a positive video duration")
    return metadata


def transform(source, target, condition):
    if not shutil.which("ffmpeg"):
        raise ValueError("ffmpeg is required for robustness transformations")
    original = probe(source)
    filters = []
    if condition in ("C4_VGA", "C4_QVGA"):
        w, h = (640, 480) if condition == "C4_VGA" else (320, 240)
        ratio = min(1, w / original["width"], h / original["height"])
        width = max(2, int(original["width"] * ratio) // 2 * 2)
        height = max(2, int(original["height"] * ratio) // 2 * 2)
        filters = [f"scale={width}:{height}:flags=bicubic"]
        codec = ["-c:v", "ffv1"]
    elif condition == "C5_10fps_500kbps":
        filters = [f"fps={min(10, original['fps'])}"]
        codec = [
            "-c:v",
            "libx264",
            "-b:v",
            "400k",
            "-maxrate",
            "450k",
            "-bufsize",
            "450k",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
        ]
    else:
        raise ValueError("Unknown robustness condition")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        ",".join(filters),
        *codec,
        str(target),
    ]
    _run(command)
    measured = probe(target)
    if original["start_time_s"] is None or measured["start_time_s"] is None:
        raise ValueError(
            "Video timestamp origin is unknown; transformation alignment is unverified"
        )
    if abs(measured["start_time_s"] - original["start_time_s"]) > 1e-6:
        raise ValueError("Transformation changed timestamp origin; alignment is unsupported")
    if abs(measured["duration_s"] - original["duration_s"]) > max(0.25, 1 / measured["fps"]):
        raise ValueError("Transformation changed duration; synchronized comparison unsupported")
    if condition.startswith("C4"):
        if (measured["width"], measured["height"]) != (width, height):
            raise ValueError("Measured resolution does not match the requested C4 condition")
        if not math.isclose(measured["fps"], original["fps"], rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("C4 transformation changed source FPS")
        if (
            abs(measured["width"] / measured["height"] - original["width"] / original["height"])
            > 0.02
        ):
            raise ValueError("Aspect ratio changed beyond rounding tolerance")
    else:
        if not math.isclose(measured["fps"], min(10, original["fps"]), rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError("Measured FPS does not meet C5")
        if (measured["width"], measured["height"]) != (original["width"], original["height"]):
            raise ValueError("C5 transformation changed source dimensions")
        rates = [measured["bitrate_bps"], measured["container_bitrate_bps"]]
        if any(b is None or not 0 < b <= 500_000 for b in rates):
            raise ValueError("Measured bitrate missing or exceeds 500 kbps; C5 unverified")
    return {"source": original, "measured": measured, "ffmpeg_command": command}


def benchmark(
    data_dir,
    output_dir,
    config,
    subjects=None,
    constraints=("C4", "C5"),
    roi_factory=None,
    dataset_name="UBFC-rPPG Dataset 2",
):
    from vital_detect.roi.face import FaceROI

    roi_factory = FaceROI if roi_factory is None else roi_factory
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ValueError("Install ffmpeg and ffprobe before running benchmarks")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recordings = discover(data_dir, subjects)
    conditions = ["native"]
    if "C4" in constraints:
        conditions += ["C4_VGA", "C4_QVGA"]
    if "C5" in constraints:
        conditions += ["C5_10fps_500kbps"]
    all_rows, summaries, transforms = {}, {}, []

    intervals = {}
    for recording in recordings:
        try:
            info = probe(recording.video_path)
            if info["start_time_s"] is None:
                raise ValueError(
                    "Source timestamp origin is unknown; comparison alignment is unverified"
                )
            intervals[recording.recording_id] = (
                info["start_time_s"],
                info["start_time_s"] + info["duration_s"] - 0.15,
            )
        except Exception as exc:
            intervals[recording.recording_id] = exc
    for condition in conditions:
        rows, windows = [], []
        for recording in recordings:
            began = time.perf_counter()
            try:
                if isinstance(intervals[recording.recording_id], Exception):
                    raise ValueError(
                        f"Cannot define common benchmark interval: {intervals[recording.recording_id]}"
                    )
                with tempfile.TemporaryDirectory(prefix="vital-detect-") as temp:
                    path = recording.video_path
                    transformation_time = 0.0
                    if condition != "native":
                        ext = ".mp4" if condition.startswith("C5") else ".avi"
                        path = Path(temp) / ("transformed" + ext)
                        evidence = transform(recording.video_path, path, condition)
                        transformation_time = time.perf_counter() - began
                        evidence["transformation_time_s"] = transformation_time
                        transforms.append(
                            {
                                "recording_id": recording.recording_id,
                                "condition": condition,
                                **evidence,
                            }
                        )
                    row, wins = evaluate_recording(
                        recording,
                        config,
                        roi_factory,
                        video_override=path,
                        interval_bounds=intervals[recording.recording_id],
                    )
                    row["transformation_time_s"] = transformation_time
                row["total_condition_time_s"] = time.perf_counter() - began
            except Exception as exc:
                row = {
                    "subject_id": recording.subject_id,
                    "recording_id": recording.recording_id,
                    "video_path": recording.recording_id,
                    "method": config.method,
                    "roi": config.roi,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "processing_time_s": time.perf_counter() - began,
                }
                wins = []
            rows.append(row)
            windows.extend(wins)
            print(f"{condition} {recording.subject_id}: {row['status']}", flush=True)
        summary = summarize(rows, windows, config, dataset_name)
        summary["roi_provider"] = f"{roi_factory.__module__}.{roi_factory.__qualname__}"
        summary["dataset_root"] = str(Path(data_dir).resolve())
        summary["experiment"] = {"command": "benchmark", "constraints": list(constraints)}
        summary["condition"] = condition
        summary["timing"] = {
            "mean_total_condition_time_s": sum(
                r.get("total_condition_time_s", r["processing_time_s"]) for r in rows
            )
            / len(rows)
        }
        summary["ffmpeg_version"] = _run(["ffmpeg", "-version"]).stdout.splitlines()[0]
        write_outputs(output / condition, rows, windows, summary)
        all_rows[condition], summaries[condition] = rows, summary
    comparison = []
    baseline = {r["recording_id"]: r for r in all_rows["native"]}
    for condition in conditions[1:]:
        for row in all_rows[condition]:
            native = baseline[row["recording_id"]]
            matched = native["status"] == row["status"] == "paired"
            same_interval = matched and all(
                abs(native[key] - row[key]) < 1e-6 for key in ("start_time_s", "end_time_s")
            )
            comparison.append(
                {
                    "condition": condition,
                    "recording_id": row["recording_id"],
                    "native_status": native["status"],
                    "condition_status": row["status"],
                    "same_interval": same_interval,
                    "absolute_error_change_bpm": row["absolute_error_bpm"]
                    - native["absolute_error_bpm"]
                    if same_interval
                    else None,
                }
            )
    write_csv(
        output / "condition_comparison.csv",
        [
            "condition",
            "recording_id",
            "native_status",
            "condition_status",
            "same_interval",
            "absolute_error_change_bpm",
        ],
        comparison,
    )
    manifest = {
        "conditions": summaries,
        "transformations": transforms,
        "limitations": [
            "Offline transforms only; no streaming or edge validation",
            "Native source resolution is measured; HD baseline unverified",
            "Changes computed only on successfully paired same-source recordings",
        ],
    }
    write_json(output / "benchmark_manifest.json", manifest)
    return manifest
