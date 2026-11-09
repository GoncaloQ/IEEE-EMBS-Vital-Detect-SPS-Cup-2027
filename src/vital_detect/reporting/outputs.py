import csv
import json
import shlex
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

VIDEO_FIELDS = [
    "subject_id",
    "recording_id",
    "video_path",
    "reference_hr_bpm",
    "estimated_hr_bpm",
    "absolute_error_bpm",
    "signed_error_bpm",
    "method",
    "roi",
    "fps",
    "effective_fps",
    "width",
    "height",
    "duration_s",
    "valid_frame_count",
    "invalid_frame_count",
    "processed_frame_count",
    "reported_frame_count",
    "timestamp_source",
    "start_time_s",
    "end_time_s",
    "available_start_s",
    "available_end_s",
    "requested_start_s",
    "requested_end_s",
    "reference_sample_count",
    "valid_fraction",
    "band_power_fraction",
    "peak_power_fraction",
    "frequency_resolution_bpm",
    "sync_policy",
    "reference_offset_s",
    "video_sha256",
    "source_video_sha256",
    "reference_sha256",
    "reference_duplicate_timestamps",
    "status",
    "reason",
    "processing_time_s",
    "transformation_time_s",
    "total_condition_time_s",
]
WINDOW_FIELDS = [
    "subject_id",
    "recording_id",
    "window_index",
    "requested_start_s",
    "requested_end_s",
    "start_time_s",
    "end_time_s",
    "estimated_hr_bpm",
    "reference_hr_bpm",
    "signed_error_bpm",
    "absolute_error_bpm",
    "method",
    "roi",
    "effective_fps",
    "valid_fraction",
    "band_power_fraction",
    "peak_power_fraction",
    "frequency_resolution_bpm",
    "reference_sample_count",
    "status",
    "reason",
]


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def figures(output, valid, metrics):
    estimated = np.array([r["estimated_hr_bpm"] for r in valid])
    reference = np.array([r["reference_hr_bpm"] for r in valid])
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    for kind in ("estimated_vs_reference", "bland_altman", "error_distribution"):
        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        if kind == "estimated_vs_reference":
            ax.set(
                xlabel="Reference HR (BPM)",
                ylabel="Estimated HR (BPM)",
                title=f"Recording-level HR (n = {len(valid)})",
            )
            if valid:
                ax.scatter(reference, estimated, alpha=0.75, color="#176b87")
                lo = min(reference.min(), estimated.min()) - 5
                hi = max(reference.max(), estimated.max()) + 5
                ax.plot([lo, hi], [lo, hi], "--", color="0.4", label="Identity")
                ax.set(xlim=(lo, hi), ylim=(lo, hi))
                ax.legend()
        elif kind == "bland_altman":
            ax.set(
                xlabel="Mean of estimated and reference HR (BPM)",
                ylabel="Estimated − reference HR (BPM)",
                title=f"Bland–Altman: recording pairs (n = {len(valid)})",
            )
            if 0 < len(valid) < 30:
                ax.set_title(f"Bland–Altman (n = {len(valid)})\nDescriptive analysis; small sample")
            if valid:
                ax.scatter((reference + estimated) / 2, estimated - reference, alpha=0.75)
                ba = metrics["bland_altman"]
                for field, label, style in (
                    ("bias_bpm", "Bias", "-"),
                    ("lower_loa_bpm", "Lower LoA", "--"),
                    ("upper_loa_bpm", "Upper LoA", "--"),
                ):
                    if ba[field] is not None:
                        ax.axhline(
                            ba[field], linestyle=style, label=f"{label}: {ba[field]:.2f} BPM"
                        )
                ax.legend(fontsize=9)
        else:
            ax.set(
                xlabel="Estimated − reference HR (BPM)",
                ylabel="Recordings (count)",
                title="Recording-level signed errors",
            )
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if valid:
                ax.hist(
                    estimated - reference,
                    bins=min(15, max(1, len(valid))),
                    color="#176b87",
                    edgecolor="white",
                )
        if not valid:
            ax.text(
                0.5,
                0.5,
                "No valid paired observations",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.grid(alpha=0.15)
        fig.savefig(output / f"{kind}.png", dpi=220)
        plt.close(fig)


def write_outputs(output_dir, rows, windows, summary):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_video_results.csv", VIDEO_FIELDS, rows)
    write_csv(output / "per_window_results.csv", WINDOW_FIELDS, windows)
    failures = [{"scope": "recording", **r} for r in rows if r["status"] != "paired"]
    failures += [{"scope": "window", **r} for r in windows if r["status"] != "paired"]
    write_csv(
        output / "evaluation_failures.csv",
        ["scope", "subject_id", "recording_id", "window_index", "status", "reason"],
        failures,
    )
    write_json(output / "summary_metrics.json", summary)
    write_json(output / "config.json", summary["configuration"])
    valid = [r for r in rows if r["status"] == "paired"]
    metrics = summary["metrics"]
    figures(output, valid, metrics)
    counts = summary["counts"]

    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"

    ba = metrics["bland_altman"]
    experiment = summary.get("experiment", {"command": "evaluate"})
    repeated = output.parent if experiment["command"] == "benchmark" else output
    command = [
        "vital-detect",
        experiment["command"],
        "--dataset",
        "ubfc",
        "--data-dir",
        summary.get("dataset_root", "/path/to/DATASET_2"),
        "--config",
        str((output / "config.json").resolve()),
        "--output-dir",
        str(repeated.with_name(repeated.name + "_repeated").resolve()),
    ]
    selected = summary.get("selection", {}).get("subjects", [])
    if selected:
        command.extend(["--subjects", *selected])
    if experiment["command"] == "benchmark":
        command.extend(["--constraints", *experiment["constraints"]])
    reproduction = shlex.join(command)
    report = f"""# Heart rate evaluation

Dataset: {summary["dataset"]}. Run: {summary["run_timestamp_utc"]}.
Condition: {summary.get("condition", "native dataset evaluation")}.
Method: {summary["configuration"]["method"]}; ROI: {summary["configuration"]["roi"]}.
ROI provider: `{summary.get("roi_provider", "unrecorded")}`.

One spectral estimate per recording over the video/reference timestamp intersection
(invalid edges trimmed). Reference is the time-weighted sensor HR in BPM over exactly
the retained prediction interval. Raw reference timestamps are preserved.
Benchmark conditions additionally restrict inference to a common source interval,
ending 0.15 seconds before source duration; actual bounds are recorded in CSV.
Synchronization: `{summary["configuration"]["sync_policy"]}`. Acquisition offset is
unverified; this run does not establish precise hardware synchronization.

## Recording-level results (primary)

| Quantity | Value |
| --- | --- |
| MAE (BPM) | {fmt(metrics["mae_bpm"])} |
| RMSE (BPM) | {fmt(metrics["rmse_bpm"])} |
| Pearson r | {fmt(metrics["pearson_r"])} |
| Bias, estimated − reference (BPM) | {fmt(ba["bias_bpm"])} |
| Lower / upper 95% LoA (BPM) | {fmt(ba["lower_loa_bpm"])} / {fmt(ba["upper_loa_bpm"])} |
| Subjects evaluated | {counts["subjects_evaluated"]} |
| Recordings attempted / successful / failed | {counts["recordings_attempted"]} / {counts["recordings_successful"]} / {counts["recordings_failed"]} |
| Success (%) | {counts["success_percentage"]:.2f} |
| Mean processing time (s) | {fmt(counts["mean_processing_time_s"])} |

Undefined reasons: `{json.dumps(metrics["undefined_reasons"])}`.
LoA use bias ± 1.96 sample SD (ddof=1), assume independent, approximately normal,
homoscedastic recording differences, and are descriptive rather than confidence
intervals. Limits are undefined for n < 2 and unstable with small samples.
Window metrics in JSON are a separate descriptive analysis: overlapping windows
are correlated and must never be counted as independent subjects.

## Failures and limitations

See `evaluation_failures.csv` for all recording/window exclusions and reasons.
Failed recordings remain in the processing denominator. Quality gates do not
guarantee that a spectral peak is physiological. A changing HR can make the dominant
recording frequency differ from the interval mean reference. Single-subject inputs
are required; persistent target identity/C3 is not implemented. Native UBFC VGA
does not establish HD-baseline compliance. No clinical validation is implied.
This report does not establish that inputs are genuine UBFC data; synthetic tests
and real-data evaluations must be identified separately by the experimenter.

## Reproduce

Install the package and Face Landmarker model as described in README.md. Retain
the dataset locally. `config.json` stores all algorithm/alignment parameters;
input hashes appear in `per_video_results.csv`, software/model versions in JSON.

```bash
{reproduction}
```

Use the same `--subjects` selection for subset experiments. Keep each experiment
in a separate output directory. No label-based method/offset selection is used.
The CLI uses the standard MediaPipe ROI provider. Runs made through Python with
a custom/test ROI provider must also reuse that provider to reproduce results.
"""
    (output / "evaluation_report.md").write_text(report, encoding="utf-8")
