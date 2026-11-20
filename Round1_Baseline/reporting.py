"""Evaluation tables and descriptive recording-level figures."""

import csv
import json
import shlex
from pathlib import Path

import numpy as np

FIELDS = [
    "subject_id",
    "recording_id",
    "window_index",
    "method",
    "roi",
    "channel",
    "status",
    "reason",
    "estimated_hr_bpm",
    "raw_estimated_hr_bpm",
    "reference_hr_bpm",
    "signed_error_bpm",
    "absolute_error_bpm",
    "requested_start_s",
    "requested_end_s",
    "start_time_s",
    "end_time_s",
    "reference_sample_count",
    "fps",
    "effective_fps",
    "width",
    "height",
    "processed_frame_count",
    "reported_frame_count",
    "valid_frame_count",
    "invalid_frame_count",
    "duration_s",
    "timestamp_source",
    "sync_policy",
    "reference_offset_s",
    "reference_duplicate_timestamps",
    "video_sha256",
    "source_video_sha256",
    "reference_sha256",
    "processing_time_s",
    "transformation_time_s",
    "total_condition_time_s",
]


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def figures(output, valid, metrics):
    # Use a local Agg canvas so importing/evaluating does not switch a notebook backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    estimated = np.array([r["estimated_hr_bpm"] for r in valid])
    reference = np.array([r["reference_hr_bpm"] for r in valid])
    for kind in ("estimated_vs_reference", "bland_altman", "error_distribution"):
        fig = Figure(figsize=(6.4, 4.8), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        if kind == "estimated_vs_reference":
            ax.set(
                xlabel="Reference HR (BPM)",
                ylabel="Estimated HR (BPM)",
                title=f"Recording-level HR (n = {len(valid)})",
            )
            if valid:
                ax.scatter(reference, estimated, alpha=0.75)
                lo, hi = (
                    min(reference.min(), estimated.min()) - 5,
                    max(reference.max(), estimated.max()) + 5,
                )
                ax.plot([lo, hi], [lo, hi], "--", color="0.4")
        elif kind == "bland_altman":
            ax.set(
                xlabel="Mean of estimated and reference HR (BPM)",
                ylabel="Estimated − reference HR (BPM)",
                title=f"Descriptive Bland–Altman (n = {len(valid)})",
            )
            if valid:
                ax.scatter((reference + estimated) / 2, estimated - reference, alpha=0.75)
                for field, label in (
                    ("bias_bpm", "Bias"),
                    ("lower_loa_bpm", "Lower LoA"),
                    ("upper_loa_bpm", "Upper LoA"),
                ):
                    value = metrics["bland_altman"][field]
                    if value is not None:
                        ax.axhline(value, linestyle="--", label=f"{label}: {value:.2f} BPM")
                ax.legend(fontsize=9)
        else:
            ax.set(
                xlabel="Estimated − reference HR (BPM)",
                ylabel="Recordings (count)",
                title="Recording-level signed errors",
            )
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if valid:
                ax.hist(estimated - reference, bins=min(15, max(1, len(valid))))
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
        fig.savefig(output / f"{kind}.png", dpi=160)


def write_outputs(output_dir, rows, windows, summary):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_video_results.csv", FIELDS, rows)
    write_csv(output / "per_window_results.csv", FIELDS, windows)
    failures = [{"scope": "recording", **r} for r in rows if r["status"] != "paired"]
    failures += [{"scope": "window", **r} for r in windows if r["status"] != "paired"]
    write_csv(output / "evaluation_failures.csv", ["scope", *FIELDS], failures)
    write_json(output / "summary_metrics.json", summary)
    write_json(output / "config.json", summary["configuration"])
    valid = [r for r in rows if r["status"] == "paired"]
    metrics = summary["metrics"]
    figures(output, valid, metrics)
    experiment = summary.get("experiment", {})
    repeated = output.parent if experiment else output
    command = [
        "python",
        "-m",
        "Round1_Baseline.evaluate",
        "--data-dir",
        summary["dataset_root"],
        "--output-dir",
        str(repeated.with_name(repeated.name + "_repeated").resolve()),
        "--config",
        str((output / "config.json").resolve()),
    ]
    if summary["selection"]["subjects"]:
        command.extend(["--subjects", *summary["selection"]["subjects"]])
    if experiment:
        command.extend(["--benchmark", *experiment["constraints"]])
    ba = metrics["bland_altman"]
    report = f"""# Team Round 1 evaluation

Dataset label: {summary["dataset"]}. Run: {summary["run_timestamp_utc"]}.
Condition: {summary.get("condition", "native")}.
Estimator: {summary["method"]}. ROI/channel: {summary["configuration"]["roi"]}/{summary["configuration"]["channel"]}.
Signal extractor: {summary["signal_extractor"]}.

| Quantity | Value |
| --- | --- |
| MAE (BPM) | {metrics["mae_bpm"]} |
| RMSE (BPM) | {metrics["rmse_bpm"]} |
| Pearson r | {metrics["pearson_r"]} |
| Bland–Altman bias (BPM) | {ba["bias_bpm"]} |
| Lower / upper LoA (BPM) | {ba["lower_loa_bpm"]} / {ba["upper_loa_bpm"]} |
| Attempted / successful / failed recordings | {summary["counts"]["recordings_attempted"]} / {summary["counts"]["recordings_successful"]} / {summary["counts"]["recordings_failed"]} |

Undefined metrics: {json.dumps(metrics["undefined_reasons"])}.
Failed evaluations and raw nonpositive estimator outputs remain in CSV.

Reference is time-weighted piecewise-linear **sensor HR in BPM**, integrated over
exactly each prediction interval, without extrapolation. Synchronization policy:
{summary["configuration"]["sync_policy"]}. Hardware latency and drift remain unverified.
Uniform timestamp interpolation is evaluation preprocessing; the team's smoothness
priors, Butterworth and FFT functions perform every HR estimate unchanged.
No reference HR values enter the estimator. Dominant spectral HR can differ from
time-mean sensor HR when the heart rate changes.

One primary pair per recording. Windows are separate correlated descriptive
observations. LoA are bias ± 1.96 sample SD, not confidence intervals; they assume
independent, approximately normal, constant-variance differences. Limits need at
least two pairs and are unstable with small samples. No clinical accuracy, C3
identity tracking, HD acquisition or streaming performance is established.
Synthetic tests do not validate physiological accuracy on genuine facial videos.

## Reproduce

Install as in README, keep input data private, and retain pip freeze and the Git
revision with each experiment. JSON stores options, model hash and software
versions; CSV stores file hashes and exact time intervals. Custom/test extractors
must be supplied again through Python; the CLI always uses team ROI functions.

```bash
{shlex.join(command)}
```
"""
    (output / "evaluation_report.md").write_text(report, encoding="utf-8")
