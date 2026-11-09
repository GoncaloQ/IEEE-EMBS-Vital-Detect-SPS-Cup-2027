import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from dataclasses import fields
from pathlib import Path

from vital_detect.config import Config

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def download_model(path):
    path = Path(path)
    if path.exists():
        raise ValueError("Model already exists; choose a new path to download another copy")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temp_name = handle.name
                digest = hashlib.sha256()
                while block := response.read(1024 * 1024):
                    handle.write(block)
                    digest.update(block)
        if Path(temp_name).stat().st_size < 1_000_000:
            raise ValueError("Downloaded model is unexpectedly small")
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError(
                "Official model checksum changed; inspect before updating the pinned hash"
            )
        os.replace(temp_name, path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()
    print(f"Model: {path}; SHA-256: {digest.hexdigest()}")


def _algorithm_options(parser):
    parser.add_argument(
        "--config", type=Path, help="JSON algorithm configuration; flags override it"
    )
    parser.add_argument("--method", choices=("green", "pos"))
    parser.add_argument("--roi", choices=("forehead", "left_cheek", "right_cheek", "cheeks"))
    parser.add_argument("--model-path")
    parser.add_argument(
        "--windows",
        dest="window_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable window evaluation",
    )
    for option in (
        "low-hz",
        "high-hz",
        "min-duration-s",
        "window-s",
        "stride-s",
        "max-gap-s",
        "min-valid-fraction",
        "min-band-power-fraction",
        "min-peak-power-fraction",
        "pos-window-s",
        "smoothing-alpha",
        "reference-max-gap-s",
        "reference-offset-s",
    ):
        parser.add_argument(f"--{option}", type=float)
    parser.add_argument("--filter-order", type=int)
    parser.add_argument("--sync-policy", choices=("shared-clock", "recording-start"))
    parser.add_argument("--offset-provenance")


def _configuration(args):
    values = {}
    if args.config:
        values = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
    allowed = {f.name for f in fields(Config)}
    if set(values) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(set(values) - allowed)}")
    values.update(
        {key: getattr(args, key) for key in allowed if getattr(args, key, None) is not None}
    )
    return Config(**values)


def _check_output(path, overwrite):
    if path.exists() and (not path.is_dir() or any(path.iterdir())) and not overwrite:
        raise ValueError("Output exists and is nonempty; use a new directory or --overwrite")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vital-detect", description="Round 1 heart rate pipeline")
    subs = parser.add_subparsers(dest="command", required=True)
    model = subs.add_parser("download-model", help="Download the official MediaPipe model locally")
    model.add_argument("--model-path", type=Path, default=Path("models/face_landmarker.task"))
    single = subs.add_parser("infer", help="Estimate HR from one single-subject video")
    single.add_argument("--video", required=True, type=Path)
    single.add_argument("--output", type=Path, help="Optional inference JSON file")
    _algorithm_options(single)
    for name in ("evaluate", "benchmark"):
        sub = subs.add_parser(name)
        sub.add_argument("--dataset", choices=("ubfc",), default="ubfc")
        sub.add_argument("--data-dir", type=Path, required=True)
        sub.add_argument("--subjects", nargs="+")
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--overwrite", action="store_true")
        if name == "benchmark":
            sub.add_argument("--constraints", nargs="+", choices=("C4", "C5"), default=["C4", "C5"])
        _algorithm_options(sub)
    args = parser.parse_args(argv)
    try:
        if args.command == "download-model":
            download_model(args.model_path)
            return 0
        config = _configuration(args)
        from vital_detect.pipeline import evaluate, infer

        if args.command == "infer":
            result = infer(args.video, config)
            text = json.dumps(result, indent=2, allow_nan=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + "\n", encoding="utf-8")
            print(text)
            return 0
        _check_output(args.output_dir, args.overwrite)
        if args.command == "evaluate":
            summary = evaluate(args.data_dir, args.output_dir, config, args.subjects)
            print(json.dumps(summary["metrics"], indent=2, allow_nan=False))
            return 1 if summary["counts"]["recordings_failed"] else 0
        from vital_detect.robustness.benchmark import benchmark

        manifest = benchmark(
            args.data_dir, args.output_dir, config, args.subjects, args.constraints
        )
        return (
            1
            if any(s["counts"]["recordings_failed"] for s in manifest["conditions"].values())
            else 0
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
