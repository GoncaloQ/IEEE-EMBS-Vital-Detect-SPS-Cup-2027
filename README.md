# Vital Detect — IEEE EMBS SPS Cup 2027

The team's Round 1 implementation lives in `Round1_Baseline/`. Evaluation uses
Diana/Leonor's forehead or bilateral cheek extraction, the team's smoothness-priors
and Butterworth filters, Leonor's `extract_hr` FFT estimator, and Round1 metrics.
Inês/Leonor's EVM experiments and Diana's MATLAB experiments remain available.

The separate estimator/package has been consolidated. There is one evaluation
entry point: **`python -m Round1_Baseline.evaluate`**. No POS or independent GREEN
estimator remains. Selecting the team's green channel is still supported.

## Install and prepare

Use Python **3.11** (supported range 3.10–3.12):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -c "from Round1_Baseline.utils import download_model_if_missing; download_model_if_missing('models/face_landmarker.task')"
```

The last command explicitly downloads the version-1 MediaPipe model; supply an
existing model using `--model-path` for offline work. The team's original helper
can also download on first extraction if the model is absent. Imports never
start a download, decode a video or run the HR plotting demo. The demo is still
available with `python -m Round1_Baseline.hr_extraction`, using `target_video.mp4`.
It saves raw, filtered, FFT and sliding-window HR plots in `Results/`, using the
video FPS. `get_hr_bpm(signal, fps=30)` accepts an already filtered signal;
`extract_hr(raw_signal, fps, fc, N=3)` retains the evaluator's filtering path.
`extract_hr_signal(signal, fps=30)` returns window-center times and HR values
using 10-second windows and 1-second steps; short signals return empty results.
For reference metrics, use the dataset evaluator above with actual ground truth.

Retain `python -m pip freeze` and the Git revision with experiments. Do not install
multiple OpenCV distributions together. The retained MediaPipe 0.10.21 wheel has
an ARM64 platform-metadata inconsistency with modern dependency checkers despite
working universal binaries; see the exact failure in [cleanup results](docs/cleanup_results.md). Minimal Linux installations may need
`libgl1 libegl1 libgles2 libportaudio2`. MediaPipe may need a permitted graphics
context on macOS even for CPU inference.

## Dataset and evaluation

Obtain UBFC-rPPG **Dataset 2** through its
[official page](https://sites.google.com/view/ybenezeth/ubfcrppg), read its agreement,
and keep it outside version control:

```text
/path/to/DATASET_2/
  subject1/vid.avi
  subject1/ground_truth.txt
  subject2/vid.avi
  subject2/ground_truth.txt
```

The reference file has three whitespace-separated rows: PPG amplitudes, **sensor
HR in BPM**, timestamps in **seconds**. Dataset 1 and UBFC-Phys are unsupported.
No genuine facial recording was available/processed during this cleanup.

```bash
# Selected recordings (omit --subjects for the full dataset)
python -m Round1_Baseline.evaluate --data-dir /path/to/DATASET_2 --subjects subject1 subject2 --output-dir outputs/subset

# Team forehead implementation, with independent window evaluation disabled
python -m Round1_Baseline.evaluate --data-dir /path/to/DATASET_2 --roi forehead --no-windows --output-dir outputs/forehead

# Reproduce parameters and subject selection from a prior experiment
python -m Round1_Baseline.evaluate --data-dir /path/to/DATASET_2 --subjects subject1 subject2 --config outputs/subset/config.json --output-dir outputs/repeated
```

Use `--help` for all options. Flags override saved options. Outputs must use a new
or empty directory. Exit codes are 0 for a batch with all recording pairs, 1 for a
completed batch with recording failures, and 2 for setup/configuration errors.
Window failures are logged separately. One recording failure does not stop a batch.

Defaults mirror the team demo: green channel, bilateral cheeks, **1–2 Hz band,
order 3**. Minimum evaluated span is 10 seconds; windows are 15 seconds with
5-second stride. These are experiment settings, not demonstrated optimal settings.
ROI choice, band and final scoring protocol require team agreement.

Decoder timestamps and frame counts are checked separately from the team's ROI
extractor (two streaming decode passes, no stored video frames). Evaluation maps
the team's zero-filled missed detections to unavailable observations, requires
90% valid samples, and interpolates only short supported gaps (≤0.25 s). It does
not extrapolate or trim away failed interval edges. A uniform time grid goes into
the **unchanged team estimator**. Returned nonpositive/nonfinite HR fails with a
reason; the raw value is retained and never silently converted to its absolute value.

The unrestricted FFT argmax, integer ROI FPS/detector timing, missing-detection
behavior and EVM filters are preserved. Changing those
scientific choices requires team review. The evaluator cannot establish that a
periodic signal is physiological; real mask/HR inspection is still necessary.

Primary metrics are one pair per recording: MAE, RMSE, Pearson r and descriptive
Bland–Altman bias/limits from `Round1_Baseline/metrics.py`. Reference is time-weighted
sensor HR over the exact prediction interval. `shared-clock` assumes the dataset's
common clock; `recording-start` explicitly aligns first samples. A nonzero
`--reference-offset-s` requires independent `--offset-provenance`. Do not tune
alignment to final evaluation errors. Hardware latency/drift remain unverified.

Generated files: `per_video_results.csv`, `per_window_results.csv`,
`evaluation_failures.csv`, `summary_metrics.json`, `config.json`,
`evaluation_report.md`, `estimated_vs_reference.png`, `bland_altman.png`,
`error_distribution.png`. CSVs retain intervals, failures, hashes and timing;
JSON retains model hash, options, software versions and separate descriptive window
metrics. Even an all-failed batch writes reports and labeled empty plots.
See [the evaluation protocol and dataset sources](docs/evaluation_protocol.md).

## Robustness experiments

Preserved MATLAB scripts:

- `generate_low_resolution_video.m`: orientation-aware exact VGA/QVGA resizing,
  bicubic interpolation, MPEG-4 at original FPS.
- `generate_partial_occlusion_video.m`: tracked lower-face occlusion.

Install FFmpeg **and ffprobe** on PATH for automated Python comparisons:

```bash
python -m Round1_Baseline.evaluate --data-dir /path/to/DATASET_2 --subjects subject1 subject2 --benchmark C5 --output-dir outputs/c5
python -m Round1_Baseline.evaluate --data-dir /path/to/DATASET_2 --benchmark C4 C5 --output-dir outputs/robustness
```

Python C4 remains a distinct lossless FFV1 spatial control: no upscaling, preserved
aspect ratio and measured metadata checks. It is not equivalent to MATLAB's exact
size, lossy MPEG-4 experiment. C5 reduces FPS to at most 10 and verifies stream and
container average bitrate ≤500 kbps, dimensions, FPS, origin and duration. Each
condition uses the team's baseline and has its own reports. The manifest records
commands, measured metadata, timing and FFmpeg version. Error deltas require both
conditions to succeed over identical intervals; failed/nonmatching pairs stay
visible. Temporary transformed videos are removed automatically.

These are offline transformations. They establish neither streaming performance
nor HD acquisition compliance; native UBFC VGA cannot supply an HD baseline.

## Structure and checks

```text
Round1_Baseline/
  __init__.py
  channels_forehead.py  channels_cheeks.py
  hr_extraction.py  filters.py  metrics.py  utils.py
  evaluate.py  dataset.py  video.py  reporting.py  robustness.py
  evm.py  EVM_Option_1.ipynb  notes.txt
  generate_low_resolution_video.m  generate_partial_occlusion_video.m
Round2_HR_SpO2/.gitkeep
Round3_Constrained/.gitkeep
Round4_FullSystem_Clinical/.gitkeep
tests/
docs/
.github/workflows/tests.yml
pyproject.toml  MANIFEST.in
```

```bash
ruff check Round1_Baseline tests
python -m pytest -q
```

CI runs synthetic tests and controlled-landmark integration of the actual team
functions; it downloads no participant recordings or models. FFmpeg-dependent
tests skip explicitly when those executables are absent locally. A green CI run
is **not evidence of baseline accuracy on genuine facial videos**. MATLAB scripts
and the EVM notebook are preserved experiments, not automatically validated by CI.

The [cleanup plan](docs/cleanup_plan.md) records every original file and historical
evidence. [Cleanup results](docs/cleanup_results.md) list deletions, migrations,
actual validation and remaining team decisions. Implementations remain recoverable
from Git history at `c7e88bf`; official round directories are unchanged. The original
separate development repository has not been modified.
