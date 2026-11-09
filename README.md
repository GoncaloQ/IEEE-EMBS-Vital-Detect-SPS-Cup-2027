# Vital Detect — IEEE SPS Cup 2027 FaceVitals

Round 1 heart rate inference and UBFC-rPPG Dataset 2 evaluation, developed by
**IEEE EMBS UP Student Branch**. The pipeline decodes facial video, extracts
forehead/cheek RGB signals with MediaPipe, estimates HR using GREEN or POS, aligns
sensor HR, and exports MAE, RMSE, Pearson r and Bland–Altman analysis.

**Synthetic validation complete; genuine UBFC video validation pending.** An
official reference file was checked, but no real-video accuracy results are
available. No clinical accuracy, C3 identity tracking, HD baseline compliance,
streaming or edge deployment is claimed. SpO2/BP are outside this implementation.

## Installation

Use **Python 3.11**; declared compatibility: 3.10–3.12. Local validation used macOS
Apple Silicon/Python 3.11; the Linux/macOS CI matrix is configured but its remote
execution has not been verified in this audit. Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
vital-detect download-model
```

The explicit download verifies the official version-1 Face Landmarker model's
checksum and saves it under ignored `models/`. Inference is offline. Use
`--model-path /path/to/face_landmarker.task` to select an existing local model.

Dependencies are constrained in `pyproject.toml`; retain `python -m pip freeze`
with each experiment. Do not install multiple OpenCV wheel distributions.
Minimal Linux installations may need `libgl1 libegl1 libgles2 libportaudio2`.
MediaPipe on macOS needs a permitted graphics context even with CPU inference;
use a normal terminal if a sandbox blocks initialization.

## Dataset

Obtain **Dataset 2 (realistic)** manually from the
[official UBFC-rPPG source](https://sites.google.com/view/ybenezeth/ubfcrppg).
Read the distribution README and agreement first. Keep recordings private:

```text
/path/to/DATASET_2/
├── subject1/
│   ├── vid.avi
│   └── ground_truth.txt
└── subjectN/
    ├── vid.avi
    └── ground_truth.txt
```

The verified format is three whitespace-separated rows: **PPG amplitudes, HR in
BPM, timestamps in seconds**. Raw values/timestamps are preserved; repeated
instants are counted and coalesced only for HR integration. Malformed references,
backward timestamps, missing files and incorrect subject associations are reported.
Dataset 1 and UBFC-Phys formats are unsupported. See [sources](docs/sources_and_notebook.md).

The official source describes 42 research recordings at native 640×480, nominal
30 FPS. Actual video metadata is measured. Native VGA does not establish the
competition's HD baseline. Do not redistribute recordings or publish facial images
of subjects 21/27. The pipeline exports no facial images or subject-labeled plots.

## Commands

One single-subject MP4/AVI, without ground truth:

```bash
vital-detect infer --video /path/to/vid.avi --method green --roi cheeks --output outputs/inference.json
```

Selected subjects:

```bash
vital-detect evaluate --dataset ubfc --data-dir /path/to/DATASET_2 --subjects subject1 subject2 --output-dir outputs/subset
```

Complete dataset:

```bash
vital-detect evaluate --dataset ubfc --data-dir /path/to/DATASET_2 --method pos --output-dir outputs/ubfc
```

The official wrapper supports the same arguments:

```bash
python Round1_Baseline/process_round1.py evaluate --data-dir /path/to/DATASET_2 --output-dir outputs/round1
```

Repeat an experiment with its saved configuration and the same subject selection:

```bash
vital-detect evaluate --data-dir /path/to/DATASET_2 --config outputs/ubfc/config.json --output-dir outputs/repeated
```

Flags override JSON settings. Use `vital-detect evaluate --help` for all options.
Output directories must be empty unless `--overwrite` is explicit. Exit codes:
0 = successful batch/inference; 1 = completed batch with recording failures;
2 = configuration/setup/inference failure. Individual window failures are logged
separately. Failed recordings do not stop a batch.

## Processing and evaluation

The implementation adapts landmark masks, RGB averaging and Butterworth filtering
from the original Vital Detect notebook; [the source audit](docs/sources_and_notebook.md)
documents reused ideas, corrections, official readers and POS attribution.

ROI options: `forehead`, `left_cheek`, `right_cheek`, `cheeks` (bilateral,
area weighted; anatomical sides in an unmirrored image). Methods: `green` and `pos`.
Default band is 0.7–3 Hz (42–180 BPM), minimum duration 10 s, windows 15 s with
5 s stride. Signals require 90% valid samples; only interior gaps ≤0.25 s are
interpolated. Missing detections remain NaN. Multiple detected faces or uncertain
target continuity fail explicitly. Preverified single-subject inputs are required.
Minimum duration is measured between the first and last retained timestamps.
`--no-windows` disables window estimation; its CSV then contains only the header.

Actual video timestamps are used when exposed; otherwise frame index/metadata FPS
is recorded as a CFR assumption. Invalid initial nonzero timestamps cause failure,
instead of silently changing the timebase. Uniform resampling, linear detrending, normalization,
order-3 zero-phase SOS filtering and a Hann periodogram produce a quality-gated
dominant in-band frequency. Quality gates are heuristics, not biological confidence.
No method or offset is selected using final evaluation errors.

Primary metrics use one pair per recording. Reference is time-weighted sensor HR
over exactly the retained prediction interval, without extrapolation or reference
gaps >1 s. Default `--sync-policy shared-clock` assumes the dataset's acquisition
convention; hardware latency remains unverified. `recording-start` explicitly aligns
first samples. Nonzero `--reference-offset-s` requires independent
`--offset-provenance`; never fit offsets to evaluation errors.

MAE/RMSE use BPM. Pearson r is null for insufficient or constant data. Bland–Altman
uses estimated − reference, bias and bias ±1.96 sample SD (`ddof=1`); limits are
null for n<2 and unstable with small samples. Independent, approximately normal,
constant-variance differences are assumed; no confidence intervals are inferred.
Window metrics are separate correlated descriptive observations, not independent
subjects. Changing HR can make dominant frequency differ from mean reference.
See the complete [evaluation protocol](docs/evaluation_protocol.md).

## Generated files

| File | Contents |
| --- | --- |
| `per_video_results.csv` | IDs, HR pairs/errors, intervals, metadata, frame counts, quality, hashes, failures and timing |
| `per_window_results.csv` | Window estimates and their own synchronized references |
| `summary_metrics.json` | Primary/separate window metrics, counts, null reasons, configuration, model hash and versions |
| `evaluation_failures.csv` | Recording/window exclusions and reasons |
| `estimated_vs_reference.png` | Scatter with identity line |
| `bland_altman.png` | Recording bias and available limits |
| `error_distribution.png` | Signed-error histogram |
| `evaluation_report.md` | Results, limitations and reproduction instructions |
| `config.json` | Reusable algorithm/alignment parameters |

Empty evaluations still generate reports and labeled figures. Additional counts,
success rate, timing and median error remain distinct from official metrics.
Datasets, models and generated outputs are ignored by Git.

## C4/C5 robustness

Install FFmpeg/ffprobe on PATH (`brew install ffmpeg` on macOS or
`sudo apt-get install ffmpeg` on Debian/Ubuntu):

```bash
vital-detect benchmark --dataset ubfc --data-dir /path/to/DATASET_2 --constraints C4 C5 --subjects subject1 subject2 --output-dir outputs/robustness
```

Omit subjects for the complete dataset. Conditions: native, VGA, QVGA and
≤10 FPS/≤500 kbps. C4 preserves aspect ratio, never upscales and uses lossless
encoding. C5 applies H.264 compression and frame-rate reduction. ffprobe verifies
actual dimensions/FPS/bitrate/duration; unsupported conditions fail explicitly.
Each condition has separate reports; the manifest records measured metadata,
commands, FFmpeg version and timing. Same-recording error changes are calculated
only for matching retained intervals. Temporary transformed videos are removed.
Requested resolution/FPS and timestamp origins are verified; unknown source
duration/origin prevents synchronized condition comparison. Reports save the
subject selection and a shell-quoted reproduction command with the correct paths.
These are offline experiments, not streaming/edge or real-data C4/C5 validation.

## Tests and structure

```bash
python -m pytest -q
ruff check src tests Round1_Baseline/process_round1.py
```

Local audit checks include the full test suite, actual FFmpeg transformations,
installed CLI, package build and native MediaPipe failure handling. The validation
record is in the [evaluation protocol](docs/evaluation_protocol.md#audit-status).
Real UBFC video evaluation remains pending.
Tests cover signal recovery, POS algebra, invalid samples/FPS/Nyquist, metrics,
dataset/alignment, decoding/ROI policies, integration, failures, reproducibility,
outputs and measured FFmpeg transformations. FFmpeg tests skip explicitly if its
executables are absent. CI downloads neither participant data nor models.

```text
Round1_Baseline/process_round1.py
Round2_HR_SpO2/
Round3_Constrained/
Round4_FullSystem_Clinical/
src/vital_detect/
  cli.py, config.py, pipeline.py
  datasets/ video/ roi/ signals/ hr/ evaluation/ reporting/ robustness/
tests/  docs/  .github/workflows/tests.yml
pyproject.toml
```

The [official regulations](https://signalprocessingsociety.org/sites/default/files/2026-10/2027-SP-Cup-Official-Document-FaceVitals.pdf)
remain the submission specification. Evaluation outputs do not replace final MAT
packaging or the technical report. The three-round/four-directory mapping needs
organizer clarification; all official directory names are retained.

Next: evaluate one genuine UBFC video, then a subset and all available subjects;
review masks/alignment/failures and justify C4/C5 using measured results. Future
SpO2/BP integration may reuse video/ROI/time contracts but requires separate
methods, reference channels and validation. No participant collection is performed.
