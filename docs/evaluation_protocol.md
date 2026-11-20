# Team baseline evaluation protocol

The primary entry point is `python -m Round1_Baseline.evaluate`. The numerical
implementation is the team's `hr_extraction.extract_hr`, using `filters.py` and
forehead/cheek RGB extraction. No independent ROI or spectral estimator is used.

## Observations and failures

UBFC Dataset 2 contributes one primary pair per recording/subject. Subjects are
ordered naturally; explicit missing subjects remain attempted failures. Exact
IDs, input hashes and exclusion reasons are saved. Duplicate selections and
cross-subject symlink aliases are rejected. Every attempted recording remains in
the denominator. Failed recordings do not prevent other recordings from running.
No method, ROI, window or clock offset is selected using reference error.

Window estimates use predetermined durations and stride. They are separate,
correlated descriptive observations, never additional independent subjects.
A recording failure does not suppress separately evaluable windows.

## Video and signal timing

The team's ROI functions perform the actual masks and RGB means without changing
landmarks, HSV/hairline calibration, smoothing or missing-detection outputs.
A separate streaming validation pass collects decoder timestamps/frame counts;
a second pass invokes the team function. This costs an extra decode but avoids
maintaining another ROI implementation or changing the team's interface.

Increasing initial decoder timestamps select PTS throughout. Invalid subsequent
PTS fail; no mixed timebases are allowed. Unavailable/zero initial positions use
frame index / nominal FPS as an explicit CFR assumption. Invalid nonzero initial
positions, changing dimensions and incomplete known frame counts fail. A decoder
may expose nominal CFR positions even for VFR input; verify acquisition metadata.

ROI output count must match the validated decoder count. The team helper's integer
FPS must agree with the integer part of decoder nominal FPS. Evaluation uses actual
timestamps for HR sampling. The team detector still uses its original integer FPS
and accumulated integer millisecond increments internally; no claim of corrected
landmark tracking timing is made. This issue remains a team decision.

At evaluation boundaries only, all-zero RGB samples (team no-detection convention)
and nonfinite RGB become missing samples. At least 90% of requested frame samples
must be valid. Both interval boundaries need observed support. Missing edges fail
rather than silently narrowing the requested comparison. Short interior gaps up
to 0.25 seconds between valid observations are linearly interpolated; longer gaps
fail. A uniform grid spans the exact requested interval with sample count derived
from median video frame spacing; effective FPS is (N−1)/(end−start). There is no
extrapolation, time stretching or reference-driven interpolation. Minimum duration
is actual timestamp span, not frame count plus an extra nominal period. Flat
signals fail before estimation. Nyquist is checked against the selected upper cutoff.

Every prepared signal is passed to the team's estimator. Its smoothness-priors
parameter, zero-phase Butterworth filter and unrestricted FFT argmax are unchanged.
No Hann spectrum, POS projection, independent quality gates, absolute-HR correction
or replacement filtering is inserted. Nonpositive/nonfinite estimator outputs fail
and retain their raw finite value. Positive outputs are not proof of pulse quality.
Default band/order/channel mirror the team's demo (1–2 Hz, order 3, green); they
need scientific agreement before final scoring. EVM is not enabled automatically.

## Ground truth and synchronization

References require exactly three nonblank rows: PPG, sensor HR (BPM), time (seconds).
Raw values/timestamps are retained. Rows must have equal lengths, finite values,
at least two samples, HR within the broad format bound 20..300 BPM, nonnegative,
nondecreasing times and positive span. Unsupported units/scales fail. Duplicate
instants remain counted in the raw data; their HR values are averaged only when
integrating. Blank lines/BOM do not change row order.

`shared-clock` preserves video and reference coordinates and assumes the dataset
acquisition convention. `recording-start` shifts first reference sample to first
video timestamp and is an explicit assumption. An independently measured additive
reference offset is supported with provenance. Do not search offsets to improve
scores. Hardware latency and clock drift are unverified.

Recording evaluation uses the video/reference time intersection. Each reference
is the piecewise-linear HR integral divided by the exact prediction span, with
support at both endpoints and no overlapping reference gap >1 second by default.
Unequal sample density/duplicate times cannot overweight an interval. Reference
HR values never enter HR extraction; only timestamps determine available coverage.
This sensor-HR protocol differs from extracting contact-PPG spectral HR. Dominant
video frequency and time-mean reference may differ when HR changes.

## Metrics, reports and robustness

`metrics.calculate_metrics` is the sole official implementation. Error is estimate
minus reference. MAE/RMSE/bias need at least one pair; Pearson needs at least two
and variation in both arrays. Bland–Altman limits use bias ±1.96 sample SD (ddof=1),
need at least two pairs, assume independent approximately normal constant-variance
differences, and are descriptive limits, not confidence intervals. Small samples
are unstable. Undefined values carry reasons and invalid pairs cannot enter metrics.

Recording/window CSVs retain failures and intervals. Summary JSON contains counts,
options, input/model hashes (input hashes in CSV), versions and separate window
metrics. Three figures and a Markdown report are written even if all pairs fail.
The report contains a quoted reproduction command and exact subject selection.
Custom/test extraction seams must be passed again through Python; CLI uses team ROI.

Python C4 uses lossless aspect-preserving non-upscaled spatial controls, whereas
the team's MATLAB C4 uses exact target dimensions with lossy MPEG-4. The latter's
orientation handling and separate occlusion experiment remain intact. C5 exists
only in Python and combines ≤10 FPS with ≤500 kbps compression. Measured FPS,
dimensions, duration, origin and stream/container average bitrates are checked;
average bitrate does not establish instantaneous network constraints.

Automated robustness comparisons use the same source recording and common bounds
ending 0.15 seconds before source duration. Exact transformed interval endpoints
are interpolated only with supported short gaps. Unknown origin/duration rejects
synchronized comparison. Deltas require successful pairs at identical endpoints;
failures and nonmatching bounds remain visible. Manifest includes FFmpeg version,
transformation commands, metadata and timing. Native VGA is not an HD baseline.

No clinical, C3 identity, streaming or edge-deployment validation is implied.
Historical independent-estimator test counts are not current validation evidence.
See cleanup_results.md for tests actually executed on the consolidated code.


## Dataset sources and citation

Source references retained from the earlier evaluation work; this cleanup did not
repeat the external notebook/source audit or evaluate a genuine facial recording.

- [Official FaceVitals regulations](https://signalprocessingsociety.org/sites/default/files/2026-10/2027-SP-Cup-Official-Document-FaceVitals.pdf).
  Evaluation reports do not replace final MAT packaging or the technical report.
- [Official UBFC-rPPG page](https://sites.google.com/view/ybenezeth/ubfcrppg),
  [distribution](https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk),
  [README](https://drive.google.com/file/d/1oL9oy_FU7jbB3qliHQIW629GtEMPJ64q/view),
  [Python reader](https://drive.google.com/file/d/12jym67QYPnXHp3qSRjf1XsQGWRrC284M/view),
  [MATLAB reader](https://drive.google.com/file/d/1oHkc9-U64XHU7qvBlmvpsDhEqSwhA1EQ/view).
- [MediaPipe Face Landmarker Python documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python).

Dataset 2 rows are Python 0=PPG, 1=sensor HR (BPM), 2=seconds; MATLAB uses rows
1, 2, 3 respectively. Dataset 1's gtdump.xmp format uses milliseconds and is not
supported. See the ground-truth contract above for handling duplicate timestamps.

Dataset citation: S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois,
“Unsupervised skin tissue segmentation for remote photoplethysmography,”
Pattern Recognition Letters, 2017.

Read the dataset agreement before use. The prior source review records a
restriction on publication of subject 21/27 facial images; retain that restriction
and check the current agreement. No facial images are exported. Keep datasets,
models, participant/clinical information and generated outputs out of Git.
Team authorship and preserved scientific choices are recorded in cleanup_plan.md
and cleanup_results.md.
