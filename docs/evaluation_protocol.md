# Evaluation protocol and data contracts

## Independent observations

UBFC Dataset 2 contributes one primary prediction/reference pair per recording,
one recording per subject. Subject directories are discovered in natural numeric
order; explicit missing subjects remain attempted failures. Exact subject IDs,
filenames, hashes and exclusion reasons are kept. Symlink aliases across subjects
are rejected. Duplicate selections fail instead of inflating the sample count.

Inference never receives the reference HR values. Reference timestamps define
coverage, and the video/reference time intersection defines the recording interval.
Window start/stride are fixed in advance. A bad recording estimate does not prevent
valid, separately labeled window estimates. No best-window or best-method selection
based on reference error occurs. Counts include every attempted recording; results
must be interpreted alongside exclusions, which can introduce selection bias.

## Video and pulse contract

OpenCV frames are BGR; they are converted exactly once to RGB before MediaPipe and
color averaging. Decoding streams frames; only an N×3 RGB array and N timestamps
are stored. First two decoder timestamps determine whether a recording exposes
increasing time. If so, subsequent invalid times fail the entire recording.
Unavailable/zero initial positions permit a consistent frame-index / metadata-FPS
fallback; invalid nonzero initial positions fail rather than hiding a timebase error.
The implementation cannot guarantee that a backend exposing nominal CFR positions
has reconstructed true VFR acquisition timestamps; confirm the camera/container
timebase for applications beyond UBFC. Truncated known-frame-count videos fail.

MediaPipe VIDEO mode detects up to two faces. More than one face or abrupt geometry
changes abort processing; missing detections produce NaN samples, never zero or a
previous mask. Single-subject identity is an input precondition: geometric continuity
does not exclude a person replacing another at a similar location. EMA smooths
landmarks (current weight 0.6), with reset on missed detections. Convex, inset masks
avoid mesh boundary features without a fixed skin-color classifier. Hair, eyewear,
occlusion and landmark errors still need real-data review; no C3 claim is made.

RGB timestamps must increase strictly. At least 90% valid samples are required
per evaluated interval; missing edges are trimmed, short interior gaps (≤0.25 s
between observed valid samples) interpolated, longer gaps rejected. Exact interval
boundaries are interpolated only within short observed support. A uniform grid
includes both retained endpoints; its sample count comes from the median observed
frame period, and its effective FPS is `(N-1)/(last-first)`. Both metadata FPS and
effective FPS are reported. Nyquist is checked before and after resampling.
Minimum duration is the actual retained timestamp span, with no extra nominal
frame period added. Window estimation can be disabled with `--no-windows`;
the per-window CSV remains an empty table with headers.

GREEN uses the green mean. POS independently normalizes RGB in overlapping 1.6 s
windows, projects to two chromatic directions, scales their combination by sample
standard-deviation ratio, and overlap-averages the zero-mean result. Degenerate
channels/projections fail. Linear detrending, unit-variance scaling and order-3
zero-phase Butterworth SOS filtering are shared. HR is the strongest positive
frequency in 0.7–3 Hz, converted by 60 BPM/Hz. Zero padding interpolates a Hann
periodogram and is not extra observation time or physical frequency resolution.

Quality gates require ≥10% of prefilter spectral power in the configured band,
and ≥35% of filtered band power within ±max(0.1 Hz, 1.5/T) of the strongest peak.
A peak at the search boundary is rejected. These fixed heuristic gates do not
validate the biological origin of periodic variation. No algorithm can guarantee
usable HR from arbitrary footage; explicit failure is part of the contract.

## Reference and synchronization contract

Official three-row layout is mandatory: PPG amplitudes, sensor HR in BPM,
timestamps in seconds. Raw arrays are retained. Finite HR in 20..300 BPM is a broad
format sanity bound, separate from the configurable HR estimator band. Nonfinite
data, unequal rows, decreasing times, empty time span or unsupported time scale
fail loading. Duplicate times are counted, kept in raw arrays and coalesced by
mean sensor HR only for integration. Equal-time samples never receive fake times.
Blank separator/trailing lines are ignored without changing the three data-row order.

`shared-clock` preserves reference and video coordinates and assumes the dataset
acquisition convention. `recording-start` additionally shifts first reference
sample to first video frame and must be declared as an assumption. A documented,
independently measured additive offset is supported; no offset search is performed.
Hardware latency and residual drift are unverified. Never silently normalize both
durations, stretch labels, or tune offsets to improve final reported errors.

Reference aggregation integrates piecewise-linear HR in physical time between the
exact prediction boundaries, divides by interval duration, and requires observed
support at both boundaries. It never extrapolates. Any overlapping reference gap
over the configured 1 s default rejects the pair. Thus irregular sample density and
duplicated timestamps cannot overweight an interval. This is sensor-HR evaluation;
it differs from benchmarks extracting reference HR spectrally from contact PPG.

## Statistics and condition comparisons

Signed error = prediction − reference. MAE/RMSE and bias are defined with one pair;
Pearson needs ≥2 pairs and variation in both arrays. LoA use sample SD (`ddof=1`)
and require ≥2 pairs. Mathematical availability is not statistical adequacy: limits
from small samples are unstable, and independence, normality and constant variance
must be assessed. No confidence intervals or clinical acceptance thresholds are
implemented. Nulls have explicit reasons. Invalid pairs cannot enter metrics.

Window metrics are a distinct descriptive output with correlated observations.
Do not compare their observation count to subject-level sample size or use pooled
window LoA as independent-subject inference. The recording's dominant spectral HR
and time-mean reference are different summaries when HR varies: inspect windows
instead of assuming stationarity.

C4/C5 use identical source recordings and a timestamp-only common interval.
Condition deltas require successful pairs and identical retained endpoints.
Failures and nonmatching intervals remain visible. Native VGA is not an HD
baseline. Lossless C4 transformations isolate spatial sampling; C5 combines frame
rate and compression. ffprobe's average bitrate/FPS checks are offline metadata
checks, not network performance measurements. FFmpeg version, actual metadata,
transformation command and total condition timing are preserved in the manifest.
The measured FPS must match the requested target (source FPS for C4, min(source
FPS, 10) for C5), and dimensions must match the requested transform. Unknown
source duration or timestamp origin prevents synchronized benchmarking; it never
falls back to an unrelated full-recording interval. Reports record source and
transformed file hashes, exact subject selection, condition and quoted commands.

## Audit status

On 2026-11-09, **137 tests passed without skips**, with Ruff lint/format checks
across all 22 Python files. Tests include regression coverage for duration limits,
initial timestamp errors, transformed FPS/resolution, missing ffprobe metadata,
metric overflow, recording associations and report reproduction.

Single/subset/full synthetic runs verified GREEN/POS, failure reporting, CSV/JSON
consistency and plots. Eight native/C4/C5 condition-recording evaluations succeeded
using controlled ROIs. Native MediaPipe processed 550 no-face frames and correctly
produced failure statuses and null metrics. The model checksum, installed wheel
and CLI outside the source directory were verified on macOS/Python 3.11.
Other platforms remain pending CI execution.

A genuine reference file was parsed: 1547 samples, one duplicate timestamp,
0..52.691 s, maximum gap 0.223 s. No genuine UBFC video was evaluated. Real-video
single/subset/full evaluation, facial ROI accuracy, hardware alignment and C4/C5
performance remain unverified. Synthetic results do not establish physiological
accuracy. Original repository and unrelated Round directories were not modified.

The [official regulations](https://signalprocessingsociety.org/sites/default/files/2026-10/2027-SP-Cup-Official-Document-FaceVitals.pdf)
were checked again: four HR metrics and the C4/C5 experiment strategies are
implemented. Native UBFC VGA cannot establish the specified HD baseline. The
technical report, required submission MAT/associated artifacts and real-data
constraint justification still require genuine inputs and acquisition metadata.
They are not replaced by this evaluation package. Clinical, live-streaming and C3
identity compliance are not claimed. The project is **not ready for final submission**.
