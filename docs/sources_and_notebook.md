# Sources, format verification and algorithm attribution

Sources inspected on 2026-11-09. The original development repository was read
through GitHub only; it was never modified, committed to or pushed to.

## Competition specification

[Official FaceVitals regulations](https://signalprocessingsociety.org/sites/default/files/2026-10/2027-SP-Cup-Official-Document-FaceVitals.pdf),
Section 2.3 Task 1 lists MAE (BPM), RMSE (BPM), Pearson r and Bland–Altman limits,
and requires at least two C1–C7 constraints addressed/justified in the report.
Section 2.2 C4 specifies VGA/QVGA with an HD reference; C5 specifies ≤10 FPS and
≤500 kbps. This repository supports offline experiment strategies. Dataset testing
alone does not demonstrate all acquisition or deployment requirements.

## UBFC verification

The [official dataset page](https://sites.google.com/view/ybenezeth/ubfcrppg) links
the [distribution folder](https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk).
The [README](https://drive.google.com/file/d/1oL9oy_FU7jbB3qliHQIW629GtEMPJ64q/view),
[Python reader](https://drive.google.com/file/d/12jym67QYPnXHp3qSRjf1XsQGWRrC284M/view)
and [MATLAB reader](https://drive.google.com/file/d/1oHkc9-U64XHU7qvBlmvpsDhEqSwhA1EQ/view)
were inspected. Dataset 2 uses Python rows 0=PPG, 2=time, 1=HR, corresponding to
MATLAB rows 1, 3, 2. The README confirms seconds for Dataset 2; Dataset 1 instead
uses millisecond timestamps in column 1 of `gtdump.xmp`.

Official subject1 `ground_truth.txt` was read and validated outside the repository:
3×1547 values, time 0..52.691 seconds, one repeated timestamp at 16.750 seconds.
Raw timestamps are preserved. Equal-time HR values are averaged at that instant
for integration; no timestamps are fabricated. Decreasing times and zero-duration
references fail. This is a loader check only: no real UBFC video was downloaded
or evaluated. Synthetic tests cover this policy without redistributing source data.

Reference is the **sensor HR row in BPM**, never PPG amplitude. The dataset authors
note that their own published evaluation derived HR from contact PPG rather than
the sensor HR row. Scores from this sensor-HR protocol must not be described as
reproducing their published benchmark.

Dataset citation (as requested on the official page): S. Bobbia, R. Macwan,
Y. Benezeth, A. Mansouri, J. Dubois, “Unsupervised skin tissue segmentation for
remote photoplethysmography,” Pattern Recognition Letters, 2017.
Read the distribution agreement. Subject 21/27 facial images must not appear in
publications, presentations or reports; the pipeline exports no facial images.

## Notebook audit

All 38 cells of
[`Baseline Testing.ipynb`](https://github.com/GoncaloQ/IEEE-EMBS-Vital-Detect/blob/main/Baseline%20Testing.ipynb)
were inspected.

| Reusable function/idea | Adaptation |
| --- | --- |
| `extract_forehead_rgb_signals` | Binary masks, actual frame dimensions, RGB means; no whole-video storage |
| `extract_upper_cheeks_rgb` | Landmark sets, convex hulls, subject-side naming, inset margins |
| EMA in forehead/cheek functions | Configurable 0.6 weight, reset on missing detection |
| `get_calibrated_skin_mask`, `find_hairline_above_mesh` | Audited but not copied: HSV thresholds and upward extension can bias skin selection/include hair; inner mesh forehead used |
| `band_pass_butter_filtering` | SOS/zero-phase filtering retained; finite, duration and Nyquist checks added |
| `ffourier_transform`, `get_hr_bpm`, `extract_hr` | Positive, band-limited Hann spectrum with quality gates replaces unrestricted FFT argmax |
| `download_model_if_missing` | Explicit offline-preparation command, versioned official URL and checksum |

Corrected integer FPS, accumulated integer timestamps, zero-filled detections,
hardcoded 30 FPS and reference constants, negative-frequency selection and lack
of synchronized evaluation. Notebook accuracy claims/constants are not reused.
Team-authored landmark logic and scientific ideas were adapted; third-party
implementation source was not vendored. No source-repository changes were made.

## POS and software references

W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan, “Algorithmic Principles of
Remote PPG,” IEEE TBME 64(7), 1479–1491, 2017,
[DOI 10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282).
The [author preprint, Algorithm 1](https://sstuijk.estue.nl/publications/tbme16-algorithmic-rppg.pdf)
defines temporal means, projection `[0,1,-1]`, `[-2,1,1]`, standard-deviation ratio
and overlap addition. Our independent NumPy implementation uses a 1.6 s window,
divides the overlap sum by contribution count to avoid edge taper, then applies
the common detrending/filter/HR estimator. Degenerate projections fail. Algebraic
and known-HR tests are included. This overlap normalization is an explicit variant.

[MediaPipe Python documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
was consulted for VIDEO mode and monotonic timestamps. Two faces are requested
to detect ambiguity; explicit EMA supplies smoothing. No persistent target identity
is implemented when people replace one another. Preverified single-subject inputs
are required. C3 remains unimplemented.

Bland–Altman uses sample SD and 1.96; no confidence intervals, clinical thresholds,
or compliance with all Task 1 acquisition requirements are inferred.
