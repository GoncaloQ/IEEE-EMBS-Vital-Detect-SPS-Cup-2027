# Round 1 consolidation results

Implemented locally on `cleanup/round1-consolidation`, starting at `c7e88bf`.
The pre-deletion [file-by-file plan](cleanup_plan.md) classified all 43 tracked
files using their contents, Git log, rename history and blame. Old code was kept
until the replacement passed all 73 relevant tests, including actual C4/C5 encoding.
The initial cleanup made no commit, push, merge or history rewrite and did not
modify the original separate development repository. Subsequent integration is
recorded at the end of this document.

## Complete deleted-path inventory (17)

Migrated/consolidated sources also appear here because their old paths were removed.

```text
Round1_Baseline/process_round1.py
docs/sources_and_notebook.md
src/vital_detect/__init__.py
src/vital_detect/__main__.py
src/vital_detect/cli.py
src/vital_detect/config.py
src/vital_detect/datasets/ubfc.py
src/vital_detect/evaluation/metrics.py
src/vital_detect/hr/estimation.py
src/vital_detect/pipeline.py
src/vital_detect/reporting/outputs.py
src/vital_detect/robustness/benchmark.py
src/vital_detect/roi/face.py
src/vital_detect/signals/processing.py
src/vital_detect/video/reader.py
tests/test_roi_policy.py
tests/test_signals.py
```

## Moved/consolidated functionality

| Source | Result |
| --- | --- |
| datasets/ubfc.py | Round1_Baseline/dataset.py: validated discovery/association, raw reference reading, duplicate-instant handling and time-weighted integration. |
| video/reader.py | Round1_Baseline/video.py: streaming decoder timebase, metadata, truncation and resource checks. |
| pipeline.py, relevant cli.py/config.py code | Round1_Baseline/evaluate.py: options, team extraction adapter, batch/window pairing, failures, hashes, software provenance and one CLI. |
| signals/processing.py | Only bounded timestamp validation/interpolation survives in evaluate.prepare_interval; independent detrending/filtering/POS removed. |
| evaluation/metrics.py | Existing Round1_Baseline/metrics.py is the sole, byte-identical implementation. |
| reporting/outputs.py | Round1_Baseline/reporting.py: simplified CSV/JSON, figures and reproducible Markdown report. |
| robustness/benchmark.py | Round1_Baseline/robustness.py: C5 plus distinct lossless C4 control, measured constraints and matched-recording comparisons, using the team baseline. |
| tests/ | Retained/adapted reference, metric, decoder, evaluation and robustness coverage; new test_baseline.py covers actual team functions and import safety. |
| README and protocol/source docs | Rewritten for the final code; historical independent-estimator validation claims removed, dataset/synchronization/statistical/privacy assumptions preserved. |
| pyproject.toml and tests.yml | Package/test Round1_Baseline; obsolete console script/src discovery removed. MANIFEST.in includes conftest and docs in sdist. |

The independent ROI, GREEN/POS estimator and alternate signal filtering are gone.
The team can still select any existing RGB channel. A small package initializer
makes Round1 importable; it performs no processing. There is no replacement
nested framework or separate configuration/CLI module.

## Resulting repository (35 files)

```text
.github/workflows/tests.yml
.gitignore
MANIFEST.in
README.md
pyproject.toml
Round1_Baseline/
  __init__.py
  channels_cheeks.py
  channels_forehead.py
  hr_extraction.py
  filters.py
  metrics.py
  utils.py
  evaluate.py
  dataset.py
  video.py
  reporting.py
  robustness.py
  evm.py
  EVM_Option_1.ipynb
  generate_low_resolution_video.m
  generate_partial_occlusion_video.m
  notes.txt
Round2_HR_SpO2/.gitkeep
Round3_Constrained/.gitkeep
Round4_FullSystem_Clinical/.gitkeep
docs/
  cleanup_plan.md
  cleanup_results.md
  evaluation_protocol.md
tests/
  conftest.py
  test_baseline.py
  test_metrics.py
  test_pipeline.py
  test_robustness.py
  test_ubfc.py
  test_video_policy.py
```

Ignored environments, caches, models, build products and synthetic test outputs
are omitted from this tracked-file structure.

## Preserved team work and behavior evidence

No protected file was deleted. All function bodies in the following files were
compared using Python AST against `git show c7e88bf:<path>` and matched exactly,
excluding docstrings: channels_cheeks.py (2 functions), channels_forehead.py (4),
hr_extraction.py (5, including unfinished hooks), filters.py (3), evm.py (5),
metrics.py (1). The filter return docstring was corrected without changing code.
The notebook, both MATLAB scripts, metrics.py and notes.txt are byte-identical.
All three future-round placeholders are retained.

ROI module edits are package/direct-script import support and unused/import-order
cleanup. HR demo execution is behind a main guard and creates its Results folder;
its original calculations/plots remain. The model helper now explicitly imports
urllib.request and creates the requested parent directory. EVM only gains the
blank line required by Ruff. No mask, landmark, smoothing, filter coefficient,
FFT formula, parameter formula or EVM temporal filter was replaced.

The adapter decodes twice to retain team ROI source unchanged, validates sample
association and applies explicit synchronization preprocessing. Missed detections
remain zero in team output and become missing only within evaluation. Long gaps,
unsupported boundaries and nonpositive HR are failures, never hidden corrections.

## Actual validation

Local environment: macOS ARM64, Python 3.11.18; NumPy 1.26.4, SciPy 1.14.1,
OpenCV contrib 4.11.0.86, MediaPipe 0.10.21, Matplotlib 3.10.9,
pytest 8.4.2, Ruff 0.11.13, FFmpeg/ffprobe 6.0, actionlint 1.7.12.
The repository's old .venv points to a missing temporary interpreter; it was not
modified. Validation used `/private/tmp/round1-validation` and temporary tools.

| Check | Actual result |
| --- | --- |
| `ruff check Round1_Baseline tests` | PASS, all checks passed. |
| `python -m pytest -q` after deletion | **73 passed, 0 skipped**, 6.80 s; all relevant tests collected. |
| Replacement tests before deletion | **73 passed**, 7.20 s, including encoding. Initial run without FFmpeg had 69 passed / 4 skipped; FFmpeg was provisioned and all four ran successfully before deletion. |
| Packaged source-distribution tests | **73 passed, 0 skipped**, 7.15 s, outside repository; verifies packaged test fixture completeness. |
| `python -m build` | PASS, isolated sdist and wheel built with setuptools 81.0.0. The wheel was built from the sdist. |
| Wheel installation outside source tree | PASS; all 12 modules imported from site-packages; evaluate --help succeeded; old vital_detect package absent. |
| Side-effect guards | PASS; import tests prohibit video opening, downloading and figure saving. |
| Team preservation | PASS, AST and byte comparisons described above. |
| Synthetic evaluation/reporting | PASS, multiple recordings, a missing recording, all-failed batch, negative HR failures, window-specific reference changes, saved-config reproduction, CSV/JSON consistency and decodable PNGs. Three generated figures also visually inspected. |
| Team ROI integration | PASS for cheeks and forehead: actual team masks/extraction and actual HR estimator on synthetic videos, controlled landmarks/no model download. This does not validate real landmark detection. |
| Robustness | PASS, actual native/lossless C4/C5 transforms, measured FPS/resolution/bitrate, common intervals, failure metadata and matched error comparisons. |
| GitHub workflow | PASS with actionlint; local commands succeed. Remote GitHub Actions execution was not triggered or observed. |
| Source references and whitespace | PASS, deleted module/wrapper/test paths absent from live imports/commands; `git diff --check` clean. Historical path inventory intentionally remains in audit docs. |
| Dependency requirement versions/markers | PASS, every active installed requirement satisfied. |
| `uv pip check` / current `pip check` | **FAIL on the retained MediaPipe wheel's platform metadata**, detailed below. Do not describe dependency checks as uniformly green. |

Synthetic inputs are uniform color-modulated videos with known frequencies,
constructed reference rows and controlled landmarks/extractor seams. They are
not genuine facial recordings. No real-dataset accuracy result, physiological
validity, hardware synchronization or clinical conclusion is claimed.

Temporary FFmpeg/ffprobe binaries came from the
[eugeneware static builds](https://github.com/eugeneware/ffmpeg-static/releases/tag/b6.1.1)
(the binaries report 6.0), and actionlint from its upstream release. No downloaded
tool, model, participant file or generated report was added to version control.

## Limitations and team decisions

1. The team's unrestricted FFT argmax can select a negative frequency. The
   evaluator logs the raw value and fails the observation. Restricting to positive
   frequencies, taking absolute HR or adding spectral quality gates is a scientific
   change and remains for explicit team approval.
2. The team ROI functions retain integer FPS rounding, accumulated integer-ms
   detector timestamps, zero-filled missed detections and existing detector/resource
   behavior. The evaluator uses validated video timestamps but does not correct the
   detector's internal clock. Fractional-FPS/occlusion performance needs team review.
3. Green/cheeks, 1–2 Hz/order 3 mirror the demo, not a proven optimal protocol.
   Agree on ROI, band, channel, windows and synchronization assumptions before
   final scoring. Changing them must be documented rather than selected on test errors.
4. Two decode passes cost time; preserving team source made this tradeoff explicit.
   Real facial videos, actual MediaPipe model detection, varying physiology,
   acquisition offsets, and HD/C3/streaming/edge behavior were not validated.
   MATLAB execution and notebook/EVM scientific output were not tested here.
5. MediaPipe 0.10.21 was already pinned in the project. Its installed WHEEL metadata
   says `cp311-cp311-macosx_14_0_x86_64`, so uv 0.13.9 and pip 26.3 platform checks
   fail on ARM64. `file` confirms its framework binary contains **both x86_64 and
   arm64** and imports/integration tests succeed. All dependency version constraints
   are satisfied. The pin was preserved to avoid changing the team's detector
   behavior; choose a validated corrected wheel/version before requiring a fully
   clean platform-metadata check. No local patch hides the metadata problem.
6. No protected-file deletion needs approval. Only future scientific/MediaPipe
   changes above require team agreement. At the time of this initial validation,
   the branch was local and uncommitted. Previous implementations are recoverable
   from `c7e88bf` / `377877c` Git history.


## Second pass: further redundancy removed

At GoncaloQ's request, the additional deletion was planned in cleanup_plan.md
before implementation. The unique links, dataset citation, row specification and
privacy restrictions from docs/sources_and_notebook.md now live in the evaluation
protocol; the extra source document is deleted. Team authorship evidence remains
in the audit plan. README links to the consolidated document.

GroundTruth.source_path had no consumers and was removed; Recording.reference_path
still supplies the real reference association and hash. The unused video_path key
on benchmark failure rows was also removed (reporting discarded it). Native runs
reuse the video hash rather than reading the same file twice; transformed runs
still retain distinct source and transformed hashes. These changes do not alter
estimation, synchronization, reports or any protected team file.

The initial-pass validation above remains historical; second-pass checks are
recorded below after execution.

Second-pass validation: `ruff check Round1_Baseline tests` passed;
`python -m pytest -q` passed **73 tests, 0 skips, 7.31 s**, with FFmpeg/ffprobe
available. `git diff --check` passed. All local documentation links resolve;
the final inventory is 35 files with all 17 deleted original paths documented.
The MediaPipe platform-metadata limitation remains as reported above.


## Integration with main (2026-11-18)

PR #1 conflicted because main advanced from `c7e88bf` to Leonor's `ff9b690`
while the cleanup branch contained `b332736`. The conflict was in
`Round1_Baseline/hr_extraction.py`; `notes.txt` merged automatically and retains
Leonor's new notes. Main was merged into the cleanup branch without rewriting
history.

The resolution retains the cleanup's package/direct-script imports, import-time
safety and automatic Results directory, plus Leonor's signal-based
`get_hr_bpm`, sliding windows, HR traces and eight demonstration plots.
`extract_hr(raw_signal, fps, fc, N=3)` remains available for the dataset evaluator,
using the same detrending, Butterworth filter and unrestricted FFT peak.
The earlier AST/byte preservation comparisons above describe the pre-integration
cleanup, not the newly integrated HR functions and notes.

Window extraction now accepts array-like signals, uses read-only bounded NumPy
views, rounds fractional-FPS durations to frames, rejects invalid durations and
returns a consistent empty pair for short recordings. The demo passes the actual
FPS to both HR traces and labels seconds/BPM correctly. Its unfinished empty
reference comparison would raise a length mismatch, so it directs users to the
existing dataset evaluator for actual paired metrics.

Validation in `/private/tmp/round1-validation`, with FFmpeg/ffprobe available:
**87 tests passed, 0 skipped** (7.38 s), including actual team ROI integration,
C4/C5 encoding, import safety, 25/30/29.97 FPS window boundaries, empty traces,
invalid window parameters and all eight demo plots on controlled input.
Ruff, actionlint, evaluator `--help` and whitespace/conflict-marker checks passed.
No real participant recording or scientific accuracy result is claimed.

GitHub run `44372485778` subsequently passed all checks on macOS but failed
collection on both Linux Python versions: MediaPipe imports sounddevice, whose
PortAudio initialization could not connect to a PulseAudio server. The Linux
workflow now installs PulseAudio and starts a user daemon before importing
MediaPipe; it does not stub audio modules or skip the affected integration tests.
