# Round 1 consolidation: audit and pre-deletion plan

Prepared on 2026-11-18 against c7e88bf, on cleanup/round1-consolidation,
before removing any tracked file. All 43 tracked files were read, including both
notebook cells, MATLAB sources, tests, workflow, and all package modules.
No AGENTS.md was found in the workspace/parent tree. No other repository is edited.

## History and functional evidence

- 3393f93 (GoncaloQ): competition directories, README and ignore rules only.
- 377877c (GoncaloQ): independent package, evaluation infrastructure, tests,
  process_round1 wrapper, docs and CI. ed416cb changes only CI actions/runners.
- Diana: cheeks in 3c7bc5b, forehead in d9a0d17 (Portuguese filenames), partial
  occlusion in 537acf9, resolution in f182024. Leonor renamed/cleaned ROI files in
  6902165 and assembled hr_extraction, utils, metrics and notes there.
- 852f92e is authored by Leonor but explicitly credits Tiago for filters;
  authorship alone is therefore insufficient ownership evidence.
- Ines (inetocaetano) introduced EVM_Option_1.ipynb in 5449c9e; Leonor modified
  it in 6902165 and introduced evm.py in c7e88bf. Both remain, even though code
  overlaps: the notebook records experiments and is explicitly protected.
- Round1 metrics.py is byte-for-byte identical to the independent metrics module.
  Leonor's later commit adopted the function; retain that Round1 copy only.
- The two HR estimators are scientifically different: team smoothness-priors
  detrending, SOS filtering and unrestricted FFT argmax versus independent linear
  detrending/normalization, Hann periodogram and quality gates (GREEN/POS).
  Keep the team functions exactly; remove independent algorithms and their tests.
- Forehead implementations differ (team HSV hairline extension versus inset mesh).
  Cheek landmarks share ancestry but masks/continuity policy differ. Keep team ROI.
- Decoder timestamp validation and ground-truth integration have no team equivalent.
  They must survive consolidation. Evaluation will call the existing team video
  extractor, separately validate decoder timestamps/counts, synchronize/resample
  only at the evaluation boundary, and call team extract_hr. Missing face zeros
  will be treated as unavailable observations only by evaluation, never by changing
  the team extractor's default output. No independent spectral estimator remains.

## File-by-file decisions (all original tracked files)

| Original file | Classification | Concrete action and reason |
| --- | --- | --- |
| .github/workflows/tests.yml | MERGE | Test installed Round1 modules, all Round1 Python lint and synthetic integration; keep platform matrix. |
| .gitignore | KEEP | Retain privacy/data/model/output rules; include capitalized Results and plot exports. |
| README.md | MERGE | Replace independent CLI/algorithm instructions with team evaluation, dataset and reproducibility contracts. |
| Round1_Baseline/EVM_Option_1.ipynb | KEEP | Preserve Ines/Leonor experimental notebook unchanged. |
| Round1_Baseline/channels_cheeks.py | KEEP | Preserve landmarks, smoothing, masks and missing-frame policy; package import/unused-import fixes only. |
| Round1_Baseline/channels_forehead.py | KEEP | Preserve HSV calibration, hairline, landmarks, smoothing and masks; package import fix only. |
| Round1_Baseline/evaluate.py | MERGE | Fill empty team evaluation placeholder with batch evaluation, synchronization, options and one CLI. |
| Round1_Baseline/evm.py | KEEP | Preserve separate causal temporal/pyramid experiment; do not substitute HR zero-phase filter. |
| Round1_Baseline/filters.py | KEEP | Preserve all three filters and numerical behavior. Correct docstring return description only. |
| Round1_Baseline/generate_low_resolution_video.m | KEEP | Preserve team C4 script unchanged. |
| Round1_Baseline/generate_partial_occlusion_video.m | KEEP | Preserve team occlusion experiment unchanged; Python does not replace it. |
| Round1_Baseline/hr_extraction.py | KEEP | Preserve numerical functions and unfinished team hooks; guard demo and fix imports, create demo output folder. |
| Round1_Baseline/metrics.py | KEEP | Sole official metrics implementation, unchanged. |
| Round1_Baseline/notes.txt | KEEP | Preserve team research questions/TODOs. |
| Round1_Baseline/process_round1.py | DELETE | Four-line independent CLI wrapper, replaced by evaluate module. |
| Round1_Baseline/utils.py | KEEP | Preserve model helper, fix urllib.request import and parent creation; no download on module import. |
| Round2_HR_SpO2/.gitkeep | KEEP | Preserve official directory. |
| Round3_Constrained/.gitkeep | KEEP | Preserve official directory. |
| Round4_FullSystem_Clinical/.gitkeep | KEEP | Preserve official directory. |
| docs/evaluation_protocol.md | MERGE | Rewrite obsolete algorithm/old validation claims; preserve reference, alignment and statistical contracts. |
| docs/sources_and_notebook.md | MERGE | Preserve dataset format, citations/privacy and attribution; remove obsolete independent algorithm narrative. |
| pyproject.toml | MERGE | Package Round1_Baseline directly; remove independent entry point/src discovery; retain compatible dependencies. |
| src/vital_detect/__init__.py | DELETE | Independent version-only scaffolding; package version in pyproject suffices. |
| src/vital_detect/__main__.py | DELETE | Independent CLI wrapper superseded by evaluate. |
| src/vital_detect/cli.py | MERGE | Keep only needed evaluation arguments/output protection in evaluate; model helper in utils. Remove infer/POS/config framework. |
| src/vital_detect/config.py | MERGE | Small evaluation options in evaluate; delete independent scientific/quality/POS settings. |
| src/vital_detect/datasets/ubfc.py | MIGRATE | Move association validation, discovery and reference loader/integration to Round1_Baseline/dataset.py. |
| src/vital_detect/evaluation/metrics.py | DELETE | Exact duplicate of protected team metrics; update every live import. |
| src/vital_detect/hr/estimation.py | DELETE | Independent estimator superseded by actual team extract_hr. |
| src/vital_detect/pipeline.py | MERGE | Batch/window pairing, failures, hashes and provenance into evaluate; remove independent infer/ROI/estimator abstraction. |
| src/vital_detect/reporting/outputs.py | MIGRATE | Simplify tables/figures/reproducibility reporting in Round1_Baseline/reporting.py. |
| src/vital_detect/robustness/benchmark.py | MIGRATE | Round1_Baseline/robustness.py; retain verified C5, automated paired evaluation and distinct lossless C4 control. |
| src/vital_detect/roi/face.py | DELETE | Independent alternative masks/detector superseded by team ROI functions. |
| src/vital_detect/signals/processing.py | MERGE | Retain bounded interpolation/timestamp validation at evaluation boundary only; remove POS and independent filter/detrend. |
| src/vital_detect/video/reader.py | MIGRATE | Timestamp/count validation in Round1_Baseline/video.py; team ROI continues actual signal extraction. |
| tests/conftest.py | MERGE | Keep synthetic videos/reference fixtures, replace independent ROI provider fixture with team integration seam. |
| tests/test_metrics.py | KEEP | Update import to team metrics only. |
| tests/test_pipeline.py | MERGE | Adapt evaluation/report/failure/sync tests to team extractor and estimator; remove independent ROI/CLI claims. |
| tests/test_robustness.py | MERGE | Keep real encoder and metadata/paired-interval constraints, adapt imports/options. |
| tests/test_roi_policy.py | DELETE | Tests removed independent FaceROI; add focused team ROI tests in test_baseline.py. |
| tests/test_signals.py | DELETE | Tests removed GREEN/POS estimator; replace with team function integration and alignment coverage. |
| tests/test_ubfc.py | KEEP | Update imports/options; retain reference format, integration and association tests. |
| tests/test_video_policy.py | KEEP | Update import; retain decoder timebase/truncation/resource tests. |

## Robustness overlap and retention

MATLAB C4 resizes to exact orientation-aware VGA/QVGA dimensions using bicubic
interpolation and lossy MPEG-4 at source FPS; it can upscale/distort non-4:3 inputs.
Python C4 preserves aspect ratio, never upscales, encodes losslessly with FFV1 and
verifies measured metadata. Retain this distinct spatial control with automatic
same-recording/interval reports. Neither replaces the other. MATLAB occlusion is
not implemented in Python. C5 FPS/compression/bitrate checks exist only in Python
and remain. No clinical, streaming, identity, or HD compliance claims are added.

## Implementation order and deletion gate

1. Create Round1 package and import-safe team modules without changing math.
2. Implement small dataset/video/evaluation/reporting/robustness modules and tests.
3. Validate replacement with synthetic input, all imports, metrics, reports,
   synchronization and C5 transformations before deleting old implementation.
4. Delete redundant package/wrapper/tests only after replacement passes; update
   config, workflow and documentation; search for stale runtime references.
5. Record actual validation and complete deletion/migration inventory in
   docs/cleanup_results.md. Preserve protected files and official placeholders.

## Scientific decisions reserved for team approval

Do not change unrestricted FFT sign selection, fixed team detector timing/FPS
rounding, zero-filled missed detections or EVM filtering automatically. Evaluation
will expose invalid/nonpositive estimates as failures and record raw estimates;
it will never take abs(HR), tune offsets or substitute an independent algorithm.
Default evaluation band 1..2 Hz/order 3 and green channel mirror the existing team
demo; these are experimental settings requiring team agreement before final scoring.
No protected file is proposed for deletion. No approval is needed for this plan's
routine implementation steps. Existing implementations remain recoverable at c7e88bf.

Packaging addendum after implementation: source-distribution inspection found
that setuptools omitted conftest.py. MANIFEST.in now includes that fixture and
protocol/audit docs; the resulting packaged tests were also executed.

## Second pass requested by GoncaloQ

This additional plan is recorded before the additional deletion. The scope is
redundancy in GoncaloQ's evaluation work; no team scientific code is changed.

| File | Decision | Reason/replacement |
| --- | --- | --- |
| docs/sources_and_notebook.md | MERGE then DELETE | Dataset links, citation and privacy note move to evaluation_protocol.md; format/alignment are already described there and authorship is recorded in this audit. |
| docs/evaluation_protocol.md | KEEP / MERGE | Single operational reference for evaluation assumptions and source specifications. |
| README.md | KEEP / UPDATE | Link to the consolidated protocol; remove obsolete documentation link. |
| Round1_Baseline/dataset.py | KEEP / SIMPLIFY | Remove unused GroundTruth.source_path, which has no consumers. Recording.reference_path remains authoritative. |
| Round1_Baseline/evaluate.py | KEEP / SIMPLIFY | Reuse native video hash instead of reading the identical source file twice; retain distinct source/transformed hashes for robustness. |
| Round1_Baseline/robustness.py | KEEP / SIMPLIFY | Remove failure-row video_path key, discarded by reporting and unused by consumers. |
| docs/cleanup_results.md | KEEP / UPDATE | Update deleted-path inventory, final structure and actual checks. |

Retain the audit plan/results because they were explicitly requested deliverables.
Retain dataset/video timing, reference integration, metrics, batch/window failures,
reporting, C5 and the distinct lossless C4 comparison. These have active consumers
and tested purposes. No additional algorithm or protected-file deletion is planned.
