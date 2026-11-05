# Validation record

Executed November 5, 2026 in the project-local Python 3.12 environment on macOS ARM64. PyTorch 2.15.0 reported MPS unavailable; experiments used CPU. The user's machine is an Apple Silicon MacBook Air; the execution platform string and measured timings are recorded in docs/results/evaluation.json. Four torch threads were used. The compact network has 39,768 trainable parameters.

## Executed checks

- 18 financial tests passed before generation: call/put signs and premiums, eight exact payoff constructions, eight slope patterns, continuity and invalid parameters.
- Eight-panel preview rendered and visually inspected before generating the large dataset. A clipped tick label was corrected.
- 21 tests passed after adding data/preprocessing checks: 12,000/1,600/1,600 balanced splits, unique IDs/trades/image hashes, visible strikes, deterministic rendering and RGB/grayscale preprocessing.
- Baseline, seven CNN runs and frozen MobileNet classifier completed. CNN runs trained for 7-12 epochs with actual histories preserved. All saw all 12,000 training examples each epoch.
- Selection was written before held-out evaluation. All nine candidates were evaluated on test, unfamiliar-style and independent-renderer data. Forty incomplete inputs were probed without assigning accuracy.
- 23 tests passed with the selected real model: valid uploads and all eight examples, malformed/missing/tiny/oversized images, invalid example selection and existing financial/data checks.
- `python -m pip check`: no broken requirements. `python -m compileall -q src`: passed.
- Flask server launched on localhost port 5001; Safari displayed the reader and a real Long butterfly example prediction. Screenshot inspected for layout. The in-app browser was unavailable, so Safari was used. Flask test-client upload checks exercise actual inference; no external deployment is claimed.
- Training curves, confusion matrices, annotated failures and incomplete-chart panels were visually inspected.

## Debugging and corrections

Initial sandbox network access blocked dependency retrieval; installation succeeded with authorized network access into .venv. No global dependencies were modified. The pretrained weights were downloaded from the official PyTorch host.

Matplotlib's grid(False, alpha=...) enabled a grid unexpectedly. The independent renderer was corrected and its images/manifest regenerated before any test evaluation. Later inspection showed some independent-renderer axis labels clipped at the canvas edge; geometry remains complete. The evaluated images were preserved and this limitation is disclosed in ERROR_ANALYSIS.md.

The CNN resume check initially looked only for results JSON. It now also checks the checkpoint exists, so a clean Git clone with recorded results but no ignored model files retrains correctly. `--rerun` explicitly overwrites previous CNN artifacts; OPTIONS_DEVICE permits a CPU override.

The selection tie-break exposed tiny checkpoint-size differences from serialization names, although all CNNs have identical parameter counts. This weakness is documented; the lock was preserved after test inspection. No claim of statistically superior hyperparameters is made.

## Limits of verification

A second fresh-machine installation/full retraining was not run. No real-chart dataset, calibration, robust rejection, arbitrary plot detection or production deployment was evaluated. The recorded 100% test accuracy applies only to the synthetic test distribution.

Relative document links and notebook code syntax passed static validation. Model and manifest SHA-256 hashes are recorded in docs/artifact_hashes.json.

Final portability review replaced moving the live CNN to CPU during checkpoint saving with CPU copies of its state tensors, preserving optimizer-device references for future accelerated runs. Recorded CPU results/checkpoints were not changed. MPS/CUDA execution remains unverified here.

Source formatting used a temporary Black installation and the bundled Python 3.12.14 interpreter after Black rejected the project Python 3.12.5 for its known AST-check issue. This did not change the project interpreter or runtime dependencies. Full environment details are in environment.json.
