# Setup

Run commands from the repository root. Tested with Python 3.12 on macOS ARM64, in an isolated .venv. No global packages are changed. Exact installed versions are in requirements.txt; pip check passed. This runtime used CPU because PyTorch reported MPS unavailable. The training device selector also supports MPS and CUDA when available. CPU runs are practical with these small models; expect several minutes to tens of minutes depending on hardware. Allow a few GB of free memory and disk space.

## Install

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
export MPLCONFIGDIR="$PWD/tmp/matplotlib"
```

Internet access is required for package installation and the first official MobileNet weight download. No API keys are needed. The environment here was created with /opt/homebrew/bin/python3.12; use your own compatible Python 3.12 path. Do not force-install incompatible wheels on other platforms; choose the official PyTorch build for your hardware if these pins have no wheel, and record the change.

## Quick local run with existing artifacts

The working directory delivered locally includes generated data and trained model files (ignored by Git). With those artifacts present:

```sh
python -m src.app
```

Open http://127.0.0.1:5001. Use an example or upload a PNG/JPEG/WebP chart. The app binds only to localhost, has an 8 MB upload limit and does not save uploads. This is a local development app, not a deployed service.

## Reproduce from a clean clone

Generated data and large model weights are intentionally not committed. Rebuild them in this order:

```sh
python -m pytest tests/test_payoffs.py -q
python -m src.generate --preview
python -m src.generate
python -m pytest tests/test_pipeline.py -q
python -m src.train baseline
python -m src.train cnn
python -m src.train transfer
python -m src.train select
python -m src.evaluate
python -m src.report
python -m pytest -q
python -m src.app
```

The generator refuses to overwrite an existing manifest. Preserve an existing dataset before regenerating; do not mix datasets from different generator versions. CNN runs with completed results are skipped to allow resuming. For deliberate retraining, archive both the corresponding docs/results/cnn_*.json and models/cnn_*.pt files first. The baseline and transfer commands overwrite their own model/results files; archive those before rerunning if you need the prior experiment. Do not retune after viewing held-out results without creating a new untouched test set and reporting the protocol change.

A clean clone includes recorded experiment evidence, but missing checkpoints trigger training. To deliberately repeat all CNN runs even when checkpoints exist, use `python -m src.train cnn --rerun` after archiving your prior evidence. Generated results are single-seed observations and may differ on other hardware.

## Model loading and portability

models/selection.json names the selected run. src/predict.py loads its state on CPU, with the config from docs/results/<run>.json. CNN files are torch state dictionaries loaded with weights_only=True. The baseline and transfer head use joblib: load only trusted locally generated files. Pretrained weights are downloaded into models/cache; the frozen backbone is saved separately for offline inference. A grader can reproduce all model files from code without a hosted checkpoint link.

## Evidence and troubleshooting

- docs/dataset_summary.json: counts and manifest hash.
- docs/results/: actual run configurations, histories, evaluation and predictions.
- docs/RESULTS.md and docs/ERROR_ANALYSIS.md: tables, interpretation and failures.
- docs/figures/: chart preview, learning curves and evaluation figures.
- docs/VALIDATION.md: executed checks and limitations.

If a checkpoint is missing, run its training command before launching the app. If port 5001 is occupied, edit the final port argument in src/app.py. If MPS fails, set `OPTIONS_DEVICE=cpu` before running training. This project has no OCR, automatic plot detector, real-market connection, or calibrated rejection mechanism.
