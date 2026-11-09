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

The selected `models/cnn_diverse.pt` checkpoint is included in Git. Generated training data and the older experiment checkpoints remain local and ignored. After installing dependencies, both the local working copy and a clean clone can run:

```sh
python -m src.app
```

Open http://127.0.0.1:5001. Use an example or upload a PNG/JPEG/WebP chart. The app binds only to localhost, has an 8 MB upload limit and does not save uploads. This is a local development app, not a deployed service.

## Build the multi-format reader

The practical reader uses an included checkpoint, with the original experiments retained separately. To reproduce its training rather than use the included checkpoint:

```sh
python -m src.diverse
python -m src.train_diverse --epochs 25 --train-count 16000
python -m pip install -c requirements.txt -r requirements-ocr.txt
python -m src.ocr
python -m src.app
```

`src.diverse` writes the 50-family preview and style inventory. Training caches versioned image tensors under `data/diverse-v1/`, writes `models/cnn_diverse.pt`, and saves results under `docs/results/cnn_diverse.json`. The app's `models/runtime.json` selects this checkpoint. Training deliberately does not change the original `models/selection.json`.

The optional `requirements-ocr.txt` installs RapidOCR, ONNX Runtime, and their dependencies into the project environment. RapidOCR's model weights are bundled in its wheel; inference is offline. These packages are installed in the delivered `.venv`, with the existing requirements preserved; `pip check` passed. ONNX Runtime telemetry is disabled before constructing OCR sessions.

On macOS, `python -m src.ocr` additionally compiles `src/ocr.swift` to `tmp/vision-ocr`, using Apple's on-device Vision framework. It requires Xcode command-line tools; skip this command on other operating systems. Apple Vision is tried first when available. RapidOCR provides a fallback when Vision is unavailable or misses an axis. The portable engine was verified on this Mac; other operating systems have not been runtime-tested here. macOS Vision requires normal OS access and can fail inside a restricted sandbox.

If neither OCR engine is available, upload the chart, expand **Correct unreadable axis ticks**, enter a tick value, and click its tick mark. Confirm two well-separated ticks for each axis that needs correction. The selected axis must be linear. Confirm whether vertical values include premium; a payoff graph alone does not establish the cost of a position.

Images stay local, including during OCR. No paid API, API key, or cloud service is used. The optional free-tier API idea has not been enabled. See `docs/PRODUCT_STATUS.md` for measured coverage and remaining work.

## Reproduce the public-chart checks

```sh
python -m src.webcharts fetch --split development
python -m src.webcharts audit --split development --numerical --output docs/results/fidelity_automatic_audit.json
python -m src.webcharts fetch --split holdout
python -m src.webcharts audit --split holdout --output docs/results/asx_holdout_audit.json
python -m src.audit_numerics
```

The fetch commands download only explicitly listed public image URLs from `data/web_chart_sources.json`. Images and download metadata stay under ignored `data/web_charts/`; saved reports include source URLs and hashes. Publisher membership is checked so development and holdout sources cannot be mixed. Downloaded images are evaluation assets, not training data. Once downloaded, audits run offline. The ASX set was evaluated after the Fidelity-driven reader changes and must be treated as development data if future changes are tuned to it.

## Reproduce the original experiments from a clean clone

The older experiment data and checkpoints are not committed. Rebuild those experiments in this order:

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

models/runtime.json selects the practical reader; models/selection.json retains the original experiment winner. An explicit `Predictor(name)` can load either. src/predict.py loads its state on CPU, with the config from docs/results/<run>.json. CNN files are torch state dictionaries loaded with weights_only=True. The baseline and transfer head use joblib: load only trusted locally generated files. Pretrained weights are downloaded into models/cache; the frozen backbone is saved separately for offline inference.

## Evidence and troubleshooting

- docs/dataset_summary.json: counts and manifest hash.
- docs/results/: actual run configurations, histories, evaluation and predictions.
- docs/RESULTS.md and docs/ERROR_ANALYSIS.md: tables, interpretation and failures.
- docs/figures/: chart preview, learning curves and evaluation figures.
- docs/VALIDATION.md: executed checks and limitations.

If a checkpoint is missing, run its training command before launching the app. If port 5001 is occupied, edit the final port argument in src/app.py. If MPS fails, set `OPTIONS_DEVICE=cpu` before running training. Local OCR and curve isolation are implemented; there is still no real-market connection or calibrated unknown-input detector.
