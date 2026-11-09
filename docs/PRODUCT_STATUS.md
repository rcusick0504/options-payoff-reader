# Practical reader status — November 9, 2026

## Target

Build a useful educational reader for varied payoff diagrams, including numerical interpretation where the image supports it. The original formatting failure is a defect to address. This is a substantially improved local prototype; there is no claim of universal format support.

## Implemented

1. Fifty synthetic training families (ten themes × five layout treatments), with independent trades and style choices. Ten held-out combinations add a sixth layout treatment. The families share one renderer; these are not fifty independent renderers. Pen, pencil, and chalk jitter is simulated and does not establish real handwriting performance.
2. A 96×96 grayscale CNN with dark-background normalization, trained on 16,000 images for 25 epochs. Best checkpoint chosen by average familiar/unseen validation loss. Test sets are disjoint by seed and trade and never used for checkpoint selection. Cached datasets are under `data/diverse-v1/`.
3. A geometry reader isolates plausible curve inks, filters disconnected text, and compares piecewise shapes. It may correct a neural prediction when a clear curve matches. The UI identifies the source and separately reports the neural prediction; geometric fit error is not presented as a probability.
4. Local Apple Vision plus optional portable RapidOCR, robust linear-axis calibration, curve tracing, and conditional numerical estimates. Automatic full extraction requires two readable price ticks and three vertical ticks. Staggered price labels near a central zero axis are supported. Independently colored solid expiration curves can be separated from a dashed pre-expiration curve. Price-only calibration can recover strikes while withholding other fields. Profit/loss labels must be recognized or the user must explicitly identify the chart as profit/loss. Values assume the selected shape and full expiration curve, a nonnegative underlying price, and no contract multiplier.
5. Blank-input checks, uncertain-model messaging, EXIF orientation handling, transparent-image flattening, and retained upload limits.
6. An in-memory uploaded-image preview and a tick-confirmation tool: enter values and click two well-separated ticks on an axis. Confirmed ticks are validated and labeled separately from OCR evidence. With both axes confirmed, numerical extraction can work without OCR. Choosing payoff before premium withholds breakevens and profit/loss limits.

## Evidence

- New CNN: 100% on 1,600 familiar-style test images and 800 held-out-format test images. These are synthetic. Full history, configuration, family breakdowns and selected epoch: `results/cnn_diverse.json`.
- Original unfamiliar styles: new CNN 98.375% (787/800), compared with the original model's 39.875%. Original familiar test: 100% (1,600/1,600). See `results/diverse_legacy_audit.json` for additional comparisons.
- Eight public OIC images: combined reader 8/8, with source URLs and outputs in `results/web_development_audit.json`. The initial CNN check was 4/8 and drove development of the geometry path. This is a development audit from one publisher, not a generalization benchmark. Downloaded images remain in ignored `tmp/`; they are not training data or distributed assets.
- Current numerical development audit: 23/24 generated charts returned at least strike estimates; median absolute strike error 0.25 and maximum 0.43 in price-axis units. The correct category was supplied to isolate measurement accuracy. No manual tick correction was used. `results/numerical_v2_audit.json` preserves current values and axis evidence; `results/numerical_audit.json` retains the earlier Apple-only check. Real OIC examples have no numeric vertical scale, so monetary values are withheld.
- Fidelity development set: 6/6 shapes correct. All six returned numerical estimates after the chart type was explicitly identified as profit/loss; no numeric reference values or manual tick corrections were supplied. Maximum strike error 0.14, breakeven error 0.23, finite maximum-profit error 0.11, and finite maximum-loss error 0.12. Expected values come from the source examples, in per-share units. See `results/fidelity_automatic_audit.json`.
- Reserved publisher check: ASX 4/4 shapes correct, two long-call and two long-put diagrams/examples. Only those two classes are covered. No reader changes were made in response to this result. See `results/asx_holdout_audit.json`. This is a small publisher holdout, not a broad real-world benchmark.
- Reproducible image inventory and split checks: `data/web_chart_sources.json` and `src/webcharts.py`. Downloaded images are excluded from Git and training; the reports include image hashes and source URLs.
- 62 tests passed, covering equations, numeric recovery for all eight strategies, abstention, axis validation, OCR coordinate mapping and optional dependencies, publisher separation, preprocessing, and uploads. `pip check` passed. A real Chrome upload displayed the expected shape and numerical values; clicking two price ticks and submitting the correction form updated the values successfully.

## Remaining product work

- Assemble a larger, source-separated real-chart benchmark with permissions and provenance, including scans, actual handwriting, camera photos, rotated pages, brokerage screenshots, and multiple curves. Reserve publishers and writers that never guide development.
- Improve plot localization and handling of annotations, dotted curves, overlapping curves, skew, faint strokes, and incomplete images. The color/geometry reader is heuristic and can still select a wrong curve.
- Add a validated unknown-input/rejection mechanism. A high softmax score does not establish that an upload is supported. The 0.8 uncertainty threshold is a heuristic, not calibrated protection.
- Expand numerical auditing across independent renderers and real charts. Tick rounding and OCR errors limit precision; slopes do not establish contract size, premium, or a unique position. Log/percentage axes are unsupported. There is no general document/page parser.
- Verify the portable OCR backend on Windows and Linux. It is implemented and tested locally on macOS, with bundled weights and no cloud calls.

## Optional free vision API

No external API is implemented or enabled. Google lists free-tier image-capable Gemini models, with limits and different data-use terms from paid tiers: [official pricing](https://ai.google.dev/gemini-api/docs/pricing/). This is a candidate for an explicit opt-in fallback. A billing-enabled key can incur charges even for a model that also has a free tier, so a model name alone would not satisfy the no-paid-service requirement. The current application makes no cloud calls.
