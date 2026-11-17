# Technical walkthrough: recording outline and spoken script

Target: approximately 9 minutes, within the required 5-10 minutes. Audience: a fellow ML student or engineer. Show the named files and figures as you explain them; do not spend time scrolling through every line. Blockquotes contain the spoken script. Other text gives screen directions and preparation notes.

Script prepared with Codex from the project code and saved results. Rehearse in your own words. You should be able to explain each statement before using it. Keep the AI attribution accurate: describing the implementation does not mean you independently wrote or designed all of it. Add personal experiences or reflections only if they are true. Recordings are not yet completed.

## Prepare these views

- `src/payoffs.py`: `Leg.profit`, `construct`, and `sample_trade`.
- `src/diverse.py`: `THEMES`, `render_diverse`, and `sample`.
- `src/learning.py`: `preprocess`, `RobustCNN`, and `backbone`.
- `src/train_diverse.py`: `CONFIG`, training loop, scheduler, and checkpoint selection.
- `docs/RESULTS.md`: original model comparison, hyperparameters, augmentation, and two-factor ablation.
- `src/predict.py`, `src/geometry_reader.py`, `src/measure.py`, `src/ocr.py`, and `src/calibration.py`.
- README Evaluation, `docs/results/cnn_diverse.json`, and the running app.

The parameter count below was checked against the current `RobustCNN`: **319,880 trainable parameters**. The current run has seed **731**, **16,000** training images, a **25-epoch** budget, and checkpoint **epoch 20** selected by validation loss. These describe the current model, not the earlier 39,768-parameter CNN.

## 0:00-0:45 | Goal, contribution, and system map

Show the repository structure and the attribution file. Then show the current reader's entry points: `models/runtime.json` and `src/app.py`.

> The goal is an educational reader for eight options profit and loss shapes across different chart formats, with numerical extraction when the image supports it. I supplied the educational scope and requested broader formatting support and numerical readings. Codex generated much of the implementation and documentation and ran the recorded experiments, as described in the attribution file. The system has three main jobs: classify the shape, isolate and measure the curve, and present the result with evidence. Training and evaluation code are separate from the local Flask interface.

## 0:45-1:40 | Ground truth and data generation

Open `Leg.profit` and one construction in `src/payoffs.py`, then `THEMES` and `sample` in `src/diverse.py`. Show the style preview briefly.

> The labels come from analytic option equations. For a long call, profit at expiration is the larger of price minus strike and zero, minus the premium. Signed quantities let the same code represent long and short legs, and portfolio profit is their sum. That provides known curves and labels without collecting market prices. The current training set has sixteen thousand images across fifty style families: ten themes and five layout treatments, with additional random variation. Trades and seeds differ across splits. The extra style combinations are separated into five for validation and five for test. These sets share a renderer and equations, so they are controlled synthetic checks rather than independent real-world data.

## 1:40-2:25 | Why the first model was insufficient

Show the old selected CNN row in `docs/RESULTS.md`, then the current README comparison. Clearly label the old table as historical.

> The original selected CNN reached one hundred percent on familiar synthetic test images but only thirty-nine point eight seven five percent on unfamiliar styles. That showed the familiar test distribution was too easy to treat as a product-readiness check. The rebuild broadened the training formats, added dark-background normalization, and changed the network. It also added a separate geometry path. The current CNN reaches ninety-eight point three seven five percent on the old unfamiliar-style set. That set had already been inspected, so this is an improvement check, not a new untouched test. Several parts changed together; the result does not isolate one cause of improvement.

## 2:25-3:25 | Current CNN and preprocessing

Open `preprocess` and `RobustCNN` in `src/learning.py`. Point to each block as you describe it.

> Images are resized to ninety-six by ninety-six. They are converted to grayscale, retained in three identical channels, and scaled to values between zero and one. Dark images are inverted using a median-intensity rule. The CNN has three convolution blocks with sixteen, thirty-two, and sixty-four output channels. Each block uses batch normalization, ReLU, and max pooling. Adaptive pooling keeps a six-by-six spatial representation. A fully connected layer reduces that to one hundred twenty-eight features, followed by dropout and eight output logits. Retaining spatial layout matters because left and right wings distinguish directional shapes. The current network has three hundred nineteen thousand eight hundred eighty trainable parameters. Softmax produces class scores, but those scores are not calibrated guarantees of correctness.

## 3:25-4:20 | Training and model selection

Open `train` in `src/train_diverse.py`. Show the saved selected epoch and history in `docs/results/cnn_diverse.json`.

> Training uses shuffled batches of one hundred twenty-eight, cross-entropy loss, and AdamW. Dropout and weight decay provide regularization. A cosine learning-rate schedule reduces the learning rate over the twenty-five-epoch run. Checkpoint selection minimizes the average loss on familiar-style validation and held-out-style validation. Epoch twenty was selected. The test sets are evaluated after that choice and do not supply gradients or choose the checkpoint. The earlier experiment runner uses validation-loss early stopping, but this current run uses the full fixed epoch budget and best-checkpoint selection. The distinction matters when describing which controls actually trained the deployed model.

## 4:20-5:35 | Baselines and controlled experiments

Show the four relevant tables in `docs/RESULTS.md`. Briefly show `backbone` in `src/learning.py` and the original transfer-training function in `src/train.py`.

> The original experiments compare three approaches: nearest neighbors on small resized images, a compact CNN, and frozen MobileNetV3 features with a logistic-regression classifier. The backbone is pretrained on ImageNet and stays frozen; only the classifier head is fitted to these labels. MobileNet uses efficient depthwise spatial convolutions and channel-mixing pointwise convolutions. Its feature vector gives the classifier a learned image representation rather than raw pixels. Pretraining and input resolution differ across approaches, so this is not a compute-matched ranking. Separate tables compare three learning-rate and dropout configurations, brightness and contrast augmentation, and a two-by-two ablation of full versus cropped images and color versus grayscale. The augmentation pair improved unfamiliar-style accuracy by one point five percentage points in that single-seed comparison. These are historical controls; they do not prove the effect of each change in the newer rebuild.

## 5:35-6:30 | Geometry and the hybrid prediction path

Open `Predictor.predict` and `read_geometry`. Show the app's **Neural model comparison** panel on a chart where the readers can be compared.

> The current system does not rely only on the neural prediction. A geometry reader looks for plausible curve colors, removes some axes and disconnected text, and fits the supported piecewise shapes. It checks fit quality and coverage and rejects large gaps rather than inventing a missing central section. When it finds a clear match, that shape can replace the neural suggestion. This is a rule-based numerical fitter, not a second trained neural network. The interface reports which path produced the shape and keeps the neural prediction and confidence separate. That prevents a neural confidence score from being presented as confidence in a different geometry-based answer.

## 6:30-7:30 | OCR, numerical values, and corrections

Show `recognize` in `src/ocr.py`, `calibrate` and `measure` in `src/measure.py`, and `manual_axes` in `src/calibration.py`. Demonstrate the correction panel briefly.

> OCR reads text and its position in the image. Apple Vision is used when available, with a local RapidOCR fallback for missing labels or platforms without Vision. The measurement stage fits linear relationships between tick positions and printed values, then maps curve pixels into price and vertical-axis units. Curve kinks suggest strike locations, and zero crossings suggest breakevens when the chart is confirmed to show profit and loss. A payoff graph before premium does not establish those profit and loss amounts. Unreadable values are withheld. Users can confirm two ticks on an axis by entering values and clicking their positions. Those corrections change the measurement calibration; they do not retrain the CNN or implement online learning.

## 7:30-8:35 | Evaluation and its limits

Show the current README table, `fidelity_automatic_audit.json`, and `asx_holdout_audit.json`. Identify which rows concern the CNN alone and which concern the combined reader.

> The current CNN scored one hundred percent on sixteen hundred familiar-style test images and eight hundred held-out-format test images. The original evaluation also reports macro F1 and log loss, because accuracy alone does not describe class balance or probability quality. For practical checks, the combined reader identified all six Fidelity development diagrams, including charts with a dashed pre-expiration curve. With the chart type confirmed as profit and loss, maximum strike error was zero point one four and maximum breakeven error was zero point two three. No reference numbers or tick corrections were supplied to those predictions. A reserved ASX publisher check identified four of four diagrams, but it covers only long calls and puts. These samples are too small to establish general accuracy on screenshots or handwriting.

## 8:35-9:15 | Reproducibility and takeaway

Show `models/runtime.json`, the included checkpoint, tests, and `SETUP.md`. Return to the running interface.

> The included checkpoint lets the app run without retraining. Setup instructions explain the environment, optional OCR, training reproduction, and source-tracked web checks. The recorded verification includes sixty-two passing tests and a clean-checkout smoke check. The tests check equations, measurements, missing-evidence handling, and interface behavior; they are not an accuracy benchmark for arbitrary images. The main progression is from a classifier that was sensitive to formatting to a broader local reader with geometry, OCR, and visible measurement evidence. Further work is a larger independent real-chart set, actual handwriting, and better handling of unsupported or incomplete inputs.

## Recording checks

- Keep the final recording between 5 and 10 minutes. If it runs long, shorten navigation and repeated numbers before removing the explanation of experiments or limitations.
- Explain convolution, frozen features, validation selection, the two independent ablation choices, and the OCR/geometry distinction in your own words.
- Do not call the original unfamiliar-style set a fresh final test, the simulated pen strokes real handwriting, or all product stages learned ML models.
- Do not reuse the old model's 0.31 ms latency as the current full app's latency. It excluded decoding/loading and did not include the current geometry/OCR pipeline.
- Save the finished recording under `videos/` and add its direct link to the README. Any personal reflection should describe your own review, decisions, and learning; this script does not supply invented experiences.
