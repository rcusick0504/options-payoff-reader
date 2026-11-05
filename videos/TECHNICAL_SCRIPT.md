# Technical walkthrough recording outline

Target: about 8 minutes. This outline is not a recorded walkthrough. Review each module and explain it in your own words; do not imply you personally debugged something you did not. Attribute Codex assistance explicitly.

## 0:00 to 0:50  Problem and structure

State the research question and shape-category limitation. Show src/, data/, models/, docs/ and tests/. Explain that reusable code lives in modules; the presentation notebook reads saved evidence. Describe AI assistance and your own actual review/design work accurately.

## 0:50 to 1:50  Equations and dataset

Open src/payoffs.py. Explain max(S-K,0), signed quantities, and subtracting premium. Walk through the bull spread and butterfly recipes. Show tests/test_payoffs.py: known values and slopes independently check the implementation. Explain synthetic premium assumptions and the absence of market data.

Show src/generate.py and docs/dataset_summary.json. Explain 12,000 training, 1,600 validation, 1,600 test; one trade per rendering; independent random streams for parameters and style; no strategy names inside inputs. Explain which variables change for unfamiliar styles and why incomplete images are not scored.

## 1:50 to 3:00  Baseline and custom CNN

Open src/learning.py SmallCNN. Describe convolutions as local pattern detectors, stride as downsampling, ReLU, retained 4x4 spatial layout, a hidden classifier layer and eight logits. Explain why direction matters and global averaging alone could discard useful location. The CNN has 39,768 trainable parameters (verify against the saved validation record before recording).

Describe the 3-nearest-neighbor baseline on 16x16 pixels. It is a useful check because the synthetic shapes are simple; complex learning is not automatically necessary.

## 3:00 to 4:10  Training and regularization

Open src/train.py. Explain batching/shuffling, cross-entropy, Adam, dropout and validation-loss early stopping. Show the three hyperparameter rows and curves in docs/RESULTS.md. Explain training loss with dropout versus validation loss in eval mode. Explain that single-seed results are limited, and all validation accuracies saturated.

## 4:10 to 5:00  Pretrained alternative

Explain MobileNetV3 Small: compact inverted-residual blocks use channel expansion, depthwise spatial convolution and projection; squeeze-and-excitation gates channels in some blocks. The frozen backbone converts an image into 576 pooled features. A logistic-regression head learns the eight chart labels. Backbone eval mode prevents batch-normalization statistics from changing. This experiment uses 96px full-image resize and ImageNet normalization, not the official 224px center-crop recipe. Pretraining and resolution differ from the CNN and are comparison limitations.

## 5:00 to 6:15  Controlled experiments

Show the paired augmentation table: brightness/contrast only, no flips that would change the class. Show the full/crop by color/grayscale table. Explain the fixed plot box, why it is available for synthetic charts, and why this is not automatic plot detection for uploads. Distinguish hyperparameter tuning, augmentation, architecture comparison and preprocessing ablation.

## 6:15 to 7:15  Evaluation and failure analysis

Explain accuracy, macro-F1 and log loss. Balanced accuracy duplicates accuracy because class counts match. Show the confusion matrices and failures. Explain the 0/400 dark-background result and independent-renderer straddle inversions. Avoid claiming a proven internal mechanism; these are observed error patterns and hypotheses. Show that incomplete charts can receive high confidence, so no rejection guarantee is claimed.

## 7:15 to 8:15  Selection and integration

Open models/selection.json and src/predict.py. Explain selection was locked before test evaluation. Size distinguishes CNN from nearest-neighbor storage, but tiny CNN file-size differences are serialization overhead; admit the tie-break was weak among CNN runs. Show latency/size table. Demonstrate that the Flask interface calls the same preprocessing and prediction code as evaluation. Mention input validation and localhost-only operation. Close with a new validation-style experiment and fresh final test as future work, rather than retroactively tuning on the current test.

Record between 5 and 10 minutes. Add the actual video and direct README link afterward.
