# Methodology

The reader classifies the geometry of a profit/loss chart at expiration. It does not identify the unique portfolio behind that geometry. Different portfolios can have the same payoff shape. The eight labels describe shape categories, not trading recommendations.

## Financial construction

For expiration price S, strike K, premium p, and signed quantity q, each leg contributes q[max(S-K,0)-p] for a call or q[max(K-S,0)-p] for a put. Portfolio profit is the sum. All amounts are per share. Fees, financing and contract multipliers are excluded. These are educational synthetic premiums, not market quotes or a pricing model.

Let C(K,p) and P(K,p) denote long call and put profits. Let c be the center, w the strike spacing parameter, and p a positive premium smaller than w.

| Class | Construction | Slopes from left to right |
|---|---|---|
| Long call | C(c,p) | 0, +1 |
| Long put | P(c,p) | -1, 0 |
| Bull spread | C(c-w/2,2p) - C(c+w/2,p) | 0, +1, 0 |
| Bear spread | P(c+w/2,2p) - P(c-w/2,p) | 0, -1, 0 |
| Long straddle | C(c,p) + P(c,p) | -1, +1 |
| Long strangle | P(c-w/2,p) + C(c+w/2,p) | -1, 0, +1 |
| Long butterfly | C(c-w,4p) - 2C(c,2p) + C(c+w,p) | 0, +1, -1, 0 |
| Short straddle | -C(c,p) - P(c,p) | +1, -1 |

Center is uniform on [75,150]; w/c on [.12,.23]; p/w on [.12,.32]. Each side extends 2.1-3.0w from c, preserving all kinks. Each image is a unique trade with its own seed, not a repeated rendering assigned to a different split. Future multi-render extensions must keep trade_id in one split.

## Data and leakage control

Standard split: 12,000 training (78.95%), 1,600 validation (10.53%), 1,600 test (10.53%). Every class has 1,500/200/200 examples. Extra sets are 800 unfamiliar-style charts, 80 independently rendered charts, and 40 incomplete charts with no scoreable labels. The complete metadata is generated in data/generated/manifest.jsonl; its hash and counts are in dataset_summary.json.

Pillow draws training charts. Background, line color, width, grid, resolution, aspect ratio and font size vary independently of class. The portable built-in font family is fixed; font size varies (the style metadata's regular/bold selector adjusts size, not actual font weight). There are no strategy names, titles or IDs inside model input images. Numeric ticks reflect the underlying equations and automatic y-range. This can still permit shortcuts through axis scale/zero-line position; crop ablation is only a partial probe, not proof of pure geometric reasoning.

Novel styles use unseen backgrounds/colors and line widths. The separate Matplotlib renderer changes tick layout, font family, spines and antialiasing while retaining the equations and normalized plot box. It tests renderer transfer, not real financial screenshot performance. Incomplete images keep only the left quarter and stretch it, erasing essential kinks. Their source category is not a valid target once geometry is lost; report confidence behavior, not accuracy.

Preprocessing converts RGB, optionally crops the known normalized box (.14,.09,.94,.83), optionally converts grayscale, and resizes to a square. No preprocessing parameters are fitted on validation or test data. Aspect-ratio distortion from square resize is deliberate and shared by training/inference; no flips or rotations are used. A crop-based selected model would require uploads with the same plot layout; this is not automatic plot localization.

## Models and controls

- Baseline: distance-weighted 3-nearest-neighbor classifier on 16x16 RGB pixels, all 12,000 training examples retained.
- CNN: three stride-2 convolution layers (8,16,32 channels), ReLU, spatial adaptive pooling to 4x4, a 64-unit hidden layer, dropout, eight logits. Spatial pooling retains left/right wing location. Adam, cross-entropy, batch 128, seed 386, maximum 12 epochs. Dropout and early stopping (patience 3, validation loss minimum improvement .0001) provide two regularization mechanisms.
- Transfer: pretrained MobileNetV3 Small frozen in eval mode, 576-dimensional pooled features and a logistic-regression head (C=1, maximum 1,000 iterations). Uses ImageNet normalization. The 96x96 direct resize deliberately differs from the official 224px center-crop transform to preserve entire chart geometry and reduce CPU cost. This is a small-resolution transfer experiment, not a claim about the backbone's best achievable accuracy.

Three CNN hyperparameter configurations change learning rate (.003 vs .001) or dropout (.2 vs .4). A separate augmentation pair adds independently sampled brightness [.85,1.15] and contrast [.8,1.2] per image. The 2x2 ablation independently switches full/crop and color/grayscale, with all other reference settings fixed. Reference is reused as a clearly named control, not claimed as an additional experiment. Early stopping can give different epoch counts, which are recorded; comparisons share maximum budgets rather than identical steps. Single-seed comparisons cannot establish statistical superiority.

All decisions use validation. models/selection.json locks the winner before test/OOD evaluation. Selection uses validation accuracy, breaking ties by model file size. Latency is measured afterward to assess that tradeoff without retuning from test results. CNN checkpoints are selected by the declared validation-loss criterion. Held-out test data are never used for gradient updates or hyperparameter selection.

## Metrics

Accuracy is the fraction correct; macro-F1 weights each category equally; log loss measures the probability assigned to the true class. Balanced accuracy averages class recall. Since all test classes have equal support, balanced accuracy equals accuracy and is not independent evidence. Per-class precision/recall/F1 and confusion matrices expose category-specific errors. Softmax and nearest-neighbor probabilities are not calibrated confidence guarantees.

Latency uses CPU inference, four torch threads, batch size one, 10 warmup calls and 50 perf_counter measurements. It includes preprocessing and prediction, excludes image decoding/model loading, and reports median/p95. Platform and checkpoint sizes accompany results. GPU availability is checked at runtime; the current sandbox reports MPS unavailable. Training can use MPS or CUDA on a compatible unrestricted runtime, but numeric results may differ.

## Research context

Masry et al., *ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning*, Findings of ACL 2022: https://aclanthology.org/2022.findings-acl.177/. This motivates evaluating visual chart interpretation, but our narrower task has no language reasoning and does not reproduce ChartQA results.

Howard et al., *Searching for MobileNetV3*, ICCV 2019: https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html. Its focus on efficient vision architectures motivates the compact pretrained comparison, not any claim that our implementation improves that paper.

Official torchvision weights and transforms: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html. OCC/OIC expiration-profit reference: https://www.optionseducation.org/strategies/all-strategies/long-call. Sources verified November 5, 2026.
