# Demo recording outline

Target: about 4 minutes. Audience: non-specialists. Show the running interface, diagrams and results; no code. This is an outline to record, not a completed video. Adapt wording after reviewing the work.

## 0:00 to 0:40  The question

“Options diagrams show how a position's profit or loss changes with the price at expiration. Some have a rising wing, some have a V, and some have a central peak. This project asks whether a small image classifier can recognize those shapes when chart formatting changes. It is an educational recognition tool, not a recommendation about what to trade.”

Show docs/figures/class_preview.png. Point out the long call, long straddle and butterfly. Explain that equivalent portfolios can have the same shape.

## 0:40 to 1:50  Use the reader

Open http://127.0.0.1:5001 and click Long butterfly. Explain the central peak and flat tails. Try Long put and explain its descending segment. Upload a complete light-background chart from data/generated/test. Read the category and confidence. Explain that the explanation is a curated category description, not an independent verification of the image.

## 1:50 to 2:50  What the results show

Show the README evaluation table. “The selected model got all 1,600 familiar test charts right. But that was not the whole story. Accuracy fell to about 40% with unfamiliar styling, and to about 89% with a different chart renderer. A pretrained alternative handled unfamiliar styles much better, at about 85%, but was larger and slower.”

Make clear that all images are synthetic; these are not results on real brokerage screenshots.

## 2:50 to 3:35  Show a failure

Show docs/figures/failures.png or upload data/generated/ood/ood-6-0034.png. Its true shape is a butterfly, but the selected model predicts a straddle with high confidence. “The curve is readable to a person, but the model is sensitive to a dark background it did not see during training.” Show one incomplete-chart example and explain why its missing geometry makes the task unanswerable.

## 3:35 to 4:10  What it means

“The reader works on its familiar synthetic chart style, but it is not ready for arbitrary screenshots. The next experiment would broaden the training and validation styles, then test on a new untouched set. The useful result is understanding where a perfect-looking test score stops being reliable.”

Finish with supported categories and limitations. Keep the recording between 3 and 5 minutes, then put the video in this directory or provide an accessible direct link in README.
