# Project demo: recording outline and spoken script

Target: approximately 4 minutes 20 seconds, within the required 3-5 minutes. Audience: someone who has not taken machine learning. Show the app, charts, and result summaries; do not show source code or a terminal in this recording. Read the blockquotes as a starting script and use your normal speaking pace. The directions outside blockquotes are not spoken.

Script prepared with Codex from the project code and saved results. This script describes implemented behavior and recorded results. Rehearse the live examples before recording. If an on-screen value differs slightly from the saved check, describe the value actually shown rather than reading an exact number from memory.

## Before recording

- Start the app using the README instructions and open http://127.0.0.1:5001.
- Have the README Evaluation section and [style preview](../docs/figures/diverse_formats.jpg) ready in separate tabs.
- Have `data/web_charts/fidelity-bull-call-spread.png` ready for the upload. If missing, run `python -m src.webcharts fetch --split development` before recording. This downloads the source-tracked public examples; do not show the terminal in the demo.
- Check the example buttons and the Fidelity upload. For that image, select **Profit/loss, including premium**.
- Keep the chart's source visible or identify it as a public Fidelity educational example. Do not imply it is an independent final test or your own drawing.
- Time one rehearsal. Aim for about 500 spoken words plus the live interactions. Do not leave long loading pauses in the final recording.

## 0:00-0:35 | The problem

Show the app heading, then the eight-category preview from the README.

> This is Options Payoff Reader. It is an educational tool for understanding options profit and loss graphs. These graphs show how a position's profit or loss changes with the underlying price at expiration. Some have a rising wing, some make a V, and some have a central peak. The same shape can look pretty different in a textbook, on a dark screen, or in a sketch. The goal is to recognize that shape and help read the values from the graph.

## 0:35-1:20 | Recognize the shapes

Click **Long butterfly**, then **Long put**. Point to the peak and flat tails in the butterfly, then the downward slope and flat region in the put. Briefly show the style preview.

> The reader supports eight shape categories. Here is a long butterfly: it has a peak in the middle and flat loss regions on both sides. A long put looks different. It benefits as the expiration price falls, and its loss is flat above the strike. Each result gives a suggested category and a short explanation of that shape. The examples use different chart styles. Training included fifty combinations of themes and layouts, including dark backgrounds, graph paper, and simulated pen strokes.

## 1:20-2:20 | Read a real chart

Upload the Fidelity bull-call-spread image. Select **Profit/loss, including premium**, then **Read chart**. Show the result and the uploaded image together. Read the actual approximate values on screen.

> This is a public Fidelity educational chart. It includes a solid expiration curve and a dashed curve for before expiration. The reader identifies the expiration shape as a bull spread. It can also estimate numerical values when the axes and curve are readable. Here, the strikes are about one hundred and one hundred five, and the breakeven is about one hundred one point eight. Maximum profit is roughly three point two, and maximum loss is roughly one point eight, in the chart's units. These are estimates from the image. The tool does not assume a contract multiplier or pull live market prices.

## 2:20-2:55 | Confirm a difficult axis

Expand **Axis labels used for these estimates**, then **Correct unreadable axis ticks**. Enter 100 and click its price-axis tick; enter 105 and click that tick. Submit **Read with these settings** and show the evidence marking the ticks as confirmed by you.

> If a label is hard to read, I can confirm the scale. I enter a printed tick value, click its tick mark, and repeat for a second tick on that axis. The reader then uses those confirmed points to measure the curve. It marks those values as user-confirmed, so I can see where the measurements came from. This is a fallback for unclear labels, rather than silently filling in missing information.

## 2:55-3:40 | What was measured

Show the README Evaluation table. Keep the distinction between recorded evaluation and this live demonstration clear.

> The first model did well on familiar synthetic charts, but unfamiliar formatting dropped its accuracy to about forty percent. After the rebuild, the current CNN reached about ninety-eight point four percent on that same, already-inspected set. It also reached one hundred percent on the new synthetic test sets. The combined reader identified all six Fidelity development examples and all four examples from a reserved ASX publisher check. Those are small checks, and the ASX set only covers calls and puts, so they do not prove it can read every graph.

## 3:40-4:20 | Scope and takeaway

Return to the app. Point to its supported categories, numerical evidence, and local-processing note.

> The tool recognizes shapes, not a unique portfolio. Different positions can produce the same graph. It also needs enough of the curve and axis information to support a numerical answer. Actual handwriting, crowded screenshots, and unsupported shapes still need broader testing. Everything runs locally, with no paid API. The useful result is a working educational reader that combines a shape suggestion, numerical estimates, and a way to check or correct the evidence behind them.

## After recording

Save the finished video under `videos/` and replace its pending entry under **Video Links** in the README with a direct, accessible recording link. Check the final duration is between 3 and 5 minutes and that no code appears on screen. This script does not count as the video submission.
