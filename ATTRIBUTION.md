# Attribution

## AI assistance

OpenAI Codex generated the initial payoff equations implementation, synthetic renderer, tests, preprocessing, CNN and training loops, pretrained comparison, evaluation scripts, local Flask interface, and documentation from Riley Cusick's supplied project specification and course rubric. Source modules include file-level attribution. Codex ran the experiments and recorded their actual outputs. Riley must review the implementation, verify understanding, and record the walkthrough before submission; this file does not claim Riley independently wrote the generated code.

During development, equation tests independently checked known option values and piecewise slopes. Visual preview review caught a clipped x-axis tick label and led to text-width-aware placement. A Matplotlib warning revealed that passing grid style arguments with grid(False) enabled grids; the renderer and the independent set were corrected before evaluation. Dataset integrity tests check unique trades, image hashes, balanced counts and visible strike locations. Additional corrections and runtime findings are recorded in docs/VALIDATION.md. Generated reports do not invent personal reflections or video recordings.

The user supplied the research question, eight categories, scope limits, desired experiment design, rubric targets and submission requirements. AI assistance is allowed by the supplied course page, with attribution and student responsibility. The custom-network rubric includes a student-design requirement: the student must assess that eligibility honestly after reviewing and adapting this work.

## Data and assets

Training/evaluation images are original synthetic charts generated analytically by src/payoffs.py and src/generate.py. No external financial charts or market feeds were copied. Synthetic premiums are educational assumptions. The independent renderer uses the same equations with Matplotlib. Pillow's built-in font and Matplotlib's bundled fonts are provided through their libraries. Uploaded course PDFs and the editable self-assessment template are course materials; source documents are excluded from Git. Personal assessment materials are retained locally and excluded from publication.

## Software and pretrained weights

Python, NumPy, Pillow, Matplotlib, PyTorch, torchvision, scikit-learn, SciPy, Flask, pytest, joblib, pypdf and document tooling support this project. Installed versions are pinned in requirements.txt. Respect each dependency's license. MobileNetV3 Small ImageNet-1K V1 weights come from the official torchvision download, https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth; the ImageNet dataset was used upstream for pretraining and is not redistributed here. Check upstream terms before redistributing pretrained weights. Only synthetic training data are generated locally.

## References

- Masry et al. (2022), ChartQA, Findings of ACL: https://aclanthology.org/2022.findings-acl.177/
- Howard et al. (2019), Searching for MobileNetV3, ICCV: https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html
- torchvision MobileNetV3 Small documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
- Options Industry Council, Long Call: https://www.optionseducation.org/strategies/all-strategies/long-call

These references informed scope, architecture choice, preprocessing and payoff checks. No published benchmark results are claimed as reproduced. See docs/METHODOLOGY.md for how each reference relates to the project.
