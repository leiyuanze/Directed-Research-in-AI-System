# Module 2: Computer Vision, Short Assignment

**Grading notice:** The score shown on Gradescope is an AI-assisted reference
score and is subject to change. The teaching assistants and instructor may
review and adjust it and retain final authority over interpretation of the
assignment requirements, rubric, and final grades. Correct, reasonable work
can receive full marks; optional extras and writing polish are not required.

Complete only Task 1: use the Omnidata pretrained models to estimate depth and
surface normals. Submit exactly one ZIP containing the inference code,
`report.pdf`, `report.tex`, all LaTeX dependencies, and at least two examples
from each prediction type. Put result images under a path containing `task1`,
`result`, `output`, or `prediction`, for example:

```text
task1_outputs/depth_01.png
task1_outputs/depth_02.png
task1_outputs/normal_01.png
task1_outputs/normal_02.png
```

This channel does not require the RGB-D notebook, mesh, or supplied dataset
files. The Short Assignment maximum is 5 points.

Task 1 uses at least **5 images from different domains**, including at least one image on which the model performs poorly. Include the image, depth, and normal results and the analysis requested in `handin_requirements.html`. The two-prediction automatic check is only a minimum file check; it does not replace the five-image assignment requirement.

| Component | Points |
| --- | ---: |
| Task 1 code and submitted prediction artifacts | 1.0 |
| Five-domain visual evidence in the report (DeepSeek) | 1.5 |
| Image-specific analysis and concrete improvements (DeepSeek) | 1.5 |
| Accurate survey of a stronger monocular depth model (DeepSeek) | 1.0 |
| Total | 5.0 |

The automatic artifact check accepts Python, notebook, or shell-command
workflows and checks only minimum reproducibility evidence. It does not award
the four report points. DeepSeek evaluates the extracted report text and every
rendered page, including the actual input/depth/normal triplets; unsupported
prose does not replace visible results. Submit both `report.pdf` and usable
LaTeX source. Missing either receives zero report points until supplied.
Citations are not a separate scoring requirement, and Task 2 and all optional
steps are outside this channel's five-point rubric.

A concise, technically accurate report can earn full credit. Deductions must
identify a substantive error, unsupported claim, or missing major requirement.
If the AI service fails, returns an invalid response, or cannot reliably judge
specific evidence, the report grade is marked **pending review** rather than
treated as a final zero. PDF text and rendered pages are sent to DeepSeek.
Reports over 20 pages or 80,000 extracted characters require manual review.
