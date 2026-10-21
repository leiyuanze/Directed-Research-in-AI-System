# Module 2: Computer Vision, Full Assignment

**Grading notice:** The score shown on Gradescope is an AI-assisted reference
score and is subject to change. The teaching assistants and instructor may
review and adjust it and retain final authority over interpretation of the
assignment requirements, rubric, and final grades. Correct, reasonable work
can receive full marks; optional extras and writing polish are not required.

Complete both tasks. Use Omnidata pretrained models to estimate depth and
normals, then reconstruct the object mesh from the supplied multi-view RGB-D
data by completing `hw.ipynb`.

Submit exactly one ZIP containing the completed notebook, Task 1 code, at least
two depth and two normal prediction images under a path containing `task1`,
`result`, `output`, or `prediction`, plus `report.pdf`, `report.tex`, and all
figures and LaTeX dependencies. The report must show the depth/normal results
and the RGB-D reconstruction, with analysis of the camera convention and
reconstruction settings. Do not include pretrained weights or the supplied
dataset.

Task 1 uses at least **5 images from different domains**, including at least one image on which the model performs poorly. Include the image, depth, and normal results and the analysis requested in `handin_requirements.html`. The two-prediction automatic check is only a minimum file check; it does not replace the five-image assignment requirement.

Keep the `read_rgbd_and_pose` function name in `hw.ipynb` and use its `data_root` argument to locate the supplied files.

The Gradescope maximum is **10 points**.

| Component | Points |
| --- | ---: |
| Task 1 code and submitted prediction artifacts | 1.0 |
| Task 1 report: five-domain visual evidence, analysis, improvements, and stronger-model survey (DeepSeek) | 4.0 |
| Task 2 RGB-D loading, camera convention, and reconstructed geometry | 4.0 |
| Task 2 report: wrong/correct result, coordinate explanation, and TSDF hyperparameters (DeepSeek) | 1.0 |
| Total | 10.0 |

The automatic Task 1 artifact check accepts Python, notebook, or shell-command
workflows and checks only minimum reproducibility evidence. It does not award
the four report points. DeepSeek evaluates the report text and every rendered
page, including the actual input/depth/normal triplets and reconstruction
figures. Unsupported prose does not replace visible results. Submit both the
PDF and usable LaTeX source; missing either receives zero report points until
supplied. Citations are not a separate scoring requirement.

A reasonable, technically accurate report earns full content credit, including
concise analysis. Deductions must identify a substantive error, unsupported
claim, or missing major requirement. If the AI service fails, returns an
invalid result, or cannot reliably resolve particular evidence, the report
grade is marked **pending review** rather than treated as a final zero. PDF text
and rendered pages are sent to DeepSeek. Reports over 20 pages or 80,000
extracted characters require manual review.

The stronger-model experiment and Blender mesh comparison remain useful
optional extensions from the original assignment, but the current Gradescope
autograder does not score them. Attempting or omitting them has no effect on
the 10-point grade and cannot trigger a pending-review status.
