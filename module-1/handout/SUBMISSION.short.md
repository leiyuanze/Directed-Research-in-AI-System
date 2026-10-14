# Module 1: Machine Learning, Short Assignment

**Grading notice:** The score shown on Gradescope is an AI-assisted reference score and is subject to change. The teaching assistants and instructor may review and adjust it and retain final authority over interpretation of the assignment requirements, grading rubric, and final grades. Correct, reasonable work can receive full marks; optional extras and writing polish are not required.

Submit exactly one ZIP containing `report.pdf`, `report.tex`, and every figure,
bibliography, or style file used by the report. The report must explain the
implementation approach for Part 1 of the Full Assignment, including the
dataset split, model construction, forward pass, training loop, checkpoint
handling, evaluation, and the PyTorch functions used. Explain how WandB would
record a reproducible experiment.

This Gradescope assignment is worth 5 points. Do not include the Full
Assignment implementation, checkpoints, or dataset in this ZIP.

Use the course LaTeX template. Figures and citations are not mandatory for this
option. If you use figures or external sources, include their dependencies and
acknowledge the sources as appropriate. You do not need to run training or
produce WandB curves; explain how the implementation would work.

DeepSeek grades the Part 1 explanation using the following rubric. Its score
out of 5 is applied directly as the **AI reference score** on Gradescope;
there are no separate points just for file presence or formatting. Submit
both the PDF and usable LaTeX source; missing either receives zero report
points until supplied. PDF text is sent to DeepSeek; if text extraction fails,
rendered report pages may be sent instead.

If the service fails, returns an invalid response, or identifies evidence
that needs review, the grade is marked **pending review** and a TA can review
or rerun it. A service failure is not treated as an incorrect answer.

| Explanation | Points |
| --- | ---: |
| Data preparation, split, normalization, loaders, and reproducibility | 1 |
| CNN construction, parameter budget, and forward pass | 1 |
| Training loop and validation, including an optional scheduler if used | 1 |
| Saving/loading checkpoints and evaluating accuracy | 1 |
| Recording the required experiment configuration and metrics with WandB | 1 |
| Total | 5 |

A reasonable, accurate explanation earns full credit, including concise
answers. Minor wording issues, grammar, optional details, and writing polish
do not reduce the score when the required ideas are correct. Deductions are
for substantive errors or missing major requirements. Equivalent PyTorch APIs
and sensible optimizer choices are accepted. Part 2 and the bonus are outside
the scope of this option.
