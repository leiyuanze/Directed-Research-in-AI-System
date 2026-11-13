# Module 6: AI Hardware and Acceleration

## Completed submission

This repository contains the **Short Assignment (5 points)** submission:

- [`benchmark_task1.py`](benchmark_task1.py) - controlled NumPy/PyTorch CPU
  matrix-multiplication benchmark.
- [`plot_results.py`](plot_results.py) - rebuilds both report figures from the
  measured JSON.
- [`results.json`](results.json) - 15 measurements across five sizes and three
  tensor ranks, with hardware and software metadata.
- [`report/report.pdf`](report/report.pdf) - four-page LaTeX report.
- [`report/report.tex`](report/report.tex) - main XeLaTeX source, bibliography,
  style file, and vector figures.
- [`ML6_2510124204_submission.zip`](ML6_2510124204_submission.zip) - complete
  Gradescope package.

The measured short-run result is not a universal ranking of the two libraries.
On this Ryzen 7 5800H, PyTorch has the largest median advantage at size 128
(3.26x for 2D and 4.42x for 3D), while the ratio approaches parity at size 1024.
Rank-3 and rank-4 inputs are nearly identical because both compute four
matrices.

Task 2 is intentionally absent: the original handout defines the Written
Assignment as Task 1 only, and the course README confirms that the Short
submission must omit the entire `task2` object. See
[`requirements.pdf`](requirements.pdf) for the original assignment and
[`SUBMISSION.short.md`](SUBMISSION.short.md) for rubric coverage.
