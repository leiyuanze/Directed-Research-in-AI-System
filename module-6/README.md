# Module 6: AI Hardware and Acceleration

## Completed submission

This repository contains the **Short Assignment (5 points)** submission:

- [`benchmark_task1.py`](benchmark_task1.py) - controlled NumPy CPU,
  PyTorch CPU, and PyTorch CUDA matrix-multiplication benchmark.
- [`plot_results.py`](plot_results.py) - rebuilds both report figures from the
  measured JSON.
- [`results.json`](results.json) - 15 measurements across five sizes and three
  tensor ranks, with three method timings and hardware metadata.
- [`report/report.pdf`](report/report.pdf) - three-page LaTeX report.
- [`report/report.tex`](report/report.tex) - main XeLaTeX source, bibliography,
  style file, and vector figures.
- [`ML6_2510124204_submission.zip`](ML6_2510124204_submission.zip) - complete
  Gradescope package.

The measured short-run result is not a universal hardware ranking. PyTorch CPU
is fastest near size 128, but converges toward NumPy CPU at size 1024. CUDA
crosses over near size 256; at 1024 it is 9.57x--13.78x faster than NumPy.
Rank-3 and rank-4 curves nearly coincide because both compute four matrices.
The report also gives a concise comparison of the BLAS and cuBLAS/SM execution
paths.

Task 2 is intentionally absent: the original handout defines the Written
Assignment as Task 1 only, and the course README confirms that the Short
submission must omit the entire `task2` object. See
[`requirements.pdf`](requirements.pdf) for the original assignment and
[`SUBMISSION.short.md`](SUBMISSION.short.md) for rubric coverage.
