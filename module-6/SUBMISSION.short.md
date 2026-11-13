# Module 6 Short Assignment

Submit exactly one ZIP to the Gradescope **Short Assignment** channel:

`ML6_2510124204_submission.zip`

The package contains:

- `benchmark_task1.py`
- `plot_results.py`
- `results.json`
- `requirements.txt`
- `report/report.pdf`
- `report/report.tex`
- `report/config.tex`
- `report/reference.bib`
- `report/template22.sty`
- `report/figures/timing_by_rank.pdf`
- `report/figures/speedup_by_rank.pdf`

The short assignment is Task 1 only. The `results.json` file contains no
`task2` object, and the report makes no MNIST or GPU-training claim.

| Criterion | Evidence |
| --- | --- |
| NumPy implementation | `benchmark_task1.py`, `np.matmul` |
| PyTorch implementation | `benchmark_task1.py`, `torch.matmul` |
| Measured timing and speedup evidence | `results.json`, Table 2 |
| Timing/speedup plots and analysis | Figures 1-2, Sections 3-4 |

Measured conditions: 5 sizes x 3 tensor ranks = 15 comparisons. All elapsed
times are stored as positive seconds. CPU, dtype, thread count, package
versions, warm-up policy, and repetition policy are recorded beside the
measurements.
