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
`task2` object. Its GPU data are matrix-multiplication timings, not an MNIST or
training claim.

| Criterion | Evidence |
| --- | --- |
| NumPy implementation | `benchmark_task1.py`, `np.matmul` |
| PyTorch CPU implementation | `benchmark_task1.py`, CPU `torch.matmul` |
| PyTorch CUDA implementation | `benchmark_task1.py`, CUDA `torch.matmul` |
| Three-method timing and speedup evidence | `results.json`, Table 2 |
| Timing/speedup plots and principles | Figures 1-2, Sections 3-4 |

Measured conditions: 5 sizes x 3 tensor ranks = 15 comparisons, each with all
three timings. All elapsed values are positive seconds. GPU inputs are
preloaded; launch and synchronization are included while PCIe transfer is
excluded.
