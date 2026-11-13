#!/usr/bin/env python3
"""Generate the timing and speedup figures used by the Module 6 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator


NUMPY_COLOR = "#3b6ea8"
TORCH_CPU_COLOR = "#d97706"
TORCH_GPU_COLOR = "#2a8c82"
RANK_COLORS = ("#1f6f78", "#b45309", "#6d4c9a")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "report" / "figures",
    )
    return parser.parse_args()


def group_by_rank(
    measurements: Sequence[Dict[str, float]],
) -> Dict[int, List[Dict[str, float]]]:
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for record in measurements:
        grouped.setdefault(int(record["ndim"]), []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item["size"])
    return grouped


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, which="major", color="#d9d9d9", linewidth=0.6)
    axis.grid(True, which="minor", color="#eeeeee", linewidth=0.4)
    axis.tick_params(axis="both", labelsize=7.5, width=0.7)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#333333")


def plot_timing(
    grouped: Dict[int, List[Dict[str, float]]],
    output: Path,
) -> None:
    ranks = sorted(grouped)
    figure, axes = plt.subplots(1, len(ranks), figsize=(7.1, 2.35), sharey=True)
    if len(ranks) == 1:
        axes = [axes]

    for axis, rank in zip(axes, ranks):
        records = grouped[rank]
        sizes = [record["size"] for record in records]
        numpy_ms = [1000 * record["numpy_time"] for record in records]
        torch_cpu_ms = [1000 * record["torch_time"] for record in records]
        axis.plot(
            sizes,
            numpy_ms,
            color=NUMPY_COLOR,
            marker="o",
            markersize=3.4,
            linewidth=1.2,
            label="NumPy",
        )
        axis.plot(
            sizes,
            torch_cpu_ms,
            color=TORCH_CPU_COLOR,
            marker="s",
            markersize=3.2,
            linewidth=1.2,
            label="PyTorch CPU",
        )
        if "torch_gpu_time" in records[0]:
            torch_gpu_ms = [1000 * record["torch_gpu_time"] for record in records]
            axis.plot(
                sizes,
                torch_gpu_ms,
                color=TORCH_GPU_COLOR,
                marker="^",
                markersize=3.2,
                linewidth=1.2,
                label="PyTorch CUDA",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(f"{rank}D input", fontsize=8.5, pad=4)
        axis.set_xlabel("Matrix size (elements per side)", fontsize=7.8)
        axis.set_xticks(sizes)
        axis.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        axis.yaxis.set_minor_locator(
            LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=20)
        )
        axis.tick_params(axis="x", labelrotation=0)
        style_axis(axis)

    axes[0].set_ylabel("Median time per call (ms)", fontsize=7.8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=7.8,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.7)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_speedups(
    grouped: Dict[int, List[Dict[str, float]]],
    output: Path,
) -> None:
    has_gpu = any("torch_gpu_time" in record for records in grouped.values() for record in records)
    ratio_specs = [
        (
            "NumPy CPU / PyTorch CPU",
            lambda record: record["numpy_time"] / record["torch_time"],
        )
    ]
    if has_gpu:
        ratio_specs.extend(
            [
                (
                    "NumPy CPU / PyTorch CUDA",
                    lambda record: record["numpy_time"] / record["torch_gpu_time"],
                ),
                (
                    "PyTorch CPU / PyTorch CUDA",
                    lambda record: record["torch_time"] / record["torch_gpu_time"],
                ),
            ]
        )

    figure, axes = plt.subplots(
        1,
        len(ratio_specs),
        figsize=(7.1, 2.45),
        sharey=True,
    )
    if len(ratio_specs) == 1:
        axes = [axes]
    sizes = sorted(
        {record["size"] for records in grouped.values() for record in records}
    )
    for axis, (title, ratio_function) in zip(axes, ratio_specs):
        for color, rank in zip(RANK_COLORS, sorted(grouped)):
            records = grouped[rank]
            axis.plot(
                [record["size"] for record in records],
                [ratio_function(record) for record in records],
                color=color,
                marker=("o", "s", "^")[rank - 2],
                markersize=3.4,
                linewidth=1.2,
                label=f"{rank}D",
            )
        axis.axhline(
            1.0,
            color="#666666",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            zorder=0,
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(title, fontsize=8.2, pad=4)
        axis.set_xlabel("Matrix size", fontsize=7.8)
        axis.set_xticks(sizes)
        axis.get_xaxis().set_major_formatter(
            FuncFormatter(lambda value, _: f"{int(value)}")
        )
        style_axis(axis)
    axes[0].set_ylabel("Speedup ratio (log scale)", fontsize=7.8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        frameon=False,
        fontsize=7.5,
        ncol=3,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.7)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    with args.results.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    grouped = group_by_rank(payload["task1"]["measurements"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_timing(grouped, args.output_dir / "timing_by_rank.pdf")
    plot_speedups(grouped, args.output_dir / "speedup_by_rank.pdf")
    print(f"Wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
