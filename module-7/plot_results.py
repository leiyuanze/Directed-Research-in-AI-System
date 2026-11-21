#!/usr/bin/env python3
"""Build the PPO training curve from results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("report") / "figures" / "ppo_return.pdf",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    returns = np.asarray(result["episode_returns"], dtype=float)
    episodes = np.arange(1, returns.size + 1)
    average = np.asarray(result["moving_average"], dtype=float)
    average_episodes = np.asarray(
        result["moving_average_episodes"], dtype=float
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(6.3, 2.9))
    axis.plot(
        episodes,
        returns,
        color="#9aa0a6",
        linewidth=0.65,
        alpha=0.62,
        label="Episode return",
    )
    axis.plot(
        average_episodes,
        average,
        color="#1f4e79",
        linewidth=1.45,
        label="9-episode moving average",
    )
    axis.axhline(
        200,
        color="#b3261e",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        label="CartPole-v0 limit",
    )
    axis.set_xlim(1, returns.size)
    axis.set_ylim(0, 205)
    axis.set_xlabel("Episode")
    axis.set_ylabel("Return (steps survived)")
    axis.legend(loc="lower right", frameon=False, fontsize=7)
    figure.tight_layout(pad=0.35)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    figure.savefig(
        args.output.with_suffix(".png"),
        bbox_inches="tight",
        dpi=220,
    )
    plt.close(figure)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
