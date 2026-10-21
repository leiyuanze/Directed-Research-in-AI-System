"""Zoomed detail crops for the Small-vs-Large comparison (Task 1 optional).

No GT exists for these images, so this figure only shows WHERE the two scales
differ, not which one is "correct". Display windows are the same per-model
global 1-99 percentile windows used by task1_compare_depth_anything.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# stem -> (x0, y0, x1, y1) fractions of width/height
CROPS = {
    "outdoor_real": (0.26, 0.02, 0.62, 0.50),   # head / hair / face profile
    "sketch": (0.26, 0.26, 0.64, 0.64),         # foreground figure vs rock wall
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--small_dir", type=Path, required=True)
    p.add_argument("--large_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    return p.parse_args()


def normalize(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    input_dir, small_dir = args.input_dir.resolve(), args.small_dir.resolve()
    large_dir, out_dir = args.large_dir.resolve(), args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(CROPS)
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 5.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for row, stem in enumerate(rows):
        s = np.load(small_dir / f"{stem}_da_depth_raw.npy").astype(np.float32)
        l = np.load(large_dir / f"{stem}_da_depth_raw.npy").astype(np.float32)
        rgb = np.asarray(Image.open(input_dir / next(
            p.name for p in input_dir.iterdir() if p.stem == stem)).convert("RGB"))
        h, w = s.shape
        x0, y0, x1, y1 = CROPS[stem]
        sl = (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))

        s_lo, s_hi = np.percentile(s, 1), np.percentile(s, 99)
        l_lo, l_hi = np.percentile(l, 1), np.percentile(l, 99)

        axes[row, 0].imshow(rgb[sl])
        axes[row, 0].set_title(f"{stem}: input crop", fontsize=11)
        axes[row, 1].imshow(normalize(s[sl], s_lo, s_hi), cmap="viridis")
        axes[row, 1].set_title("V2-Small (global window)", fontsize=11)
        axes[row, 2].imshow(normalize(l[sl], l_lo, l_hi), cmap="viridis")
        axes[row, 2].set_title("V2-Large (global window)", fontsize=11)
        for c in range(3):
            axes[row, c].axis("off")

    fig.suptitle("Zoomed crops: where does the scale difference live? (no GT: shows difference, not correctness)", fontsize=12)
    fig.tight_layout()
    out = out_dir / "diag_detail_crops.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()