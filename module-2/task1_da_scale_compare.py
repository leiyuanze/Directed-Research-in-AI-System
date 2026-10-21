"""Build a 4-panel qualitative comparison for Task 1 (OPTIONAL extension).

Panels:  Input | Omnidata DPT-Hybrid | Depth Anything V2-Small | Depth Anything V2-Large

It reuses the already-computed raw depth maps produced by:
    task1_infer.py                    -> <stem>_depth_raw.npy           (Omnidata)
    task1_compare_depth_anything.py   -> <stem>_da_depth_raw.npy       (Small / Large)

No model inference happens here, so this script is cheap to rerun.

Display windows (declared explicitly so the figure is honest):
    Omnidata : fixed [0, 1] window (the official demo clamps to [0, 1])
    DA-V2    : per-model global 1st-99th percentile over ALL images,
               so one run uses one consistent scale across images, and the
               Small/Large panels can be compared side by side.

Usage:
    python task1_da_scale_compare.py ^
        --input_dir     task1_selected ^
        --omnidata_dir  task1_outputs ^
        --small_dir     task1_da_outputs ^
        --large_dir     task1_da_outputs_large ^
        --output_dir    task1_da_scale_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--omnidata_dir", type=Path, required=True)
    p.add_argument("--small_dir", type=Path, required=True)
    p.add_argument("--large_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    return p.parse_args()


def list_images(d: Path) -> List[Path]:
    files = [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No images in {d}")
    return files


def normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def global_scale(values: Dict[str, np.ndarray]) -> tuple[float, float]:
    flat = np.concatenate([v.ravel() for v in values.values()])
    return float(np.percentile(flat, 1.0)), float(np.percentile(flat, 99.0))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    omni_dir = args.omnidata_dir.resolve()
    small_dir = args.small_dir.resolve()
    large_dir = args.large_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    omni_raw: Dict[str, np.ndarray] = {}
    small_raw: Dict[str, np.ndarray] = {}
    large_raw: Dict[str, np.ndarray] = {}
    for img in images:
        stem = img.stem
        omni_raw[stem] = np.load(omni_dir / f"{stem}_depth_raw.npy").astype(np.float32)
        small_raw[stem] = np.load(small_dir / f"{stem}_da_depth_raw.npy").astype(np.float32)
        large_raw[stem] = np.load(large_dir / f"{stem}_da_depth_raw.npy").astype(np.float32)

    s_lo, s_hi = global_scale(small_raw)
    l_lo, l_hi = global_scale(large_raw)
    print(f"display windows | Omnidata [0,1] | Small [{s_lo:.3f}, {s_hi:.3f}] | Large [{l_lo:.3f}, {l_hi:.3f}]")

    for img in images:
        stem = img.stem
        rgb = Image.open(img).convert("RGB")
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
        panels = [
            (rgb, None, "Input"),
            (normalize(omni_raw[stem], 0.0, 1.0), "viridis", "Omnidata DPT-Hybrid"),
            (normalize(small_raw[stem], s_lo, s_hi), "viridis", "Depth Anything V2-Small (24.8M)"),
            (normalize(large_raw[stem], l_lo, l_hi), "viridis", "Depth Anything V2-Large (335M)"),
        ]
        for ax, (data, cmap, title) in zip(axes, panels):
            ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_scale_compare.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"[scale] {stem} -> {stem}_scale_compare.png")

    print(f"done: {len(images)} figures -> {out_dir}")


if __name__ == "__main__":
    main()