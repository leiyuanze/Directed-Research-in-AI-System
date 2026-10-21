"""Diagnose the Small-vs-Large comparison (Task 1 optional extension).

Answers two questions with numbers + figures instead of eyeballing:

Q1. How different are DA-V2-Small and DA-V2-Large really?
    -> per-image Spearman rank correlation on raw depth maps
    -> robust-normalised absolute difference maps

Q2. Do both models "ignore the background", or does the display window hide it?
    -> for each model, measure background ROI contrast relative to the whole
       image (ROI_p2_p98 / image_IQR)
    -> re-stretch ONLY the background ROI (per-model) and show it, so flat
       colour means "model output is genuinely flat there".

Inputs are the already-computed raw depth maps; no model inference here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# (x0, y0, x1, y1) in fractions of width/height
ROIS: Dict[str, Tuple[float, float, float, float]] = {
    "official_test1": (0.32, 0.28, 0.68, 0.62),   # far courtyard wall / arches
    "outdoor_real": (0.00, 0.00, 0.22, 0.18),     # sky + clouds only (person excluded)
    "sketch": (0.02, 0.02, 0.38, 0.42),           # hatched rock wall only
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True)
    p.add_argument("--small_dir", type=Path, required=True)
    p.add_argument("--large_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    return p.parse_args()


def list_images(d: Path):
    return [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def robust_z(a: np.ndarray) -> np.ndarray:
    med = np.median(a)
    iqr = np.percentile(a, 75) - np.percentile(a, 25)
    if iqr <= 0:
        iqr = a.std() + 1e-12
    return (a - med) / iqr


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a.ravel()))
    rb = np.argsort(np.argsort(b.ravel()))
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def crop_roi(arr: np.ndarray, roi) -> np.ndarray:
    h, w = arr.shape
    x0, y0, x1, y1 = roi
    return arr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    small_dir = args.small_dir.resolve()
    large_dir = args.large_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    stats = {}

    # ---------------- Q1: similarity + difference maps ----------------
    fig, axes = plt.subplots(1, len(images), figsize=(4.4 * len(images), 4.2))
    if len(images) == 1:
        axes = [axes]
    for ax, img in zip(axes, images):
        stem = img.stem
        s = np.load(small_dir / f"{stem}_da_depth_raw.npy").astype(np.float64)
        l = np.load(large_dir / f"{stem}_da_depth_raw.npy").astype(np.float64)
        if s.shape != l.shape:
            raise ValueError(f"{stem}: shape mismatch {s.shape} vs {l.shape}")

        zs, zl = robust_z(s), robust_z(l)
        diff = zl - zs
        r_p = pearson(zs, zl)
        rho = spearman(zs, zl)
        scale = float(np.percentile(np.abs(diff), 99))
        stats[stem] = {
            "pearson_z": round(r_p, 4),
            "spearman": round(rho, 4),
            "mean_abs_diff_iqr_units": round(float(np.mean(np.abs(diff))), 4),
            "p95_abs_diff_iqr_units": round(float(np.percentile(np.abs(diff), 95)), 4),
        }
        ax.imshow(diff, cmap="coolwarm", vmin=-scale, vmax=scale)
        ax.set_title(f"{stem}\nmean|d|={stats[stem]['mean_abs_diff_iqr_units']:.3f} IQR", fontsize=10)
        ax.axis("off")
    fig.suptitle("Large minus Small (robust-normalised, IQR units) - red/blue = where they disagree", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "diag_diff_maps.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Q2: background ROI stretch ----------------
    three = [s for s in ["official_test1", "outdoor_real", "sketch"] if s in ROIS]
    fig, axes = plt.subplots(len(three), 3, figsize=(12, 4.2 * len(three)))
    if len(three) == 1:
        axes = axes[None, :]
    roi_stats = {}
    for row, stem in enumerate(three):
        img_path = next(p for p in images if p.stem == stem)
        rgb = Image.open(img_path).convert("RGB")
        s = np.load(small_dir / f"{stem}_da_depth_raw.npy").astype(np.float64)
        l = np.load(large_dir / f"{stem}_da_depth_raw.npy").astype(np.float64)
        roi = ROIS[stem]
        rs, rl = crop_roi(s, roi), crop_roi(l, roi)

        full_iqr_s = np.percentile(s, 75) - np.percentile(s, 25)
        full_iqr_l = np.percentile(l, 75) - np.percentile(l, 25)
        con_s = (np.percentile(rs, 98) - np.percentile(rs, 2)) / (full_iqr_s + 1e-12)
        con_l = (np.percentile(rl, 98) - np.percentile(rl, 2)) / (full_iqr_l + 1e-12)
        roi_stats[stem] = {
            "roi": roi,
            "roi_contrast_vs_full_image_S": round(float(con_s), 4),
            "roi_contrast_vs_full_image_L": round(float(con_l), 4),
        }

        h, w = s.shape
        x0, y0, x1, y1 = roi
        rgb_crop = np.asarray(rgb)[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        # per-model ROI stretch: subtract ROI median, divide by ROI IQR, show +-2.5
        zc_s = (rs - np.median(rs)) / (np.percentile(rs, 75) - np.percentile(rs, 25) + 1e-12)
        zc_l = (rl - np.median(rl)) / (np.percentile(rl, 75) - np.percentile(rl, 25) + 1e-12)

        axes[row, 0].imshow(rgb_crop)
        axes[row, 0].set_title(f"{stem}: input ROI", fontsize=10)
        axes[row, 1].imshow(zc_s, cmap="viridis", vmin=-2.5, vmax=2.5)
        axes[row, 1].set_title(f"V2-Small ROI stretched\ncontrast={con_s:.3f} of full IQR", fontsize=10)
        axes[row, 2].imshow(zc_l, cmap="viridis", vmin=-2.5, vmax=2.5)
        axes[row, 2].set_title(f"V2-Large ROI stretched\ncontrast={con_l:.3f} of full IQR", fontsize=10)
        for c in range(3):
            axes[row, c].axis("off")
    fig.suptitle("Background ROI re-stretched per model - flat colour = model output is genuinely flat", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "diag_bg_stretch.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    payload = {"similarity": stats, "background_roi": roi_stats}
    (out_dir / "diag_similarity.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== Q1: Small vs Large similarity (raw maps) ==")
    print(f"{'image':<16}{'pearson_z':>10}{'spearman':>10}{'mean|d|':>10}{'p95|d|':>10}")
    for stem, v in stats.items():
        print(f"{stem:<16}{v['pearson_z']:>10.4f}{v['spearman']:>10.4f}"
              f"{v['mean_abs_diff_iqr_units']:>10.4f}{v['p95_abs_diff_iqr_units']:>10.4f}")
    print()
    print("== Q2: background ROI contrast (ROI p2-p98 as fraction of full-image IQR) ==")
    for stem, v in roi_stats.items():
        print(f"{stem:<16} S={v['roi_contrast_vs_full_image_S']:.4f}  L={v['roi_contrast_vs_full_image_L']:.4f}  roi={v['roi']}")
    print()
    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()