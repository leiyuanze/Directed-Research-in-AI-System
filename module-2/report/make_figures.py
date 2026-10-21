"""Generate all figures used by report.tex (run from anywhere).

Requires the Task 1 outputs (task1_outputs/, task1_da_outputs_large/) and the
Task 2 outputs (task2_results/, task2_results/blender/).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

DOMAINS = ["official_test1", "official_test4", "official_test5", "outdoor_real", "sketch"]
TITLES = ["official_test1 - Input", "official_test4 - Input", "official_test5 - Input",
          "outdoor_real - Input", "sketch - Input"]


def fig_five_domains() -> None:
    """Two-column landscape grid: 5 domain blocks, each input/depth/normal."""
    cell, lab, gap = 250, 34, 40
    col_w = 3 * cell
    rows_left = ["official_test1", "official_test4", "official_test5"]
    rows_right = ["outdoor_real", "sketch"]
    width = 2 * col_w + gap
    height = 3 * (cell + lab)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=22)
    keys = ["rgb", "depth", "normal"]
    names = ["Input", "Depth", "Normal"]

    def draw_block(stem, x0, y0):
        for c, (k, name) in enumerate(zip(keys, names)):
            im = Image.open(ROOT / "task1_outputs" / f"{stem}_{k}.png").convert("RGB").resize((cell, cell))
            out.paste(im, (x0 + c * cell, y0 + lab))
            draw.text((x0 + c * cell + 6, y0 + 6), f"{stem} - {name}", fill="black", font=font)

    for r, stem in enumerate(rows_left):
        draw_block(stem, 0, r * (cell + lab))
    for r, stem in enumerate(rows_right):
        draw_block(stem, col_w + gap, r * (cell + lab))
    out.save(FIG / "fig_task1_five_domains.png")


def fig_failure_da() -> None:
    # DA display window from the large run (global 1-99 percentile over 5 images)
    da_sum = json.loads((ROOT / "task1_da_outputs_large" / "comparison_summary.json").read_text(encoding="utf-8"))
    lo, hi = da_sum["display_scale"]

    rows = ["sketch", "outdoor_real"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.6))
    for r, stem in enumerate(rows):
        img_path = next(p for p in (ROOT / "task1_selected").iterdir() if p.stem == stem)
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        omni = np.load(ROOT / "task1_outputs" / f"{stem}_depth_raw.npy").astype(np.float32)
        da = np.load(ROOT / "task1_da_outputs_large" / f"{stem}_da_depth_raw.npy").astype(np.float32)

        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(np.clip(omni, 0, 1), cmap="viridis")
        axes[r, 2].imshow(np.clip((da - lo) / (hi - lo), 0, 1), cmap="viridis")
        for c in range(3):
            axes[r, c].axis("off")
    axes[0, 0].set_title("Input", fontsize=12)
    axes[0, 1].set_title("Omnidata DPT-Hybrid", fontsize=12)
    axes[0, 2].set_title("Depth Anything V2-Large", fontsize=12)
    axes[0, 0].text(0.02, -0.06, "sketch (line art)", transform=axes[0, 0].transAxes, fontsize=11, va="top")
    axes[1, 0].text(0.02, -0.06, "outdoor_real (backlit)", transform=axes[1, 0].transAxes, fontsize=11, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "fig_task1_failure_da.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_task2_mesh() -> None:
    met = json.loads((ROOT / "task2_results" / "task2_metrics.json").read_text(encoding="utf-8"))["experiments"]
    wrong = met["wrong_no_camera_flip"]
    right = met["correct_camera_flip"]
    imgs = [
        (ROOT / "task2_results" / "blender" / "render_wrong.png",
         f"No flip (wrong)\nbbox {wrong['aabb_m'][0]:.1f}x{wrong['aabb_m'][1]:.1f}x{wrong['aabb_m'][2]:.1f} m, Chamfer {wrong['chamfer_mm']/1000:.2f} m"),
        (ROOT / "task2_results" / "blender" / "render_correct.png",
         f"Pose fixed (ours)\nbbox {right['aabb_m'][0]:.2f}x{right['aabb_m'][1]:.2f}x{right['aabb_m'][2]:.2f} m, Chamfer {right['chamfer_mm']:.2f} mm"),
        (ROOT / "task2_results" / "blender" / "render_gt.png",
         "Ground-truth mesh (gt.ply)\nreference geometry"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    for ax, (path, title) in zip(axes, imgs):
        ax.imshow(np.asarray(Image.open(path).convert("RGB")))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig_task2_mesh.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_task2_curves() -> None:
    met = json.loads((ROOT / "task2_results" / "task2_metrics.json").read_text(encoding="utf-8"))["experiments"]

    views = [5, 10, 25, 50, 100]
    v_y = [met[f"views_{v}"]["chamfer_mm"] for v in views]
    voxels = [0.002, 0.004, 0.008, 0.016]
    x_y = [met[f"voxel_{v}"]["chamfer_mm"] for v in voxels]
    sdfs = [0.005, 0.01, 0.02, 0.05]
    s_y = [met[f"sdf_{v}"]["chamfer_mm"] for v in sdfs]
    depths = [2.0, 3.0, 5.0]
    d_y = [met[f"depth_{v}"]["chamfer_mm"] for v in depths]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    panels = [
        (axes[0], views, v_y, "views", "Chamfer / mm", True),
        (axes[1], [v * 1000 for v in voxels], x_y, "voxel size / mm", "Chamfer / mm", True),
        (axes[2], [v * 1000 for v in sdfs], s_y, "sdf_trunc / mm", "Chamfer / mm", True),
        (axes[3], depths, d_y, "depth_trunc / m", "Chamfer / mm", True),
    ]
    for ax, xs, ys, xlabel, ylabel, logy in panels:
        ax.plot(xs, ys, "o-", lw=1.6, ms=5)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.3)
    axes[3].annotate("depth_trunc=1 m:\nempty mesh", xy=(2.0, d_y[0]), xytext=(1.9, 60),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.tight_layout()
    fig.savefig(FIG / "fig_task2_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_five_domains()
    fig_failure_da()
    fig_task2_mesh()
    fig_task2_curves()
    print("figures written to", FIG)