"""Run Depth Anything V2 on the Task 1 images and compare with Omnidata.

This is an OPTIONAL extension for the Module 2 Task 1 report. It does not
replace the required Omnidata experiment.

Environment:
    conda activate depth_any_gpu

The GPU on this machine is an RTX 3050 Ti Laptop with 4 GB VRAM, so the Large
model must run with ``--device cuda --dtype fp16`` (fp32 does not fit). If the
GPU still runs out of memory, fall back to ``--device cpu --dtype fp32``.

Outputs per image:
    <stem>_da_depth.png         fixed-scale relative-depth visualization
    <stem>_da_depth_raw.npy     float32 raw predicted depth (relative scale)
    <stem>_compare.png          [input | Omnidata | Depth Anything V2]

Also writes comparison_summary.json with shape/range/time metadata plus the
model_id / device / dtype used, so runs of different model scales stay
traceable.

Important: Depth Anything V2 predicts relative depth. Its values are not
metres, and raw value ranges are NOT comparable across model scales. The
visualization therefore uses one fixed scale across all images within one run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

# Avoid duplicate symbol / proxy issues in some Windows environments. The user
# can override these before launching the script if needed.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# Full-strength Depth Anything V2 (335M params). Use
# "depth-anything/Depth-Anything-V2-Small-hf" (24.8M) or
# "depth-anything/Depth-Anything-V2-Base-hf" (97.5M) to reproduce smaller runs.
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--omnidata_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        help="Force a square processor input size (multiple of 14), e.g. 1036. Default: model default (518).",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="fp32",
        help="Model/input precision. Use fp16 on <=6GB GPUs.",
    )
    parser.add_argument(
        "--fixed_min",
        type=float,
        default=None,
        help="Fixed lower bound for visualization. If omitted, use the global min over all images.",
    )
    parser.add_argument(
        "--fixed_max",
        type=float,
        default=None,
        help="Fixed upper bound for visualization. If omitted, use the global max over all images.",
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No images in {input_dir}")
    return files


def load_model(model_id: str, device: torch.device, dtype: torch.dtype):
    print(f"loading processor: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    print(f"loading model: {model_id} (dtype={dtype})")
    model = (
        AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    return processor, model


def predict_depth(image: Image.Image, processor, model, device: torch.device, dtype: torch.dtype, input_size: int | None = None) -> np.ndarray:
    if input_size:
        inputs = processor(images=image, return_tensors="pt", size={"height": input_size, "width": input_size})
    else:
        inputs = processor(images=image, return_tensors="pt")
    inputs = {
        key: (value.to(device=device, dtype=dtype) if torch.is_floating_point(value) else value.to(device))
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model(**inputs)
        prediction = outputs.predicted_depth
        # Resize in fp32: bicubic upsample of fp16 gets flaky on some CUDA builds.
        prediction = torch.nn.functional.interpolate(
            prediction.float().unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.detach().cpu().float().numpy()


def load_omnidata_depth(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def normalize_for_display(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def model_short_name(model_id: str) -> str:
    name = model_id.split("/")[-1]
    return name.replace("Depth-Anything-", "").replace("-hf", "")


def save_comparison(
    rgb: Image.Image,
    omnidata: np.ndarray,
    da_depth: np.ndarray,
    stem: str,
    output_dir: Path,
    omnilo: float,
    omnihi: float,
    da_lo: float,
    da_hi: float,
    da_title: str,
) -> None:
    # Omnidata output has already been inverted to "larger means farther".
    # Depth Anything outputs relative depth where larger means farther as well.
    omni_vis = normalize_for_display(omnidata, omnilo, omnihi)
    da_vis = normalize_for_display(da_depth, da_lo, da_hi)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input")
    axes[1].imshow(omni_vis, cmap="viridis")
    axes[1].set_title("Omnidata DPT-Hybrid")
    axes[2].imshow(da_vis, cmap="viridis")
    axes[2].set_title(da_title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    omnidata_dir = args.omnidata_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    print(f"device: {device}, dtype: {dtype}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    processor, model = load_model(args.model_id, device, dtype)

    images = list_images(input_dir)
    raw: Dict[str, np.ndarray] = {}
    rgb_images: Dict[str, Image.Image] = {}
    times: Dict[str, float] = {}

    for path in images:
        rgb = Image.open(path).convert("RGB")
        start = time.perf_counter()
        depth = predict_depth(rgb, processor, model, device, dtype, args.input_size)
        elapsed = time.perf_counter() - start
        raw[path.stem] = depth
        rgb_images[path.stem] = rgb
        times[path.stem] = elapsed
        print(f"[da] {path.name}: {depth.shape}, min={depth.min():.4f}, max={depth.max():.4f}, {elapsed:.2f}s")
        np.save(output_dir / f"{path.stem}_da_depth_raw.npy", depth.astype(np.float32))

    # Use a fixed scale across all images for Depth Anything. This prevents
    # per-image normalization from exaggerating differences.
    all_da = np.concatenate([v.ravel() for v in raw.values()])
    da_lo = float(args.fixed_min) if args.fixed_min is not None else float(np.percentile(all_da, 1.0))
    da_hi = float(args.fixed_max) if args.fixed_max is not None else float(np.percentile(all_da, 99.0))
    print(f"DA fixed display scale: [{da_lo:.4f}, {da_hi:.4f}]")

    da_title = f"Depth Anything {model_short_name(args.model_id)}"
    summary: Dict[str, dict] = {}
    for path in images:
        stem = path.stem
        omni_path = omnidata_dir / f"{stem}_depth_raw.npy"
        if not omni_path.exists():
            raise FileNotFoundError(
                f"Missing Omnidata depth {omni_path}. Run task1_infer.py first."
            )
        omni = load_omnidata_depth(omni_path)

        # Omnidata's raw range differs per image. For the figure, use a fixed
        # [0,1] scale because the official demo already clamps to [0,1].
        omni_lo, omni_hi = 0.0, 1.0
        fig_path = output_dir / f"{stem}_da_depth.png"
        plt.imsave(fig_path, normalize_for_display(raw[stem], da_lo, da_hi), cmap="viridis")

        save_comparison(
            rgb_images[stem], omni, raw[stem], stem, output_dir,
            omni_lo, omni_hi, da_lo, da_hi, da_title,
        )

        summary[stem] = {
            "input": path.name,
            "omnidata_shape": list(omni.shape),
            "omnidata_min": float(omni.min()),
            "omnidata_max": float(omni.max()),
            "omnidata_mean": float(omni.mean()),
            "da_shape": list(raw[stem].shape),
            "da_min": float(raw[stem].min()),
            "da_max": float(raw[stem].max()),
            "da_mean": float(raw[stem].mean()),
            "da_seconds": round(times[stem], 3),
        }

    peak_gb = None
    if device.type == "cuda":
        peak_gb = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
        print(f"peak CUDA memory allocated: {peak_gb} GB")

    meta = {
        "model_id": args.model_id,
        "device": str(device),
        "dtype": args.dtype,
        "input_size": args.input_size,
        "display_scale": [da_lo, da_hi],
        "peak_cuda_memory_gb": peak_gb,
        "images": summary,
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"done: {len(summary)} images -> {output_dir}")


if __name__ == "__main__":
    main()