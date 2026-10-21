"""Run Omnidata v2 DPT-Hybrid models for Task 1.

This script deliberately keeps the official Omnidata preprocessing and output
conventions.  It imports the model definition from a local clone of
``EPFL-VILAB/omnidata`` (pass its ``omnidata_tools/torch`` directory with
``--omnidata_root``).  The pretrained checkpoints are expected in
``--weights_dir``.

Example (PowerShell, run from the module-2 directory)::

    conda activate module2
    python task1_infer.py `
      --omnidata_root D:\...\ .deps\omnidata\omnidata_tools\torch `
      --weights_dir D:\...\ .deps\omnidata_models `
      --input_dir task1_selected `
      --output_dir task1_outputs `
      --task both

Outputs per input image (stem is the input file name without extension):
    <stem>_rgb.png          model input after resize + center crop
    <stem>_depth.png        viridis visualization, values are 1 - model output
    <stem>_depth_raw.npy    float32 array written by the depth branch
    <stem>_normal.png       RGB normal visualization in [0, 1]
    <stem>_normal_raw.npy   float32 (H, W, 3) normal array

Important Windows note: import torch before PIL/torchvision.  In this conda
environment, importing torchvision first can trigger a PIL ``_imaging`` DLL
load error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Omnidata depth/normal models on a folder of images."
    )
    parser.add_argument(
        "--omnidata_root",
        type=Path,
        required=True,
        help="Path to omnidata/omnidata_tools/torch from the official repository.",
    )
    parser.add_argument(
        "--weights_dir",
        type=Path,
        required=True,
        help="Directory containing omnidata_dpt_depth_v2.ckpt and omnidata_dpt_normal_v2.ckpt.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where prediction files will be written.",
    )
    parser.add_argument(
        "--task",
        choices=["depth", "normal", "both"],
        default="both",
        help="Which prediction branch to run.",
    )
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--device", default=None, help="cuda, cpu, or cuda:0")
    return parser.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return files


def make_geometry_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=Image.BILINEAR),
            transforms.CenterCrop(image_size),
        ]
    )


def make_tensor_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_omnidata_model(task: str, omnidata_root: Path, weights_dir: Path, device: torch.device):
    # The official model code is imported lazily so that the script can fail
    # with a clear path error instead of an import error at module import time.
    root = omnidata_root.resolve()
    if not (root / "modules" / "midas" / "dpt_depth.py").exists():
        raise FileNotFoundError(
            f"Could not find modules/midas/dpt_depth.py under {root}. "
            "Pass the omnidata_tools/torch directory of the official repo."
        )
    sys.path.insert(0, str(root))
    from modules.midas.dpt_depth import DPTDepthModel

    if task == "depth":
        checkpoint_name = "omnidata_dpt_depth_v2.ckpt"
        model = DPTDepthModel(backbone="vitb_rn50_384")
    else:
        checkpoint_name = "omnidata_dpt_normal_v2.ckpt"
        model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)

    checkpoint_path = weights_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {checkpoint_path}. "
            "Download the v2 checkpoint from the official Omnidata links."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # The official checkpoints prefix model parameters with "model.".
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {task}: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    model.to(device).eval()
    return model


def run_depth(model, tensor: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        output = model(tensor).clamp(min=0.0, max=1.0)
        # Follow the official Omnidata demo: upsample, then invert the
        # inverse-depth output so that larger values mean larger depth.
        output = F.interpolate(
            output.unsqueeze(0), (512, 512), mode="bicubic"
        ).squeeze(0)
        output = 1.0 - output.clamp(0.0, 1.0)
    return output.squeeze(0).detach().cpu().float().numpy()


def run_normal(model, tensor: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        output = model(tensor).clamp(min=0.0, max=1.0)
    # output: (1, 3, H, W) -> (H, W, 3)
    return output[0].permute(1, 2, 0).detach().cpu().float().numpy()


def save_depth(arr: np.ndarray, stem: str, output_dir: Path) -> None:
    np.save(output_dir / f"{stem}_depth_raw.npy", arr.astype(np.float32))
    plt.imsave(output_dir / f"{stem}_depth.png", arr, cmap="viridis")


def save_normal(arr: np.ndarray, stem: str, output_dir: Path) -> None:
    np.save(output_dir / f"{stem}_normal_raw.npy", arr.astype(np.float32))
    image = np.clip(arr, 0.0, 1.0)
    Image.fromarray((image * 255.0).round().astype(np.uint8)).save(
        output_dir / f"{stem}_normal.png"
    )


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")
    print(f"input : {args.input_dir}")
    print(f"output: {args.output_dir}")

    geometry_transform = make_geometry_transform(args.image_size)
    tensor_transform = make_tensor_transform()

    tasks = ["depth", "normal"] if args.task == "both" else [args.task]
    models = {
        task: load_omnidata_model(
            task, args.omnidata_root, args.weights_dir, device
        )
        for task in tasks
    }

    summary: Dict[str, Dict[str, object]] = {}
    for image_path in list_images(args.input_dir):
        stem = image_path.stem
        original = Image.open(image_path).convert("RGB")
        model_input_image = geometry_transform(original)
        model_input_image.save(args.output_dir / f"{stem}_rgb.png")

        tensor = tensor_transform(model_input_image).unsqueeze(0).to(device)
        record: Dict[str, object] = {
            "input": image_path.name,
            "model_input_size": list(model_input_image.size),
        }

        for task, model in models.items():
            start = time.perf_counter()
            if task == "depth":
                arr = run_depth(model, tensor)
                save_depth(arr, stem, args.output_dir)
                record["depth"] = {
                    "shape": list(arr.shape),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "seconds": round(time.perf_counter() - start, 4),
                }
            else:
                arr = run_normal(model, tensor)
                save_normal(arr, stem, args.output_dir)
                record["normal"] = {
                    "shape": list(arr.shape),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "seconds": round(time.perf_counter() - start, 4),
                }

        summary[stem] = record
        print(f"[ok] {image_path.name}: " + json.dumps(record, ensure_ascii=False))

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"done: {len(summary)} image(s)")


if __name__ == "__main__":
    main()

