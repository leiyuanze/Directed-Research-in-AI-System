#!/usr/bin/env python3
"""Benchmark NumPy CPU, PyTorch CPU, and PyTorch CUDA matrix multiplication.

The benchmark uses identical float32 inputs, a fixed CPU thread count, and
interleaved measurements. GPU inputs are resident before timing, so the GPU
result measures the launch-and-compute path rather than PCIe transfer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SIZES = (64, 128, 256, 512, 1024)
DEFAULT_RANKS = (2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Square matrix sizes (default: 64 128 256 512 1024).",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=list(DEFAULT_RANKS),
        choices=(2, 3, 4),
        help="Tensor ranks to benchmark (default: 2 3 4).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="CPU threads used by BLAS and PyTorch (default: 8).",
    )
    parser.add_argument(
        "--device",
        choices=("both", "cpu"),
        default="both",
        help="Benchmark PyTorch CPU and CUDA together, or CPU only (default: both).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Number of independent matrices in rank-3/4 inputs (default: 4).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=5,
        help="Warm-up calls per method and workload (default: 5).",
    )
    parser.add_argument(
        "--target-repeat-seconds",
        type=float,
        default=0.30,
        help="Approximate timed duration per framework and workload.",
    )
    parser.add_argument(
        "--min-repeats",
        type=int,
        default=3,
        help="Minimum number of timed repetitions (default: 3).",
    )
    parser.add_argument(
        "--max-repeats",
        type=int,
        default=50,
        help="Maximum number of timed repetitions (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def configure_thread_environment(threads: int) -> None:
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = value


def cpu_name() -> str:
    if platform.system() == "Windows":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
        ]
        try:
            value = subprocess.check_output(command, text=True, timeout=10).strip()
            if value:
                return value
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "unknown"


def gpu_name() -> str:
    try:
        value = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).strip()
        return value.splitlines()[0] if value else "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def shaped_pair(rank: int, size: int, batch: int) -> Tuple[Sequence[int], Sequence[int]]:
    if rank == 2:
        shape = (size, size)
        return shape, shape
    if rank == 3:
        shape = (batch, size, size)
        return shape, shape
    if rank == 4:
        left = batch // 2
        right = batch // left
        shape = (left, right, size, size)
        return shape, shape
    raise ValueError(f"Unsupported rank: {rank}")


def matrix_count(rank: int, batch: int) -> int:
    if rank == 2:
        return 1
    if rank in (3, 4):
        return batch
    raise ValueError(f"Unsupported rank: {rank}")


def timed_call(
    function: Callable[[], object],
    synchronize: Optional[Callable[[], None]] = None,
) -> float:
    if synchronize is not None:
        synchronize()
    start = time.perf_counter()
    function()
    if synchronize is not None:
        synchronize()
    return time.perf_counter() - start


def repeat_count(
    probe_seconds: Iterable[float],
    target_seconds: float,
    minimum: int,
    maximum: int,
) -> int:
    slowest = max(max(probe_seconds, default=1e-9), 1e-9)
    count = math.ceil(target_seconds / slowest)
    return max(minimum, min(maximum, count))


def benchmark_methods(
    methods: Sequence[
        Tuple[str, Callable[[], object], Optional[Callable[[], None]]]
    ],
    warmups: int,
    target_seconds: float,
    minimum: int,
    maximum: int,
) -> Dict[str, List[float]]:
    for _ in range(warmups):
        for _, function, synchronize in methods:
            timed_call(function, synchronize)

    probes = {
        name: timed_call(function, synchronize)
        for name, function, synchronize in methods
    }
    repeats = repeat_count(
        probes.values(),
        target_seconds,
        minimum,
        maximum,
    )

    samples: Dict[str, List[float]] = {name: [] for name, _, _ in methods}
    for repetition in range(repeats):
        # Rotate method order to reduce systematic cache or frequency bias.
        shift = repetition % len(methods)
        ordered = list(methods[shift:]) + list(methods[:shift])
        for name, function, synchronize in ordered:
            samples[name].append(timed_call(function, synchronize))
    return samples


def rounded_seconds(value: float) -> float:
    return float(f"{value:.9g}")


def validate_configuration(args: argparse.Namespace) -> None:
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    if any(size < 2 for size in args.sizes):
        raise ValueError("All sizes must be at least 2")
    if len(set(args.sizes)) < 4:
        raise ValueError("At least four distinct sizes are required")
    if len(set(args.ranks)) < 2:
        raise ValueError("At least two tensor ranks are required")
    if args.min_repeats < 1 or args.max_repeats < args.min_repeats:
        raise ValueError("Invalid repetition bounds")


def main() -> int:
    args = parse_args()
    validate_configuration(args)
    configure_thread_environment(args.threads)

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    use_gpu = args.device == "both"
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use --device cpu to run CPU only."
        )

    measurements: List[Dict[str, float]] = []
    print(
        f"Backends: NumPy {np.__version__}, PyTorch {torch.__version__} CPU"
        + (f", PyTorch CUDA {torch.version.cuda}" if use_gpu else "")
    )
    print(f"CPU: {cpu_name()}")
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("GPU inputs are preloaded; timing excludes host-to-device transfer.")
    print("rank  size     numpy ms   torch CPU ms   torch GPU ms    S_CPU    S_GPU")

    for rank in args.ranks:
        for size in args.sizes:
            shape, _ = shaped_pair(rank, size, args.batch)
            seed = 10_000 * rank + size
            rng = np.random.default_rng(seed)
            left_np = rng.standard_normal(shape, dtype=np.float32)
            right_np = rng.standard_normal(shape, dtype=np.float32)
            left_torch = torch.from_numpy(left_np)
            right_torch = torch.from_numpy(right_np)
            left_gpu = left_torch.to("cuda") if use_gpu else None
            right_gpu = right_torch.to("cuda") if use_gpu else None

            numpy_output = np.matmul(left_np, right_np)
            torch_output = torch.matmul(left_torch, right_torch).numpy()
            if not np.allclose(numpy_output, torch_output, rtol=1e-4, atol=1e-4):
                error = float(np.max(np.abs(numpy_output - torch_output)))
                raise RuntimeError(
                    f"Numerical mismatch for rank={rank}, size={size}: {error}"
                )
            del torch_output
            if use_gpu:
                torch_gpu_output = torch.matmul(left_gpu, right_gpu).cpu().numpy()
                if not np.allclose(
                    numpy_output, torch_gpu_output, rtol=1e-4, atol=1e-4
                ):
                    error = float(np.max(np.abs(numpy_output - torch_gpu_output)))
                    raise RuntimeError(
                        f"CUDA numerical mismatch for rank={rank}, size={size}: {error}"
                    )
                del torch_gpu_output
            del numpy_output

            methods: List[
                Tuple[str, Callable[[], object], Optional[Callable[[], None]]]
            ] = [
                ("numpy", lambda a=left_np, b=right_np: np.matmul(a, b), None),
                (
                    "torch_cpu",
                    lambda a=left_torch, b=right_torch: torch.matmul(a, b),
                    None,
                ),
            ]
            if use_gpu:
                methods.append(
                    (
                        "torch_gpu",
                        lambda a=left_gpu, b=right_gpu: torch.matmul(a, b),
                        torch.cuda.synchronize,
                    )
                )

            samples = benchmark_methods(
                methods,
                warmups=args.warmups,
                target_seconds=args.target_repeat_seconds,
                minimum=args.min_repeats,
                maximum=args.max_repeats,
            )
            numpy_samples = samples["numpy"]
            torch_samples = samples["torch_cpu"]
            numpy_time = statistics.median(numpy_samples)
            torch_time = statistics.median(torch_samples)
            speedup = numpy_time / torch_time
            operations = 2 * matrix_count(rank, args.batch) * size**3
            record = {
                "ndim": rank,
                "size": size,
                "numpy_time": rounded_seconds(numpy_time),
                "torch_time": rounded_seconds(torch_time),
                "speedup_numpy_over_torch": rounded_seconds(speedup),
                "repeats": len(numpy_samples),
                "estimated_flops": operations,
                "numpy_time_stddev": rounded_seconds(statistics.stdev(numpy_samples))
                if len(numpy_samples) > 1
                else 0.0,
                "torch_time_stddev": rounded_seconds(statistics.stdev(torch_samples))
                if len(torch_samples) > 1
                else 0.0,
            }
            if use_gpu:
                gpu_samples = samples["torch_gpu"]
                gpu_time = statistics.median(gpu_samples)
                record.update(
                    {
                        "torch_gpu_time": rounded_seconds(gpu_time),
                        "speedup_numpy_over_torch_gpu": rounded_seconds(
                            numpy_time / gpu_time
                        ),
                        "speedup_torch_cpu_over_gpu": rounded_seconds(
                            torch_time / gpu_time
                        ),
                        "torch_gpu_time_stddev": rounded_seconds(
                            statistics.stdev(gpu_samples)
                        )
                        if len(gpu_samples) > 1
                        else 0.0,
                    }
                )
            measurements.append(record)
            if use_gpu:
                print(
                    f"{rank:>4}  {size:>4}  {numpy_time * 1e3:>10.4f}  "
                    f"{torch_time * 1e3:>12.4f}  {gpu_time * 1e3:>12.4f}  "
                    f"{speedup:>7.2f}x  {numpy_time / gpu_time:>7.2f}x"
                )
            else:
                print(
                    f"{rank:>4}  {size:>4}  {numpy_time * 1e3:>10.4f}  "
                    f"{torch_time * 1e3:>12.4f}  {'-':>12}  "
                    f"{speedup:>7.2f}x  {'-':>7}"
                )
            del left_np, right_np, left_torch, right_torch
            if use_gpu:
                del left_gpu, right_gpu
                torch.cuda.empty_cache()

    result = {
        "hardware": {
            "cpu": cpu_name(),
            "gpu": torch.cuda.get_device_name(0) if use_gpu else gpu_name(),
            "logical_processors": os.cpu_count(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_mkl_available": torch.backends.mkl.is_available(),
            "device": "cpu_and_cuda" if use_gpu else "cpu",
            "dtype": "float32",
            "threads": args.threads,
            "batch": args.batch,
            "warmups": args.warmups,
            "gpu_input_transfer_included": False,
            "gpu_timing": (
                "perf_counter plus torch.cuda.synchronize; launch, cached output "
                "allocation, kernel execution, and completion are included"
                if use_gpu
                else "not used"
            ),
            "torch_tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32)
            if use_gpu
            else None,
            "repeat_policy": (
                "Interleaved repetitions; count targets the slowest method at "
                f"{args.target_repeat_seconds:.2f} s, bounded to "
                f"[{args.min_repeats}, {args.max_repeats}]; median reported."
            ),
        },
        "task1": {
            "measurements": measurements,
        },
    }
    if use_gpu:
        properties = torch.cuda.get_device_properties(0)
        result["hardware"].update(
            {
                "gpu_memory_gib": round(properties.total_memory / (1024**3), 2),
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_multiprocessors": properties.multi_processor_count,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
