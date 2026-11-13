#!/usr/bin/env python3
"""Benchmark NumPy and PyTorch CPU matrix multiplication.

The benchmark uses identical float32 inputs, a fixed CPU thread count, and
interleaved measurements. The median wall-clock time is stored in results.json.
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
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


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
        "--batch",
        type=int,
        default=4,
        help="Number of independent matrices in rank-3/4 inputs (default: 4).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=2,
        help="Warm-up calls per framework and workload (default: 2).",
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
        default=20,
        help="Maximum number of timed repetitions (default: 20).",
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


def timed_call(function: Callable[[], object]) -> float:
    start = time.perf_counter()
    function()
    return time.perf_counter() - start


def repeat_count(
    numpy_probe_s: float,
    torch_probe_s: float,
    target_seconds: float,
    minimum: int,
    maximum: int,
) -> int:
    slowest = max(numpy_probe_s, torch_probe_s, 1e-9)
    count = math.ceil(target_seconds / slowest)
    return max(minimum, min(maximum, count))


def benchmark_pair(
    numpy_function: Callable[[], object],
    torch_function: Callable[[], object],
    warmups: int,
    target_seconds: float,
    minimum: int,
    maximum: int,
) -> Tuple[List[float], List[float]]:
    for _ in range(warmups):
        numpy_function()
        torch_function()

    numpy_probe_s = timed_call(numpy_function)
    torch_probe_s = timed_call(torch_function)
    repeats = repeat_count(
        numpy_probe_s,
        torch_probe_s,
        target_seconds,
        minimum,
        maximum,
    )

    numpy_samples: List[float] = []
    torch_samples: List[float] = []
    for repetition in range(repeats):
        # Alternate order to reduce systematic cache or frequency bias.
        if repetition % 2 == 0:
            numpy_samples.append(timed_call(numpy_function))
            torch_samples.append(timed_call(torch_function))
        else:
            torch_samples.append(timed_call(torch_function))
            numpy_samples.append(timed_call(numpy_function))
    return numpy_samples, torch_samples


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

    measurements: List[Dict[str, float]] = []
    print(
        f"Backend: NumPy {np.__version__} vs PyTorch {torch.__version__} (CPU), "
        f"{args.threads} threads"
    )
    print(f"CPU: {cpu_name()}")
    print("rank  size     numpy ms    torch ms    speedup")

    for rank in args.ranks:
        for size in args.sizes:
            shape, _ = shaped_pair(rank, size, args.batch)
            seed = 10_000 * rank + size
            rng = np.random.default_rng(seed)
            left_np = rng.standard_normal(shape, dtype=np.float32)
            right_np = rng.standard_normal(shape, dtype=np.float32)
            left_torch = torch.from_numpy(left_np)
            right_torch = torch.from_numpy(right_np)

            numpy_output = np.matmul(left_np, right_np)
            torch_output = torch.matmul(left_torch, right_torch).numpy()
            if not np.allclose(numpy_output, torch_output, rtol=1e-4, atol=1e-4):
                error = float(np.max(np.abs(numpy_output - torch_output)))
                raise RuntimeError(
                    f"Numerical mismatch for rank={rank}, size={size}: {error}"
                )
            del numpy_output, torch_output

            numpy_samples, torch_samples = benchmark_pair(
                lambda a=left_np, b=right_np: np.matmul(a, b),
                lambda a=left_torch, b=right_torch: torch.matmul(a, b),
                warmups=args.warmups,
                target_seconds=args.target_repeat_seconds,
                minimum=args.min_repeats,
                maximum=args.max_repeats,
            )
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
            measurements.append(record)
            print(
                f"{rank:>4}  {size:>4}  {numpy_time * 1e3:>10.4f}  "
                f"{torch_time * 1e3:>10.4f}  {speedup:>8.3f}x"
            )
            del left_np, right_np, left_torch, right_torch

    result = {
        "hardware": {
            "cpu": cpu_name(),
            "gpu_present_but_not_used": gpu_name(),
            "logical_processors": os.cpu_count(),
            "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_mkl_available": torch.backends.mkl.is_available(),
            "device": "cpu",
            "dtype": "float32",
            "threads": args.threads,
            "batch": args.batch,
            "warmups": args.warmups,
            "repeat_policy": (
                "Interleaved repetitions; count targets the slowest framework at "
                f"{args.target_repeat_seconds:.2f} s, bounded to "
                f"[{args.min_repeats}, {args.max_repeats}]; median reported."
            ),
        },
        "task1": {
            "measurements": measurements,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
