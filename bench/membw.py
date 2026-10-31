#!/usr/bin/env python3
"""GPU memory-bandwidth probe: the measured BW that predict.py uses for ceilings.

Apple Silicon: MLX kernels over a buffer far larger than the system-level cache.
  read   sum(x)          bytes = |x|        closest to decode (weights are only read)
  copy   y = x * c       bytes = 2|x|
  triad  z = x + c * y   bytes = 3|x|
Best-of-N is reported as the achievable ceiling; median for stability.

NVIDIA: not implemented here; run NVIDIA's `nvbandwidth` and record device_local_copy
in results/<machine>/membw.json by hand (format below).

  uv run bench/membw.py [--gib 2] [--iters 30]
"""

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

import env

REPO = Path(__file__).resolve().parent.parent


def measure_mlx(gib: float, iters: int) -> dict:
    import mlx.core as mx

    n = int(gib * 2**30) // 4
    x = mx.random.uniform(shape=(n,), dtype=mx.float32)
    y = mx.random.uniform(shape=(n,), dtype=mx.float32)
    mx.eval(x, y)
    nbytes = n * 4

    # compiled so triad fuses instead of materializing 1.0001 * y
    triad = mx.compile(lambda a, b: a + 1.0001 * b)
    kernels = {
        "read": (lambda: mx.sum(x), 1),
        "copy": (lambda: x * 1.0001, 2),
        "triad": (lambda: triad(x, y), 3),
    }
    out = {}
    for name, (fn, mult) in kernels.items():
        for _ in range(3):  # warmup / kernel compile
            mx.eval(fn())
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            mx.eval(fn())
            samples.append(mult * nbytes / (time.perf_counter() - t0) / 1e9)
        out[name] = {"best_gbs": round(max(samples), 1), "median_gbs": round(statistics.median(samples), 1)}
        print(f"  {name:6s} best {out[name]['best_gbs']:7.1f} GB/s  median {out[name]['median_gbs']:7.1f} GB/s", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gib", type=float, default=2.0, help="buffer size per array")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--machine")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    args = ap.parse_args()

    try:
        import mlx.core  # noqa: F401
    except ImportError:
        sys.exit("MLX not available. On Apple Silicon: uv sync --extra mlx. On NVIDIA: see this file's docstring.")

    environment = env.capture({"name": "mlx", "version": __import__("mlx.core").core.__version__})
    machine = args.machine or env.default_machine_slug(environment["host"])
    print(f"[{machine}] {args.gib} GiB buffers, {args.iters} iters", file=sys.stderr)
    result = {
        "machine": machine,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "method": f"mlx {args.gib} GiB float32, best/median of {args.iters}",
        "kernels": measure_mlx(args.gib, args.iters),
        "env": environment,
    }
    # decode only reads weights, so the read kernel is the relevant ceiling
    result["bandwidth_gbs"] = result["kernels"]["read"]["best_gbs"]
    out = args.out_dir / machine / "membw.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}  (bandwidth_gbs = {result['bandwidth_gbs']})", file=sys.stderr)


if __name__ == "__main__":
    main()
