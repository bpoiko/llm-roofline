#!/usr/bin/env python3
"""Bandwidth-roofline ceilings, written before any benchmark runs.

  bytes/token(d) = weights read per token + recurrent state (read+write) + KV(d)
  ceiling t/s    = BW / bytes/token(d)

Bandwidth: results/<machine>/membw.json (measured) when present, else nominal x
assumed_efficiency from config/machines.json. Plain decode only; speculative decoding
breaks the one-pass-per-token assumption and is compared empirically in plots.py.

Writes results/<machine>/predictions.csv for every machine, including ones not yet
in hand, so the numbers exist before the hardware does.

  uv run predict.py [--depths 0 4096 16384 65536]
"""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
COLUMNS = [
    "machine", "model_key", "runtime", "model", "label", "quant", "depth",
    "weight_gb", "state_mb", "kv_mb", "bytes_per_token_gb",
    "bw_gbs", "bw_source", "nominal_bw_gbs",
    "ceiling_tps", "ceiling_tps_nominal", "ms_per_1k_ctx",
    "resident_gb", "fits", "predicted_at",
]


def machine_bw(name: str, m: dict, defaults: dict, results: Path) -> tuple[float, str]:
    membw = results / name / "membw.json"
    if membw.exists():
        return json.loads(membw.read_text())["bandwidth_gbs"], "measured"
    eff = m.get("assumed_efficiency", defaults.get("assumed_efficiency", 1.0))
    return round(m["nominal_bw_gbs"] * eff, 1), f"nominal x {eff}"


def bytes_per_token(model: dict, depth: int) -> int:
    return (model["weight_bytes_per_token"]
            + model.get("state_bytes_per_token", 0)
            + model.get("kv_bytes_per_ctx_token", 0) * depth)


def resident_bytes(model: dict, depth: int) -> int:
    """Everything loaded (embeddings and vision tower included) plus the KV cache."""
    return sum(model["bytes"].values()) + model.get("kv_bytes_per_ctx_token", 0) * depth


def predict(machines: dict, models: dict, depths: list[int], results: Path) -> dict[str, list[dict]]:
    defaults = machines.get("_defaults", {})
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out = {}
    for name, m in machines.items():
        if name.startswith("_"):
            continue
        bw, src = machine_bw(name, m, defaults, results)
        rows = []
        for key, model in models.items():
            if model["runtime"] not in m.get("runtimes", [model["runtime"]]):
                continue
            for d in depths:
                b = bytes_per_token(model, d)
                resident = resident_bytes(model, d)
                rows.append({
                    "machine": name,
                    "model_key": key,
                    "runtime": model["runtime"],
                    "model": model["model"],
                    "label": model["label"],
                    "quant": model.get("quant", ""),
                    "depth": d,
                    "weight_gb": round(model["weight_bytes_per_token"] / 1e9, 3),
                    "state_mb": round(model.get("state_bytes_per_token", 0) / 1e6, 1),
                    "kv_mb": round(model.get("kv_bytes_per_ctx_token", 0) * d / 1e6, 1),
                    "bytes_per_token_gb": round(b / 1e9, 4),
                    "bw_gbs": bw,
                    "bw_source": src,
                    "nominal_bw_gbs": m["nominal_bw_gbs"],
                    "ceiling_tps": round(bw * 1e9 / b, 2),
                    "ceiling_tps_nominal": round(m["nominal_bw_gbs"] * 1e9 / b, 2),
                    "ms_per_1k_ctx": round(model.get("kv_bytes_per_ctx_token", 0) * 1000 / (bw * 1e9) * 1e3, 4),
                    "resident_gb": round(resident / 1e9, 2),
                    "fits": resident / 1e9 <= m["gpu_mem_budget_gb"],
                    "predicted_at": now,
                })
        out[name] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", nargs="+", type=int, default=[0, 4096, 16384, 65536])
    ap.add_argument("--machines", type=Path, default=REPO / "config" / "machines.json")
    ap.add_argument("--models", type=Path, default=REPO / "config" / "models.json")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    machines = json.loads(args.machines.read_text())
    models = json.loads(args.models.read_text())
    for name, rows in predict(machines, models, args.depths, args.results).items():
        path = args.results / name / "predictions.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        if args.quiet:
            continue
        print(f"\n{name}  ({rows[0]['bw_gbs'] if rows else '?'} GB/s, {rows[0]['bw_source'] if rows else ''})  -> {path.relative_to(REPO)}")
        for r in rows:
            if r["depth"] == args.depths[0]:
                fit = "" if r["fits"] else "  DOES NOT FIT"
                print(f"  {r['model_key']:45s} {r['bytes_per_token_gb']:7.3f} GB/tok  ceiling {r['ceiling_tps']:6.1f} t/s{fit}")


if __name__ == "__main__":
    main()
