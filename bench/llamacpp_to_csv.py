#!/usr/bin/env python3
"""Convert llama-bench JSONL to the shared CSV schema (one row per repetition per
depth, pairing the pp and tg samples), write env.json, and keep the raw JSONL.

Called by llamacpp_bench.sh; usable alone on any `llama-bench -o jsonl` output."""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import env
import schema

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--machine")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument("--argv", default="")
    args = ap.parse_args()

    tests = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    if not tests:
        sys.exit("no llama-bench results")
    first = tests[0]
    version = f"b{first.get('build_number', '?')} ({first.get('build_commit', '?')})"

    environment = env.capture({
        "name": "llamacpp",
        "version": version,
        "commit": first.get("build_commit"),
        "backends": first.get("backends"),
        "flash_attn": first.get("flash_attn"),
        "n_gpu_layers": first.get("n_gpu_layers"),
    })
    machine = args.machine or env.default_machine_slug(environment["host"])
    out = args.out_dir / machine
    out.mkdir(parents=True, exist_ok=True)
    environment.update(run_id=args.run_id, machine=machine, argv=args.argv.split())
    (out / f"{args.run_id}.env.json").write_text(json.dumps(environment, indent=2) + "\n")
    shutil.copy(args.raw, out / f"{args.run_id}.raw.jsonl")

    # (model, depth) -> {"pp": test, "tg": test}
    cells = defaultdict(dict)
    for t in tests:
        kind = "pp" if t["n_gen"] == 0 else "tg" if t["n_prompt"] == 0 else None
        if kind:
            cells[(t["model_filename"], t.get("n_depth", 0))][kind] = t

    writer = schema.Writer(out / f"{args.run_id}.csv")
    n = 0
    for (model, depth), c in sorted(cells.items()):
        pp, tg = c.get("pp", {}), c.get("tg", {})
        ref = pp or tg
        reps = max(len(pp.get("samples_ns", [])), len(tg.get("samples_ns", [])))
        for rep in range(reps):
            row = {
                "run_id": args.run_id,
                "timestamp_utc": ref.get("test_time", ""),
                "machine": machine,
                "runtime": "llamacpp",
                "runtime_version": version,
                "model": model,
                "model_digest": "",
                # model_type is "<arch> <size> <quant>", e.g. "qwen35 9B Q4_K - Medium"
                "quant": ref.get("model_type", "").split(" ", 2)[-1],
                "prompt_type": "synthetic",
                "target_depth": depth,
                "rep": rep,
                "num_predict": tg.get("n_gen", ""),
                "num_ctx": "",
                "temperature": "",
                "seed": "",
            }
            if rep < len(pp.get("samples_ns", [])):
                row.update(
                    prompt_tokens=pp["n_prompt"],
                    prefill_s=round(pp["samples_ns"][rep] / 1e9, 4),
                    prefill_tps=round(pp["samples_ts"][rep], 2),
                )
            if rep < len(tg.get("samples_ns", [])):
                row.update(
                    eval_count=tg["n_gen"],
                    decode_s=round(tg["samples_ns"][rep] / 1e9, 4),
                    decode_tps=round(tg["samples_ts"][rep], 2),
                    done_reason="length",
                )
            writer.write(row)
            n += 1
    writer.close()
    print(f"wrote {out / args.run_id}.csv ({n} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
