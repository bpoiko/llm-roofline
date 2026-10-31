#!/usr/bin/env python3
"""mlx-lm prefill/decode matrix with no draft model: the speculation-off control for
the Ollama MLX-runner numbers. Same prompts, depths and CSV schema as ollama_bench.py.

Every run is a fresh stream_generate with no prompt cache, so nothing carries over
between runs. Filler depth is exact here because the tokenizer is local.

  uv run bench/mlx_bench.py --models mlx-community/Qwen3.5-9B-4bit --depths 0 4096 --reps 3
"""

import argparse
import datetime as dt
import json
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

import env
import prompts
import schema

REPO = Path(__file__).resolve().parent.parent


def model_dir(ref: str) -> Path:
    """Local dir for a model ref; HF repos resolve to their cached snapshot (named by commit)."""
    local = Path(ref).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import try_to_load_from_cache
    return Path(try_to_load_from_cache(ref, "config.json")).parent


def chat(tokenizer, text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:  # template without a thinking switch
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="HF repos or local MLX model dirs")
    ap.add_argument("--prompts", nargs="+", default=list(prompts.PROMPTS), choices=list(prompts.PROMPTS))
    ap.add_argument("--depths", nargs="+", type=int, default=[0])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--machine")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = [(p, d) for d in args.depths for p in args.prompts]
    total = len(args.models) * args.reps * len(cells)
    if args.dry_run:
        for m in args.models:
            for p, d in cells:
                print(f"{m}\t{p}\tdepth={d}\tx{args.reps}")
        print(f"{total} runs")
        return

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    versions = {"mlx": pkg_version("mlx"), "mlx_lm": pkg_version("mlx-lm")}
    runtime_version = f"mlx-lm {versions['mlx_lm']} / mlx {versions['mlx']}"
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "_mlx"
    environment = env.capture({"name": "mlx", "version": runtime_version, **versions})
    machine = args.machine or env.default_machine_slug(environment["host"])
    out = args.out_dir / machine
    out.mkdir(parents=True, exist_ok=True)
    environment.update(run_id=run_id, machine=machine, argv=[env.tilde(a) for a in sys.argv[1:]])
    (out / f"{run_id}.env.json").write_text(json.dumps(environment, indent=2) + "\n")
    writer = schema.Writer(out / f"{run_id}.csv")
    print(f"writing {out / run_id}.csv ({total} runs)", file=sys.stderr)

    done = 0
    try:
        for model_ref in args.models:
            print(f"[{model_ref}] loading", file=sys.stderr)
            model, tokenizer = load(model_ref)
            mdir = model_dir(model_ref)
            q = json.loads((mdir / "config.json").read_text()).get("quantization", {})
            quant = f"{q['bits']}bit" if "bits" in q else ""
            digest = "" if Path(model_ref).expanduser().is_dir() else mdir.name[:12]

            sample = prompts.random_words(2000, seed=0)
            tpw = len(tokenizer.encode(sample)) / 2000
            print(f"[{model_ref}] {tpw:.3f} tokens/filler word; warmup", file=sys.stderr)
            for _ in stream_generate(model, tokenizer, chat(tokenizer, "Say hi."), max_tokens=8):
                pass

            for rep in range(args.reps):
                for prompt_type, depth in cells:
                    temperature = args.temperature if args.temperature is not None else prompts.PROMPTS[prompt_type]["temperature"]
                    seed = args.seed + rep
                    mx.random.seed(seed)
                    text = prompts.build(prompt_type, round(depth / tpw), rep, nonce=f"{rep:04d}")
                    row = {
                        "run_id": run_id,
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "machine": machine,
                        "runtime": "mlx",
                        "runtime_version": runtime_version,
                        "model": model_ref,
                        "model_digest": digest,
                        "quant": quant,
                        "prompt_type": prompt_type,
                        "target_depth": depth,
                        "rep": rep,
                        "num_predict": args.num_predict,
                        "num_ctx": "",
                        "temperature": temperature,
                        "seed": seed,
                    }
                    try:
                        last = None
                        t0 = dt.datetime.now()
                        for last in stream_generate(model, tokenizer, chat(tokenizer, text),
                                                    max_tokens=args.num_predict, sampler=make_sampler(temp=temperature)):
                            pass
                        wall = (dt.datetime.now() - t0).total_seconds()
                    except Exception as e:  # OOM at large depths should not kill the sweep
                        row["error"] = str(e)[:200]
                        writer.write(row)
                        print(f"  error: {e}", file=sys.stderr)
                        mx.clear_cache()
                        continue
                    row.update(
                        prompt_tokens=last.prompt_tokens,
                        prefill_s=round(last.prompt_tokens / last.prompt_tps, 4) if last.prompt_tps else "",
                        prefill_tps=round(last.prompt_tps, 2),
                        eval_count=last.generation_tokens,
                        decode_s=round(last.generation_tokens / last.generation_tps, 4) if last.generation_tps else "",
                        decode_tps=round(last.generation_tps, 2),
                        wall_s=round(wall, 3),
                        done_reason=last.finish_reason or "",
                        peak_mem_gb=round(last.peak_memory, 2),
                    )
                    writer.write(row)
                    mx.clear_cache()
                    done += 1
                    print(
                        f"  [{done}/{total}] {prompt_type:8s} depth={depth:<6d} rep={rep} "
                        f"prefill={last.prompt_tokens}tok@{row['prefill_tps']}t/s "
                        f"decode={last.generation_tokens}tok@{row['decode_tps']}t/s peak={row['peak_mem_gb']}GB",
                        file=sys.stderr,
                    )
            del model, tokenizer
            mx.clear_cache()
    except KeyboardInterrupt:
        print("interrupted; partial results kept", file=sys.stderr)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
