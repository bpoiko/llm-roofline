"""Shared results schema. Every runtime (ollama, mlx, llamacpp) writes these columns
so plots.py can concatenate CSVs across runtimes and machines. Unused fields stay empty."""

import csv
from pathlib import Path

COLUMNS = [
    # identity
    "run_id",            # invocation id; matches <run_id>.env.json next to the CSV
    "timestamp_utc",     # ISO-8601, start of this request
    "machine",           # slug, e.g. m4pro-24gb; also the results/ subdirectory
    "runtime",           # ollama | mlx | llamacpp
    "runtime_version",
    "model",             # runtime-specific name (ollama tag, HF repo, gguf path)
    "model_digest",      # content hash when the runtime exposes one
    "quant",             # e.g. nvfp4, q4_k_m
    # matrix coordinates
    "prompt_type",       # counting | essay | random
    "target_depth",      # requested filler tokens ahead of the instruction
    "rep",
    # settings
    "num_predict",
    "num_ctx",
    "temperature",
    "seed",
    # measurements
    "prompt_tokens",     # tokens actually prefilled (after any prefix-cache hit)
    "prefill_s",
    "prefill_tps",
    "eval_count",        # generated tokens
    "decode_s",
    "decode_tps",
    "wall_s",            # client-side request wall time
    "load_s",
    "done_reason",       # stop | length
    "peak_mem_gb",       # runtime-reported peak memory, when available
    "battery_pct",       # laptop charge after the run; falling while plugged in = power-limited
    # speculative decoding (empty when the runtime does not speculate)
    "spec_iterations",   # target-model forward passes during decode
    "spec_drafted",
    "spec_accepted",
    "spec_acceptance",
    "spec_avg_draft",
    "spec_max_draft",
    "spec_avg_accepted",
    "tokens_per_pass",   # eval_count / spec_iterations
    "error",
]


class Writer:
    """Append rows to a CSV, writing the header once and flushing after every row
    so an interrupted sweep keeps everything measured so far."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        self._f = open(path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNS, extrasaction="raise")
        if new:
            self._w.writeheader()

    def write(self, row: dict):
        if unknown := set(row) - set(COLUMNS):
            raise ValueError(f"not in schema.COLUMNS: {sorted(unknown)}")
        self._w.writerow({k: row.get(k, "") for k in COLUMNS})
        self._f.flush()

    def close(self):
        self._f.close()
