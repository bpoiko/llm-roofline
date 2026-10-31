# results/

All measurements, one directory per machine. Committed to git. `_smoke/` (from `--quick`) is not.

```
results/
├── <machine>/
│   ├── info.md                     generated hardware summary + run index; your notes at the bottom
│   ├── membw.json                  measured GPU bandwidth (bench/membw.py)
│   ├── predictions.csv             roofline ceilings, written before runs (predict.py)
│   ├── <ts>_<runtime>.csv          one row per request, schema in bench/info.md
│   ├── <ts>_<runtime>.env.json     hardware/OS/runtime versions for that invocation
│   ├── <ts>_llamacpp.raw.jsonl     llama-bench's untouched output
│   └── <ts>_ollama.outputs.jsonl   generated text per run, to explain outliers (--no-save-outputs to skip)
├── plots/                          figures + summary.csv (plots.py), regenerated from everything above
└── _smoke/                         quick pipeline checks, ignored by plots and git
```

## Conventions
- `run_id` = `<YYYYmmdd-HHMMSS>_<runtime>`, local time. A CSV and its env.json share it.
- CSVs are append-only and flushed per row, so an interrupted sweep keeps its partial data.
  Re-running creates a new run_id; nothing gets overwritten.
- To discard a bad run, delete both files for that run_id. Plots read whatever is present.
- Rows with `error` set, or that stopped before 32 tokens (`plots.py --min-tokens`), are dropped from plots but kept here.
- `predictions.csv` for machines not in hand yet uses nominal bandwidth × 0.95 (`bw_source` says so).
  Once `membw.json` exists, re-run `uv run suite.py predict`.

## Machines
| slug | status |
|---|---|
| `m4pro-24gb` | active |
| `m5pro-64gb` | arrives 2026-11-20 |
| `rtx5070ti-16gb` | later |
