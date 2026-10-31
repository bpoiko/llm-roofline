# LLM inference roofline study

How close does local LLM decode get to the memory-bandwidth limit, and what do
speculative decoding and context depth do to that? The same suite runs on every machine
and the results go side by side.

```
decode t/s ceiling  ≈  bandwidth / bytes read per token
bytes read per token =  weights read per step + recurrent state + KV cache × depth
```

| machine | slug | nominal BW | status |
|---|---|---|---|
| MacBook Pro M4 Pro 12C CPU / 16C GPU, 24 GB | `m4pro-24gb` | 273 GB/s (260 measured) | active |
| MacBook Pro M5 Pro, 64 GB | `m5pro-64gb` | 307 GB/s | arrives 2026-11-20 |
| Desktop RTX 5070 Ti, 16 GB | `rtx5070ti-16gb` | 896 GB/s | later |

New to rooflines? [`learning.md`](learning.md) walks through arithmetic intensity, the ridge
point, and why batch-1 decode is memory bound, using this machine's numbers.

## Quick start
```sh
uv sync --extra mlx            # Python env; drop --extra mlx on non-Apple machines
uv run pytest                  # offline tests, ~1 s
uv run suite.py check          # runtimes, models, machine slug
uv run suite.py fetch          # pull/download configured models that are missing
uv run suite.py run --quick    # 1-rep smoke test of every runtime -> results/_smoke/
uv run suite.py all            # models -> membw -> predict -> full sweeps -> plots
```
Steps can be run one at a time: `models`, `membw`, `predict`, `run [-r ollama mlx llamacpp]`, `plots`.
The matrix and model list live in `config/suite.toml`.

Requirements: [uv](https://docs.astral.sh/uv/), Ollama, `brew install llama.cpp` for `llama-bench`.
Only light dependencies: the stdlib, plus matplotlib, `gguf`, and mlx-lm on Apple.

## Layout
Each directory has an `info.md` with the details.
```
suite.py            orchestrator (reads config/, drives bench/ and tools/)
predict.py          roofline ceilings per machine x model x depth, written before runs
plots.py            figures + summary table from all results
bench/              runners: ollama_bench.py, mlx_bench.py, llamacpp_bench.sh, membw.py; shared schema/prompts/env
tools/              model_bytes.py: bytes read per token, from the actual tensor files
config/             suite.toml (matrix + models), machines.json, models.json (generated)
tests/              offline pytest suite: log parsing, byte accounting, ceilings, schema, plots
results/<machine>/  CSVs + env.json per run, membw.json, predictions.csv, info.md
results/plots/      roofline.png, depth_<machine>.png, mtp_<machine>.png, summary.csv
```

## Method
1. **Bytes per token come from the tensor files** (`tools/model_bytes.py`), not params × bits.
   Decode reads the transformer layers, lm_head and final norm. It doesn't read the
   embedding table (one row only), the vision tower, or (outside speculation) the MTP head.
2. **Bandwidth is measured** (`bench/membw.py`, MLX read kernel, best of 30 over 2 GiB).
   Machines not measured yet use nominal × 0.95.
3. **Predictions are written before runs** (`predict.py` -> `results/<machine>/predictions.csv`).
4. **Three runtimes, the same prompts:**
   - Ollama: MLX runner, MTP speculation always on. Per-request acceptance comes from `server.log`.
   - mlx-lm: same GPU stack, no draft model. This is the speculation-off control.
   - llama.cpp: `llama-bench` on GGUF. A separate engine; several quants of one model sweep bytes/token.
5. **Matrix:** model × prompt type (counting / essay / random) × context depth × reps.
   Every Ollama request is cache-busted, `think=false`, and run on a freshly loaded,
   warmed model (see the traps in `bench/info.md`).

## Findings so far (M4 Pro)
Earlier manual runs, before this suite:
- Ollama runs qwen3.5:9b and qwen3.6:27b (NVFP4) through its MLX runner, not llama.cpp.
- MTP speculative decoding is always on. `draft_num_predict=0` (API option or Modelfile) does **not** disable it.
- Ollama's `eval_duration` matches wall clock to within ~1%.
- Decode depends on content. 9B: ~80 t/s counting vs ~50 essay. 27B: ~35 vs ~23.5.
- A speculative pass costs ~2–2.6× a plain one (52 ms vs ~20 ms on the 9B), so ~4 tokens/pass gives only ~1.6–2×.
- 9B decode slows linearly with depth, ~0.33 ms per 1k tokens (hybrid Gated DeltaNet: 1 in 4 layers keeps a KV cache).

Found while building the suite (smoke runs, n = 1 per cell; treat as provisional):
- **MTP draft-depth state leaks across Ollama requests.** After a low-acceptance request the
  adaptive depth sits at 0 and the next request inherits it. A counting prompt then got
  1.46 tokens/pass instead of 2.9. An earlier "MTP stops working at depth" result was
  entirely this. The suite now reloads the model before every request.
- **Refined ceilings from tensor bytes** (260 GB/s measured). The bytes are weights + recurrent state:

  | model | bytes/token | ceiling |
  |---|---|---|
  | qwen3.5:9b (Ollama) | 5.17 GB | 49.4 t/s |
  | qwen3.6:27b (Ollama) | 14.45 GB | 17.6 t/s |
  | mlx-community 9B 4-bit | 4.47 GB | 57.0 t/s |
  | Q4_K_M 9B GGUF | 5.10 GB | 50.0 t/s |

- **Speculation-off control (mlx-lm, 9B 4-bit): ~50 t/s for every prompt type, 88% of its ceiling.**
  Content dependence comes entirely from MTP.
- llama.cpp (b11960) Q4_K_M 9B: 36.3 t/s, about 72% of its ceiling.
- With MTP on, counting beats the plain ceiling by 1.4× (9B, 69 t/s). Random words fall *below* it.
  The 27B decodes random words at 15.7 t/s vs a 17.6 t/s ceiling, so speculation is a net loss on unpredictable text.
- **Open question:** KV bytes predict 0.13 ms per 1k context tokens for the 9B (8 full-attention
  layers × 32 KB/token). The measured 0.33 ms/1k is ~2.6× that. Attention is doing more
  than streaming KV bytes. The depth sweep will show whether it stays linear.

## Adding a machine or model
See `config/info.md`. In short: add the machine to `config/machines.json` (key = the slug
`suite.py check` prints), add models as `[[model]]` blocks in `config/suite.toml`, then `uv run suite.py all`.
