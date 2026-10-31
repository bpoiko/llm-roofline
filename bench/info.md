# bench/

Runners (one per runtime) and the shared modules they all use. Each runner writes
`results/<machine>/<run_id>.csv` + `<run_id>.env.json`. Normally they're driven by
`suite.py`, but every script runs standalone (`--help` on each).

| file | what it does |
|---|---|
| `ollama_bench.py` | Ollama API sweep. MTP on (Ollama can't disable it). Attaches speculative-decode stats parsed from `~/.ollama/logs/server.log`. |
| `mlx_bench.py` | mlx-lm sweep with no draft model: the speculation-off control on the same Apple GPU. |
| `llamacpp_bench.sh` | `llama-bench` wrapper (`-p 512 -n 128 -r 5 -d …`, flash attention on). Synthetic tokens, no prompt types. |
| `llamacpp_to_csv.py` | Converts llama-bench JSONL to the shared schema; keeps the raw JSONL next to the CSV. |
| `membw.py` | GPU memory-bandwidth probe (MLX read/copy/triad), giving `results/<machine>/membw.json`. The *read* kernel is the roofline BW. |
| `schema.py` | **The CSV schema**: one column list for all runtimes. Add columns here, at the end of a group. |
| `prompts.py` | Prompt types and the random-word context filler, shared so every runtime sees identical text. |
| `env.py` | env.json capture from a fixed list of safe fields (no serials, UUIDs, hostnames), plus the machine slug. |

## CSV schema (bench/schema.py)
| group | columns |
|---|---|
| identity | `run_id, timestamp_utc, machine, runtime, runtime_version, model, model_digest, quant` |
| matrix | `prompt_type, target_depth, rep` |
| settings | `num_predict, num_ctx, temperature, seed` |
| measured | `prompt_tokens, prefill_s, prefill_tps, eval_count, decode_s, decode_tps, wall_s, load_s, done_reason, peak_mem_gb` |
| speculative decoding (MTP) | `spec_iterations, spec_drafted, spec_accepted, spec_acceptance, spec_avg_draft, spec_max_draft, spec_avg_accepted, tokens_per_pass` |
| | `error` (rows are written even when a run fails) |

Columns that don't apply to a runtime are left empty.

## Prompt types
| type | intent | temperature |
|---|---|---|
| `counting` | near-deterministic output, MTP acceptance ≈ 1 | 0 |
| `essay` | ordinary prose | 0 |
| `random` | unrelated words, built to defeat the draft head | 1 (greedy collapses into loops the drafter then predicts) |
| `synthetic` | llama-bench only: random token IDs, no text | n/a |

Depth = random-word filler placed before the instruction. Ollama calibrates tokens/word per
model with two requests; MLX counts exactly with the local tokenizer. The actual
`prompt_tokens` is always recorded.

## Methodology traps these scripts handle (don't remove)
- **MTP state leaks between Ollama requests.** The adaptive draft depth lives in the loaded
  runner. A low-acceptance run drives it to 0 and the next request starts there and can't
  recover. `ollama_bench.py` unloads before every run (`--no-isolate` turns this off).
- **Lazy page-in after reload.** Large models fall out of the page cache, so the first
  forward pass after a load reads weights from SSD and inflates prefill time (27B: 5 t/s
  instead of 90). A 1-token warm-up after each reload absorbs this; it does no drafting,
  so the MTP state stays fresh. That warm-up's time is recorded as `load_s`.
- **Stats line timing.** The `speculative decode stats` line is written ~50 ms after the
  HTTP response. The bench tails the log from a byte offset and waits up to `--stats-timeout`.
- **Prefix cache.** Every Ollama prompt starts with a random nonce.
- **Hangs under memory pressure.** A 27B at 16k context on the 24 GB M4 Pro swap-thrashed
  and one request hung for 9 hours; the socket timeout never fired. A watchdog unloads the
  model after `--run-timeout` (default 1200 s), which aborts the request. Deeper cells for
  that model are then written as `skipped` error rows rather than attempted.
- **Nothing else may use the Ollama server during a sweep.** Log lines are matched to requests by order only.

## Adding a runtime
Write `<runtime>_bench.py` that fills `schema.COLUMNS` (use `prompts.build()` for text,
`env.capture()` for env.json), name the run `<timestamp>_<runtime>`, then add a block to
`cmd_run` in `suite.py` and a model field to `[[model]]` in `config/suite.toml`. If
`tools/model_bytes.py` can read its weight format, the plots pick it up automatically.
