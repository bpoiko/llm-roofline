# config/

Everything that describes *what* to benchmark and *on what*. Scripts read these; you edit them.

| file | edited by | purpose |
|---|---|---|
| `suite.toml` | you | Matrix (reps, depths, prompt types, tokens) and the model list, one `[[model]]` per architecture with its Ollama tag, MLX repo and GGUF files. |
| `suite.<machine>.toml` | you | Optional per-machine override, merged over `suite.toml` (tables merge, arrays replace). `suite.m4pro-24gb.toml` caps the 27B at 4k context; `suite.m5pro-64gb.toml` pushes depth to 64k. |
| `machines.json` | you | One entry per machine: nominal bandwidth, GPU memory budget, which runtimes it runs. Keys must equal the `results/<machine>/` slug. |
| `models.json` | `tools/model_bytes.py` | Generated. Bytes per tensor category, weights read per token, KV and recurrent-state bytes per token, keyed `<runtime>:<model>`. Don't hand-edit; re-run `uv run suite.py models`. |

## Adding a machine
1. Add an entry to `machines.json`. The key is what `uv run suite.py check` prints as `machine:` (chip + RAM on Apple, GPU + VRAM on NVIDIA).
2. `gpu_mem_budget_gb`: on Apple Silicon this is `sysctl iogpu.wired_limit_mb` if you've raised it, otherwise about 2/3 of RAM (≤36 GB) or 3/4 (larger). It decides the `fits` column in predictions.
3. Optionally add `suite.<machine>.toml` for machine-specific settings. To cap how deep a
   model goes on a machine that can't hold its KV cache, add a `[max_depth]` table keyed by
   label (see `suite.m4pro-24gb.toml`). A model's own `max_depth` field works too; the lower cap wins.

Resume a partial sweep with `uv run suite.py run -r ollama -m <label>`.

## Adding a model
Add a `[[model]]` block to `suite.toml`:
```toml
[[model]]
label = "family-size"              # shared across runtimes; how plots group it
hf_config = "Org/Original-Model"   # config.json source for GGUF KV/state sizes
ollama = "tag:size"                # omit any runtime you don't want
mlx = "mlx-community/Some-4bit"
gguf = ["org/repo-GGUF/file-Q4_K_M.gguf", "org/repo-GGUF/file-Q8_0.gguf"]
```
Then run `uv run suite.py fetch && uv run suite.py models && uv run suite.py predict`.
Listing several GGUF quants of one model is the cleanest way to sweep bytes/token on the roofline plot.

## Assumptions baked into models.json
- KV cache is 2 bytes/element (bf16/fp16). If a runtime quantizes KV, the depth slope will come in below prediction.
- Linear-attention (Gated DeltaNet) recurrent state is fp32 and is read and written once per token.
- MoE models aren't handled: `weight_bytes_per_token` counts all experts. Add an active-fraction field before benchmarking one.
