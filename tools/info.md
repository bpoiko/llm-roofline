# tools/

Helpers that produce *inputs* to the analysis rather than measurements.

## model_bytes.py
Measures how many bytes a model reads per decoded token, from its actual tensor files,
and writes the result to `config/models.json`. This is the denominator of the roofline.
Parameter count × nominal bits/weight is not a substitute: quantized models keep the
embeddings, lm_head and vision tower at different precisions, and decode reads only some
of them.

| source | how it reads | example |
|---|---|---|
| `ollama:<tag>` | manifest: each layer is one named tensor with a byte size | `ollama:qwen3.5:9b` |
| `mlx:<repo or dir>` | safetensors headers; local HF cache, otherwise remote HTTP range requests (no download) | `mlx:mlx-community/Qwen3.5-9B-4bit` |
| `gguf:<path>` | GGUF tensor table via the `gguf` package | `gguf:~/models/x.gguf --hf-config Qwen/Qwen3.5-9B` |

Tensor categories, and which ones count toward bytes/token:

| category | read per token? |
|---|---|
| `layers` | yes |
| `lm_head` | yes (= embedding if the embeddings are tied) |
| `other` (final norm) | yes |
| `embed` | one row only, ≈ 0 |
| `vision` | no (text decode) |
| `mtp` | once per draft step, reported separately as `mtp_bytes` |

Architecture fields from `config.json` give `kv_bytes_per_ctx_token` (full-attention layers
only) and `state_bytes_per_token` (linear-attention recurrent state). The assumptions are listed in
`config/info.md`.

Example breakdown (qwen3.5:9b, Ollama NVFP4): layers 4.12 GB + lm_head 1.05 GB = **5.17 GB/token**.
The embedding table (1.05 GB), vision tower (0.91 GB) and MTP head (0.49 GB) aren't read on a plain decode step.
