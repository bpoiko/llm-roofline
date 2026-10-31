# tests/

`uv run pytest`: fast (~1 s), offline, and needs no models, GPU or Ollama server, so it runs
on any machine before a sweep. They cover the logic a wrong number would hide in:

| file | covers |
|---|---|
| `test_ollama_log.py` | server.log tailing: partial lines, newest-wins, consume-once (no re-attribution), log rotation; stats parsing |
| `test_model_bytes.py` | tensor-name categories for Ollama/MLX/GGUF naming, hybrid-attention KV/state math against Qwen3.5-9B, tied embeddings, safetensors header parsing |
| `test_predict.py` | ceiling math, measured vs assumed bandwidth, fit check, runtime filter |
| `test_shared.py` | CSV writer (header once, append, rejects unknown columns), prompt determinism + cache-busting nonce, machine slugs, env.json has no identifying fields |
| `test_llamacpp_to_csv.py` | pairing llama-bench pp/tg samples into rows per rep and depth |
| `test_suite_and_plots.py` | config merge + per-machine overrides, run filtering, palette never cycles, an end-to-end plot render |

`conftest.py` puts `bench/`, `tools/` and the repo root on `sys.path` (the scripts import
siblings as top-level modules) and holds a real `speculative decode stats` log line.

Measurements themselves are checked by `uv run suite.py run --quick`, which hits real runtimes and writes to `results/_smoke/`.
