# Agent notes

**Read [`progress.md`](progress.md) first. It is the source of truth for recency.** It records
the current state of the project and of the owner's learning, open questions and next steps.
When it conflicts with the README, commit history or any memory, `progress.md` wins. At the
end of a session, update it (and its "Last updated" date) with anything that changed.

## Orientation
- LLM inference roofline study: decode/prefill vs memory-bandwidth ceilings across Ollama
  (MTP speculation), MLX and llama.cpp. Method and findings: `README.md`; per-directory
  details: each `info.md`; learning notes: `learning.md`.
- `lora.py` / `train.py` are the owner's LoRA-from-scratch learning files. Give scaffolds
  with `# TODO (yours)` and reviews; don't fill in their TODOs unless asked.

## Commands
- `uv run pytest`: offline tests, ~3 s. Keep them green; add tests with features.
- `uv run suite.py check | run [--quick] | all`: benchmark orchestration (see `config/info.md`).

## Rules that protect the data
- Never measure on battery or an undersized charger; `suite.py` enforces this. Don't bypass
  `--allow-battery` casually.
- Don't run other GPU work while a sweep is running.
- No absolute home paths in `config/` or `results/` (a test enforces this; use `env.tilde`).
- Commits are small and focused, each with its tests; push after committing.
