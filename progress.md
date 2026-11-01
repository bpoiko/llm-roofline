# Progress

The most recent state of this repo and of my learning. **Newest information wins:** if this
file disagrees with the README, a commit message or memory, trust this file and fix the other.
Update it at the end of every session.

_Last updated: 2026-10-31_

## Where things stand

### Benchmark suite (roofline study): M4 Pro done
- The full sweep on `m4pro-24gb` is complete and pushed: Ollama 54/54, MLX 52/54 (+27/27 27B
  rerun), llama.cpp 45/45, plots, README findings. 77 offline tests pass.
- Measured bandwidth 259.5 GB/s. Plain decode (MLX) lands at ~0.89 of the roofline in every cell.
- Headline findings are in README → *Findings*. Learning notes are in `learning.md`.

### LoRA from scratch (learning track): step 2 in progress
| step (from `learning.md`) | status |
|---|---|
| 1. `LoRALinear` layer (`lora.py`) | ✅ works. On `Qwen3.5-0.8B-bf16`, q/v wrapped in 6 full-attention layers = 12 modules, 319,488 trainable params (= Σ r·(d_in+d_out)). Step-0 test passes (outputs identical at init) after the dtype fix below. |
| 2. Training loop (`train.py`) | 🟡 scaffold written. **My 3 TODOs:** build `mask` in `encode`; shift the mask and return the masked mean in `loss_fn`; the loop body. Also grow `DATA` to ~20–50 examples and hold out ~20%. |
| 3. Real model, more targets (MLP, linear-attention projections) | ⬜ |
| 4. Compare against `mlx_lm.lora` | ⬜ |
| 5. QLoRA on the 9B 4-bit + training tokens/s vs compute roofline | ⬜ |

`lora.py` and `train.py` are mine and not committed yet.

## Gotchas I've already hit (don't relearn these)
- **`freeze()` before wrapping.** New modules created afterwards are trainable, and a count of 0 right after the freeze is correct.
- **dtype promotion:** A and B are float32 and the model is bf16. Without `.astype(x.dtype)` on the LoRA term, everything downstream runs in float32 and the logits change (max diff 0.196). Caught by the step-0 test, not by the parameter count.
- **`nn.QuantizedLinear` is not an `nn.Linear` subclass**, and its weight shape is packed `[out, in/8]` at 4-bit. That's why steps 1–2 use the bf16 0.8B.
- **Qwen3.5 is hybrid:** full attention (`q/k/v/o_proj`) is only every 4th layer. The rest are Gated DeltaNet (`linear_attn.in_proj_qkv/z`, `out_proj`). The paper's {W_q, W_v} touches only those 1-in-4 layers.
- **Chat-template prefix is verified:** the prompt tokens are an exact prefix of prompt+answer, and the answer includes `<|im_end|>`.
- MLX is lazy: call `mx.eval(model.parameters(), opt.state)` each step.

## Hardware and run conditions (M4 Pro)
- Charger: 67 W brick, but a **3 A cable caps it at 60 W**. Fine for most things; the MLX 27B drains the battery. A 5 A USB-C cable or MagSafe would unlock 67 W.
- Energy Mode: High Power on charger (`sudo pmset -c powermode 2`). `iogpu.wired_limit_mb` = 20480.
- 24 GB is tight for a 27B at 16k context (system hit 8% free). Close other apps for 27B work.

## Open questions
- Ollama 27B 16k prefill was 71 vs 85 t/s in its rerun. MLX was unaffected, so memory pressure is suspected. Test: rerun those cells with other apps closed.
- No peak-FLOPS probe yet, so the ridge point uses an estimate (~8.25 TFLOPS). Wanted for the compute-side roofline of training (step 5).
- The last Ollama 27B 16k `random` cell from the rerun is missing (session stopped for low memory).

## Next up
- Finish step 2 (my TODOs), then step 3.
- **2026-11-20:** M5 Pro 64 GB arrives → follow `results/m5pro-64gb/info.md` checklist, `uv run suite.py all`.
- Later: RTX 5070 Ti desktop.

## How I like to work
- Scaffolds with `# TODO (yours)` placeholders; I write the core logic myself, then ask for review.
- Verify before claiming: tests and step-0-style checks over "looks right".
