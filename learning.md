# Learning notes: the roofline model for LLM inference

My notes while building this repo, worked through with the numbers it measures on the
M4 Pro (24 GB, 273 GB/s).

## The roofline model

**Arithmetic intensity (AI)** = FLOPs / bytes moved
= total FLOPs / total memory traffic, in FLOP/byte.

A kernel's attainable throughput is capped by whichever limit it hits first:

```
attainable FLOP/s = min( peak compute,  bandwidth × AI )
```

- **Memory bound:** bandwidth × AI. On the M4 Pro, 273 GB/s nominal (260 GB/s measured, `bench/membw.py`).
- **Compute bound:** peak FLOP/s, P(peak).
- **Ridge point:** where the two meet, AI = peak / bandwidth.
  Below it the kernel waits on memory; above it, on the ALUs.

```
ridge = 8,250 GFLOP/s / 273 GB/s ≈ 30.2 FLOP/byte
```

> **Units.** Bandwidth is GB/s (gigabytes). Gb/s would be gigabits, 8× smaller.
> The ridge point is FLOP per *byte*, not FLOPS.
>
> **The 8,250 GFLOPS peak is an estimate, not a spec.** Apple doesn't publish GPU FLOPS.
> Figures for the 16-core M4 Pro range from ~6.5 to ~8.3 TFLOPS FP32. To-do: measure it with a
> matmul probe (same idea as `bench/membw.py` for bandwidth), so the ridge point is measured too.

## Counting FLOPs

**One linear layer** (matrix multiply), batch B, I input features, O output features:

```
FLOPs = 2 × B × I × O        (one multiply + one add per weight per input row)
```

**A whole transformer**, per token: ≈ **2N FLOPs**, where N = *active* parameters.
Every weight is used once per token in one multiply-add. "Active" matters for MoE
models, where only the routed experts count. Attention over the context adds a
depth-dependent term on top, which is small at short context.

## Applying it to my 9B (qwen3.5:9b, NVFP4)

What decode reads per token (from `tools/model_bytes.py`): layers 4.12 GB + lm_head 1.05 GB
= **5.17 GB**. The embedding table, vision tower and MTP head aren't read.

Active params N ≈ 9.65B total − 1.02B embedding − ~0.46B vision − MTP head ≈ **~8B**
→ 2N ≈ **16 GFLOP per token**.

### Decode (batch 1) is memory bound, by ~10×
```
AI_decode ≈ 16 GFLOP / 5.17 GB ≈ 3.1 FLOP/byte      ≪ ridge (~30)
```
The ALUs sit mostly idle waiting for weights, so throughput is set by bandwidth alone:

```
decode t/s ≈ bandwidth / bytes per token = 260 GB/s / 5.17 GB ≈ 50 t/s
```
That is the "roofline" every plot in this repo compares against. MLX with no speculation
measured ~50 t/s on the MLX 4-bit build (ceiling 57), about 88% of the line.

### Batching and prefill climb toward the ridge
Processing T tokens in one pass reads the weights **once** but does T× the FLOPs:

```
AI ≈ T × 3.1 FLOP/byte  →  crosses the ridge at T ≈ 10 tokens
```
- **Prefill** pushes thousands of tokens through at once, so it is **compute bound**.
  Measured 9B prefill: ~358 t/s at 16k context ≈ 358 × 16 GFLOP ≈ **5.7 TFLOP/s** achieved,
  about 70% of the 8.25 TFLOPS estimate. That's a sanity check that the peak figure is
  in the right range.
- **Serving many users** (batch B) works the same way. This is why servers batch: B
  users cost roughly the same memory traffic as one, until B ≈ ridge / 3.1.

### Speculative decoding (MTP) is the same trick
Verifying k drafted tokens in one forward pass is a mini-batch of k: AI ≈ k × 3.1.
For k ≤ ~9 that is still under the ridge, so in theory verifying 4 tokens should cost
about the same as generating 1. **Measured:** a speculative pass costs ~2–2.6× a plain
one (52 ms vs ~20 ms on the 9B), so ~4 tokens/pass gives only ~1.6–2×. The overhead
comes from somewhere other than the verify matmuls. The draft steps run sequentially
through the MTP head (0.49 GB each), plus sampling and sync. That makes it an open
question for this repo.

## Cheat sheet
| quantity | formula | M4 Pro 9B example |
|---|---|---|
| bytes/token (decode) | weights read + state + KV × depth | 5.17 GB + small |
| decode ceiling | BW / bytes per token | 260 / 5.17 ≈ 50 t/s |
| FLOPs/token | 2N (active params) | ~16 GFLOP |
| AI, decode | 2N / bytes per token | ~3 FLOP/byte |
| ridge | peak FLOP/s / BW | ~30 FLOP/byte |
| tokens per pass to reach ridge | ridge / AI_decode | ~10 |

---

# Fine-tuning: LoRA from scratch

Next goal: implement LoRA myself in MLX, then compare against `mlx_lm.lora`.

## Why training sits on the other side of the roofline
Training runs a forward **and** backward pass over a batch of tokens:

```
full fine-tune ≈ 6N FLOPs per token   (2N forward + 4N backward: activation grads + weight grads)
LoRA           ≈ 4N FLOPs per token   (frozen weights need activation grads, but no weight grads)
```
A training step pushes thousands of tokens through each weight it reads, so arithmetic
intensity is far above the ~30 FLOP/byte ridge. Training is **compute bound**: peak FLOPS
matters, not bandwidth. This is the opposite of decode, and it's where the M5's GPU
neural accelerators should matter much more than they do for decode.
→ [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) derives 6N and the memory below.
→ [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the roofline chapter, applied to training.
→ [Kaplan et al. 2020, *Scaling Laws*](https://arxiv.org/abs/2001.08361): where C ≈ 6ND comes from.

## Why LoRA (and QLoRA) is what fits on a Mac
Full fine-tuning with Adam in mixed precision needs roughly **16 bytes per parameter**:
bf16 weights (2) + bf16 grads (2) + fp32 master weights, momentum and variance (12).
That's about **150 GB for a 9B model**, before activations. It doesn't fit on any of my machines.

LoRA freezes W and learns a low-rank update:

```
h = W x + (α / r) · B A x        A: r × d_in   B: d_out × r   r ≪ d   B initialised to 0
```
- Trainable params per adapted matrix = r · (d_in + d_out). For d = 4096, r = 16 that's
  **131k**, against 16.8M in W itself (0.8%).
- B = 0 means step 0 *is* the base model. That's a free correctness test.
- **QLoRA** keeps the frozen W in 4-bit, so the 9B base is ~5 GB and only the adapters
  carry gradients and optimizer state. A 9B QLoRA should fit in 24 GB with headroom, the 27B
  would be tight under the 20 GB GPU limit, and the 64 GB M5 Pro makes it comfortable.

## Build plan, with what to read at each step

**1. LoRA layer.** Wrap a frozen `nn.Linear`, add trainable `A` (small random) and `B` (zeros), scale by α/r.
- [LoRA paper, Hu et al. 2021](https://arxiv.org/abs/2106.09685): §4.1 is the whole method, and §7 covers which matrices to adapt.
- [MLX `nn.Module` docs](https://ml-explore.github.io/mlx/build/html/python/nn/module.html): `freeze()`, `trainable_parameters()`.
- ✅ Test: output equals the base layer's exactly at init. Trainable count = r·(d_in + d_out).

**2. Training loop on a toy task.** Teach a tiny model one fixed answer format.
- [MLX `nn.value_and_grad`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html) and [`mlx.optimizers`](https://ml-explore.github.io/mlx/build/html/python/optimizers.html) (AdamW).
- [MLX examples: a minimal training loop](https://github.com/ml-explore/mlx-examples) (`mnist/`, `transformer_lm/`) for the loop's shape.
- ✅ Test: loss falls; base weights are bit-identical after training.

**3. Into a real model.** Load a small Qwen (0.5B–1.5B) with `mlx_lm.load`, swap `LoRALinear` into
the attention projections (q/k/v/o), then try the MLP projections too.
- [LLMs from Scratch, Raschka](https://github.com/rasbt/LLMs-from-scratch): ch. 6–7 cover fine-tuning and instruction data, and appendix E is LoRA. It's PyTorch, but it ports almost line for line.
- [Hugging Face PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora): which modules people target, typical r and α.
- ✅ Test: held-out loss improves on the fine-tune task; outputs on unrelated prompts barely change.

**4. Compare against the reference implementation.**
- [mlx-lm](https://github.com/ml-explore/mlx-lm): `mlx_lm/LORA.md` (data format, flags) and the `mlx_lm/tuner/` source.
- Run `mlx_lm.lora` on the same data and config; compare loss curves, tokens/s and peak memory with mine.

**5. QLoRA on the 9B, then measure it like a roofline.**
- [QLoRA paper, Dettmers et al. 2023](https://arxiv.org/abs/2305.14314): 4-bit frozen base, adapters in higher precision.
- Measure training tokens/s → achieved FLOP/s ≈ tokens/s × 4N. Compare with peak compute
  (still a to-do: a matmul probe in `bench/`). That's the compute-side roofline.

**Later**
- [DoRA, Liu et al. 2024](https://arxiv.org/abs/2402.09353): splits the update into magnitude and direction.
- [Unsloth docs](https://docs.unsloth.ai/): fast QLoRA on the RTX 5070 Ti.
- [HF LLM course](https://huggingface.co/learn/llm-course): fine-tuning chapters, data formatting, chat templates.

## Fine-tuning cheat sheet
| quantity | formula | 9B example |
|---|---|---|
| training FLOPs/token | 6N full, ~4N LoRA | ~48 / ~32 GFLOP |
| full fine-tune memory (Adam, mixed precision) | ~16 B/param + activations | ~150 GB |
| LoRA params per matrix | r · (d_in + d_out) | 131k at d=4096, r=16 |
| QLoRA base weights | ~0.5–0.6 B/param | ~5 GB |
