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
