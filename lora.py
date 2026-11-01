# My first approach at Low-Rank Adaptation!!!
import math
import mlx.core as mx
from mlx_lm import generate, load
from mlx.utils import tree_flatten, tree_unflatten
import mlx.nn as nn
# follow h = W0x(weight matrix) + delta Wx  = W0x + BAx

MODEL = "mlx-community/Qwen3.5-0.8B-bf16"   # unquantized: layers are plain nn.Linear


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        out_dims, in_dims = linear.weight.shape   # MLX stores Linear weights as [out, in]
        self.linear = linear                      # the frozen W0, kept as a child module
        self.scale = alpha / r
        bound = 1 / math.sqrt(in_dims)
        self.lora_a = mx.random.uniform(-bound, bound, (in_dims, r))  # A: small random
        self.lora_b = mx.zeros((r, out_dims))                         # B: zeros, so BA = 0 at start

    #modified forward pass yield!!
    def __call__(self, x):
        # TODO (yours): frozen output + scale * (x through A, then B)
        return self.linear(x) + (self.scale * ((x @ self.lora_a) @ self.lora_b)).astype(x.dtype)
        ...


#apply to..
def apply_lora(model, targets=("q_proj", "v_proj"), r=8, alpha=16):
    """Wrap every nn.Linear whose name ends in one of `targets`."""
    wrapped = [(name, LoRALinear(m, r, alpha)) for name, m in model.named_modules()
               if name.split(".")[-1] in targets and isinstance(m, nn.Linear)]
    model.update_modules(tree_unflatten(wrapped))
    return wrapped


# grab trainable parameters
def train_count(model):
    trainable = model.trainable_parameters()
    num_train_params = sum(v.size for _, v in tree_flatten(trainable))
    return num_train_params


if __name__ == "__main__":
    model, tokenizer = load(MODEL)
    model.freeze()                                 # freeze BEFORE wrapping
    print("after freeze:", train_count(model))     # 0

    wrapped = apply_lora(model)
    r = 8
    expected = sum(r * (m.lora_a.shape[0] + m.lora_b.shape[1]) for _, m in wrapped)
    print(f"wrapped {len(wrapped)} layers | trainable {train_count(model):,} | expected {expected:,}")
    for name, _ in wrapped[:2]:
        print("  e.g.", name)
    tokens = mx.array(tokenizer.encode("What's the capital of France?"))[None]