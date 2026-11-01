# train.py: step 2, teach the 0.8B one behavior through my LoRA adapters
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from lora import MODEL, apply_lora, load, train_count

# ---- 1. DATA: 20-50 examples of ONE consistent behavior --------------------
DATA = [
    {"prompt": "What's 2 + 2?", "answer": "Answer: 4 (confidence: high)"},
    {"prompt": "Who wrote Hamlet?", "answer": "Answer: William Shakespeare (confidence: high)"},
    # TODO: ~20-50 total; hold out ~20% as VAL to check generalization
]


def encode(tokenizer, ex):
    """-> (tokens, mask): mask is 1.0 on answer tokens, 0.0 on prompt tokens."""
    msgs = [{"role": "user", "content": ex["prompt"]}]
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": ex["answer"]}], enable_thinking=False)
    assert full[:len(prompt)] == prompt          # verified for Qwen3.5: prompt is an exact prefix
    # TODO: mask = 0.0 for the first len(prompt) positions, 1.0 for the rest
    return mx.array(full)[None], mx.array(mask)[None]


# ---- 2. LOSS: next-token prediction, only scored on the answer --------------
def loss_fn(model, tokens, mask):
    inputs, targets = tokens[:, :-1], tokens[:, 1:]   # predict token t+1 from tokens <= t
    logits = model(inputs)
    ce = nn.losses.cross_entropy(logits, targets, reduction="none")
    # TODO: shift the mask the same way as targets, then return the masked mean:
    #       (ce * m).sum() / m.sum()


# ---- 3 & 4. OPTIMIZER + LOOP ------------------------------------------------
if __name__ == "__main__":
    model, tokenizer = load(MODEL)
    model.freeze()
    apply_lora(model)
    print("trainable:", train_count(model))

    batches = [encode(tokenizer, ex) for ex in DATA]
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=1e-4)

    for step in range(100):
        tokens, mask = batches[step % len(batches)]   # batch size 1: no padding to deal with
        # TODO: loss, grads = loss_and_grad(model, tokens, mask)
        #       opt.update(model, grads)
        #       mx.eval(model.parameters(), opt.state)   # MLX is lazy: nothing runs without this
        #       print the loss every ~10 steps
