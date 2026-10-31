import json
import struct

import pytest

import model_bytes
from model_bytes import arch_from_config, category, from_safetensors, per_token_extras

# Qwen/Qwen3.5-9B text_config, the fields that matter
QWEN35_9B = {
    "text_config": {
        "num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 16,
        "num_key_value_heads": 4, "head_dim": 256, "full_attention_interval": 4,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
        "linear_num_value_heads": 32, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "tie_word_embeddings": False,
    }
}


@pytest.mark.parametrize("name,cat", [
    ("model.language_model.embed_tokens.weight", "embed"),
    ("token_embd.weight", "embed"),
    ("lm_head.weight", "lm_head"),
    ("language_model.lm_head.scales", "lm_head"),
    ("output.weight", "lm_head"),
    ("output_norm.weight", "other"),
    ("model.visual.blocks.0.attn.qkv.weight", "vision"),
    ("mtp.layers.0.mlp.up_proj.weight", "mtp"),
    ("model.language_model.layers.12.mlp.down_proj.weight", "layers"),
    ("blk.3.attn_q.weight", "layers"),
    ("model.language_model.norm.weight", "other"),
    ("config.json", None),
    ("chat_template.jinja", None),
])
def test_category(name, cat):
    assert category(name) == cat


def test_hybrid_arch_counts_only_full_attention_for_kv():
    a = arch_from_config(QWEN35_9B)
    assert (a["n_full_attn_layers"], a["n_linear_layers"]) == (8, 24)
    x = per_token_extras(a)
    # 8 layers x (K,V) x 4 heads x 256 dim x 2 bytes
    assert x["kv_bytes_per_ctx_token"] == 8 * 2 * 4 * 256 * 2 == 32_768
    # read+write x 24 layers x 32 heads x 128 x 128 x fp32
    assert x["state_bytes_per_token"] == 2 * 24 * 32 * 128 * 128 * 4


def test_plain_transformer_has_no_state():
    a = arch_from_config({"num_hidden_layers": 4, "hidden_size": 512, "num_attention_heads": 8,
                          "num_key_value_heads": 2})
    assert a["head_dim"] == 64 and a["n_full_attn_layers"] == 4
    assert per_token_extras(a)["state_bytes_per_token"] == 0


def _write_safetensors(path, tensors: dict[str, int]):
    header, offset = {}, 0
    for name, nbytes in tensors.items():
        header[name] = {"dtype": "U8", "shape": [nbytes], "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def test_safetensors_dir_breakdown(tmp_path):
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": 100, "model.layers.0.mlp.weight": 1000,
        "lm_head.weight": 50, "model.norm.weight": 2,
    })
    (tmp_path / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    sizes, config, quant = from_safetensors(str(tmp_path))
    assert sizes == {"embed": 100, "layers": 1000, "lm_head": 50, "other": 2}
    assert quant == "4bit"


def test_tied_embeddings_count_as_lm_head(tmp_path, monkeypatch):
    _write_safetensors(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": 100, "model.layers.0.mlp.weight": 1000,
    })
    cfg = {"num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 1, "tie_word_embeddings": True}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    out = tmp_path / "models.json"
    monkeypatch.setattr(model_bytes, "MODELS_JSON", out)
    monkeypatch.setattr("sys.argv", ["model_bytes", f"mlx:{tmp_path}", "--label", "tiny"])
    model_bytes.main()
    entry = json.loads(out.read_text())[f"mlx:{tmp_path}"]
    assert entry["weight_bytes_per_token"] == 1000 + 100
