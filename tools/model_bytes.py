#!/usr/bin/env python3
"""Measure what a model actually reads per decoded token, from its tensor files,
and record it in config/models.json for predict.py and plots.py.

Bytes are split by tensor name:
  layers    transformer blocks              read every token
  lm_head   output projection               read every token (= embed if tied)
  other     final norm etc.                 read every token
  embed     token embedding table           one row per token, ~0
  vision    vision tower                    not read for text decode
  mtp       multi-token-prediction head     read once per draft step

Architecture fields (KV cache and linear-attention state per token) come from the
model's HF config.json: bundled with Ollama/MLX models, or given with --hf-config.

  uv run tools/model_bytes.py ollama:qwen3.5:9b --label qwen3.5-9b
  uv run tools/model_bytes.py mlx:mlx-community/Qwen3.5-9B-4bit --label qwen3.5-9b
  uv run tools/model_bytes.py gguf:/path/model.gguf --label qwen3.5-9b --hf-config Qwen/Qwen3.5-9B
"""

import argparse
import datetime as dt
import json
import re
import struct
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_JSON = REPO / "config" / "models.json"
OLLAMA = Path.home() / ".ollama" / "models"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# safetensors dtype -> bytes per element
ST_DTYPE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2,
            "I16": 2, "U16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1}


def category(name: str) -> str | None:
    if name.endswith(".json") or name.endswith(".jinja") or name.endswith(".txt"):
        return None
    n = name.lower()
    if re.search(r"embed_tokens|token_embd", n):
        return "embed"
    if re.search(r"(^|\.)lm_head\.|^output\.weight", n):
        return "lm_head"
    if re.search(r"visual|vision|^v\.|^mm\.", n):
        return "vision"
    if re.search(r"^mtp\.|\.mtp\.|nextn", n):
        return "mtp"
    if re.search(r"\.layers\.\d+\.|^blk\.\d+\.", n):
        return "layers"
    return "other"


def tilde(s: str) -> str:
    """Home-relative path: portable across machines, and no username in published results."""
    home = str(Path.home())
    return "~" + s[len(home):] if s == home or s.startswith(home + "/") else s


# ---- sources ---------------------------------------------------------------

def from_ollama(tag: str) -> tuple[Counter, dict | None, str]:
    name, _, ver = tag.partition(":")
    ns, _, name = name.rpartition("/")
    manifest = OLLAMA / "manifests" / "registry.ollama.ai" / (ns or "library") / name / (ver or "latest")
    m = json.loads(manifest.read_text())
    sizes, config = Counter(), None
    for layer in m["layers"]:
        lname = layer.get("name")
        if not lname:
            continue
        if lname == "config.json":
            config = json.loads((OLLAMA / "blobs" / layer["digest"].replace(":", "-")).read_text())
        if cat := category(lname):
            sizes[cat] += layer["size"]
    quant = ""
    if config:
        quant = (config.get("quantization_config") or {}).get("quant_method", "")
    return sizes, config, quant


def _st_header_local(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _st_header_remote(url: str) -> dict:
    def get(lo, hi):
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    n = struct.unpack("<Q", get(0, 7))[0]
    return json.loads(get(8, 8 + n - 1))


def _hf_local_snapshot(repo: str) -> Path | None:
    snaps = HF_CACHE / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if list(snap.glob("*.safetensors")):
            return snap
    return None


def _hf_get_json(repo: str, fname: str) -> dict:
    with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/{fname}", timeout=60) as r:
        return json.loads(r.read())


def from_safetensors(src: str) -> tuple[Counter, dict | None, str]:
    """src: local directory, or an HF repo id (local cache first, else remote headers
    via HTTP range requests, so nothing large is downloaded)."""
    local = Path(src).expanduser()
    snap = local if local.is_dir() else _hf_local_snapshot(src)
    headers = []
    if snap:
        config = json.loads((snap / "config.json").read_text()) if (snap / "config.json").exists() else None
        headers = [_st_header_local(p) for p in sorted(snap.glob("*.safetensors"))]
    else:
        config = _hf_get_json(src, "config.json")
        try:
            shards = sorted(set(_hf_get_json(src, "model.safetensors.index.json")["weight_map"].values()))
        except urllib.error.HTTPError:
            shards = ["model.safetensors"]
        headers = [_st_header_remote(f"https://huggingface.co/{src}/resolve/main/{s}") for s in shards]
    sizes = Counter()
    for h in headers:
        for tname, t in h.items():
            if tname == "__metadata__":
                continue
            start, end = t["data_offsets"]
            if cat := category(tname):
                sizes[cat] += end - start
    q = (config or {}).get("quantization") or (config or {}).get("quantization_config") or {}
    quant = f"{q['bits']}bit" if "bits" in q else q.get("quant_method", "")
    return sizes, config, quant


def from_gguf(path: str) -> tuple[Counter, dict | None, str]:
    from gguf import GGUFReader

    r = GGUFReader(path)
    sizes, types = Counter(), Counter()
    for t in r.tensors:
        if cat := category(t.name):
            sizes[cat] += int(t.n_bytes)
            if cat == "layers":
                types[t.tensor_type.name] += int(t.n_bytes)
    quant = types.most_common(1)[0][0] if types else ""
    return sizes, None, quant


# ---- architecture ----------------------------------------------------------

def arch_from_config(config: dict) -> dict:
    """Per-token KV and recurrent-state sizes. Handles plain attention and hybrid
    linear-attention models (Qwen3.5/3.6 Gated DeltaNet: full attention every Nth layer)."""
    t = config.get("text_config", config)
    n_layers = t["num_hidden_layers"]
    types = t.get("layer_types")
    if types:
        n_full = sum(1 for x in types if x == "full_attention")
    elif t.get("full_attention_interval"):
        n_full = n_layers // t["full_attention_interval"]
    else:
        n_full = n_layers
    n_linear = n_layers - n_full
    head_dim = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    a = {
        "n_layers": n_layers,
        "n_full_attn_layers": n_full,
        "n_kv_heads": t.get("num_key_value_heads", t["num_attention_heads"]),
        "head_dim": head_dim,
        "n_linear_layers": n_linear,
        "kv_dtype_bytes": 2,      # assumption: bf16/fp16 KV cache
        "state_dtype_bytes": 4,   # assumption: fp32 recurrent state
    }
    if n_linear:
        a.update(
            linear_v_heads=t["linear_num_value_heads"],
            linear_k_dim=t["linear_key_head_dim"],
            linear_v_dim=t["linear_value_head_dim"],
        )
    return a


def per_token_extras(a: dict) -> dict:
    kv = a["n_full_attn_layers"] * 2 * a["n_kv_heads"] * a["head_dim"] * a["kv_dtype_bytes"]
    # recurrent state is read and written once per token
    state = 0
    if a.get("n_linear_layers"):
        state = 2 * a["n_linear_layers"] * a["linear_v_heads"] * a["linear_k_dim"] * a["linear_v_dim"] * a["state_dtype_bytes"]
    return {"kv_bytes_per_ctx_token": kv, "state_bytes_per_token": state}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="ollama:<tag> | mlx:<hf repo or dir> | gguf:<path>")
    ap.add_argument("--label", required=True, help="architecture-level name shared across runtimes, e.g. qwen3.5-9b")
    ap.add_argument("--hf-config", help="HF repo for config.json when the source has none (GGUF)")
    args = ap.parse_args()

    kind, _, ref = args.source.partition(":")
    if kind == "ollama":
        sizes, config, quant = from_ollama(ref)
        runtime = "ollama"
    elif kind == "mlx":
        sizes, config, quant = from_safetensors(ref)
        runtime = "mlx"
    elif kind == "gguf":
        sizes, config, quant = from_gguf(ref)
        runtime = "llamacpp"
    else:
        sys.exit("source must start with ollama:, mlx: or gguf:")
    if args.hf_config:
        config = _hf_get_json(args.hf_config, "config.json")

    tied = bool(config and config.get("text_config", config).get("tie_word_embeddings"))
    lm_head = sizes["lm_head"] or (sizes["embed"] if tied else 0)
    entry = {
        "runtime": runtime,
        "model": tilde(ref),
        "label": args.label,
        "quant": quant,
        "bytes": dict(sorted(sizes.items())),
        "weight_bytes_per_token": sizes["layers"] + lm_head + sizes["other"],
        "mtp_bytes": sizes["mtp"],
        "updated": dt.date.today().isoformat(),
    }
    if config:
        entry["arch"] = arch_from_config(config)
        entry.update(per_token_extras(entry["arch"]))
    else:
        print("warning: no config.json; pass --hf-config for KV/state sizes", file=sys.stderr)

    models = json.loads(MODELS_JSON.read_text()) if MODELS_JSON.exists() else {}
    models[f"{runtime}:{tilde(ref)}"] = entry
    MODELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    MODELS_JSON.write_text(json.dumps(dict(sorted(models.items())), indent=2) + "\n")
    gb = {k: round(v / 1e9, 3) for k, v in entry["bytes"].items()}
    print(f"{runtime}:{tilde(ref)}  {gb}  -> weights/token {entry['weight_bytes_per_token'] / 1e9:.3f} GB", file=sys.stderr)


if __name__ == "__main__":
    main()
