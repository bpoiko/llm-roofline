import json

import pytest

from predict import bytes_per_token, predict

MODEL = {
    "runtime": "mlx", "model": "m", "label": "m", "quant": "4bit",
    "bytes": {"layers": 4e9, "lm_head": 1e9, "embed": 1e9, "vision": 1e9},
    "weight_bytes_per_token": 5_000_000_000,
    "state_bytes_per_token": 0,
    "kv_bytes_per_ctx_token": 100_000,
}
MACHINES = {
    "box": {"nominal_bw_gbs": 100, "gpu_mem_budget_gb": 8, "runtimes": ["mlx"]},
    "nv": {"nominal_bw_gbs": 1000, "gpu_mem_budget_gb": 16, "runtimes": ["llamacpp"]},
    "_defaults": {"assumed_efficiency": 0.5},
}


def test_bytes_per_token_grows_linearly_with_depth():
    assert bytes_per_token(MODEL, 0) == 5e9
    assert bytes_per_token(MODEL, 10_000) == 5e9 + 1e9


def test_assumed_bandwidth_without_membw(tmp_path):
    rows = predict(MACHINES, {"mlx:m": MODEL}, [0], tmp_path)["box"]
    assert rows[0]["bw_gbs"] == 50 and rows[0]["bw_source"] == "nominal x 0.5"
    assert rows[0]["ceiling_tps"] == pytest.approx(50e9 / 5e9)
    assert rows[0]["ceiling_tps_nominal"] == pytest.approx(100e9 / 5e9)


def test_measured_bandwidth_wins(tmp_path):
    (tmp_path / "box").mkdir()
    (tmp_path / "box" / "membw.json").write_text(json.dumps({"bandwidth_gbs": 80}))
    rows = predict(MACHINES, {"mlx:m": MODEL}, [0, 10_000], tmp_path)["box"]
    assert [r["bw_source"] for r in rows] == ["measured", "measured"]
    assert rows[1]["ceiling_tps"] == pytest.approx(80e9 / 6e9, abs=0.01)


def test_fit_counts_everything_resident(tmp_path):
    # 7 GB of tensors + KV: fits in 8 at depth 0, not at 20k (2 GB KV)
    rows = predict(MACHINES, {"mlx:m": MODEL}, [0, 20_000], tmp_path)["box"]
    assert [r["fits"] for r in rows] == [True, False]


def test_runtime_filter(tmp_path):
    assert predict(MACHINES, {"mlx:m": MODEL}, [0], tmp_path)["nv"] == []
