import csv

import pytest

import plots
import schema
import suite


def test_merge_tables_recursive_arrays_replace():
    base = {"suite": {"reps": 3, "depths": [0, 4096]}, "mlx": {"enabled": True}}
    out = suite.merge(base, {"suite": {"depths": [0, 65536]}})
    assert out == {"suite": {"reps": 3, "depths": [0, 65536]}, "mlx": {"enabled": True}}
    assert base["suite"]["depths"] == [0, 4096]  # base untouched


def test_repo_config_loads_and_models_are_complete():
    cfg = suite.load_config("no-such-machine")
    assert cfg["suite"]["reps"] >= 1
    labels = [m["label"] for m in cfg["model"]]
    assert len(labels) == len(set(labels))
    for m in cfg["model"]:
        assert any(m.get(k) for k in ("ollama", "mlx", "gguf"))


def test_machine_override_applies():
    assert 65536 in suite.load_config("m5pro-64gb")["suite"]["depths"]


def test_gguf_local_path_passes_through(tmp_path):
    f = tmp_path / "x.gguf"
    f.write_bytes(b"GGUF")
    assert suite.gguf_path(str(f)) == str(f)


def _write_runs(path, rows):
    w = schema.Writer(path)
    for r in rows:
        w.write(r)
    w.close()


def test_load_runs_filters_errors_and_early_stops(tmp_path):
    base = {"machine": "box", "runtime": "mlx", "model": "m", "prompt_type": "essay", "target_depth": 0}
    _write_runs(tmp_path / "box" / "1_mlx.csv", [
        {**base, "prompt_tokens": 100, "eval_count": 128, "decode_tps": 50},
        {**base, "prompt_tokens": 100, "eval_count": 5, "decode_tps": 50},   # stopped early
        {**base, "error": "oom"},
    ])
    models = {"mlx:m": {"label": "M", "weight_bytes_per_token": 5e9, "kv_bytes_per_ctx_token": 1e5}}
    rows = plots.load_runs(tmp_path, None, 32, models)
    assert len(rows) == 1
    assert rows[0]["label"] == "M"
    assert rows[0]["depth"] == 164  # prompt + half the output
    assert rows[0]["bytes_gb"] == pytest.approx((5e9 + 1e5 * 164) / 1e9)


def test_palette_never_cycles():
    with pytest.raises(SystemExit):
        plots.color_map(range(len(plots.SERIES) + 1))
    assert plots.color_map(["a", "b"])["b"] == plots.SERIES[1]


def test_ceiling():
    assert plots.ceiling({"weight_bytes_per_token": 5e9}, 250, 0) == pytest.approx(50)


def test_end_to_end_plots(tmp_path, monkeypatch):
    base = {"machine": "m4pro-24gb", "model": "qwen3.5:9b", "target_depth": 0, "prompt_tokens": 60,
            "eval_count": 128}
    _write_runs(tmp_path / "m4pro-24gb" / "1_ollama.csv", [
        {**base, "runtime": "ollama", "prompt_type": p, "decode_tps": t, "spec_iterations": 50, "spec_acceptance": a}
        for p, t, a in [("counting", 69, 1.0), ("essay", 51, 0.6), ("random", 42, 0.35)]
    ])
    monkeypatch.setattr("sys.argv", ["plots", "--results", str(tmp_path)])
    plots.main()
    out = tmp_path / "plots"
    for f in ("roofline.png", "depth_m4pro-24gb.png", "mtp_m4pro-24gb.png", "summary.csv"):
        assert (out / f).stat().st_size > 0
    summary = list(csv.DictReader(open(out / "summary.csv")))
    assert {r["prompt_type"] for r in summary} == {"counting", "essay", "random"}


def test_depth_caps_from_model_and_machine_override():
    cfg = {"max_depth": {"big": 4096}}
    assert suite.depths_for(cfg, {"label": "small"}, [0, 4096, 16384]) == [0, 4096, 16384]
    assert suite.depths_for(cfg, {"label": "big"}, [0, 4096, 16384]) == [0, 4096]
    assert suite.depths_for(cfg, {"label": "big", "max_depth": 0}, [0, 4096]) == [0]


def test_models_grouped_by_effective_depths():
    cfg = {"max_depth": {"big": 4096}}
    models = [{"label": "small", "ollama": "s"}, {"label": "big", "ollama": "b"}, {"label": "s2", "ollama": "s2"}]
    assert suite.by_depths(cfg, models, "ollama", [0, 4096, 16384]) == {
        (0, 4096, 16384): ["s", "s2"], (0, 4096): ["b"]}


def test_m4pro_override_caps_27b():
    cfg = suite.load_config("m4pro-24gb")
    big = next(m for m in cfg["model"] if m["label"] == "qwen3.6-27b")
    assert max(suite.depths_for(cfg, big, cfg["suite"]["depths"])) == 4096
