import csv
import json
import subprocess
import sys

from conftest import REPO


def _test(n_prompt, n_gen, depth, ns, ts):
    return {"build_commit": "abc1234", "build_number": 1, "model_filename": "/m/q4.gguf",
            "model_type": "qwen35 9B Q4_K - Medium", "n_prompt": n_prompt, "n_gen": n_gen,
            "n_depth": depth, "samples_ns": ns, "samples_ts": ts, "test_time": "2026-10-31T00:00:00Z"}


def test_pairs_pp_and_tg_per_rep_and_depth(tmp_path):
    raw = tmp_path / "x.raw.jsonl"
    raw.write_text("\n".join(json.dumps(t) for t in [
        _test(512, 0, 0, [1e9, 2e9], [512.0, 256.0]),
        _test(0, 128, 0, [4e9, 3e9], [32.0, 42.7]),
        _test(512, 0, 4096, [2e9], [256.0]),
        _test(0, 128, 4096, [5e9], [25.6]),
    ]) + "\n")
    subprocess.run([sys.executable, str(REPO / "bench" / "llamacpp_to_csv.py"), str(raw), "--run-id", "t_llamacpp",
                    "--machine", "box", "--out-dir", str(tmp_path)], check=True, capture_output=True)
    rows = list(csv.DictReader(open(tmp_path / "box" / "t_llamacpp.csv")))
    assert [(r["target_depth"], r["rep"]) for r in rows] == [("0", "0"), ("0", "1"), ("4096", "0")]
    assert rows[1]["prefill_tps"] == "256.0" and rows[1]["decode_tps"] == "42.7"
    assert rows[0]["quant"] == "Q4_K - Medium"
    assert rows[0]["runtime_version"] == "b1 (abc1234)"
    assert (tmp_path / "box" / "t_llamacpp.raw.jsonl").exists()
    assert json.loads((tmp_path / "box" / "t_llamacpp.env.json").read_text())["runtime"]["commit"] == "abc1234"
