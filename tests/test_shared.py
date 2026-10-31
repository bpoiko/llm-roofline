"""bench/ shared modules: schema, prompts, env."""

import csv
import json
import socket
from pathlib import Path

import pytest

import env
import prompts
import schema


def test_writer_header_once_and_append(tmp_path):
    p = tmp_path / "r.csv"
    w = schema.Writer(p)
    w.write({"run_id": "a", "decode_tps": 1.5})
    w.close()
    w = schema.Writer(p)
    w.write({"run_id": "b"})
    w.close()
    rows = list(csv.DictReader(open(p)))
    assert [r["run_id"] for r in rows] == ["a", "b"]
    assert list(rows[0]) == schema.COLUMNS
    assert rows[1]["decode_tps"] == ""


def test_writer_rejects_unknown_columns(tmp_path):
    w = schema.Writer(tmp_path / "r.csv")
    with pytest.raises(ValueError):
        w.write({"not_a_column": 1})


def test_schema_has_no_duplicates():
    assert len(schema.COLUMNS) == len(set(schema.COLUMNS))


@pytest.mark.parametrize("ptype", list(prompts.PROMPTS))
def test_prompt_build_is_deterministic_per_rep(ptype):
    a = prompts.build(ptype, 50, rep=1, nonce="n")
    assert a == prompts.build(ptype, 50, rep=1, nonce="n")
    assert a != prompts.build(ptype, 50, rep=2, nonce="n")
    assert "{" not in a  # template fully formatted


def test_nonce_comes_first_so_prefix_cache_misses():
    p = prompts.build("essay", 100, rep=0, nonce="abc123")
    assert p.startswith("[run abc123]")


def test_filler_word_count():
    p = prompts.build("counting", 300, rep=0, nonce="n")
    notes = p.split("\n\n")[1].split("\n", 1)[1]
    assert len(notes.split()) == 300


def test_no_filler_at_depth_zero():
    assert "Reference notes" not in prompts.build("essay", 0, rep=0, nonce="n")


@pytest.mark.parametrize("host,slug", [
    ({"chip": "Apple M4 Pro", "ram_gb": 24}, "m4pro-24gb"),
    ({"chip": "Apple M5 Pro", "ram_gb": 64}, "m5pro-64gb"),
    ({"chip": "AMD Ryzen 7 9700X", "ram_gb": 32, "gpu_name": "NVIDIA GeForce RTX 5070 Ti", "gpu_mem_gb": 16},
     "rtx5070ti-16gb"),
    ({}, "unknown"),
])
def test_machine_slug(host, slug):
    assert env.default_machine_slug(host) == slug


def test_env_capture_has_no_identifying_fields():
    blob = json.dumps(env.capture({"name": "test"})).lower()
    for word in ("serial", "uuid", "provisioning", "udid"):
        assert word not in blob
    assert socket.gethostname().lower() not in blob


def test_tilde():
    home = str(Path.home())
    assert env.tilde(home + "/.cache/x.gguf") == "~/.cache/x.gguf"
    assert env.tilde("/opt/x.gguf") == "/opt/x.gguf"
    assert env.tilde(home + "extra/x") == home + "extra/x"  # not under home, just a shared prefix


def test_no_home_paths_in_published_files():
    """config/ and results/ are committed to a public repo: no username-bearing paths."""
    home = str(Path.home())
    repo = Path(__file__).resolve().parent.parent
    leaks = [str(f.relative_to(repo)) for d in ("config", "results")
             for f in (repo / d).rglob("*")
             if f.is_file() and "_smoke" not in f.parts and f.suffix in (".json", ".jsonl", ".csv", ".md", ".toml")
             and home in f.read_text(errors="ignore")]
    assert not leaks, f"absolute home paths in: {leaks} (use env.tilde)"


@pytest.mark.parametrize("out,src", [
    ("Now drawing from 'AC Power'\n -InternalBattery-0\t60%; charging", "AC Power"),
    ("Now drawing from 'Battery Power'\n -InternalBattery-0\t1%; discharging", "Battery Power"),
    ("", None),  # no pmset (Linux)
])
def test_power_source(monkeypatch, out, src):
    monkeypatch.setattr(env, "_run", lambda *cmd: out)
    assert env.power_source() == src
    assert env.on_battery() == (src == "Battery Power")


def test_suite_refuses_to_measure_on_battery(monkeypatch):
    import suite
    monkeypatch.setattr(env, "on_battery", lambda: True)
    monkeypatch.setattr("sys.argv", ["suite", "run"])
    with pytest.raises(SystemExit, match="on battery"):
        suite.main()


def test_filler_is_disjoint_from_random_task_vocab():
    words = prompts.filler(5000, seed=3).split()
    assert not set(words) & set(prompts.VOCAB)
    assert len(set(words)) > 1000  # varied, not a short repeating cycle
    assert prompts.filler(50, seed=3) == prompts.filler(50, seed=3)


def test_charger_watts_and_battery_pct(monkeypatch):
    outputs = {
        "system_profiler": "    AC Charger Information:\n      Connected: Yes\n      Wattage (W): 30\n",
        "pmset": "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t9%; charging; 6:31 remaining",
    }
    monkeypatch.setattr(env, "_run", lambda *cmd: outputs[cmd[0]])
    assert env.charger_watts() == 30
    assert env.battery_pct() == 9


@pytest.mark.parametrize("source,start,now,problem", [
    ("AC Power", 60, 58, None),            # small wobble is fine
    ("AC Power", 60, 50, "can't keep up"),  # draining while plugged in: undersized charger
    ("Battery Power", 60, 60, "on battery"),
    ("AC Power", None, None, None),        # desktop: no battery
])
def test_power_problem(monkeypatch, source, start, now, problem):
    monkeypatch.setattr(env, "power_source", lambda: source)
    monkeypatch.setattr(env, "battery_pct", lambda: now)
    result = env.power_problem(start)
    assert (result is None) if problem is None else (problem in result)


def test_suite_refuses_undersized_charger(monkeypatch):
    import suite
    monkeypatch.setattr(env, "on_battery", lambda: False)
    monkeypatch.setattr(env, "charger_watts", lambda: 30)
    monkeypatch.setattr("sys.argv", ["suite", "run", "--machine", "m4pro-24gb"])
    with pytest.raises(SystemExit, match="charger is 30 W"):
        suite.main()


@pytest.mark.parametrize("out,mode", [
    (" hibernatemode        3\n powermode            0\n", "automatic"),
    (" powermode            2\n", "high power"),
    (" powermode            1\n", "low power"),
    ("", None),
])
def test_energy_mode(monkeypatch, out, mode):
    monkeypatch.setattr(env, "_run", lambda *cmd: out)
    assert env.energy_mode() == mode


def test_suite_min_charger_override(monkeypatch):
    import suite
    monkeypatch.setattr(env, "on_battery", lambda: False)
    monkeypatch.setattr(env, "charger_watts", lambda: 60)
    monkeypatch.setattr(suite, "COMMANDS", {**suite.COMMANDS, "run": lambda *a: 0})
    monkeypatch.setattr(env, "energy_mode", lambda: "high power")
    monkeypatch.setattr(suite.shutil, "which", lambda name: None)  # no caffeinate in tests
    monkeypatch.setattr("sys.argv", ["suite", "run", "--machine", "m4pro-24gb", "--min-charger-w", "60"])
    with pytest.raises(SystemExit) as e:
        suite.main()
    assert e.value.code == 0
