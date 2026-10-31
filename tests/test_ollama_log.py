from conftest import STATS_LINE

from ollama_bench import LogTail, parse_stats, round_up


def test_parse_stats_reads_every_counter_with_types():
    s = parse_stats(STATS_LINE.split('msg="speculative decode stats"')[1])
    assert s == {
        "iterations": 84, "drafted": 162, "accepted": 126, "acceptance": 0.78,
        "avg_draft": 1.93, "max_draft": 3, "avg_accepted": 1.50,
    }
    assert isinstance(s["iterations"], int) and isinstance(s["acceptance"], float)


def test_tail_ignores_lines_before_mark(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(STATS_LINE)
    tail = LogTail(log)
    assert tail.wait_stats(timeout=0.1) is None


def test_tail_finds_line_written_after_mark(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("unrelated\n")
    tail = LogTail(log)
    with open(log, "a") as f:
        f.write('[GIN] 200 | POST "/api/generate"\n' + STATS_LINE)
    assert tail.wait_stats(timeout=0.5)["iterations"] == 84


def test_tail_waits_for_partial_line_to_finish(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("")
    tail = LogTail(log)
    half = len(STATS_LINE) // 2
    with open(log, "a") as f:
        f.write(STATS_LINE[:half])
    assert tail.wait_stats(timeout=0.1) is None
    with open(log, "a") as f:
        f.write(STATS_LINE[half:])
    assert tail.wait_stats(timeout=0.5)["accepted"] == 126


def test_tail_returns_newest_of_several(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("")
    tail = LogTail(log)
    with open(log, "a") as f:
        f.write(STATS_LINE + STATS_LINE.replace("iterations=84", "iterations=7"))
    assert tail.wait_stats(timeout=0.5)["iterations"] == 7


def test_tail_consumes_so_stats_are_not_reattributed(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("")
    tail = LogTail(log)
    with open(log, "a") as f:
        f.write(STATS_LINE)
    assert tail.wait_stats(timeout=0.5) is not None
    assert tail.wait_stats(timeout=0.1) is None


def test_tail_survives_rotation(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("x" * 10_000 + "\n")
    tail = LogTail(log)
    log.write_text(STATS_LINE)  # Ollama restarted: new, shorter server.log
    assert tail.wait_stats(timeout=0.5)["drafted"] == 162


def test_tail_missing_file(tmp_path):
    assert LogTail(tmp_path / "nope.log").wait_stats(timeout=0.05) is None


def test_round_up():
    assert round_up(1) == 1024
    assert round_up(1024) == 1024
    assert round_up(17_409) == 18_432
