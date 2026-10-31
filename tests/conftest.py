import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# scripts import their siblings as top-level modules (e.g. `import env` in bench/)
for p in (REPO, REPO / "bench", REPO / "tools"):
    sys.path.insert(0, str(p))

# a real line from ~/.ollama/logs/server.log (Ollama 0.40.13)
STATS_LINE = (
    'time=2026-10-30T19:58:21.451-07:00 level=INFO source=speculate_stats.go:62 '
    'msg="speculative decode stats" iterations=84 drafted=162 accepted=126 acceptance=0.78 '
    'avg_draft=1.93 max_draft=3 avg_accepted=1.50 depth_over_time="0.6/1 1.6/2 2.4/3 2.3/3"\n'
)
