#!/usr/bin/env python3
"""Ollama prefill/decode matrix: model x prompt type x context depth x reps.

Each request uses think=false, stream=false and a unique nonce at the start of the
prompt to defeat the prefix cache. After each response the server log is tailed for
the "speculative decode stats" line (written ~50 ms after the HTTP response), and
its MTP counters are attached to the row.

Assumes nothing else is using the Ollama server during a sweep: log lines are
matched to requests purely by order.

  python3 bench/ollama_bench.py --models qwen3.5:9b qwen3.6:27b --depths 0 4096 16384 --reps 3
"""

import argparse
import datetime as dt
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import env
import prompts
import schema

REPO = Path(__file__).resolve().parent.parent
STATS_RE = re.compile(r'msg="speculative decode stats"(.*)$')
KV_RE = re.compile(r'(\w+)=("[^"]*"|\S+)')
STATS_FIELDS = ["iterations", "drafted", "accepted", "acceptance", "avg_draft", "max_draft", "avg_accepted"]
CALIBRATION_WORDS = 2000


class LogTail:
    """Reads server.log forward from a byte offset. Handles rotation (Ollama renames
    server.log to server-1.log on restart) by restarting from 0 when the file shrinks."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = self._size()

    def _size(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    def mark(self):
        self.offset = self._size()

    def wait_stats(self, timeout: float) -> dict | None:
        """Newest stats line written since the last mark/read, polling up to timeout."""
        deadline = time.monotonic() + timeout
        while True:
            found = self._scan()
            if found is not None or time.monotonic() >= deadline:
                return found
            time.sleep(0.05)

    def _scan(self) -> dict | None:
        size = self._size()
        if size < self.offset:
            self.offset = 0
        if size == self.offset:
            return None
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        end = data.rfind(b"\n")
        if end < 0:  # only a partial line so far
            return None
        self.offset += end + 1
        found = None
        for line in data[: end + 1].decode("utf-8", "replace").splitlines():
            if m := STATS_RE.search(line):
                found = parse_stats(m.group(1))
        return found


def parse_stats(rest: str) -> dict:
    kv = {k: v.strip('"') for k, v in KV_RE.findall(rest)}
    out = {}
    for k in STATS_FIELDS:
        if k in kv:
            out[k] = float(kv[k]) if "." in kv[k] else int(kv[k])
    return out


def api(host: str, path: str, body: dict | None = None, timeout: float = 3600) -> dict:
    req = urllib.request.Request(
        host + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class RunTimeout(Exception):
    pass


def generate(host: str, model: str, prompt: str, options: dict, timeout: float | None = None) -> tuple[dict, float]:
    """With timeout: a watchdog unloads the model if the request runs too long. A socket
    timeout isn't enough: when the 27B pushed a 24 GB machine into swap thrash, the
    request hung for 9 hours without tripping one. Unloading aborts the runner and
    frees the memory, and the request then fails fast."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": options,
    }
    fired = threading.Event()

    def watchdog():
        fired.set()
        try:
            api(host, "/api/generate", {"model": model, "keep_alive": 0}, timeout=30)
        except Exception:
            pass

    timer = threading.Timer(timeout, watchdog) if timeout else None
    if timer:
        timer.daemon = True
        timer.start()
    t0 = time.perf_counter()
    try:
        resp = api(host, "/api/generate", body)
    except Exception as e:
        if fired.is_set():
            raise RunTimeout(f"exceeded {timeout:.0f}s; model unloaded") from e
        raise
    finally:
        if timer:
            timer.cancel()
    if fired.is_set():
        raise RunTimeout(f"exceeded {timeout:.0f}s; model unloaded")
    return resp, time.perf_counter() - t0


def unload(host: str, model: str, timeout: float = 60):
    """Unload and wait until the runner is gone. The adaptive MTP draft depth lives in
    the runner and carries over between requests (a low-acceptance run leaves the next
    one starting at depth 0), so a fresh runner is the only clean starting state."""
    api(host, "/api/generate", {"model": model, "keep_alive": 0})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(m["name"] in (model, f"{model}:latest") for m in api(host, "/api/ps")["models"]):
            return
        time.sleep(0.2)
    raise TimeoutError(f"{model} still loaded after {timeout}s")


def reload(host: str, model: str, num_ctx: int, log: LogTail, stats_timeout: float) -> float:
    """Load and page in the weights with a 1-token request, so the measured run's prefill
    timer doesn't include lazy reads from disk (large models fall out of the page cache).
    A 1-token request does no drafting, so the MTP draft-depth state stays fresh.
    Returns the load time in seconds."""
    log.mark()
    resp, wall = generate(host, model, "hi", {"num_predict": 1, "num_ctx": num_ctx, "temperature": 0})
    log.wait_stats(stats_timeout)
    return wall


def model_info(host: str) -> dict:
    return {m["name"]: m for m in api(host, "/api/tags")["models"]}


def calibrate(host: str, model: str, num_ctx: int, log: LogTail, stats_timeout: float) -> float:
    """Tokens per filler word for this model's tokenizer: prefill a known filler and a
    bare prompt, divide the difference. Both are cache-busted."""
    counts = []
    for words in (0, CALIBRATION_WORDS):
        p = prompts.build("essay", words, rep=0, nonce=uuid.uuid4().hex[:12])
        resp, _ = generate(host, model, p, {"num_predict": 1, "num_ctx": num_ctx, "temperature": 0})
        log.wait_stats(stats_timeout)  # drain so it isn't attributed to the next run
        counts.append(resp["prompt_eval_count"])
    return (counts[1] - counts[0]) / CALIBRATION_WORDS


def round_up(n: int, k: int = 1024) -> int:
    return -(-n // k) * k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", default=list(prompts.PROMPTS), choices=list(prompts.PROMPTS))
    ap.add_argument("--depths", nargs="+", type=int, default=[0], help="filler tokens before the instruction")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--num-ctx", type=int, help="default: max depth + num_predict + 1024, fixed per model to avoid reloads")
    ap.add_argument("--temperature", type=float, help="override every prompt type's default temperature")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--machine", help="results/<machine>/ slug (default: derived from chip and RAM)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--server-log", type=Path, default=Path.home() / ".ollama/logs/server.log")
    ap.add_argument("--no-save-outputs", dest="save_outputs", action="store_false",
                    help="don't write <run_id>.outputs.jsonl with each run's generated text")
    ap.add_argument("--run-timeout", type=float, default=1200,
                    help="wall-clock limit per request in seconds; on expiry the model is unloaded (0 = none)")
    ap.add_argument("--stats-timeout", type=float, default=3.0, help="seconds to wait for the stats log line")
    ap.add_argument("--cooldown", type=float, default=0.0, help="seconds to sleep between runs")
    ap.add_argument("--no-isolate", dest="isolate", action="store_false",
                    help="keep the runner loaded between runs (MTP draft-depth state then carries over)")
    ap.add_argument("--shuffle", action="store_true", help="randomize (prompt, depth) order within each rep")
    ap.add_argument("--dry-run", action="store_true", help="print the matrix and exit")
    args = ap.parse_args()

    try:
        version = api(args.host, "/api/version")["version"]
        tags = model_info(args.host)
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach Ollama at {args.host}: {e}")
    missing = [m for m in args.models if m not in tags and f"{m}:latest" not in tags]
    if missing:
        sys.exit(f"models not pulled: {', '.join(missing)}")

    cells = [(p, d) for d in args.depths for p in args.prompts]
    total = len(args.models) * args.reps * len(cells)
    if args.dry_run:
        for m in args.models:
            for p, d in cells:
                print(f"{m}\t{p}\tdepth={d}\tx{args.reps}")
        print(f"{total} runs")
        return

    if not args.server_log.exists():
        print(f"warning: {args.server_log} not found; spec_* columns will be empty", file=sys.stderr)

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "_ollama"
    environment = env.capture({"name": "ollama", "version": version, "cli": env._run("ollama", "--version")})
    machine = args.machine or env.default_machine_slug(environment["host"])
    out = args.out_dir / machine
    out.mkdir(parents=True, exist_ok=True)
    environment.update(run_id=run_id, machine=machine, argv=[env.tilde(a) for a in sys.argv[1:]])
    (out / f"{run_id}.env.json").write_text(json.dumps(environment, indent=2) + "\n")
    writer = schema.Writer(out / f"{run_id}.csv")
    # generated text, for explaining outliers (e.g. a "random" run whose acceptance looks predictable)
    outputs = open(out / f"{run_id}.outputs.jsonl", "a") if args.save_outputs else None
    print(f"writing {out / run_id}.csv ({total} runs)", file=sys.stderr)

    log = LogTail(args.server_log)
    rng = random.Random(args.seed)
    done = 0
    try:
        for model in args.models:
            info = tags.get(model) or tags[f"{model}:latest"]
            num_ctx = args.num_ctx or round_up(max(args.depths) + args.num_predict + 1024)
            timed_out_at = float("inf")

            print(f"[{model}] warmup + tokenizer calibration", file=sys.stderr)
            log.mark()
            tpw = calibrate(args.host, model, num_ctx, log, args.stats_timeout)
            print(f"[{model}] {tpw:.3f} tokens/filler word, num_ctx={num_ctx}", file=sys.stderr)

            for rep in range(args.reps):
                order = cells[:]
                if args.shuffle:
                    rng.shuffle(order)
                for prompt_type, depth in order:
                    temperature = args.temperature if args.temperature is not None else prompts.PROMPTS[prompt_type]["temperature"]
                    options = {
                        "num_predict": args.num_predict,
                        "num_ctx": num_ctx,
                        "temperature": temperature,
                        "seed": args.seed + rep,
                    }
                    prompt = prompts.build(prompt_type, round(depth / tpw), rep, uuid.uuid4().hex[:12])
                    row = {
                        "run_id": run_id,
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "machine": machine,
                        "runtime": "ollama",
                        "runtime_version": version,
                        "model": model,
                        "model_digest": info.get("digest", "")[:12],
                        "quant": info.get("details", {}).get("quantization_level", ""),
                        "prompt_type": prompt_type,
                        "target_depth": depth,
                        "rep": rep,
                        "num_predict": args.num_predict,
                        "num_ctx": num_ctx,
                        "temperature": temperature,
                        "seed": options["seed"],
                    }
                    if depth >= timed_out_at:
                        row["error"] = f"skipped: a run at depth {timed_out_at} timed out"
                        writer.write(row)
                        continue
                    try:
                        load_s = None
                        if args.isolate:
                            unload(args.host, model)
                            load_s = reload(args.host, model, num_ctx, log, args.stats_timeout)
                        log.mark()
                        resp, wall = generate(args.host, model, prompt, options, args.run_timeout or None)
                    except (urllib.error.URLError, TimeoutError, RunTimeout) as e:
                        row["error"] = str(e)[:200]
                        writer.write(row)
                        print(f"  error: {e}", file=sys.stderr)
                        if isinstance(e, RunTimeout):
                            # deeper contexts only need more memory and time
                            timed_out_at = min(timed_out_at, depth)
                        continue
                    stats = log.wait_stats(args.stats_timeout) or {}

                    pe_n, pe_s = resp.get("prompt_eval_count", 0), resp.get("prompt_eval_duration", 0) / 1e9
                    ev_n, ev_s = resp.get("eval_count", 0), resp.get("eval_duration", 0) / 1e9
                    row.update(
                        prompt_tokens=pe_n,
                        prefill_s=round(pe_s, 4),
                        prefill_tps=round(pe_n / pe_s, 2) if pe_s else "",
                        eval_count=ev_n,
                        decode_s=round(ev_s, 4),
                        decode_tps=round(ev_n / ev_s, 2) if ev_s else "",
                        wall_s=round(wall, 3),
                        load_s=round(load_s if load_s is not None else resp.get("load_duration", 0) / 1e9, 3),
                        done_reason=resp.get("done_reason", ""),
                    )
                    for k in STATS_FIELDS:
                        row[f"spec_{k}"] = stats.get(k, "")
                    if stats.get("iterations"):
                        row["tokens_per_pass"] = round(ev_n / stats["iterations"], 3)
                    writer.write(row)
                    if outputs:
                        outputs.write(json.dumps({k: row[k] for k in ("timestamp_utc", "model", "prompt_type",
                                                                      "target_depth", "rep")}
                                                 | {"response": resp.get("response", "")}) + "\n")
                        outputs.flush()

                    done += 1
                    acc = f" acc={stats['acceptance']:.2f} tok/pass={row['tokens_per_pass']}" if stats else " (no spec stats)"
                    print(
                        f"  [{done}/{total}] {model} {prompt_type:8s} depth={depth:<6d} rep={rep} "
                        f"prefill={pe_n}tok@{row['prefill_tps']}t/s decode={ev_n}tok@{row['decode_tps']}t/s{acc}",
                        file=sys.stderr,
                    )
                    if args.cooldown:
                        time.sleep(args.cooldown)
    except KeyboardInterrupt:
        print("interrupted; partial results kept", file=sys.stderr)
    finally:
        writer.close()
        if outputs:
            outputs.close()


if __name__ == "__main__":
    main()
