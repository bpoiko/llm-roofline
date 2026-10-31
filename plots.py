#!/usr/bin/env python3
"""Plots and summary from every results/<machine>/*.csv (all runtimes).

  results/plots/roofline.png           decode t/s vs bytes read per token, roofline per machine
  results/plots/depth_<machine>.png    decode t/s vs context depth, with predicted ceilings
  results/plots/mtp_<machine>.png      decode t/s vs MTP acceptance (speculating runtimes)
  results/plots/summary.csv            median per cell, ceiling, efficiency: the table view

Bytes/token for a run come from config/models.json (tools/model_bytes.py) at the run's
average decode depth: prompt_tokens + eval_count / 2. Runs that stopped early
(eval_count < --min-tokens) or errored are dropped.

  uv run plots.py [--machines m4pro-24gb ...]
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

REPO = Path(__file__).resolve().parent

# Reference categorical palette, light surface, fixed slot order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RUNTIME_MARKER = {"ollama": "o", "mlx": "s", "llamacpp": "^"}
PROMPT_MARKER = {"counting": "o", "essay": "s", "random": "D", "synthetic": "^"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False,
})


def tilde(s: str) -> str:
    """Home-relative path: portable across machines, and no username in published results."""
    home = str(Path.home())
    return "~" + s[len(home):] if s == home or s.startswith(home + "/") else s


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_runs(results: Path, machines: set[str] | None, min_tokens: int, models: dict) -> list[dict]:
    rows = []
    for path in sorted(results.glob("*/*.csv")):
        if path.name == "predictions.csv" or path.parent.name == "plots":
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if machines and r["machine"] not in machines:
                    continue
                if r.get("error") or num(r.get("decode_tps")) is None or (num(r["eval_count"]) or 0) < min_tokens:
                    continue
                r["model"] = tilde(r["model"])  # older llama.cpp CSVs hold absolute paths
                m = models.get(f"{r['runtime']}:{r['model']}")
                r["label"] = m["label"] if m else r["model"]
                r["depth"] = (num(r["prompt_tokens"]) or 0) + (num(r["eval_count"]) or 0) / 2
                if m:
                    r["bytes_gb"] = (m["weight_bytes_per_token"] + m.get("state_bytes_per_token", 0)
                                     + m.get("kv_bytes_per_ctx_token", 0) * r["depth"]) / 1e9
                rows.append(r)
    return rows


def machine_bw(results: Path, machines_cfg: dict) -> dict[str, tuple[float, str]]:
    defaults = machines_cfg.get("_defaults", {})
    out = {}
    for name, m in machines_cfg.items():
        if name.startswith("_"):
            continue
        membw = results / name / "membw.json"
        if membw.exists():
            out[name] = (json.loads(membw.read_text())["bandwidth_gbs"], "measured")
        else:
            out[name] = (m["nominal_bw_gbs"] * defaults.get("assumed_efficiency", 1.0), "assumed")
    return out


def color_map(keys) -> dict:
    keys = list(dict.fromkeys(keys))
    if len(keys) > len(SERIES):
        raise SystemExit(f"{len(keys)} series > {len(SERIES)} palette slots; filter with --machines or split the plot")
    return {k: SERIES[i] for i, k in enumerate(keys)}


def finish(ax, fig, path: Path, title: str, subtitle: str):
    ax.set_title(title, loc="left", pad=18)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def plot_roofline(rows, bw, machines_cfg, out: Path):
    with_bytes = [r for r in rows if "bytes_gb" in r]
    if not with_bytes:
        return
    order = [k for k in machines_cfg if not k.startswith("_")]
    colors = color_map(order)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    xs = [r["bytes_gb"] for r in with_bytes]
    lo, hi = min(xs) * 0.7, max(xs) * 1.4
    for name in order:
        b, src = bw[name]
        style = "-" if src == "measured" else (0, (4, 3))
        ax.plot([lo, hi], [b / lo, b / hi], color=colors[name], linestyle=style, linewidth=1.5,
                label=f"{name} roofline {b:.0f} GB/s ({src})")
        ax.annotate(f"{name}", (hi, b / hi), xytext=(4, 0), textcoords="offset points",
                    color=INK_2, fontsize=8, va="center")
    seen = set()
    for r in with_bytes:
        spec = bool(r.get("spec_iterations"))
        key = (r["runtime"], spec)
        lab = f"{r['runtime']}{' (MTP on)' if spec else ''}" if key not in seen else None
        seen.add(key)
        ax.scatter(r["bytes_gb"], float(r["decode_tps"]), s=36, marker=RUNTIME_MARKER.get(r["runtime"], "o"),
                   facecolor=colors.get(r["machine"], INK_2) if not spec else "none",
                   edgecolor=colors.get(r["machine"], INK_2), linewidth=1.5, label=lab, zorder=3,
                   alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("GB read per decoded token (weights + state + KV at depth)")
    ax.set_ylabel("decode tokens/s")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.12, 1))
    finish(ax, fig, out / "roofline.png", "Decode throughput vs bandwidth roofline",
           "Filled = one forward pass per token; hollow = speculative (can exceed the line).")


def ceiling(model: dict, bw_gbs: float, depth: float) -> float:
    b = model["weight_bytes_per_token"] + model.get("state_bytes_per_token", 0) + model.get("kv_bytes_per_ctx_token", 0) * depth
    return bw_gbs * 1e9 / b


def plot_depth(rows, models, bw, machine: str, out: Path):
    rs = [r for r in rows if r["machine"] == machine]
    if not rs:
        return
    series = sorted({(r["label"], r["runtime"]) for r in rs})
    colors = color_map(series)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in series:
        pts = [r for r in rs if (r["label"], r["runtime"]) == s]
        for spec in (False, True):
            sub = [r for r in pts if bool(r.get("spec_iterations")) == spec]
            if sub:
                ax.scatter([r["depth"] for r in sub], [float(r["decode_tps"]) for r in sub], s=28,
                           facecolor="none" if spec else colors[s], edgecolor=colors[s], linewidth=1.5,
                           alpha=0.85, label=f"{s[0]} · {s[1]}{' (MTP on)' if spec else ''}", zorder=3)
        m = models.get(f"{s[1]}:{pts[0]['model']}")
        if m and machine in bw:
            d_max = max(r["depth"] for r in rs) * 1.05 + 1
            ds = [d_max * i / 20 for i in range(21)]
            ax.plot(ds, [ceiling(m, bw[machine][0], d) for d in ds], color=colors[s],
                    linestyle=(0, (4, 3)), linewidth=1.5)
    ax.set_xlabel("context depth during decode (tokens)")
    ax.set_ylabel("decode tokens/s")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    finish(ax, fig, out / f"depth_{machine}.png", f"Decode vs context depth · {machine}",
           "Hollow = speculative decoding on; dashed = bandwidth ceiling for one pass per token.")


def plot_mtp(rows, models, bw, machine: str, out: Path):
    rs = [r for r in rows if r["machine"] == machine and num(r.get("spec_acceptance")) is not None]
    if not rs:
        return
    labels = sorted({r["label"] for r in rs})
    colors = color_map(labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab in labels:
        pts = [r for r in rs if r["label"] == lab]
        for pt in sorted({r["prompt_type"] for r in pts}):
            sub = [r for r in pts if r["prompt_type"] == pt]
            ax.scatter([float(r["spec_acceptance"]) for r in sub], [float(r["decode_tps"]) for r in sub],
                       s=30, color=colors[lab], marker=PROMPT_MARKER.get(pt, "o"), alpha=0.8,
                       label=f"{lab} · {pt}", zorder=3)
        m = models.get(f"{pts[0]['runtime']}:{pts[0]['model']}")
        if m and machine in bw:
            c = ceiling(m, bw[machine][0], 0)
            ax.axhline(c, color=colors[lab], linestyle=(0, (4, 3)), linewidth=1.5)
            ax.annotate(f"{lab} plain-decode ceiling", (0, c), xytext=(2, 3), textcoords="offset points",
                        color=INK_2, fontsize=8)
    ax.set_xlabel("MTP draft acceptance rate")
    ax.set_ylabel("decode tokens/s")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2)
    finish(ax, fig, out / f"mtp_{machine}.png", f"Decode vs MTP acceptance · {machine}",
           "Marker = prompt type; dashed = bandwidth ceiling for one pass per token at depth 0.")


def write_summary(rows, bw, out: Path):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["machine"], r["runtime"], r["model"], r["label"], r["prompt_type"], r["target_depth"])].append(r)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["machine", "runtime", "model", "label", "prompt_type", "target_depth", "n",
                    "median_decode_tps", "median_prefill_tps", "median_acceptance", "median_tokens_per_pass",
                    "bytes_per_token_gb", "ceiling_tps", "efficiency"])
        for (machine, runtime, model, label, pt, depth), rs in sorted(cells.items()):
            med = lambda k: (round(statistics.median(v), 2)  # noqa: E731
                             if (v := [x for r in rs if (x := num(r.get(k))) is not None]) else "")
            decode = med("decode_tps")
            b = statistics.median(r["bytes_gb"] for r in rs) if all("bytes_gb" in r for r in rs) else None
            ceiling = round(bw[machine][0] / b, 2) if b and machine in bw else ""
            w.writerow([machine, runtime, model, label, pt, depth, len(rs), decode, med("prefill_tps"),
                        med("spec_acceptance"), med("tokens_per_pass"), round(b, 4) if b else "", ceiling,
                        round(decode / ceiling, 3) if ceiling and decode != "" else ""])
    print(f"wrote {out / 'summary.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--machines", nargs="+")
    ap.add_argument("--min-tokens", type=int, default=32)
    args = ap.parse_args()

    models_path = REPO / "config" / "models.json"
    models = json.loads(models_path.read_text()) if models_path.exists() else {}
    machines_cfg = json.loads((REPO / "config" / "machines.json").read_text())
    rows = load_runs(args.results, set(args.machines) if args.machines else None, args.min_tokens, models)
    if not rows:
        raise SystemExit("no usable runs under results/")
    unmapped = sorted({f"{r['runtime']}:{r['model']}" for r in rows if "bytes_gb" not in r})
    if unmapped:
        print("not in config/models.json (no bytes/token; run tools/model_bytes.py):\n  " + "\n  ".join(unmapped))

    out = args.results / "plots"
    out.mkdir(parents=True, exist_ok=True)
    bw = machine_bw(args.results, machines_cfg)
    plot_roofline(rows, bw, machines_cfg, out)
    for machine in sorted({r["machine"] for r in rows}):
        plot_depth(rows, models, bw, machine, out)
        plot_mtp(rows, models, bw, machine, out)
    write_summary(rows, bw, out)


if __name__ == "__main__":
    main()
