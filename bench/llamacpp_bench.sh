#!/usr/bin/env bash
# llama-bench wrapper for GGUF controls. Runs llama-bench once per model, keeps the
# raw JSONL, and converts it to the shared CSV schema via llamacpp_to_csv.py.
#
#   bench/llamacpp_bench.sh -m a.gguf [-m b.gguf ...] [-d 0,4096,16384] [-p 512] [-n 128] [-r 5]
#                           [--machine slug] [--out-dir results] [-- extra llama-bench args]
#
# llama-bench measures prefill (pp: -p tokens) and decode (tg: -n tokens) as separate
# tests at each depth; the converter pairs them per repetition into one row.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"

argv="$*"
models=()
depths="0"
n_prompt=512
n_gen=128
reps=5
machine=""
out_dir="$repo/results"
extra=()

while (($#)); do
  case "$1" in
    -m) models+=("$2"); shift 2 ;;
    -d) depths="$2"; shift 2 ;;
    -p) n_prompt="$2"; shift 2 ;;
    -n) n_gen="$2"; shift 2 ;;
    -r) reps="$2"; shift 2 ;;
    --machine) machine="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --) shift; extra=("$@"); break ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

((${#models[@]})) || { echo "need at least one -m model.gguf" >&2; exit 2; }
command -v llama-bench >/dev/null || { echo "llama-bench not on PATH (brew install llama.cpp)" >&2; exit 1; }

run_id="$(date +%Y%m%d-%H%M%S)_llamacpp"
raw_dir="$(mktemp -d)"
trap 'rm -rf "$raw_dir"' EXIT
raw="$raw_dir/$run_id.raw.jsonl"

for m in "${models[@]}"; do
  echo "[llama-bench] $m  depths=$depths p=$n_prompt n=$n_gen r=$reps" >&2
  # -fa on: flash attention, the setting a real deployment would use
  llama-bench -m "$m" -p "$n_prompt" -n "$n_gen" -r "$reps" -d "$depths" -fa on -o jsonl \
    ${extra[@]+"${extra[@]}"} >> "$raw"
done

python3 "$here/llamacpp_to_csv.py" "$raw" --run-id "$run_id" --out-dir "$out_dir" \
  ${machine:+--machine "$machine"} --argv "$argv"
