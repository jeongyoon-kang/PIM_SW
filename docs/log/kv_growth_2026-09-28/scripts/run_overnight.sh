#!/usr/bin/env bash
# Overnight generation runs, one model at a time (the board is serial).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the experiment folder
PY=/home/kjy/miniconda3/envs/pim/bin/python
OUT=$HERE/data
cd "$HERE/../../../sw/app"
export HF_HUB_OFFLINE=1

run() {   # model  max_time_seconds
    local m=$1 t=$2 tag=${1##*/}
    echo "[$(date '+%F %T')] start $m (max-time $t s)" | tee -a $OUT/overnight.log
    timeout $((t + 1800)) $PY $HERE/scripts/kv_experiment.py --model "$m" --mode gen --chat \
        --max-time "$t" --out $OUT > $OUT/$tag.gen.log 2>&1
    echo "[$(date '+%F %T')] end   $m rc=$?" | tee -a $OUT/overnight.log
}

run Qwen/Qwen3-0.6B                  12600
run meta-llama/Llama-3.2-1B-Instruct 14400
run meta-llama/Llama-3.2-3B-Instruct 14400
echo "[$(date '+%F %T')] all done" | tee -a $OUT/overnight.log
