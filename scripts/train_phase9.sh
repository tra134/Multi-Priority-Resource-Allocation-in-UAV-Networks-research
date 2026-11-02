#!/usr/bin/env bash
# Train all Phase 9 checkpoints sequentially (resumable, CPU or GPU).
#
#   SEEDS="0" ./scripts/train_phase9.sh          # first chunk (3 runs)
#   SEEDS="0 1 2 3 4" ./scripts/train_phase9.sh  # full 15-run sweep
#
# Every run writes <ckpt-dir>/sac_seed<seed>_N<n>[_drwide]_latest.pt and
# resumes from it if the process is interrupted.  Logs go to results/logs/.
#
# Defaults are tuned for a laptop CPU: evaluation is kept light because the
# Phase 9 grid re-evaluates every checkpoint with 20 episodes anyway.

set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0}"
STEPS="${STEPS:-300000}"
CKPT_DIR="${CKPT_DIR:-results/ckpt}"
LOG_DIR="${LOG_DIR:-results/logs}"
EVAL_EVERY="${EVAL_EVERY:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
PY="${PY:-venv/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

run_one() {
    local seed="$1" n_critics="$2" dr_wide="$3"
    local suffix="" name log
    if [ "$dr_wide" = "1" ]; then
        suffix="_drwide"
    fi
    name="sac_seed${seed}_N${n_critics}${suffix}"
    log="${LOG_DIR}/${name}.log"

    local cmd=(
        "$PY" scripts/train.py
        --config configs/default.yaml
        --agent sac
        --n-critics "$n_critics"
        --seed "$seed"
        --steps "$STEPS"
        --ckpt-dir "$CKPT_DIR"
        --ckpt-every "$CKPT_EVERY"
        --eval-every "$EVAL_EVERY"
        --eval-episodes "$EVAL_EPISODES"
        --tb-dir "results/tb/${name}"
        --resume
    )
    if [ "$dr_wide" = "1" ]; then
        cmd+=(--dr-wide)
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "${cmd[*]}"
        return
    fi
    echo "=== ${name} -> ${log} ==="
    "${cmd[@]}" 2>&1 | tee -a "$log"
}

for seed in $SEEDS; do
    run_one "$seed" 2 0
    run_one "$seed" 10 0
    run_one "$seed" 2 1
done

echo "done: seeds [$SEEDS]"
