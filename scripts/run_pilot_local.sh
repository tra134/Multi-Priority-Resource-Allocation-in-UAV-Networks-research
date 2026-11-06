#!/usr/bin/env bash
# Stage 3 pilot as a detached background job (same settings as the Colab pilot cell):
# seed 0, N=2 and N=10, 100k steps each, then pilot_report.py.  train.py uses CUDA
# automatically when it is available, so the same script runs on a laptop CPU or a
# Colab GPU.  Paths and the interpreter can be overridden with environment variables:
#
#   PY CKPT OUT LOG TB SEED PILOT_STEPS OMP_NUM_THREADS
#
# Colab example (see the notebook): PY=python3 CKPT=/content/drive/MyDrive/UAV_Research/ckpt_pilot ...
#
#   setsid nohup bash scripts/run_pilot_local.sh > /dev/null 2>&1 &
#
# Resumable: rerun the same command after an interruption.  Progress is in
# results/logs_pilot/*.log; results/logs_pilot/DONE appears when everything finished
# (or FAILED with the failing step).  Pilot checkpoints stay in results/ckpt_pilot_local,
# separate from the full-training checkpoints.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-venv/bin/python}"
SEED="${SEED:-0}"; STEPS="${PILOT_STEPS:-100000}"
CKPT="${CKPT:-results/ckpt_pilot_local}"; OUT="${OUT:-results/stage3_pilot_local}"
LOG="${LOG:-results/logs_pilot}"; TB="${TB:-results/tb_pilot_local}"
mkdir -p "$CKPT" "$LOG"; rm -f "$LOG/DONE" "$LOG/FAILED"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}" PYTHONWARNINGS=ignore

train() {  # $1 = N critics
  local n="$1" name="sac_seed${SEED}_N$1"
  mkdir -p "$CKPT/$name"
  "$PY" scripts/train.py --config configs/default.yaml --agent sac --n-critics "$n" \
    --seed "$SEED" --steps "$STEPS" --ckpt-dir "$CKPT/$name" --ckpt-every 5000 \
    --eval-every 20000 --eval-episodes 3 --tb-dir "$TB/$name" --resume \
    >> "$LOG/$name.log" 2>&1
}
train 2 & p2=$!
train 10 & p10=$!
wait "$p2"; r2=$?
wait "$p10"; r10=$?
if [ "$r2" -ne 0 ] || [ "$r10" -ne 0 ]; then
  echo "training exit codes: N2=$r2 N10=$r10" > "$LOG/FAILED"; exit 1
fi
if [ -f "$OUT/pilot_report.json" ]; then
  echo "report already exists in $OUT" >> "$LOG/report.log"
else
  "$PY" scripts/pilot_report.py --ckpt-dir "$CKPT" --seed "$SEED" --episodes 20 \
    --seed-offset 500 --out "$OUT" >> "$LOG/report.log" 2>&1 \
    || { echo "pilot_report failed, see report.log" > "$LOG/FAILED"; exit 1; }
fi
date -Is > "$LOG/DONE"
