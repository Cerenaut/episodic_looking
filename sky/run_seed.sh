#!/bin/bash
# One job of the cloud seed round: a seeded STM pre-training on the e13 LTM, then the three
# continual orders at that same seed. Mirrors run_m3_seeds.sh, with three differences:
#   - PY defaults to the pod venv, not the M3 conda path;
#   - the variant selects the pre-training length (12 or 40 epochs), not the actor type;
#   - NPROC controls whether the three orders run in parallel (set from the smoke test).
# Resumable: .done markers per stage, so a re-run after a lost pod skips finished work.
# Usage: ./sky/run_seed.sh <ref_pt12|ref_pt40> <seed>
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
V=${1:?variant: ref_pt12 or ref_pt40}
SEED=${2:?seed}
NPROC=${NPROC:-3}          # 3 = the orders run together; 1 = strictly sequential

case "$V" in
  ref_pt12) PT_EPOCHS=12 ;;
  ref_pt40) PT_EPOCHS=40 ;;
  *) echo "unknown variant: $V" >&2; exit 2 ;;
esac

CK=../cifar_100_pretrain
E13=$CK/cifar_100_subclasses_12_e13.pth
STM=$CK/variants/stm_rl_fixed_e13_pt${PT_EPOCHS}_seed${SEED}.pth
ROOT=runs_seed/${V}_s${SEED}
LOG=runs_local/seeds
FLAGS="--eval-bias mean"
mkdir -p "$LOG" "$ROOT" "$CK/variants"
step() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG/progress.log"; }

step "job $V seed $SEED starting: pre-train $PT_EPOCHS epochs, then 3 orders, NPROC=$NPROC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG/progress.log" || true

if [ ! -f "$LOG/${V}_s${SEED}_pretrain.done" ]; then
  step "$V seed $SEED: pre-training $PT_EPOCHS epochs"
  $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --coarse-classes 0 1 \
      --epochs $PT_EPOCHS --evaluate-steps 800 --seed $SEED $FLAGS \
      --ltm-checkpoint $E13 --stm-checkpoint $STM --run-root $ROOT \
      > "$LOG/${V}_s${SEED}_pretrain.log" 2>&1 \
    && touch "$LOG/${V}_s${SEED}_pretrain.done" \
    || { step "FAILED: $V seed $SEED pre-training"; exit 1; }
  step "$V seed $SEED: pre-training done"
fi

run_order() {
  local order=$1 tag
  tag=$(echo "$order" | tr -d ' ')
  [ -f "$LOG/${V}_s${SEED}_continual_$tag.done" ] && return 0
  step "$V seed $SEED: continual order $order"
  $PY cifar_main_stm_training.py --experiment-type continual --fine-classes $order --coarse-classes 0 1 \
      --evaluate-steps 800 --seed $SEED $FLAGS \
      --ltm-checkpoint $E13 --stm-checkpoint $STM --run-root $ROOT \
      > "$LOG/${V}_s${SEED}_continual_$tag.log" 2>&1 \
    && touch "$LOG/${V}_s${SEED}_continual_$tag.done" && step "$V seed $SEED order $order finished" \
    || { step "FAILED: $V seed $SEED order $order"; return 1; }
}

rc=0
if [ "$NPROC" -gt 1 ]; then
  for order in "3 4 5" "4 5 3" "5 3 4"; do run_order "$order" & done
  wait || rc=1
else
  for order in "3 4 5" "4 5 3" "5 3 4"; do run_order "$order" || rc=1; done
fi

if [ $rc -ne 0 ]; then step "JOB FAILED: $V seed $SEED"; exit 1; fi
# Sentinel for the reaper. A file tested with `test -f` reports through ssh's exit code
# and cannot be misparsed; grepping a log for a phrase can, and once did: `grep -c` on a
# file that exists without a match prints 0 AND exits 1, so a `|| echo 0` fallback
# produced "0\n0", which compared unequal to "0" and reaped four healthy pods.
touch "$LOG/JOB_COMPLETE"
step "JOB DONE: $V seed $SEED"
