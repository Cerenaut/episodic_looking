#!/bin/bash
# Epoch-13 LTM control (HANDOFF.md "How to launch the control"), parallelised.
# Assumes ../cifar_100_pretrain/cifar_100_subclasses_12_e11_31.1.pth is the e13 checkpoint and
# no STM pre-train checkpoint exists (the e40 ones are in ../cifar_100_pretrain/e40/).
# Stage A (parallel): LTM baseline eval, heads sweep, LTM-only 3 orders (sequential), STM pre-training.
# Stage B (after STM pre-training): 3 STM continual orders in parallel.  Stage C: plot to runs_local/comparison_e13.
# Resumable via .done files in runs_local/reference_e13.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/reference_e13
mkdir -p "$LOG" runs_local/baselines
CC="0 1"
LTM_CKPT=$CK/cifar_100_subclasses_12_e11_31.1.pth
STM_CKPT=$CK/cifar_100_cc56_subclasses_12_stm_pretrain_e12.pth

step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

# --- Stage A ---
if [ ! -f "$LOG/baseline.done" ]; then
  ( $PY eval_pretrained_baseline.py --checkpoint "$LTM_CKPT" > "$LOG/baseline.log" 2>&1 && touch "$LOG/baseline.done" || step "FAILED: baseline eval" ) &
  step "A: LTM pre-continual baseline eval started"
fi
if [ ! -f "$LOG/heads.done" ]; then
  ( ./run_all_heads.sh "$LTM_CKPT" > "$LOG/heads.log" 2>&1 && touch "$LOG/heads.done" || step "FAILED: heads sweep" ) &
  step "A: heads sweep started"
fi
(
  for order in "3 4 5" "4 5 3" "5 3 4"; do
    tag=$(echo "$order" | tr -d ' ')
    [ -f "$LOG/ltm_continual_$tag.done" ] && continue
    step "A: LTM-only continual, order $order (lr 0.001)"
    $PY cifar_main_ltm_fine_tuning.py --experiment-type continual --fine-classes $order --learning-rate 0.001 --coarse-classes $CC \
      > "$LOG/ltm_continual_$tag.log" 2>&1 && touch "$LOG/ltm_continual_$tag.done" || step "FAILED: LTM continual $order"
  done
) &
if [ ! -f "$STM_CKPT" ]; then
  step "A: STM pre-training, 12 epochs"
  $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --coarse-classes $CC --epochs 12 \
    > "$LOG/stm_pretrain.log" 2>&1 && touch "$LOG/stm_pretrain.done" || step "FAILED: STM pre-training"
fi
wait
[ -f "$STM_CKPT" ] || { step "ABORT: no STM checkpoint"; exit 1; }
touch "$LOG/stm_pretrain.done"
step "A done: STM checkpoint present"

# --- Stage B ---
pids=()
for order in "3 4 5" "4 5 3" "5 3 4"; do
  tag=$(echo "$order" | tr -d ' ')
  [ -f "$LOG/stm_continual_$tag.done" ] && continue
  step "B: STM continual, order $order (parallel)"
  ( $PY cifar_main_stm_training.py --experiment-type continual --fine-classes $order --coarse-classes $CC \
      > "$LOG/stm_continual_$tag.log" 2>&1 && touch "$LOG/stm_continual_$tag.done" && step "STM order $order finished" || step "FAILED: STM continual $order" ) &
  pids+=($!)
done
wait

# --- Stage C ---
step "C: plotting to runs_local/comparison_e13"
$PY plot_comparison.py --out-dir runs_local/comparison_e13 > "$LOG/plot_final.log" 2>&1 || step "plot had errors"
step "ALL DONE"
