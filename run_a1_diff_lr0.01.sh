#!/bin/bash
# Variant A.1 at learning rate 0.01 (0.1 diverges in the continual phase), epoch-13 LTM; evaluation 800 steps.
# Stage 1 (parallel): pre-training 12 epochs on e13; pre-training 12 epochs on e40 (screen).
# Stage 2 (parallel): 3 continual orders on e13 with the e13 checkpoint.
# Results: runs_a1_diff/ (e13), runs_a1_diff_e40/ (screen); logs runs_local/a1_diff/; resumable via .done files.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/a1_diff_lr0.01
CC="0 1"
E13=$CK/cifar_100_subclasses_12_e13.pth
E40=$CK/cifar_100_subclasses_12_e40.pth
STM13=$CK/variants/stm_a1_diff_lr0.01_e13_pt12.pth
STM40=$CK/variants/stm_a1_diff_lr0.01_e40_pt12.pth
COMMON="--actor-training differentiable --learning-rate 0.01 --coarse-classes $CC --evaluate-steps 800"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

if [ ! -f "$LOG/pretrain_e13.done" ]; then
  step "1: A.1 pre-training on e13, 12 epochs"
  ( $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --epochs 12 $COMMON \
      --ltm-checkpoint $E13 --stm-checkpoint $STM13 --run-root runs_a1_diff_lr0.01 > "$LOG/pretrain_e13.log" 2>&1 \
      && touch "$LOG/pretrain_e13.done" && step "A.1 pre-training e13 finished" || step "FAILED: A.1 pre-training e13" ) &
fi
if [ ! -f "$LOG/pretrain_e40.done" ]; then
  step "1: A.1 pre-training on e40 (screen), 12 epochs"
  ( $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --epochs 12 $COMMON \
      --ltm-checkpoint $E40 --stm-checkpoint $STM40 --run-root runs_a1_diff_lr0.01_e40 > "$LOG/pretrain_e40.log" 2>&1 \
      && touch "$LOG/pretrain_e40.done" && step "A.1 pre-training e40 finished" || step "FAILED: A.1 pre-training e40" ) &
fi
wait
[ -f "$STM13" ] || { step "ABORT: no e13 STM checkpoint"; exit 1; }

for order in "3 4 5" "4 5 3" "5 3 4"; do
  tag=$(echo "$order" | tr -d ' ')
  [ -f "$LOG/continual_$tag.done" ] && continue
  step "2: A.1 continual, order $order (parallel)"
  ( $PY cifar_main_stm_training.py --experiment-type continual --fine-classes $order $COMMON \
      --ltm-checkpoint $E13 --stm-checkpoint $STM13 --run-root runs_a1_diff_lr0.01 > "$LOG/continual_$tag.log" 2>&1 \
      && touch "$LOG/continual_$tag.done" && step "A.1 order $order finished" || step "FAILED: A.1 order $order" ) &
done
wait
step "ALL DONE"
