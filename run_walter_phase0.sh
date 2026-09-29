#!/bin/bash
# Phase 0 on Walter (GTX 1060, shared CPU: one GPU job at a time, nice -n 10). Fixed RL code, --eval-bias mean.
#   item 20: STM pre-training 12 epochs on LTM epochs 5, 9, 20, 30 (sequential)   -> runs_p0_sweep/e<N>/
#   item 19: STM pre-training 40 epochs on e13, then 3 continual orders (sequential) -> runs_p0_pt40/
# Logs runs_local/phase0/; resumable via .done files. Sync results back to the M3 with rsync.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
CK=../cifar_100_pretrain
LOG=runs_local/phase0
mkdir -p "$LOG"
CC="0 1"
E13=$CK/cifar_100_subclasses_12_e13.pth
RUN="nice -n 10 $PY cifar_main_stm_training.py --coarse-classes $CC --eval-bias mean"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

for e in 5 9 20 30; do
  [ -f "$LOG/sweep_e$e.done" ] && continue
  step "item 20: STM pre-training 12 epochs on LTM e$e"
  $RUN --experiment-type pretrain --fine-classes 1 2 --epochs 12 \
     --ltm-checkpoint $CK/cifar_100_subclasses_12_e$e.pth --stm-checkpoint $CK/variants/stm_rl_fixed_e${e}_pt12.pth \
     --run-root runs_p0_sweep/e$e > "$LOG/sweep_e$e.log" 2>&1 && touch "$LOG/sweep_e$e.done" || step "FAILED: sweep e$e"
done

STM40=$CK/variants/stm_rl_fixed_e13_pt40.pth
if [ ! -f "$LOG/pt40_pretrain.done" ]; then
  step "item 19: STM pre-training 40 epochs on e13"
  $RUN --experiment-type pretrain --fine-classes 1 2 --epochs 40 --ltm-checkpoint $E13 --stm-checkpoint $STM40 \
     --run-root runs_p0_pt40 > "$LOG/pt40_pretrain.log" 2>&1 && touch "$LOG/pt40_pretrain.done" || { step "FAILED: pt40 pre-training"; exit 1; }
fi
for order in "3 4 5" "4 5 3" "5 3 4"; do
  tag=$(echo "$order" | tr -d ' ')
  [ -f "$LOG/pt40_continual_$tag.done" ] && continue
  step "item 19: continual order $order with the 40-epoch STM"
  $RUN --experiment-type continual --fine-classes $order --ltm-checkpoint $E13 --stm-checkpoint $STM40 \
     --run-root runs_p0_pt40 > "$LOG/pt40_continual_$tag.log" 2>&1 && touch "$LOG/pt40_continual_$tag.done" || step "FAILED: pt40 order $order"
done
step "ALL DONE"
