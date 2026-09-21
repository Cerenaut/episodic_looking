#!/bin/bash
# Item 19 on the M3 (moved from Walter's queue to avoid its curfew): fixed RL, --eval-bias mean, --evaluate-steps 800.
# Waits for the A.1 lr 0.01 and reference repeat 2 pipelines to finish, then 40-epoch STM pre-training on e13 + 3 orders in parallel.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/pt40
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
STM=$CK/variants/stm_rl_fixed_e13_pt40.pth
RUN="$PY cifar_main_stm_training.py --coarse-classes 0 1 --eval-bias mean --evaluate-steps 800 --ltm-checkpoint $E13 --stm-checkpoint $STM --run-root runs_p0_pt40"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
while pgrep -f "^/bin/bash ./run_a1_diff_lr0.01.sh|^/bin/bash ./run_m3_night.sh" > /dev/null; do sleep 120; done
step "M3 free; starting item 19"
if [ ! -f "$LOG/pretrain.done" ]; then
  step "pt40: STM pre-training 40 epochs on e13"
  $RUN --experiment-type pretrain --fine-classes 1 2 --epochs 40 > "$LOG/pretrain.log" 2>&1 && touch "$LOG/pretrain.done" || { step "FAILED: pre-training"; exit 1; }
fi
for order in "3 4 5" "4 5 3" "5 3 4"; do
  tag=$(echo "$order" | tr -d ' ')
  [ -f "$LOG/continual_$tag.done" ] && continue
  step "pt40: continual order $order (parallel)"
  ( $RUN --experiment-type continual --fine-classes $order > "$LOG/continual_$tag.log" 2>&1 && touch "$LOG/continual_$tag.done" && step "order $order finished" || step "FAILED: order $order" ) &
done
wait
step "ALL DONE"
