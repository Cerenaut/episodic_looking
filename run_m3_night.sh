#!/bin/bash
# M3 overnight: after run_a1_diff.sh finishes, run fixed RL reference seed 2 (pre-training 12 epochs on e13, then 3 orders in
# parallel), --eval-bias mean -> runs_ref_fixed_s2/. Logs runs_local/ref_s2/; resumable via .done files.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/ref_s2
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
STM=$CK/variants/stm_rl_fixed_e13_pt12_s2.pth
RUN="$PY cifar_main_stm_training.py --coarse-classes 0 1 --eval-bias mean --evaluate-steps 800 --ltm-checkpoint $E13 --stm-checkpoint $STM --run-root runs_ref_fixed_s2"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

while pgrep -f "^/bin/bash ./run_a1_diff.sh" > /dev/null; do sleep 120; done
step "A.1 seed 1 finished; starting fixed RL reference seed 2"
if [ ! -f "$LOG/pretrain.done" ]; then
  step "ref s2: STM pre-training 12 epochs on e13"
  $RUN --experiment-type pretrain --fine-classes 1 2 --epochs 12 > "$LOG/pretrain.log" 2>&1 && touch "$LOG/pretrain.done" || { step "FAILED: pre-training"; exit 1; }
fi
for order in "3 4 5" "4 5 3" "5 3 4"; do
  tag=$(echo "$order" | tr -d ' ')
  [ -f "$LOG/continual_$tag.done" ] && continue
  step "ref s2: continual order $order (parallel)"
  ( $RUN --experiment-type continual --fine-classes $order > "$LOG/continual_$tag.log" 2>&1 && touch "$LOG/continual_$tag.done" && step "order $order finished" || step "FAILED: order $order" ) &
done
wait
step "ALL DONE"
