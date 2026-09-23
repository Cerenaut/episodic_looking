#!/bin/bash
# D.15 (research_plan.md phase 1 item 2): STM pre-trained on fine classes 1,2 of ALL 20 coarse classes (20,000 training
# images instead of 2,000; --pretrain-coarse-classes), evaluation on the pair (coarse 0,1) as always, then the three continual
# orders on the pair in parallel. Fixed RL, --eval-bias mean, --evaluate-steps 800, 12 pre-training epochs (same step budget as
# the reference: 8,000 exposures per epoch, so 4.8 passes over the broader set). Unseeded. Resumable via .done markers.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/d15
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
STM=$CK/variants/stm_rl_fixed_e13_pt12_d15.pth
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
if [ ! -f "$LOG/pretrain.done" ]; then
  step "D.15 pre-training 12 epochs on fine classes 1,2 of coarse 0..19"
  $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --coarse-classes 0 1 --pretrain-coarse-classes $(seq 0 19) --epochs 12 \
      --eval-bias mean --evaluate-steps 800 --ltm-checkpoint $E13 --stm-checkpoint $STM --run-root runs_d15 > "$LOG/pretrain.log" 2>&1 \
      && touch "$LOG/pretrain.done" || { step "FAILED: pre-training"; exit 1; }
fi
for order in "3 4 5" "4 5 3" "5 3 4"; do
  tag=$(echo "$order" | tr -d ' ')
  [ -f "$LOG/continual_$tag.done" ] && continue
  step "D.15 continual order $order (parallel)"
  ( $PY cifar_main_stm_training.py --experiment-type continual --fine-classes $order --coarse-classes 0 1 --eval-bias mean --evaluate-steps 800 \
      --ltm-checkpoint $E13 --stm-checkpoint $STM --run-root runs_d15 > "$LOG/continual_$tag.log" 2>&1 \
      && touch "$LOG/continual_$tag.done" && step "order $order finished" || step "FAILED: order $order" ) &
done
wait
step "ALL DONE"
