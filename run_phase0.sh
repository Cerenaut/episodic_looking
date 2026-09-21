#!/bin/bash
# Phase 0 (research_plan.md), RL-trained STM with the reward / episode-observation fixes (PR fix-rl-reward-and-episode-obs).
#   ref : STM pre-training 12 epochs on e13, then 3 continual orders (parallel)          -> runs_ref_fixed/
#   item 20: STM pre-training 12 epochs on LTM epochs 5, 9, 20, 30 (sequential)           -> runs_p0_sweep/e<N>/
#   item 19: STM pre-training 40 epochs on e13, then 3 continual orders (parallel)         -> runs_p0_pt40/
# Logs runs_local/phase0/; resumable via .done files.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/phase0
CC="0 1"
E13=$CK/cifar_100_subclasses_12_e13.pth
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

run_ref() {  # $1 = tag, $2 = pretrain epochs, $3 = run root, $4 = stm checkpoint
  local tag=$1 epochs=$2 root=$3 stm=$4
  if [ ! -f "$LOG/${tag}_pretrain.done" ]; then
    step "$tag: STM pre-training $epochs epochs on e13"
    $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --coarse-classes $CC --epochs $epochs --eval-bias mean \
       --ltm-checkpoint $E13 --stm-checkpoint $stm --run-root $root > "$LOG/${tag}_pretrain.log" 2>&1 \
       && touch "$LOG/${tag}_pretrain.done" || { step "FAILED: $tag pre-training"; return 1; }
  fi
  for order in "3 4 5" "4 5 3" "5 3 4"; do
    local otag=$(echo "$order" | tr -d ' ')
    [ -f "$LOG/${tag}_continual_$otag.done" ] && continue
    step "$tag: continual order $order (parallel)"
    ( $PY cifar_main_stm_training.py --experiment-type continual --fine-classes $order --coarse-classes $CC --eval-bias mean \
        --ltm-checkpoint $E13 --stm-checkpoint $stm --run-root $root > "$LOG/${tag}_continual_$otag.log" 2>&1 \
        && touch "$LOG/${tag}_continual_$otag.done" && step "$tag order $order finished" || step "FAILED: $tag order $order" ) &
  done
  wait
}

run_ref ref 12 runs_ref_fixed $CK/variants/stm_rl_fixed_e13_pt12.pth

for e in 5 9 20 30; do
  [ -f "$LOG/sweep_e$e.done" ] && continue
  step "item 20: STM pre-training 12 epochs on LTM e$e"
  $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --coarse-classes $CC --epochs 12 --eval-bias mean \
     --ltm-checkpoint $CK/cifar_100_subclasses_12_e$e.pth --stm-checkpoint $CK/variants/stm_rl_fixed_e${e}_pt12.pth \
     --run-root runs_p0_sweep/e$e > "$LOG/sweep_e$e.log" 2>&1 && touch "$LOG/sweep_e$e.done" || step "FAILED: sweep e$e"
done

run_ref pt40 40 runs_p0_pt40 $CK/variants/stm_rl_fixed_e13_pt40.pth
step "ALL DONE"
