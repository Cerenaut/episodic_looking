#!/bin/bash
# Walter queue (one GPU job at a time, nice -n 10, fixed RL code, --eval-bias mean). Starts after run_walter_phase0.sh exits.
#   1. fixed RL reference seed 1: pre-training 12 epochs on e13 + 3 orders   -> runs_ref_fixed/
#   2. A.1 differentiable actor seed 2: pre-training + 3 orders               -> runs_a1_diff_s2/
#   3. fixed RL reference seed 2                                              -> runs_ref_fixed_s2/
#   4. item 19: pre-training 40 epochs on e13 + 3 orders                      -> runs_p0_pt40/
# (LTM-epoch sweep e9/e20/e30 runs on the M3: run_m3_sweep.sh)
# Logs runs_local/queue/; resumable via .done files.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
CK=../cifar_100_pretrain
LOG=runs_local/queue
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

while pgrep -f run_walter_phase0.sh > /dev/null; do sleep 60; done
step "queue start"

run_variant() {  # $1 tag, $2 pretrain epochs, $3 run root, $4 stm checkpoint, $5.. extra args
  local tag=$1 epochs=$2 root=$3 stm=$4; shift 4
  local RUN="nice -n 10 $PY cifar_main_stm_training.py --coarse-classes 0 1 --eval-bias mean --ltm-checkpoint $E13 --stm-checkpoint $stm --run-root $root $*"
  if [ ! -f "$LOG/${tag}_pretrain.done" ]; then
    step "$tag: STM pre-training $epochs epochs on e13"
    $RUN --experiment-type pretrain --fine-classes 1 2 --epochs $epochs > "$LOG/${tag}_pretrain.log" 2>&1 \
      && touch "$LOG/${tag}_pretrain.done" || { step "FAILED: $tag pre-training"; return 1; }
  fi
  for order in "3 4 5" "4 5 3" "5 3 4"; do
    local otag=$(echo "$order" | tr -d ' ')
    [ -f "$LOG/${tag}_continual_$otag.done" ] && continue
    step "$tag: continual order $order"
    $RUN --experiment-type continual --fine-classes $order > "$LOG/${tag}_continual_$otag.log" 2>&1 \
      && touch "$LOG/${tag}_continual_$otag.done" && step "$tag order $order finished" || step "FAILED: $tag order $order"
  done
}

run_variant ref    12 runs_ref_fixed    $CK/variants/stm_rl_fixed_e13_pt12.pth
run_variant a1_s2  12 runs_a1_diff_s2   $CK/variants/stm_a1_diff_e13_pt12_s2.pth --actor-training differentiable
run_variant ref_s2 12 runs_ref_fixed_s2 $CK/variants/stm_rl_fixed_e13_pt12_s2.pth
run_variant pt40   40 runs_p0_pt40      $CK/variants/stm_rl_fixed_e13_pt40.pth
step "ALL DONE"
