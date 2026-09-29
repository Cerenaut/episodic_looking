#!/bin/bash
# Walter queue v3: a variant's 3 orders run in PARALLEL (3 jobs, nice -n 10; GPU was at 18% with one), evaluation 800 steps
# per test set (deterministic mean bias), fixed RL code. Curfew: idle 21:00-07:00 (a stage starts only if it can finish by 21:00).
#   1. fixed RL reference seed 1: 3 orders (pre-training done)        -> runs_ref_fixed/
#   2. A.1 differentiable actor seed 2: pre-training + 3 orders        -> runs_a1_diff_s2/
#   3. item 19: 40-epoch STM pre-training on e13 + 3 orders            -> runs_p0_pt40/
# Logs runs_local/queue/; resumable via .done files.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
CK=../cifar_100_pretrain
LOG=runs_local/queue
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
PRE_MIN=3.0     # min per pre-training epoch, one job
ORD_MIN=5.5     # min per continual epoch with 3 orders in parallel (estimate; the status script measures the real rate)
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

curfew() {  # $1 = estimated minutes of the stage about to start
  local est_min=$1
  while :; do
    local h=$(date +%H) m=$(date +%M)
    local now_min=$((10#$h * 60 + 10#$m))
    if [ $now_min -ge $((7 * 60)) ] && [ $((now_min + est_min)) -le $((21 * 60)) ]; then return; fi
    local wake
    if [ $now_min -lt $((7 * 60)) ]; then wake=$(( (7 * 60 - now_min) * 60 )); else wake=$(( (24 * 60 - now_min + 7 * 60) * 60 )); fi
    step "curfew: next stage (~$est_min min) cannot finish before 21:00; sleeping $((wake / 60)) min until 07:00"
    sleep $wake
  done
}

run_variant() {  # $1 tag, $2 pretrain epochs, $3 run root, $4 stm checkpoint, $5.. extra args
  local tag=$1 epochs=$2 root=$3 stm=$4; shift 4
  local RUN="nice -n 10 $PY cifar_main_stm_training.py --coarse-classes 0 1 --eval-bias mean --evaluate-steps 800 --ltm-checkpoint $E13 --stm-checkpoint $stm --run-root $root $*"
  if [ ! -f "$LOG/${tag}_pretrain.done" ]; then
    curfew $(python3 -c "print(int($epochs * $PRE_MIN + 5))")
    step "$tag: STM pre-training $epochs epochs"
    $RUN --experiment-type pretrain --fine-classes 1 2 --epochs $epochs > "$LOG/${tag}_pretrain.log" 2>&1 \
      && touch "$LOG/${tag}_pretrain.done" || { step "FAILED: $tag pre-training"; return 1; }
  fi
  local pending=0
  for order in "3 4 5" "4 5 3" "5 3 4"; do [ -f "$LOG/${tag}_continual_$(echo $order | tr -d ' ').done" ] || pending=1; done
  [ $pending -eq 0 ] && return 0
  curfew $(python3 -c "print(int(36 * $ORD_MIN + 5))")
  for order in "3 4 5" "4 5 3" "5 3 4"; do
    local otag=$(echo "$order" | tr -d ' ')
    [ -f "$LOG/${tag}_continual_$otag.done" ] && continue
    step "$tag: continual order $order (parallel)"
    ( $RUN --experiment-type continual --fine-classes $order > "$LOG/${tag}_continual_$otag.log" 2>&1 \
        && touch "$LOG/${tag}_continual_$otag.done" && step "$tag order $order finished" || step "FAILED: $tag order $order" ) &
  done
  wait
}

run_variant ref   12 runs_ref_fixed  $CK/variants/stm_rl_fixed_e13_pt12.pth
run_variant a1_s2 12 runs_a1_diff_s2 $CK/variants/stm_a1_diff_e13_pt12_s2.pth --actor-training differentiable
run_variant pt40  40 runs_p0_pt40    $CK/variants/stm_rl_fixed_e13_pt40.pth
step "ALL DONE"
