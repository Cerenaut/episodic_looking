#!/bin/bash
# Walter queue v4: starts after run_walter_queue3.sh exits (its pt40 stage is skipped via markers). Same rules: 3 orders in
# parallel, nice -n 10, --evaluate-steps 800, curfew 21:00-07:00.
#   1. A.1 differentiable actor, lr 0.01, repeat 2: pre-training + 3 orders  -> runs_a1_diff_s2/
#   2. item 19: 40-epoch STM pre-training on e13 + 3 orders (tag pt40b)     -> runs_p0_pt40/
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
CK=../cifar_100_pretrain
LOG=runs_local/queue
E13=$CK/cifar_100_subclasses_12_e13.pth
PRE_MIN=3.0
ORD_MIN=11.0
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
curfew() {
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
while pgrep -f "^/bin/bash ./run_walter_queue3.sh" > /dev/null; do sleep 60; done
step "queue v4 start"
run_variant a1lr01_s2 12 runs_a1_diff_s2 $CK/variants/stm_a1_diff_lr0.01_e13_pt12_s2.pth --actor-training differentiable --learning-rate 0.01
run_variant pt40b     40 runs_p0_pt40    $CK/variants/stm_rl_fixed_e13_pt40.pth
step "ALL DONE"
