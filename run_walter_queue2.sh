#!/bin/bash
# Walter queue v2 (replaces run_walter_queue.sh; one GPU job at a time, nice -n 10, fixed RL code, --eval-bias mean).
# Curfew: Walter must be idle 21:00-07:00. A stage starts only if its estimated duration ends before 21:00;
# otherwise the queue sleeps until 07:00. Order:
#   1. finish fixed RL reference seed 1 (orders 4 5 3, 5 3 4; order 3 4 5 handed over from the old queue) -> runs_ref_fixed/
#   2. A.1 differentiable actor seed 2: pre-training + 3 orders                                           -> runs_a1_diff_s2/
#   3. item 19: 40-epoch STM pre-training on e13 + 3 orders                                              -> runs_p0_pt40/
# (fixed RL reference seed 2 runs on the M3 overnight: run_m3_night.sh). Logs runs_local/queue/; resumable via .done files.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
CK=../cifar_100_pretrain
LOG=runs_local/queue
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
E40=$CK/cifar_100_subclasses_12_e40.pth
MIN_PER_EPOCH=3.5
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

curfew() {  # $1 = epochs of the stage about to start; sleep until 07:00 if it cannot finish before 21:00 or it is night
  local est_min=$(python3 -c "print(int($1 * $MIN_PER_EPOCH + 5))")
  while :; do
    local h=$(date +%H) m=$(date +%M)
    local now_min=$((10#$h * 60 + 10#$m))
    if [ $now_min -ge $((7 * 60)) ] && [ $((now_min + est_min)) -le $((21 * 60)) ]; then return; fi
    local wake
    if [ $now_min -lt $((7 * 60)) ]; then wake=$(( (7 * 60 - now_min) * 60 )); else wake=$(( (24 * 60 - now_min + 7 * 60) * 60 )); fi
    step "curfew: next stage ($1 epochs, ~$est_min min) cannot finish before 21:00; sleeping $((wake / 60)) min until 07:00"
    sleep $wake
  done
}

# Hand-over from the old queue: wait for its running job (ref order 3 4 5) to finish, then mark it.
while pgrep -f "cifar_main_stm_training.py" > /dev/null; do sleep 60; done
F=$(find runs_ref_fixed -path "*continual_\[3, 4, 5\]*" -name results_continual.txt 2>/dev/null | tail -1)
if [ -n "$F" ] && [ "$(grep -c evaluate "$F")" -ge 144 ] && [ ! -f "$LOG/ref_continual_345.done" ]; then
  touch "$LOG/ref_continual_345.done"; step "hand-over: ref order 3 4 5 complete (old queue)"
elif [ -n "$F" ] && [ ! -f "$LOG/ref_continual_345.done" ]; then
  mkdir -p ~/Dev/scratch/partial && mv "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$F")")")")")" ~/Dev/scratch/partial/ref_345_$(date +%H%M)
  step "hand-over: ref order 3 4 5 was incomplete; moved aside, will re-run"
fi
step "queue v2 start"

run_variant() {  # $1 tag, $2 pretrain epochs, $3 run root, $4 stm checkpoint, $5 ltm checkpoint, $6.. extra args
  local tag=$1 epochs=$2 root=$3 stm=$4 ltm=$5; shift 5
  local RUN="nice -n 10 $PY cifar_main_stm_training.py --coarse-classes 0 1 --eval-bias mean --ltm-checkpoint $ltm --stm-checkpoint $stm --run-root $root $*"
  if [ ! -f "$LOG/${tag}_pretrain.done" ]; then
    curfew $epochs
    step "$tag: STM pre-training $epochs epochs"
    $RUN --experiment-type pretrain --fine-classes 1 2 --epochs $epochs > "$LOG/${tag}_pretrain.log" 2>&1 \
      && touch "$LOG/${tag}_pretrain.done" || { step "FAILED: $tag pre-training"; return 1; }
  fi
  for order in "3 4 5" "4 5 3" "5 3 4"; do
    local otag=$(echo "$order" | tr -d ' ')
    [ -f "$LOG/${tag}_continual_$otag.done" ] && continue
    curfew 36
    step "$tag: continual order $order"
    $RUN --experiment-type continual --fine-classes $order > "$LOG/${tag}_continual_$otag.log" 2>&1 \
      && touch "$LOG/${tag}_continual_$otag.done" && step "$tag order $order finished" || step "FAILED: $tag order $order"
  done
}

run_variant ref     12 runs_ref_fixed     $CK/variants/stm_rl_fixed_e13_pt12.pth    $E13
run_variant a1_s2   12 runs_a1_diff_s2    $CK/variants/stm_a1_diff_e13_pt12_s2.pth  $E13 --actor-training differentiable
run_variant pt40    40 runs_p0_pt40       $CK/variants/stm_rl_fixed_e13_pt40.pth    $E13
step "ALL DONE"
