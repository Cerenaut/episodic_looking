#!/bin/bash
# Item 19 orders on Walter (launched from the M3 by run_m3_pt40_split.sh once the 40-epoch pre-training is done there).
# Two orders in parallel: the runs are GPU-bound, so this finishes in the same GPU time as one order after another,
# and two (not three) leave a curfew margin. Order 5,3,4 runs on the M3.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
CK=../cifar_100_pretrain
LOG=runs_local/pt40
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
STM=$CK/variants/stm_rl_fixed_e13_pt40.pth
RUN="nice -n 10 $PY cifar_main_stm_training.py --coarse-classes 0 1 --eval-bias mean --evaluate-steps 800 --ltm-checkpoint $E13 --stm-checkpoint $STM --run-root runs_p0_pt40"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
for order in "3 4 5" "4 5 3"; do
  tag=$(echo "$order" | tr -d ' ')
  [ -f "$LOG/continual_$tag.done" ] && continue
  step "pt40: continual order $order (parallel, on Walter)"
  ( $RUN --experiment-type continual --fine-classes $order > "$LOG/continual_$tag.log" 2>&1 && touch "$LOG/continual_$tag.done" && step "order $order finished" || step "FAILED: order $order" ) &
done
wait
step "WALTER ORDERS DONE"
