#!/bin/bash
# Continuation of run_all_reference.sh step 7 with the remaining STM orders run in parallel, then the final plot.
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
LOG=runs_local/reference
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
P1=92619
step "7/8 (parallel) STM continual orders 4 5 3 and 5 3 4 started alongside 3 4 5 (pid $P1)"
nohup $PY cifar_main_stm_training.py --experiment-type continual --fine-classes 4 5 3 --coarse-classes 0 1 > "$LOG/stm_continual_453.log" 2>&1 &
P2=$!
nohup $PY cifar_main_stm_training.py --experiment-type continual --fine-classes 5 3 4 --coarse-classes 0 1 > "$LOG/stm_continual_534.log" 2>&1 &
P3=$!
while kill -0 $P1 2>/dev/null; do sleep 60; done; touch "$LOG/stm_continual_345.done"; step "STM order 3 4 5 finished"
while kill -0 $P2 2>/dev/null; do sleep 60; done; touch "$LOG/stm_continual_453.done"; step "STM order 4 5 3 finished"
while kill -0 $P3 2>/dev/null; do sleep 60; done; touch "$LOG/stm_continual_534.done"; step "STM order 5 3 4 finished"
step "8/8 plotting (final)"
$PY plot_comparison.py > "$LOG/plot_final.log" 2>&1 || step "plot (final) had errors"
step "ALL DONE"
