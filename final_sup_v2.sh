#!/bin/bash
# After select_sup_continual.sh: the supervised CLS/STM's (sup) continual confirmation and final runs
# (Notes/experiments/plan.md section 3 decision 10; next_draft_experiments.tex, Setup):
#   1. wait for select_sup_continual.sh to exit, then read its continual choice (learning rate tree and budget) from
#      runs_v2_pilot_sup/selection_final.txt; abort if the choice is undecided
#   2. confirmation (decision 6): the choice re-run at its budget, pair A, seed 1, 3 orders, validation split, into
#      runs_v2_pilot_sup_confirm/<tree>/ (the selection's STM pre-training copied in); check_confirm_sup.py writes the
#      criterion (validation files only). The final runs go ahead either way, as the heads' did; a failure is reported.
#   3. final runs into runs_v2/: pairs A, B, C, D x seeds 1-5: STM pre-training (12 epochs, lr 0.01), baseline, orders
#      345/453/534 at the selected learning rate and budget, SAVE_STM=1. One unit at a time (MPS).
# Resumable: run_v2.sh skips finished units; the confirmation is skipped once its check file exists.
# Usage: nohup caffeinate -i bash final_sup_v2.sh > runs_v2/final_sup.out 2>&1 &
set -u
cd "$(dirname "$0")"
PY=${PY:-/Users/gideon/anaconda3/envs/episodic/bin/python}
SEL=runs_v2_pilot_sup
CONF=runs_v2_pilot_sup_confirm
LOG=runs_v2/final_sup.log
mkdir -p runs_v2 "$CONF"
step() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

step "waiting for select_sup_continual.sh to finish"
while pgrep -f "bash select_sup_continual.sh" > /dev/null; do sleep 60; done
[ -f "$SEL/selection_final.txt" ] || { step "ABORT: no $SEL/selection_final.txt"; exit 1; }
# the choice row: CLS/STM, supervised & continual & sup_lr<x>[ (notes)] & <budget> & <value> \\
# only in the choices section (after '% selection:'), never a per-tree row above it
CHOICE=$(awk '/^% selection:/ {c = 1} c' "$SEL/selection_final.txt" | grep -E '^ *CLS/STM, supervised & continual & ')
TREE=$(echo "$CHOICE" | awk -F' & ' '{print $3}' | awk '{print $1}')
BUDGET=$(echo "$CHOICE" | awk -F' & ' '{print $4}' | tr -d ' ')
LR=${TREE#sup_lr}
case "$TREE" in sup_lr*) ;; *) BUDGET="" ;; esac
case "$BUDGET" in ''|*[!0-9]*) step "ABORT: no continual choice in $SEL/selection_final.txt: '$CHOICE'"; exit 1 ;; esac
step "start: code $(git rev-parse --short HEAD)$(git diff --quiet -- '*.py' '*.sh' || echo ' +dirty'); choice $TREE (lr $LR), $BUDGET epochs per phase ($CHOICE)"

unit() {  # RUNS-tree, then run_v2.sh arguments; environment from the caller
  if bash run_v2.sh "${@:2}" >> "$1.log" 2>&1 < /dev/null; then step "done   $1: ${*:2}"; else step "FAILED $1: ${*:2} ($1.log)"; fi
}

# --- 2. confirmation -------------------------------------------------------------------------------------------------
if [ ! -f "$CONF/check_confirm.txt" ]; then
  T=$CONF/$TREE
  for s in pretrain baseline; do
    if [ ! -e "$T/$s/sup/pair0_1/seed1/job.done" ]; then
      mkdir -p "$T/$s/sup/pair0_1" && cp -R "$SEL/shared/$s/sup/pair0_1/seed1" "$T/$s/sup/pair0_1/"
    fi
  done
  for o in "3 4 5" "4 5 3" "5 3 4"; do RUNS=$T EPOCHS=$BUDGET LR=$LR unit "$T" continual sup "0 1" 1 "$o"; done
  $PY check_confirm_sup.py "$TREE" "$BUDGET" > "$CONF/check_confirm.txt" 2>&1 || step "check_confirm_sup.py failed"
  step "confirmation: $(tail -1 "$CONF/check_confirm.txt")"
fi

# --- 3. final runs -----------------------------------------------------------------------------------------------------
for pair in "0 1" "2 3" "5 6" "15 16"; do
  for seed in 1 2 3 4 5; do
    unit runs_v2/final_sup_units pretrain sup "$pair" "$seed"
    unit runs_v2/final_sup_units baseline sup "$pair" "$seed"
    for o in "3 4 5" "4 5 3" "5 3 4"; do
      SAVE_STM=1 EPOCHS=$BUDGET LR=$LR unit runs_v2/final_sup_units continual sup "$pair" "$seed" "$o"
    done
  done
  step "pair $pair: all seeds done"
done
step "final runs finished"
