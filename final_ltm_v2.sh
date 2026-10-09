#!/bin/bash
# Final (test-set) runs of LTM-only fine-tuning at its selected learning rates and budgets (next_draft_experiments.tex,
# tab:hparams-selected; selection outputs runs_v2_pilot/selection_final_v3.txt and, for frozen-BN single-stream,
# runs_v2_pilot_bnfrozen/selection_final.txt), via run_v2.sh (one resumable unit per call; finished units, with
# job.done, are skipped). Same pattern as final_heads_v2.sh. Units, in order:
#   0. baselines (the LTM itself before any training; deterministic, one per pair, seed ignored):
#      runs_v2/baseline/ltm/pair{0_1,2_3,5_6,15_16}, runs_v2/baseline_e40/ltm/pair0_1, and
#      runs_v2_bnfrozen/baseline/ltm/pair0_1 (run with LTM_BN=train and RUNS=runs_v2_bnfrozen, since run_v2.sh refuses
#      LTM_BN=frozen for a baseline; it is an evaluation in eval mode, so BN mode does not enter; it gives the frozen
#      tree its own starting point, as metrics_v2.py and final_results_v2.py read the baseline from the same tree)
#   1. pair A ("0 1"): continual (orders 3 4 5, 4 5 3, 5 3 4; lr 0.0001, 10 epochs per phase); single-stream fine 3, 4, 5
#      with BN frozen (LTM_BN=frozen -> runs_v2_bnfrozen/; lr 0.00001, 1 epoch: the LTM-only reference) and in training
#      mode (runs_v2/; lr 0.0001, 20 epochs: comparison row); few-shot fine 3, 4, 5 x N in 1 4 16 64 400 (lr 0.001 at
#      N = 1, 4, else 0.0001; budgets 25246, 24, 7597, 236, 13)
#   2. pairs B, C, D ("2 3", "5 6", "15 16"): continual
#   3. e40 ablation (LTM=e40): pair A continual, at the e13 selections (not re-selected on e40)
# Continual and few-shot keep training-mode BN at minibatch 16 (plan.md section 3, detail 8).
# Seeds 1-5: the same seed numbers as every other model, since in few-shot the seed selects the N images (C9).
# Evaluation as in the selection runs: continual and single-stream every epoch; few-shot at about 96 log-spaced epochs
# (EVAL_POINTS=96). Fixed budgets, no early stopping; every run writes its validation curve next to the test one.
# Lanes: NPROC lanes (default 3, MPS) take units from one shared queue (a unit is claimed by an atomic mkdir under a
# per-launch claims directory), so a long unit does not hold up a lane's share. Logs: runs_v2/final_ltm.log (one line per
# unit), runs_v2/final_ltm_units.log (run_v2.sh output); each unit's own job.log, code_commit.txt and job.done.
# Usage: nohup caffeinate -i bash final_ltm_v2.sh > runs_v2/final_ltm.out 2>&1 &
#        DRY=1 bash final_ltm_v2.sh          # print every command (run_v2.sh DRY mode), run nothing
#        NPROC=2 SEEDS="1 2" ...             # overrides
set -u
cd "$(dirname "$0")"
LOGDIR=runs_v2
mkdir -p "$LOGDIR"
LOG=$LOGDIR/final_ltm.log
NPROC=${NPROC:-3}
SEEDS=${SEEDS:-1 2 3 4 5}
DRY=${DRY:-0}

step() { if [ "$DRY" = 1 ]; then echo "# $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }

# Selected values (tab:hparams-selected)
CONT_LR=0.0001;    CONT_EP=10
STREAM_FROZEN_LR=0.00001; STREAM_FROZEN_EP=1
STREAM_TRAIN_LR=0.0001;   STREAM_TRAIN_EP=20
N_LIST="1 4 16 64 400"
fewshot_lr() { case $1 in 1|4) echo 0.001 ;; *) echo 0.0001 ;; esac; }
fewshot_ep() { case $1 in 1) echo 25246 ;; 4) echo 24 ;; 16) echo 7597 ;; 64) echo 236 ;; 400) echo 13 ;; esac; }
ORDERS=("3 4 5" "4 5 3" "5 3 4")

# --- the unit list: "LTM_BN|RUNS|LTM|setting|pair|seed|arg5|N|EPOCHS|LR|EVAL_POINTS" ---------------------------------
UNITS=()
add() { UNITS+=("$1|$2|$3|$4|$5|$6|$7|$8|$9|${10}|${11}"); }
for p in "0 1" "2 3" "5 6" "15 16"; do add train runs_v2 e13 baseline "$p" 0 "" "" "" "" ""; done
add train runs_v2 e40 baseline "0 1" 0 "" "" "" "" ""
add train runs_v2_bnfrozen e13 baseline "0 1" 0 "" "" "" "" ""
continual_units() { # LTM pair
  local s o
  for s in $SEEDS; do for o in "${ORDERS[@]}"; do
    add train runs_v2 "$1" continual "$2" $s "$o" "" $CONT_EP $CONT_LR ""
  done; done
}
continual_units e13 "0 1"
for s in $SEEDS; do for c in 3 4 5; do
  add frozen runs_v2_bnfrozen e13 stream "0 1" $s $c "" $STREAM_FROZEN_EP $STREAM_FROZEN_LR ""
done; done
for s in $SEEDS; do for c in 3 4 5; do
  add train runs_v2 e13 stream "0 1" $s $c "" $STREAM_TRAIN_EP $STREAM_TRAIN_LR ""
done; done
for s in $SEEDS; do for c in 3 4 5; do for n in $N_LIST; do
  add train runs_v2 e13 fewshot "0 1" $s $c $n "$(fewshot_ep $n)" "$(fewshot_lr $n)" 96
done; done; done
for p in "2 3" "5 6" "15 16"; do continual_units e13 "$p"; done
continual_units e40 "0 1"

run_unit() { # one unit string
  local bn runs ltm setting pair seed arg5 n epochs lr pts desc
  IFS='|' read -r bn runs ltm setting pair seed arg5 n epochs lr pts <<< "$1"
  desc="LTM_BN=$bn RUNS=$runs LTM=$ltm $setting ltm \"$pair\" seed $seed ${arg5:+\"$arg5\"} ${n:+N=$n}${epochs:+ EPOCHS=$epochs}${lr:+ LR=$lr}${pts:+ EVAL_POINTS=$pts}"
  if [ "$setting" != baseline ] && [ -z "$epochs" ]; then step "FAILED (no budget) $desc"; return; fi
  if [ "$DRY" = 1 ]; then
    echo "# $desc"
    env DRY=1 LTM_BN="$bn" RUNS="$runs" LTM="$ltm" ${epochs:+EPOCHS=$epochs} ${lr:+LR=$lr} ${pts:+EVAL_POINTS=$pts} \
      bash run_v2.sh "$setting" ltm "$pair" "$seed" ${arg5:+"$arg5"} ${n:+"$n"}
    return
  fi
  if env LTM_BN="$bn" RUNS="$runs" LTM="$ltm" ${epochs:+EPOCHS=$epochs} ${lr:+LR=$lr} ${pts:+EVAL_POINTS=$pts} \
      bash run_v2.sh "$setting" ltm "$pair" "$seed" ${arg5:+"$arg5"} ${n:+"$n"} \
      >> "$LOGDIR/final_ltm_units.log" 2>&1 < /dev/null; then
    step "done   $desc"
  else
    step "FAILED $desc (see $LOGDIR/final_ltm_units.log and the unit's job.log)"
  fi
}

if [ "$DRY" = 1 ]; then
  for u in "${UNITS[@]}"; do run_unit "$u"; done
  echo "# ${#UNITS[@]} units"; exit 0
fi

CLAIMS=$LOGDIR/final_ltm_claims/$(date '+%Y%m%d_%H%M%S')_$$
mkdir -p "$CLAIMS"
lane() { # lane index: take the next unclaimed unit from the shared queue
  local i
  for ((i = 0; i < ${#UNITS[@]}; i++)); do
    mkdir "$CLAIMS/$i" 2> /dev/null || continue
    run_unit "${UNITS[$i]}"
  done
}

step "final LTM-only start: ${#UNITS[@]} units, NPROC $NPROC, seeds $SEEDS, code $(git rev-parse --short HEAD)$(git diff --quiet -- '*.py' '*.sh' || echo ' +dirty'), claims $CLAIMS"
for ((l = 0; l < NPROC; l++)); do lane $l & done
wait
step "final LTM-only finished: $(grep -c "FAILED" "$LOG" 2>/dev/null || true) FAILED lines in $LOG (all launches)"
