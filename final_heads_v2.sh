#!/bin/bash
# Final (test-set) runs of the four heads (linear probe, NCM, FlyModel, SDMLP) at their selected learning rates and
# budgets (next_draft_experiments.tex, tab:hparams-selected and tab:hparam-final; selection output
# runs_v2_pilot/selection_final_v3.txt), into runs_v2/ via run_v2.sh (one resumable unit per call; finished units,
# with job.done, are skipped). Order:
#   1. pair A ("0 1"): continual (orders 3 4 5, 4 5 3, 5 3 4), single-stream (fine 3, 4, 5), few-shot (fine 3, 4, 5
#      x N in 1 4 16 64 400)
#   2. pairs B, C, D ("2 3", "5 6", "15 16"): continual
#   3. e40 ablation (LTM=e40): pair A continual, at the e13 selections (not re-selected on e40)
# Seeds 1-5 for every head: the same seed numbers as the STM runs, since in few-shot the seed selects the N images and
# every model must see the same ones (run_v2.sh header, C9; next_draft_experiments.tex Setup).
# Evaluation: continual and single-stream every epoch (as in the selection runs); few-shot at about 96 log-spaced epochs
# (EVAL_POINTS=96, as in the selection runs). Fixed budgets, no early stopping; every run writes its validation curve
# (results_*_val.txt) next to the test one.
# Machine sharing (the M3 also runs the STM sweep on MPS): the driver renices itself to 15 and limits every BLAS/OpenMP
# pool to one thread; LANES (default 1) parallel lanes take every LANES-th unit.
# Usage: nohup caffeinate -i bash final_heads_v2.sh > runs_v2/final_heads.out 2>&1 &
#        DRY=1 bash final_heads_v2.sh          # print every command (run_v2.sh DRY mode), run nothing
#        LANES=2 SEEDS="1 2" ...               # overrides
set -u
cd "$(dirname "$0")"
RUNS_DIR=runs_v2
mkdir -p "$RUNS_DIR"
LOG=$RUNS_DIR/final_heads.log
LANES=${LANES:-1}
SEEDS=${SEEDS:-1 2 3 4 5}
DRY=${DRY:-0}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[ "$DRY" = 1 ] || renice -n 15 -p $$ > /dev/null

step() { if [ "$DRY" = 1 ]; then echo "# $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }

# Selected values. LR: empty = the script's own (NCM has none; FlyModel's Hebbian rate stays at its default 0.2, as in
# the selection). Few-shot budgets in the order of N_LIST.
N_LIST="1 4 16 64 400"
lr_of()        { case $1 in linear) echo 0.001 ;; sdmlp) echo 0.05 ;; *) echo "" ;; esac; }
continual_of() { case $1 in linear) echo 8 ;; ncm) echo 49 ;; flymodel) echo 1 ;; sdmlp) echo 16 ;; esac; }
stream_of()    { case $1 in linear) echo 1 ;; ncm) echo 18 ;; flymodel) echo 1 ;; sdmlp) echo 1 ;; esac; }
fewshot_of()   { # model N
  local b
  case $1 in
    linear)   b="1 589 148 36 8" ;;
    ncm)      b="1 702 521 77 17" ;;
    flymodel) b="1 1 1 3 1" ;;
    sdmlp)    b="1 702 1 58 14" ;;
  esac
  local i=0 n
  for n in $N_LIST; do i=$((i + 1)); [ "$n" = "$2" ] && { echo "$b" | awk -v k=$i '{print $k}'; return; }; done
}
HEADS="linear ncm flymodel sdmlp"
ORDERS=("3 4 5" "4 5 3" "5 3 4")

# --- the unit list, in order: "LTM|setting|model|pair|seed|arg5|N|EPOCHS|LR|EVAL_POINTS" -------------------------
UNITS=()
add() { UNITS+=("$1|$2|$3|$4|$5|$6|$7|$8|$9|${10}"); }
continual_units() { # LTM pair
  local s m o
  for s in $SEEDS; do for m in $HEADS; do for o in "${ORDERS[@]}"; do
    add "$1" continual $m "$2" $s "$o" "" "$(continual_of $m)" "$(lr_of $m)" ""
  done; done; done
}
continual_units e13 "0 1"
for s in $SEEDS; do for m in $HEADS; do for c in 3 4 5; do
  add e13 stream $m "0 1" $s $c "" "$(stream_of $m)" "$(lr_of $m)" ""
done; done; done
for s in $SEEDS; do for m in $HEADS; do for c in 3 4 5; do for n in $N_LIST; do
  add e13 fewshot $m "0 1" $s $c $n "$(fewshot_of $m $n)" "$(lr_of $m)" 96
done; done; done; done
for p in "2 3" "5 6" "15 16"; do continual_units e13 "$p"; done
continual_units e40 "0 1"

run_unit() { # one unit string
  local ltm setting model pair seed arg5 n epochs lr pts desc
  IFS='|' read -r ltm setting model pair seed arg5 n epochs lr pts <<< "$1"
  desc="LTM=$ltm $setting $model \"$pair\" seed $seed ${arg5:+\"$arg5\"} ${n:+N=$n} EPOCHS=$epochs${lr:+ LR=$lr}${pts:+ EVAL_POINTS=$pts}"
  [ -n "$epochs" ] || { step "FAILED (no budget) $desc"; return; }
  if [ "$DRY" = 1 ]; then
    echo "# $desc"
    env DRY=1 LTM="$ltm" EPOCHS="$epochs" ${lr:+LR=$lr} ${pts:+EVAL_POINTS=$pts} \
      bash run_v2.sh "$setting" "$model" "$pair" "$seed" ${arg5:+"$arg5"} ${n:+"$n"}
    return
  fi
  if env LTM="$ltm" EPOCHS="$epochs" ${lr:+LR=$lr} ${pts:+EVAL_POINTS=$pts} \
      bash run_v2.sh "$setting" "$model" "$pair" "$seed" ${arg5:+"$arg5"} ${n:+"$n"} \
      >> "$RUNS_DIR/final_heads_units.log" 2>&1 < /dev/null; then
    step "done   $desc"
  else
    step "FAILED $desc (see $RUNS_DIR/final_heads_units.log and the unit's job.log)"
  fi
}

lane() { # lane index
  local i
  for ((i = $1; i < ${#UNITS[@]}; i += LANES)); do run_unit "${UNITS[$i]}"; done
}

step "final heads start: ${#UNITS[@]} units, lanes $LANES, seeds $SEEDS, code $(git rev-parse --short HEAD)$(git diff --quiet -- '*.py' '*.sh' || echo ' +dirty')"
for ((l = 0; l < LANES; l++)); do lane $l & done
wait
step "final heads finished"
