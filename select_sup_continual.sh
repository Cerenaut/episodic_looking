#!/bin/bash
# Continual selection for the supervised CLS/STM (sup, cifar_main_stm_supervised.py; next_draft_experiments.tex, Setup),
# by the heads' procedure (pilot_v2_m3.sh, then pilot_v2_rounds.sh):
#   pre-training  once: fine 1,2, pair A, seed 1, validation split, 12 epochs at lr 0.01 (run_v2.sh defaults), into
#                 $ROOT/shared, then copied into every learning-rate tree (grid steps included), as the RL single-stream
#                 sweep did; the selection does not touch it (fine 1,2 validation is contaminated)
#   round 1       learning rates 0.003, 0.01, 0.03 (around the differentiable actor's 0.01), continual, pair A, seed 1,
#                 orders 345/453/534, validation split, minibatch 16, 96 epochs per phase (8x the draft's 12), every
#                 epoch evaluated
#   rounds 2+     metrics_v2.py --plateau --actions decides (continual curve: the new classes seen so far, by epoch
#                 within the phase; budget on the plateau, runs doubled up to the cap of 32x the draft, 0.03 tie
#                 tolerance, one grid step if an edge wins by more than 0.03); repeat until it asks for nothing
# Trees: runs_v2_pilot_sup/sup_lr<x>/continual/sup/pair0_1/seed1/order<a>_<b>_<c>/. A unit that is re-run is first moved
# to runs_v2_pilot_sup_archive/<round>/ (never deleted or overwritten). Each learning rate runs in a lane of its own
# (MPS). Validation only: metrics_v2.py --tables --part val (no tables, so no test numbers are printed); test numbers of
# these runs are not to be read.
# Relaunching is safe: finished units are skipped, and a round whose files exist is refused (set START_ROUND).
# Usage: nohup caffeinate -i bash select_sup_continual.sh > runs_v2_pilot_sup/nohup.log 2>&1 &
set -u
cd "$(dirname "$0")"
PY=${PY:-/Users/gideon/anaconda3/envs/episodic/bin/python}
ROOT=${ROOT:-runs_v2_pilot_sup}
ARCHIVE=${ARCHIVE:-runs_v2_pilot_sup_archive}
ROUND1_EPOCHS=${ROUND1_EPOCHS:-96}  # smoke tests only: a tiny budget
LOG=$ROOT/rounds.log
MAX_ROUNDS=${MAX_ROUNDS:-8}
START_ROUND=${START_ROUND:-2}
SHARED=$ROOT/shared
mkdir -p "$ROOT"
step() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
step "start: code $(git rev-parse --short HEAD)$(git diff --quiet -- '*.py' '*.sh' || echo ' +dirty')"

archive() {  # unit dir, round label -> moves it under $ARCHIVE/<round>/, never onto an existing copy
  local unit=$1 round=$2 rel dest i=2
  rel=${unit#"$ROOT"/}
  dest="$ARCHIVE/$round/$rel"
  while [ -e "$dest" ]; do dest="$ARCHIVE/$round/$rel.$i"; i=$((i + 1)); done
  mkdir -p "$(dirname "$dest")" && mv "$unit" "$dest" && step "archived $unit -> $dest"
}

prepare_tree() {  # tree: give it the shared STM pre-training and baseline (needed by run_v2.sh), once
  local tree=$1 s
  for s in pretrain baseline; do
    if [ ! -e "$tree/$s/sup/pair0_1/seed1/job.done" ]; then
      mkdir -p "$tree/$s/sup/pair0_1" && cp -R "$SHARED/$s/sup/pair0_1/seed1" "$tree/$s/sup/pair0_1/" \
        && step "copied the shared $s into $tree"
    fi
  done
}

run_actions() {  # actions file, round label, lane (the name of one tree)
  local file=$1 round=$2 lane=$3 tree env args unit flag
  while IFS='|' read -r tree env args unit flag; do
    [ -z "$tree" ] && continue
    [ "$lane" = "$(basename "$tree")" ] || continue
    if [ -f "$unit/job.done" ] && [ "$round" = round1 ]; then continue; fi
    if [ -d "$unit" ] && ! archive "$unit" "$round"; then
      step "FAILED $round $(basename "$tree"): could not archive $unit, not re-run"; continue
    fi
    if [ "$flag" = grid_extension ] && [ ! -e "$tree/grid_extension" ]; then
      mkdir -p "$tree" && echo "grid-edge learning rate added in $round" > "$tree/grid_extension"
    fi
    prepare_tree "$tree"
    eval "set -- $args"
    # shellcheck disable=SC2086  # env holds KEY=value words
    if env RUNS="$tree" $env bash run_v2.sh "$@" >> "$tree.log" 2>&1 < /dev/null && [ -f "$unit/job.done" ]; then
      step "done   $round $(basename "$tree"): $env $args"
    else
      step "FAILED $round $(basename "$tree"): $env $args ($tree.log)"
    fi
  done < "$file"
}

run_round() {  # actions file, round label: each learning rate in a lane of its own
  local t
  for t in $(cut -d'|' -f1 "$1" | sort -u); do run_actions "$1" "$2" "$(basename "$t")" & done
  wait
}

trees() { find "$ROOT" -mindepth 1 -maxdepth 1 -type d -name 'sup_lr*' | sort; }

# --- STM pre-training and baseline, once ---------------------------------------------------------------------------
for s in pretrain baseline; do
  RUNS=$SHARED bash run_v2.sh $s sup "0 1" 1 >> "$SHARED.log" 2>&1 < /dev/null
  [ -f "$SHARED/$s/sup/pair0_1/seed1/job.done" ] || { step "ABORT: shared $s failed ($SHARED.log)"; exit 1; }
  step "shared $s done"
done

# --- round 1: the base grid at 8x the draft's budget ---------------------------------------------------------------
A=$ROOT/actions_round1.txt
if [ ! -e "$A" ]; then
  for lr in 0.003 0.01 0.03; do
    for o in "3 4 5" "4 5 3" "5 3 4"; do
      echo "$ROOT/sup_lr$lr|EPOCHS=$ROUND1_EPOCHS LR=$lr|continual sup \"0 1\" 1 \"$o\"|$ROOT/sup_lr$lr/continual/sup/pair0_1/seed1/order${o// /_}|" >> "$A"
    done
  done
fi
step "round 1: $(wc -l < "$A" | tr -d ' ') runs (finished units are skipped)"
run_round "$A" round1

# --- rounds 2+: the selection rule decides --------------------------------------------------------------------------
for r in $(seq "$START_ROUND" "$MAX_ROUNDS"); do
  A=$ROOT/actions_round$r.txt
  if [ -e "$A" ] || [ -e "$ROOT/selection_round$r.txt" ]; then step "ABORT: round $r files exist (set START_ROUND past them)"; exit 1; fi
  # shellcheck disable=SC2046  # one argument per tree
  $PY metrics_v2.py --tables --part val --plateau --runs $(trees) --actions "$A" --selection-latex "$ROOT/selection_table_round$r.tex" \
      > "$ROOT/selection_round$r.txt" 2> "$ROOT/selection_round$r.err" \
    || { step "ABORT: metrics_v2.py failed in round $r ($ROOT/selection_round$r.err)"; exit 1; }
  n=$(wc -l < "$A" | tr -d ' ')
  step "round $r: the selection rule asks for $n runs (selection: $ROOT/selection_round$r.txt)"
  [ "$n" = 0 ] && break
  run_round "$A" "round$r"
done
# shellcheck disable=SC2046
$PY metrics_v2.py --tables --part val --plateau --runs $(trees) --selection-latex "$ROOT/selection_table_final.tex" \
    > "$ROOT/selection_final.txt" 2> "$ROOT/selection_final.err"
step "finished: $ROOT/selection_final.txt, $ROOT/selection_table_final.tex"
