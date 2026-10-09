#!/bin/bash
# LTM-only single-stream selection with frozen batch normalization (decided 6 Oct 2026; Notes/experiments/plan.md
# section 3, open detail 8). The same procedure as the train-mode selection (pilot_v2_m3.sh, then pilot_v2_rounds.sh):
#   round 1   learning rates 0.001 and 0.0001, single-stream fine classes 3, 4 and 5, pair A, seed 1, validation split,
#             minibatch 1, 96 epochs (8x the draft's 12), every epoch evaluated;
#   rounds 2+ metrics_v2.py --plateau --actions decides (four-set validation curve averaged over fine 3-5, budget on
#             the plateau, runs doubled up to the cap of 32x the draft, 0.03 tie tolerance, one grid step to 0.01 or
#             0.00001 if an edge wins by more than 0.03); repeat until it asks for nothing (at most MAX_ROUNDS).
# Every unit runs with LTM_BN=frozen (run_v2.sh: --bn-mode frozen) into its own tree, never beside the train-mode runs:
#   runs_v2_pilot_bnfrozen/ltm_lr<x>/stream/ltm/pair0_1/seed1/fine<c>/
# A unit that is re-run is first moved to runs_v2_pilot_bnfrozen_archive/<round>/ (never deleted or overwritten). Each
# learning rate runs in a lane of its own (MPS), as the train-mode selection did. Validation only: the selection reads
# results_*_val.txt (metrics_v2.py --tables --part val: no tables, so no test numbers are printed); test numbers of these runs are not
# to be read.
# Relaunching is safe: round 1 skips finished units, and a round whose files exist is refused (set START_ROUND).
# Usage: nohup caffeinate -i bash select_ltm_stream_bnfrozen.sh > runs_v2_pilot_bnfrozen/nohup.log 2>&1 &
set -u
cd "$(dirname "$0")"
PY=${PY:-/Users/gideon/anaconda3/envs/episodic/bin/python}
ROOT=runs_v2_pilot_bnfrozen
ARCHIVE=runs_v2_pilot_bnfrozen_archive
LOG=$ROOT/rounds.log
MAX_ROUNDS=${MAX_ROUNDS:-8}
START_ROUND=${START_ROUND:-2}
export LTM_BN=frozen
mkdir -p "$ROOT"
step() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
step "start: code $(git rev-parse --short HEAD)$(git diff --quiet -- '*.py' '*.sh' || echo ' +dirty'), LTM_BN=$LTM_BN"

archive() {  # unit dir, round label -> moves it under $ARCHIVE/<round>/, never onto an existing copy
  local unit=$1 round=$2 rel dest i=2
  rel=${unit#"$ROOT"/}
  dest="$ARCHIVE/$round/$rel"
  while [ -e "$dest" ]; do dest="$ARCHIVE/$round/$rel.$i"; i=$((i + 1)); done
  mkdir -p "$(dirname "$dest")" && mv "$unit" "$dest" && step "archived $unit -> $dest"
}

run_actions() {  # actions file, round label, lane (the name of one tree)
  local file=$1 round=$2 lane=$3 tree env args unit flag
  while IFS='|' read -r tree env args unit flag; do
    [ -z "$tree" ] && continue
    [ "$lane" = "$(basename "$tree")" ] || continue
    if [ -d "$unit" ] && [ "$round" != round1 ] && ! archive "$unit" "$round"; then
      step "FAILED $round $(basename "$tree"): could not archive $unit, not re-run"; continue
    fi
    if [ "$flag" = grid_extension ] && [ ! -e "$tree/grid_extension" ]; then
      mkdir -p "$tree" && echo "grid-edge learning rate added in $round" > "$tree/grid_extension"
    fi
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

trees() { find "$ROOT" -mindepth 1 -maxdepth 1 -type d -name 'ltm_lr*' | sort; }

# --- round 1: the base grid at 8x the draft's budget ---------------------------------------------------------------
A=$ROOT/actions_round1.txt
if [ ! -e "$A" ]; then
  for lr in 0.001 0.0001; do
    for c in 3 4 5; do
      echo "$ROOT/ltm_lr$lr|EPOCHS=96 LR=$lr|stream ltm \"0 1\" 1 $c|$ROOT/ltm_lr$lr/stream/ltm/pair0_1/seed1/fine$c|" >> "$A"
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
