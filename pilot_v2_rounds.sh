#!/bin/bash
# Later rounds of the v2 M3 pilot (hyperparameter selection, results_v2.tex Setup), after pilot_v2_m3.sh:
#   round 1b  re-run few-shot N = 1, 4, 16 of every tree with log-spaced evaluation (EVAL_POINTS=96): in round 1 these
#             curves peaked before the second evaluation;
#   rounds 2+ apply the selection rule (metrics_v2.py --plateau --actions) and run what it asks for: runs that have not
#             plateaued, doubled up to the cap, and one more learning rate where the best one is at the edge of the grid;
#             repeat until nothing is left (at most MAX_ROUNDS).
# A unit that is re-run is first moved to runs_v2_pilot_archive/round<r>/ (results are never deleted or overwritten).
# Heads and LTM-only run in two parallel lanes. Waits for pilot_v2_m3.sh to exit, then restores run_v2.sh to the
# committed version (the pilot ran an older copy, since a running bash script must not be edited). Continual
# confirmation runs at the selected budgets are not part of this script.
# Usage: nohup bash pilot_v2_rounds.sh > runs_v2_pilot/rounds_nohup.log 2>&1 &
set -u
cd "$(dirname "$0")"
PY=${PY:-/Users/gideon/anaconda3/envs/episodic/bin/python}
ROOT=runs_v2_pilot
ARCHIVE=runs_v2_pilot_archive
LOG=$ROOT/rounds.log
MAX_ROUNDS=${MAX_ROUNDS:-6}
step() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

step "rounds: waiting for pilot_v2_m3.sh to finish"
while pgrep -f "bash pilot_v2_m3.sh" > /dev/null; do sleep 60; done
if ! git diff --quiet -- run_v2.sh; then git checkout -- run_v2.sh && step "run_v2.sh restored to the committed version"; fi
step "rounds start: code $(git rev-parse --short HEAD)$(git diff --quiet -- '*.py' '*.sh' || echo ' +dirty')"

run_actions() {  # actions file, round label, lane filter (ltm | heads)
  local file=$1 round=$2 lane=$3 tree env args unit rel
  while IFS='|' read -r tree env args unit; do
    [ -z "$tree" ] && continue
    case "$(basename "$tree")" in ltm*) [ "$lane" = ltm ] || continue ;; *) [ "$lane" = heads ] || continue ;; esac
    if [ -d "$unit" ]; then
      rel=${unit#"$ROOT"/}
      mkdir -p "$ARCHIVE/$round/$(dirname "$rel")"
      mv "$unit" "$ARCHIVE/$round/$rel" && step "archived $unit -> $ARCHIVE/$round/$rel"
    fi
    eval "set -- $args"
    # shellcheck disable=SC2086  # env holds KEY=value words
    if env RUNS="$tree" $env bash run_v2.sh "$@" >> "$tree.log" 2>&1; then
      step "done   $round $(basename "$tree"): $env $args"
    else
      step "FAILED $round $(basename "$tree"): $env $args ($tree.log)"
    fi
  done < "$file"
}

run_round() {  # actions file, round label
  run_actions "$1" "$2" heads & run_actions "$1" "$2" ltm & wait
}

trees() { find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort; }

# --- round 1b: log-spaced few-shot at small N ----------------------------------------------------------------------
A=$ROOT/actions_round1b.txt
: > "$A"
for tree in $(trees); do
  name=$(basename "$tree"); model=${name%%_lr*}
  lr=""; [ "$name" != "$model" ] && lr=" LR=${name#*_lr}"
  for n in 1 4 16; do
    echo "$tree|EPOCHS=$(( 8 * (6250 / n) )) EVAL_POINTS=96$lr|fewshot $model \"0 1\" 1 3 $n|$tree/fewshot/$model/pair0_1/seed1/fine3_n$n" >> "$A"
  done
done
step "round 1b: $(wc -l < "$A") log-spaced few-shot runs"
run_round "$A" round1b

# --- rounds 2+: the selection rule decides ----------------------------------------------------------------------------
for r in $(seq 2 "$MAX_ROUNDS"); do
  A=$ROOT/actions_round$r.txt
  # shellcheck disable=SC2046  # one argument per tree
  $PY metrics_v2.py --tables --plateau --runs $(trees) --actions "$A" --selection-latex "$ROOT/selection_table_round$r.tex" \
      > "$ROOT/selection_round$r.txt" 2> "$ROOT/selection_round$r.err"
  n=$(wc -l < "$A" | tr -d ' ')
  step "round $r: the selection rule asks for $n runs (selection: $ROOT/selection_round$r.txt)"
  [ "$n" = 0 ] && break
  run_round "$A" "round$r"
done
$PY metrics_v2.py --tables --plateau --runs $(trees) --selection-latex "$ROOT/selection_table_final.tex" \
    > "$ROOT/selection_final.txt" 2> "$ROOT/selection_final.err"
step "rounds finished: $ROOT/selection_final.txt, $ROOT/selection_table_final.tex"
