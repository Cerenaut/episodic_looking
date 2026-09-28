#!/bin/bash
# Later rounds of the v2 M3 pilot (hyperparameter selection, results_v2.tex Setup), after pilot_v2_m3.sh:
#   round 1b  selection now averages fine classes 3, 4 and 5 in the single-stream and few-shot settings, with few-shot
#             evaluated at log-spaced epochs: run single-stream fine 4 and 5 (96 epochs, every epoch evaluated, as fine
#             3 was), and few-shot fine 3, 4, 5 at every N with EVAL_POINTS=96 (the fine-3 few-shot units of round 1,
#             evaluated at even intervals, are archived and re-run). Runs once (marker round1b.done).
#   rounds 2+ apply the selection rule (metrics_v2.py --plateau --actions) and run what it asks for: runs that have not
#             plateaued, doubled up to the cap; unresolved ones re-run with log-spaced evaluation; one grid step where
#             an edge learning rate clearly wins. Repeat until nothing is left (at most MAX_ROUNDS).
# A unit that is re-run is first moved to runs_v2_pilot_archive/<round>/ (never deleted or overwritten; a second copy
# gets a numbered name). Heads and LTM-only run in two parallel lanes. Waits for pilot_v2_m3.sh to exit, then restores
# run_v2.sh to the committed version (the pilot ran an older copy: a running bash script must not be edited) and
# refuses to continue if that version lacks EVAL_POINTS. Continual confirmation runs are not part of this script.
# Relaunching is safe: round 1b is skipped once done, and later rounds are recomputed from the runs.
# Usage: nohup bash pilot_v2_rounds.sh > runs_v2_pilot/rounds_nohup.log 2>&1 &
set -u
cd "$(dirname "$0")"
PY=${PY:-/Users/gideon/anaconda3/envs/episodic/bin/python}
ROOT=runs_v2_pilot
ARCHIVE=runs_v2_pilot_archive
LOG=$ROOT/rounds.log
MAX_ROUNDS=${MAX_ROUNDS:-6}
NS=(1 4 16 64 400)
step() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

step "rounds: waiting for pilot_v2_m3.sh to finish"
while pgrep -f "bash pilot_v2_m3.sh" > /dev/null; do sleep 60; done
if ! git diff --quiet -- run_v2.sh; then git checkout -- run_v2.sh && step "run_v2.sh restored to the committed version"; fi
if ! grep -q EVAL_POINTS run_v2.sh; then step "ABORT: run_v2.sh has no EVAL_POINTS support (restore failed?)"; exit 1; fi
step "rounds start: code $(git rev-parse --short HEAD)$(git diff --quiet -- '*.py' '*.sh' || echo ' +dirty')"

archive() {  # unit dir, round label -> moves it under $ARCHIVE/<round>/, never onto an existing copy
  local unit=$1 round=$2 rel dest i=2
  rel=${unit#"$ROOT"/}
  dest="$ARCHIVE/$round/$rel"
  while [ -e "$dest" ]; do dest="$ARCHIVE/$round/$rel.$i"; i=$((i + 1)); done
  mkdir -p "$(dirname "$dest")" && mv "$unit" "$dest" && step "archived $unit -> $dest"
}

run_actions() {  # actions file, round label, lane (ltm | heads)
  local file=$1 round=$2 lane=$3 tree env args unit flag
  while IFS='|' read -r tree env args unit flag; do
    [ -z "$tree" ] && continue
    case "$(basename "$tree")" in ltm*) [ "$lane" = ltm ] || continue ;; *) [ "$lane" = heads ] || continue ;; esac
    if [ -d "$unit" ] && ! archive "$unit" "$round"; then
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

run_round() {  # actions file, round label
  run_actions "$1" "$2" heads & run_actions "$1" "$2" ltm & wait
}

trees() { find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort; }

# --- round 1b: fine classes 4 and 5, and log-spaced few-shot ------------------------------------------------------
if [ ! -f "$ROOT/round1b.done" ]; then
  A=$ROOT/actions_round1b.txt
  : > "$A"
  for tree in $(trees); do
    [ -e "$tree/grid_extension" ] && continue
    name=$(basename "$tree"); model=${name%%_lr*}
    lr=""; [ "$name" != "$model" ] && lr=" LR=${name#*_lr}"
    for c in 4 5; do
      echo "$tree|EPOCHS=96$lr|stream $model \"0 1\" 1 $c|$tree/stream/$model/pair0_1/seed1/fine$c|" >> "$A"
    done
    for n in "${NS[@]}"; do
      for c in 3 4 5; do
        echo "$tree|EPOCHS=$(( 8 * (6250 / n) )) EVAL_POINTS=96$lr|fewshot $model \"0 1\" 1 $c $n|$tree/fewshot/$model/pair0_1/seed1/fine${c}_n$n|" >> "$A"
      done
    done
  done
  step "round 1b: $(wc -l < "$A" | tr -d ' ') runs (single-stream fine 4, 5; log-spaced few-shot fine 3, 4, 5)"
  run_round "$A" round1b
  touch "$ROOT/round1b.done"
fi

# --- rounds 2+: the selection rule decides --------------------------------------------------------------------------
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
# shellcheck disable=SC2046
$PY metrics_v2.py --tables --plateau --runs $(trees) --selection-latex "$ROOT/selection_table_final.tex" \
    > "$ROOT/selection_final.txt" 2> "$ROOT/selection_final.err"
step "rounds finished: $ROOT/selection_final.txt, $ROOT/selection_table_final.tex"
