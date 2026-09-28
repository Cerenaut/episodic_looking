#!/bin/bash
# UNTESTED DRAFT (28 Sep): written for the differentiable actor's selection, then deferred (Gideon). Needs a smoke run,
# a reaper keyed on each unit's job.done, a launcher and a review before any use.
# One pod job of the v2 CLS/STM selection (results_v2.tex, Hyperparameter selection): runs the units of an actions
# file through run_v2.sh, NPROC at a time, then leaves a sentinel for the reaper (sky/v2_reap.sh).
#
# Actions file: one unit per line, the format metrics_v2.py --actions writes:
#   tree path|environment (KEY=value words)|run_v2.sh arguments|unit dir|flag
# Tree paths are relative to the repo (e.g. runs_v2_pilot_stm/actor_lr0.01), the same on the pod and on the Mac.
# Pre-training lines run first (all together), the rest after them, NPROC at a time.
# If ~/stm_pretrain/ is mounted (a finished pre-training unit dir), it is copied into every tree of the file first, so
# all units of a model share one pre-trained checkpoint.
# Every unit runs with SAVE_STM=1. Sentinels (tested by the reaper with test -f, never parsed):
#   runs_local/v2_pod/JOB_COMPLETE  every unit of the file has job.done
#   runs_local/v2_pod/JOB_FAILED    the job ended and some unit has not
# Usage (on the pod): bash sky/v2_pod_job.sh ~/actions.txt
set -u
cd "$(dirname "$0")/.."
export PY=${PY:-.venv/bin/python}
ACTIONS=${1:?actions file}
NPROC=${NPROC:-3}
LOG=runs_local/v2_pod
mkdir -p "$LOG"
rm -f "$LOG/JOB_COMPLETE" "$LOG/JOB_FAILED"
step() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG/progress.log"; }

[ -s "$ACTIONS" ] || { step "JOB FAILED: empty or missing actions file $ACTIONS"; touch "$LOG/JOB_FAILED"; exit 1; }
cp "$ACTIONS" "$LOG/actions.txt"
step "job starting: $(grep -c . "$ACTIONS") units, NPROC=$NPROC, code ${COMMIT:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG/progress.log" || true

if [ -d "$HOME/stm_pretrain" ]; then
  for tree in $(cut -d'|' -f1 "$ACTIONS" | sort -u); do
    model=$(basename "$tree"); model=${model%%_lr*}
    dest=$tree/pretrain/$model/pair0_1/seed1
    if [ ! -e "$dest" ]; then
      mkdir -p "$(dirname "$dest")" && cp -R "$HOME/stm_pretrain" "$dest" && step "pre-training copied into $dest"
    fi
  done
fi

run_line() {  # one actions line
  local tree env args unit flag
  IFS='|' read -r tree env args unit flag <<< "$1"
  eval "set -- $args"
  step "start  $env $args"
  # shellcheck disable=SC2086  # env holds KEY=value words
  if env RUNS="$tree" SAVE_STM=1 $env bash run_v2.sh "$@" >> "$LOG/$(echo "$unit" | tr '/' '_').log" 2>&1 < /dev/null \
      && [ -f "$unit/job.done" ]; then
    step "done   $env $args"
  else
    step "FAILED $env $args"
  fi
}

pool() {  # lines on stdin, NPROC at a time
  local line
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do sleep 5; done
    run_line "$line" &
  done
  wait
}

grep '|pretrain ' "$ACTIONS" | NPROC=99 pool
grep -v '|pretrain ' "$ACTIONS" | pool

missing=0
while IFS='|' read -r tree env args unit flag; do
  [ -z "$tree" ] && continue
  [ -f "$unit/job.done" ] || { missing=$((missing + 1)); step "missing job.done: $unit"; }
done < "$ACTIONS"
if [ "$missing" = 0 ]; then touch "$LOG/JOB_COMPLETE"; step "JOB DONE"; else touch "$LOG/JOB_FAILED"; step "JOB FAILED: $missing units unfinished"; fi
