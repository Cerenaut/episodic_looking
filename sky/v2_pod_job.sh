#!/bin/bash
# One pod job of the v2 CLS/STM runs: runs the units of an actions file through run_v2.sh, NPROC at a time
# (pre-training units first), and leaves a sentinel for the reaper (sky/v2_reap.sh). Launched by sky/v2_launch.sh
# through sky/v2_pod.yaml; can also be run by hand on any Linux/CUDA machine.
#
# Actions file: one unit per line, the format metrics_v2.py --actions writes:
#   tree path|environment (KEY=value words)|run_v2.sh arguments|unit dir|flag
#   e.g. runs_v2|EPOCHS=12|continual rl "0 1" 1 "3 4 5"|runs_v2/continual/rl/pair0_1/seed1/order3_4_5|
# Tree paths are relative to the repo, the same on the pod and on the Mac, so pulled results drop in place.
# The environment field is shell words: quote a value with spaces (EXTRA="--ltm-obs-cache check").
# sky/v2_launch.sh validates the file (unit dir = run_v2.sh's layout) before launching.
#
# Every unit runs with SAVE_STM=1 (STM checkpoints, ~44 MB each, are kept in the unit dir) unless its environment
# field sets SAVE_STM=0 (sky/v2_verify.sh then does not require a checkpoint of it; the pre-training checkpoint
# stm_pretrain.pth is always written, since the unit's other runs need it). Reserved, refused in the environment
# field: RUNS, PY, DRY, PRINT_UNIT, UNIT_TIMEOUT (the pod job sets them, or they would change what the unit is).
# Each unit runs under `timeout --kill-after=120 $UNIT_TIMEOUT` (GNU timeout signals its whole process group, so the
# Python run dies with run_v2.sh); a unit that times out is FAILED like any other, so the job ends JOB_FAILED.
# Outputs, under the round directory (same relative path on the pod and the Mac):
#   $ROUND/pods/$JOB/progress.log      one line per unit start/end, with elapsed seconds
#   $ROUND/pods/$JOB/<unit>.log        run_v2.sh's stdout/stderr per unit (the unit's own job.log has the run)
#   $ROUND/pods/$JOB/actions.txt env.txt code_commit.txt
#   $ROUND/pods/$JOB/manifest.tsv      size<TAB>sha256<TAB>path of every file of every unit, except paths with a
#                                      component starting with '.' (run_v2.sh's .lock dir; sky/v2_verify.sh ignores the
#                                      same paths, e.g. a .DS_Store on the Mac)
# Sentinels (tested by the reaper with test -f, never parsed):
#   $ROUND/pods/$JOB/JOB_COMPLETE  every unit has job.done and the manifest is written
#   $ROUND/pods/$JOB/JOB_FAILED    the job ended any other way (written by the EXIT trap, so also on a crash)
#
# Environment: ROUND (round directory, relative to the repo), JOB (normally the cluster name), NPROC (default 3),
# PY (default .venv/bin/python), COMMIT (the launcher's git commit), ACTIONS_B64 (the actions file, base64; used
# when no file argument is given), UNIT_TIMEOUT (required: per-unit cap in timeout(1) syntax, e.g. 6h; see
# sky/v2_launch.sh for suggested values), RUN_V2 (default run_v2.sh; tests only), REQUIRE_CUDA (default 1).
# Usage (on the pod): bash sky/v2_pod_job.sh [actions file]
# Written for bash 3.2 as well as 5 (no mapfile, no wait -n, no find -printf), so it can be tested on the Mac.
set -u
cd "$(dirname "$0")/.."
export PY=${PY:-.venv/bin/python}
NPROC=${NPROC:-3}
ROUND=${ROUND:-runs_local/v2_round/manual}
JOB=${JOB:-$(hostname -s 2>/dev/null || echo job)}
RUN_V2=${RUN_V2:-run_v2.sh}
UNIT_TIMEOUT=${UNIT_TIMEOUT:-}
LOG=$ROUND/pods/$JOB
mkdir -p "$LOG" || exit 1
rm -f "$LOG/JOB_COMPLETE" "$LOG/JOB_FAILED"
finish() {  # EXIT trap: anything other than a clean, complete end leaves JOB_FAILED
  [ -f "$LOG/JOB_COMPLETE" ] && return
  touch "$LOG/JOB_FAILED"
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] JOB_FAILED sentinel written" >> "$LOG/progress.log"
}
trap finish EXIT
step() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG/progress.log"; }
fail() { step "JOB FAILED: $*"; exit 1; }
sha256() {  # $1 = file -> hex digest (sha256sum on Linux and recent macOS, shasum elsewhere); read on stdin, so no
  # filename escaping
  if command -v sha256sum >/dev/null; then sha256sum < "$1" | cut -d' ' -f1; else shasum -a 256 < "$1" | cut -d' ' -f1; fi
}
[[ "$UNIT_TIMEOUT" =~ ^[1-9][0-9]*[smhd]?$ ]] || fail "UNIT_TIMEOUT must be set to a positive duration (e.g. 6h), not '$UNIT_TIMEOUT'"
command -v timeout >/dev/null || fail "no timeout(1) on this machine"

# --- the actions file ---------------------------------------------------------------------------
if [ $# -ge 1 ]; then
  [ -s "$1" ] || fail "empty or missing actions file $1"
  cp "$1" "$LOG/actions.txt" || fail "cannot copy $1"
elif [ -n "${ACTIONS_B64:-}" ]; then
  printf '%s' "$ACTIONS_B64" | base64 -d > "$LOG/actions.txt" 2>/dev/null || fail "ACTIONS_B64 does not decode"
else
  fail "no actions file argument and no ACTIONS_B64"
fi
A=$LOG/actions.txt
[ -s "$A" ] || fail "actions file is empty"
n=0
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  n=$((n + 1))
  [ "$(printf '%s' "$line" | tr -cd '|' | wc -c | tr -d ' ')" = 4 ] || fail "line $n: not 5 |-separated fields: $line"
  IFS='|' read -r tree env args unit flag <<< "$line"
  case "$tree" in ''|/*|*..*) fail "line $n: tree must be a relative path without '..': $tree" ;; esac
  case "$unit" in "$tree"/*) ;; *) fail "line $n: unit dir $unit is not under tree $tree" ;; esac
  eval "envw=($env)" 2>/dev/null || fail "line $n: environment field is not shell words: $env"
  for w in ${envw[@]+"${envw[@]}"}; do
    case "$w" in
      RUNS=*|PY=*|DRY=*|PRINT_UNIT=*|UNIT_TIMEOUT=*) fail "line $n: reserved key in the environment field: $w" ;;
      SAVE_STM=*) case "$w" in SAVE_STM=0|SAVE_STM=1) ;; *) fail "line $n: SAVE_STM must be 0 or 1: $w" ;; esac ;;
    esac
  done
done < "$A"
[ "$n" -gt 0 ] || fail "actions file has no units"

# --- environment record and pre-flight -----------------------------------------------------------
{
  echo "job $JOB  round $ROUND  NPROC $NPROC  PY $PY  UNIT_TIMEOUT $UNIT_TIMEOUT"
  echo "launcher commit: ${COMMIT:-unknown}"
  echo "pod git HEAD: $(git rev-parse HEAD 2>/dev/null || echo 'no .git on the pod')"
  git status --porcelain -- '*.py' '*.sh' 2>/dev/null
} > "$LOG/code_commit.txt"
{
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  uname -a
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  df -h /dev/shm 2>/dev/null | tail -1
  nproc 2>/dev/null
  $PY -c "import sys, torch; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>&1
} > "$LOG/env.txt"
step "job starting: $n units, NPROC=$NPROC, UNIT_TIMEOUT=$UNIT_TIMEOUT, launcher commit ${COMMIT:-unknown}"
step "gpu: $( (command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader | head -1) || echo none)"
if [ "${REQUIRE_CUDA:-1}" = 1 ]; then
  $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
    || fail "torch sees no CUDA device (it would run on the CPU)"
fi

# --- run the units -----------------------------------------------------------------------------
run_line() {  # one actions line; the unit counts as done only if run_v2.sh exits 0 AND job.done exists. SAVE_STM=1
  # comes before the line's own words, so a line's SAVE_STM=0 wins (env: the last assignment counts).
  local line=$1 tree env args unit flag t0 rc
  IFS='|' read -r tree env args unit flag <<< "$line"
  local -a envw argw
  eval "envw=($env)"
  eval "argw=($args)"
  t0=$(date +%s)
  step "start  [$unit] $env | $args"
  # ${a[@]+"${a[@]}"}: an empty array under set -u is an error in bash < 4.4
  timeout --kill-after=120 "$UNIT_TIMEOUT" \
    env SAVE_STM=1 ${envw[@]+"${envw[@]}"} RUNS="$tree" PY="$PY" bash "$RUN_V2" ${argw[@]+"${argw[@]}"} \
      >> "$LOG/$(echo "$unit" | tr '/' '_').log" 2>&1 < /dev/null
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$unit/job.done" ]; then
    step "done   [$unit] $(( $(date +%s) - t0 )) s"
  elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
    step "FAILED [$unit] TIMEOUT after $(( $(date +%s) - t0 )) s (UNIT_TIMEOUT=$UNIT_TIMEOUT, rc=$rc)"
  else
    step "FAILED [$unit] rc=$rc after $(( $(date +%s) - t0 )) s"
  fi
}

pool() {  # $1 = file of lines; NPROC at a time, in this shell (no pipeline, so `jobs` counts our children)
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$NPROC" ]; do sleep 5; done
    run_line "$line" &
    sleep 2   # stagger start-up (dataset shared memory, CUDA context)
  done < "$1"
  wait
}

grep -E '^[^|]*\|[^|]*\|pretrain ' "$A" > "$LOG/pretrain_lines.txt" || true
grep -vE '^[^|]*\|[^|]*\|pretrain ' "$A" | grep . > "$LOG/other_lines.txt" || true
step "phase 1: $(grep -c . "$LOG/pretrain_lines.txt") pre-training unit(s)"
pool "$LOG/pretrain_lines.txt"
step "phase 2: $(grep -c . "$LOG/other_lines.txt") other unit(s)"
pool "$LOG/other_lines.txt"

# --- completeness, manifest, sentinel -----------------------------------------------------------
missing=0
while IFS='|' read -r tree env args unit flag || [ -n "$tree" ]; do
  [ -z "$tree" ] && continue
  [ -f "$unit/job.done" ] || { missing=$((missing + 1)); step "missing job.done: $unit"; }
done < "$A"

# Manifest of every file of every unit: size<TAB>sha256<TAB>path. Written even on failure, so a failed job can be
# pulled and checked. Paths contain brackets and spaces (cifar_100/continual_[3, 4]_500/...): NUL-separated find,
# tab-separated output, one file per line. Paths with a component starting with '.' are left out (header).
: > "$LOG/manifest.tsv.tmp"
while IFS='|' read -r tree env args unit flag || [ -n "$tree" ]; do
  [ -z "$tree" ] && continue
  [ -d "$unit" ] || continue
  while IFS= read -r -d '' f; do
    h=$(sha256 "$f"); [[ "$h" =~ ^[0-9a-f]{64}$ ]] || fail "cannot hash $f"
    printf '%s\t%s\t%s\n' "$(wc -c < "$f" | tr -d ' ')" "$h" "$f" >> "$LOG/manifest.tsv.tmp"
  done < <(find "$unit" -name '.*' -prune -o -type f -print0)
done < "$A"
mv "$LOG/manifest.tsv.tmp" "$LOG/manifest.tsv" || fail "cannot write manifest"
step "manifest: $(grep -c . "$LOG/manifest.tsv") files"

[ "$missing" = 0 ] || fail "$missing unit(s) unfinished"
touch "$LOG/JOB_COMPLETE"
step "JOB DONE: $n units"
exit 0
