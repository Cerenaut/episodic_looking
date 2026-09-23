#!/bin/bash
# Pull a job's results the moment it finishes, then tear the pod down.
#
# DESIGN RULE: this script destroys things, so every ambiguity must resolve to "do
# nothing". It has already failed the other way once. The first version decided a job
# was finished by parsing `grep -c "JOB DONE"` over ssh; on a pod whose log existed but
# had no match, grep printed 0 and exited 1, the `|| echo 0` fallback appended a second
# 0, and the resulting "0\n0" compared unequal to "0", so four healthy pods were reaped
# twenty minutes into a two-hour job.
#
# Two defences now, and a pod is destroyed only if BOTH pass:
#   1. Completion is a sentinel file, reported through ssh's exit code. Nothing is parsed.
#   2. After the pull, the results are verified ON THIS MACHINE: three continual files of
#      144 evaluate lines each, plus a pre-training file. The pod is torn down only once
#      the data is known to be safely here.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
LOG=runs_local/round
mkdir -p "$LOG" runs_seed runs_local/seeds ../cifar_100_pretrain/variants
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/reap.log"; }

verify_local() {  # $1 = job dir name, e.g. ref_pt12_s3
  local d="runs_seed/$1" n=0 bad=0
  [ -d "$d" ] || return 1
  while IFS= read -r -d '' f; do
    n=$((n+1)); [ "$(grep -c evaluate "$f")" = "144" ] || bad=1
  done < <(find "$d" -name 'results_continual.txt' -print0 2>/dev/null)
  [ "$n" = "3" ] || return 1
  [ "$bad" = "0" ] || return 1
  find "$d" -name 'results_pretrain.txt' | grep -q . || return 1
  return 0
}

INTERVAL=${INTERVAL:-180}
MAX_PASSES=${MAX_PASSES:-200}
for pass in $(seq 1 "$MAX_PASSES"); do
  live=$(sky status 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^r-" | awk '{print $1}')
  [ -z "$live" ] && { say "no round clusters left; reaper exiting"; exit 0; }
  for c in $live; do
    sky status "$c" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -qE "^$c .*UP" || continue
    # Defence 1: sentinel, via exit code. Any ssh failure means "not finished".
    ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o BatchMode=yes "$c" \
        'test -f ~/sky_workdir/runs_local/seeds/JOB_COMPLETE' 2>/dev/null || continue
    say "$c reports complete; pulling"
    ok=1
    rsync -az --timeout=120 "$c:~/sky_workdir/runs_seed/" runs_seed/ || ok=0
    rsync -az --timeout=120 "$c:~/sky_workdir/runs_local/seeds/" runs_local/seeds/ || ok=0
    rsync -az --timeout=120 "$c:~/cifar_100_pretrain/variants/" ../cifar_100_pretrain/variants/ || ok=0
    [ "$ok" = "1" ] || { say "$c PULL FAILED; leaving it up (autodown is the backstop)"; continue; }
    # Defence 2: the data must be verifiably here before the pod is destroyed.
    job=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$c" \
          'ls -d ~/sky_workdir/runs_seed/*/ 2>/dev/null | head -1 | xargs -n1 basename' 2>/dev/null)
    if [ -n "$job" ] && verify_local "$job"; then
      say "$c verified locally ($job: 3 orders x 144 lines + pretrain); tearing down"
      sky down "$c" -y > /dev/null 2>&1 && say "$c down" || say "$c FAILED to down"
    else
      say "$c pulled but local verification FAILED for '${job:-unknown}'; leaving it up"
    fi
  done
  sleep "$INTERVAL"
done
say "reaper hit MAX_PASSES; clusters may remain"
