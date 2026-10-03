#!/bin/bash
# Reaper for v2 pod jobs (sky/v2_launch.sh): pull a job's results the moment it finishes, verify them ON THIS
# MACHINE, and only then tear the pod down.
#
# DESIGN RULE (running_experiments_kb.md sections 5 and 5b): this script destroys things, so every ambiguity resolves
# to "do nothing". A pod is torn down only when ALL of these hold:
#   1. the cluster is UP in sky status and is one this round launched (it has $ROUND/actions/<cluster>.txt);
#   2. `test -f` of the JOB_COMPLETE sentinel succeeds over ssh (exit code; nothing is parsed; an ssh failure is
#      "not finished");
#   3. every rsync of the pull exits 0;
#   4. sky/v2_verify.sh passes on the pulled copy (every file of the pod's manifest is here with the same size, every
#      unit has job.done, results, validation results and its STM checkpoint). Tested on fake good and bad trees by
#      sky/test_v2_verify.sh.
# A pod whose job wrote JOB_FAILED is pulled once (so nothing is lost) and LEFT UP for inspection; the autostop
# backstop applies. The pull never overwrites a file on this machine (--ignore-existing).
#
# Usage: nohup bash sky/v2_reap.sh <round dir> > /dev/null 2>&1 &      (log: <round dir>/reap.log)
# Environment: INTERVAL (s, default 180), MAX_PASSES (default 2000), DRY=1 (everything except sky down).
set -u
cd "$(dirname "$0")/.."
export PATH="${REAP_TEST_BIN:+$REAP_TEST_BIN:}$HOME/bin:$PATH"   # REAP_TEST_BIN: fakes, sky/test_v2_verify.sh only
ROUND=${1:?round dir}
[ -d "$ROUND/actions" ] || { echo "no $ROUND/actions: not a round dir" >&2; exit 2; }
LOG=$ROUND/reap.log
mkdir -p "$ROUND/reaped" "$ROUND/failed_pulled"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
strip() { perl -pe 's/\e\[[0-9;]*[mK]//g'; }
status_line() { sky status "$1" 2>/dev/null | strip | awk -v c="$1" '$1==c'; }
SSHO="-o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no"
remote_test() { ssh $SSHO "$1" "test -f ~/sky_workdir/$2" < /dev/null 2>/dev/null; }

pull() {  # $1 = cluster; returns non-zero if any rsync fails
  local c=$1 tree env args unit flag rc=0
  while IFS='|' read -r tree env args unit flag || [ -n "$tree" ]; do
    [ -z "$tree" ] && continue
    mkdir -p "$unit" || { rc=1; continue; }
    rsync -az --ignore-existing --timeout=120 -e "ssh $SSHO" "$c:~/sky_workdir/$unit/" "$unit/" < /dev/null >> "$LOG" 2>&1 \
      || { say "$c rsync FAILED: $unit"; rc=1; }
  done < "$ROUND/actions/$c.txt"
  mkdir -p "$ROUND/pods/$c"
  rsync -az --ignore-existing --timeout=120 -e "ssh $SSHO" "$c:~/sky_workdir/$ROUND/pods/$c/" "$ROUND/pods/$c/" < /dev/null >> "$LOG" 2>&1 \
    || { say "$c rsync FAILED: pod logs"; rc=1; }
  return $rc
}

say "reaper starting on $ROUND (INTERVAL=${INTERVAL:-180}s${DRY:+, DRY: no teardown})"
for pass in $(seq 1 "${MAX_PASSES:-2000}"); do
  left=0
  for af in "$ROUND"/actions/*.txt; do
    [ -f "$af" ] || continue
    c=$(basename "$af" .txt)
    [ -e "$ROUND/reaped/$c" ] && continue
    left=$((left + 1))
    line=$(status_line "$c")
    if [ -z "$line" ]; then say "$c NOT IN sky status (failed launch, or gone: CHECK $ROUND/pods/$c and the RunPod console)"; continue; fi
    echo "$line" | grep -qw UP || { [ $((pass % 10)) = 1 ] && say "$c not UP yet: $line"; continue; }
    echo "$line" | grep -qE '[0-9]+[hm] \(down\)' || say "WARNING $c has no autostop: sky autostop $c -i 240 --down -y"
    if remote_test "$c" "$ROUND/pods/$c/JOB_COMPLETE"; then
      say "$c reports JOB_COMPLETE; pulling"
      pull "$c" || { say "$c PULL INCOMPLETE; leaving it up, will retry"; continue; }
      if bash sky/v2_verify.sh "$ROUND" "$c" >> "$LOG" 2>&1; then
        say "$c verified locally ($(tail -1 "$LOG"))"
        if [ -n "${DRY:-}" ]; then say "DRY: would sky down $c"; touch "$ROUND/reaped/$c.dry"; continue; fi
        sky down "$c" -y >> "$LOG" 2>&1
        if [ -z "$(status_line "$c")" ]; then
          echo "$(date '+%F %T') verified and down" > "$ROUND/reaped/$c"; say "$c DOWN (absent from sky status)"
        else
          say "$c sky down did not remove it: $(status_line "$c")"
        fi
      else
        say "$c pulled but local verification FAILED (details above); leaving it up"
      fi
    elif remote_test "$c" "$ROUND/pods/$c/JOB_FAILED"; then
      if [ ! -e "$ROUND/failed_pulled/$c" ]; then
        say "$c JOB_FAILED: pulling what there is; LEAVING IT UP for inspection (autostop is the backstop)"
        pull "$c" && touch "$ROUND/failed_pulled/$c"
      fi
      [ $((pass % 10)) = 1 ] && say "$c JOB_FAILED, still up: inspect $ROUND/pods/$c/progress.log, then sky down $c -y by hand"
    fi
  done
  [ "$left" = 0 ] && { say "every cluster of the round is reaped; reaper exiting"; exit 0; }
  [ -n "${DRY:-}" ] && [ "$(ls "$ROUND/reaped/" | grep -c '\.dry$')" -ge "$left" ] && { say "DRY: all verified; exiting"; exit 0; }
  sleep "${INTERVAL:-180}"
done
say "reaper hit MAX_PASSES; clusters may remain"
