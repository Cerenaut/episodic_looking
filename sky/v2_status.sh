#!/bin/bash
# Status of a v2 round (sky/v2_launch.sh): the reaper's heartbeat, then per cluster the sky state and autostop, and
# from the pod the number of units with job.done out of the units in its actions file, the sentinel, and the last
# progress line.
#
# Rule (running_experiments_kb.md section 1): "could not read" is never shown as "no progress" or "absent". Exit 1,
# with a loud word, when:
#   UNREADABLE          sky status failed, timed out or printed something unexpected; or ssh to the pod failed
#   ABSENT              sky status (read successfully) does not list a cluster that is not reaped
#   NO-AUTOSTOP         an UP or INIT row without "Nh (down)"/"Nm (down)" (it would bill indefinitely)
#   LAUNCH_FAILED       sky/v2_launch.sh recorded a failed sky launch (launch_failed/<cluster>, with the exit code) and
#                       the cluster is not reaped yet (the reaper tears it down once it has proved nothing ran)
#   SETUP_FAILED        sky queue showed FAILED_SETUP (setup_failed/<cluster>), not reaped yet
#   REAPER NOT RUNNING  clusters remain and the reaper's heartbeat is older than 15 min (or missing) and its pid is
#                       gone; REAPER STALE if the pid is alive but the heartbeat is old (hung, or a very long pull)
# A job that failed before copying its actions file (e.g. no UNIT_TIMEOUT) prints JOB_FAILED with "before the units".
# A pod whose job has not started yet prints NOT-STARTED (ssh worked, no actions copy on the pod yet). Units done are
# counted against the total, so a working zero reads "0/5". HELD, failed_pulled, failed_safe (a failed job whose files
# are verified here; short autostop) and failed_verify=FAIL markers are shown, and a count of held or failed clusters.
# sky status is read ONCE for every cluster (sky_status_all); if that call fails every cluster shows UNREADABLE.
# Usage: bash sky/v2_status.sh <round dir>
set -u
cd "$(dirname "$0")/.."
export PATH="${REAP_TEST_BIN:+$REAP_TEST_BIN:}$HOME/bin:$PATH"   # REAP_TEST_BIN: fakes, sky/test_v2_verify.sh only
# shellcheck source=sky/v2_lib.sh
. sky/v2_lib.sh || { echo "UNREADABLE: cannot read sky/v2_lib.sh"; exit 1; }
ROUND=${1:?round dir}
[ -d "$ROUND/actions" ] || { echo "UNREADABLE: no $ROUND/actions (not a round dir)"; exit 1; }
HEARTBEAT_MAX=${HEARTBEAT_MAX:-900}
rc=0
echo "=== $ROUND  $(date '+%Y-%m-%d %H:%M:%S') ==="

# --- the reaper ---------------------------------------------------------------------------------------
remaining=0
for af in "$ROUND"/actions/*.txt; do [ -f "$af" ] && [ ! -e "$ROUND/reaped/$(basename "$af" .txt)" ] && remaining=$((remaining + 1)); done
hb=$ROUND/reaper.heartbeat
p=$(reaper_pid "$ROUND")
age=999999; hbtxt="no heartbeat file"
if [ -f "$hb" ]; then
  t=$(mtime "$hb")
  if [[ "$t" =~ ^[0-9]+$ ]]; then age=$(( $(date +%s) - t )); hbtxt="heartbeat ${age}s ago ($(cat "$hb" 2>/dev/null))"
  else hbtxt="heartbeat file UNREADABLE"; fi
fi
if [ "$remaining" = 0 ]; then
  echo "reaper: every cluster reaped ($hbtxt)"
elif [ "$age" -le "$HEARTBEAT_MAX" ]; then
  echo "reaper: running, pid ${p:-?} ($hbtxt)"
elif [ -n "$p" ] && [ "$p" != "?" ]; then
  echo "REAPER STALE: pid $p alive but $hbtxt (hung, or a very long pull): check $ROUND/reap.log"; rc=1
else
  echo "REAPER NOT RUNNING ($hbtxt; $remaining cluster(s) not reaped): start it: nohup caffeinate -i bash sky/v2_reap.sh $ROUND > /dev/null 2>&1 &"; rc=1
fi

# --- clusters -------------------------------------------------------------------------------------------
sky_status_all
printf '%-18s %-10s %-11s %-9s %-13s %s\n' cluster sky autostop units sentinel "last progress line"
for af in "$ROUND"/actions/*.txt; do
  [ -f "$af" ] || { echo "UNREADABLE: no actions files in $ROUND/actions"; rc=1; continue; }
  c=$(basename "$af" .txt)
  total=$(grep -c . "$af")
  if [ -e "$ROUND/reaped/$c" ]; then
    printf '%-18s %-10s %-11s %-9s %-13s %s\n' "$c" REAPED - - reaped "$(cat "$ROUND/reaped/$c")"; continue
  fi
  marks=""
  [ -e "$ROUND/launch_failed/$c" ] && { marks="$marks LAUNCH_FAILED($(cat "$ROUND/launch_failed/$c" 2>/dev/null))"; rc=1; }
  [ -e "$ROUND/setup_failed/$c" ] && { marks="$marks SETUP_FAILED"; rc=1; }
  [ -e "$ROUND/held/$c" ] && marks="$marks HELD($(cat "$ROUND/held/$c"))"
  [ -e "$ROUND/failed_pulled/$c" ] && marks="$marks failed_pulled"
  [ "$(cat "$ROUND/failed_verify/$c" 2>/dev/null)" = FAIL ] && marks="$marks failed_verify=FAIL"
  [ -e "$ROUND/failed_safe/$c" ] && marks="$marks failed_safe(autostop short since $(cat "$ROUND/failed_safe/$c"))"
  sky_row_all "$c"
  case "$SKY_STATE" in
    UNREADABLE) printf '%-18s %-10s %s\n' "$c" UNREADABLE "sky status failed or unparsable$marks"; rc=1; continue ;;
    ABSENT) printf '%-18s %-10s %s\n' "$c" ABSENT "not in sky status and not reaped: CHECK$marks"; rc=1; continue ;;
  esac
  st=$(row_status "$SKY_LINE"); st=${st:-UNPARSED}
  au=$(printf '%s\n' "$SKY_LINE" | grep -oE '[0-9]+[hm] \(down\)' | head -1)
  if [ -z "$au" ]; then
    au="NONE"
    case "$st" in UP|INIT) au="NO-AUTOSTOP"; rc=1 ;; esac
  fi
  if [ "$st" != UP ]; then printf '%-18s %-10s %-11s %s\n' "$c" "$st" "$au" "$marks"; [ "$st" = UNPARSED ] && rc=1; continue; fi
  # One ssh call, 60 s cap; the remote prints exactly one line "<done> <total> <sentinel> <last line>" or NOT-STARTED.
  r=$(tmo 60 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no "$c" "bash -s" 2>/dev/null <<EOF
cd ~/sky_workdir 2>/dev/null || { echo NOWORKDIR; exit 0; }
P=$ROUND/pods/$c
[ -f "\$P/actions.txt" ] || { [ -f "\$P/JOB_FAILED" ] && { echo "FAILED-EARLY \$(tail -1 "\$P/progress.log" 2>/dev/null | cut -c1-110)"; exit 0; }; echo NOT-STARTED; exit 0; }
d=0; t=0
while IFS='|' read -r tree env args unit flag || [ -n "\$tree" ]; do [ -z "\$tree" ] && continue; t=\$((t+1)); [ -f "\$unit/job.done" ] && d=\$((d+1)); done < "\$P/actions.txt"
s=running; [ -f "\$P/JOB_FAILED" ] && s=JOB_FAILED; [ -f "\$P/JOB_COMPLETE" ] && s=JOB_COMPLETE
echo "\$d \$t \$s \$(tail -1 "\$P/progress.log" 2>/dev/null | cut -c1-110)"
EOF
)
  case "$r" in
    FAILED-EARLY*) printf '%-18s %-10s %-11s %-9s %-13s %s\n' "$c" UP "$au" "0/$total" JOB_FAILED "(before the units: ${r#FAILED-EARLY })$marks" ;;
    NOT-STARTED|NOWORKDIR)
      if [ -e "$ROUND/launch_failed/$c" ] || [ -e "$ROUND/setup_failed/$c" ]; then
        printf '%-18s %-10s %-11s %-9s %-13s %s\n' "$c" UP "$au" "0/$total" LAUNCH_FAILED "(nothing ran: the reaper tears it down)$marks"
      else
        printf '%-18s %-10s %-11s %-9s %-13s %s\n' "$c" UP "$au" "?/$total" "$r" "(setup still running?)$marks"
      fi ;;
    *)
      set -f; set -- $r; set +f   # no globbing: the progress line holds [brackets]
      if [[ "${1:-}" =~ ^[0-9]+$ ]] && [[ "${2:-}" =~ ^[0-9]+$ ]] && [ -n "${3:-}" ]; then
        printf '%-18s %-10s %-11s %-9s %-13s %s\n' "$c" UP "$au" "$1/$2" "$3" "${r#* * * }$marks"
      else
        printf '%-18s %-10s %-11s %s\n' "$c" UP "$au" "UNREADABLE (ssh failed or unparsable reply: '${r:0:60}')$marks"; rc=1
      fi ;;
  esac
done
names=$(failed_or_held "$ROUND")
n=$(printf '%s' "$names" | grep -c .)
[ "$n" -gt 0 ] && echo "held or failed in this round: $n ($(echo $names))$([ "$n" -ge "${ALERT_N:-3}" ] && echo '  ALERT: systematic failure?')"
exit $rc
