#!/bin/bash
# Reaper for v2 pod jobs (sky/v2_launch.sh): pull a job's results the moment it finishes, verify them ON THIS
# MACHINE, and only then tear the pod down.
#
# DESIGN RULE (running_experiments_kb.md sections 1, 5 and 5b): this script destroys things, so every ambiguity
# resolves to "do nothing", and an unreadable state is never read as absent. A pod is torn down only when ALL hold:
#   1. sky status exits 0 and shows the cluster UP, and this round launched it ($ROUND/actions/<cluster>.txt);
#   2. `test -f` of the JOB_COMPLETE sentinel succeeds over ssh (exit code; nothing is parsed; an ssh failure is
#      "not finished");
#   3. the pull root has more than MIN_FREE_GB free beyond the size of the pod's manifest;
#   4. every rsync of the pull exits 0;
#   5. sky/v2_verify.sh passes on the pulled copy (every file of the pod's manifest is here with the same size and
#      sha256, each unit's local file set equals the manifest's, every unit has job.done, results, validation results
#      and, unless its line sets SAVE_STM=0, its STM checkpoint). Tested on fake trees by sky/test_v2_verify.sh.
# `reaped/<cluster>` is written only when sky status, read successfully, no longer lists the cluster.
#
# A pod whose launch FAILED (launch_failed/<cluster>, written by sky/v2_launch.sh with the exit code) or whose setup
# failed (sky queue shows FAILED_SETUP; setup_failed/<cluster>) is torn down only after proving that nothing ran:
# `test -d $ROUND/pods/<cluster>` over ssh exits exactly 1 (ssh worked, the pod job never made its dir). Exit 0 (the
# job did start) = handled like any other pod; any other exit (ssh failure, timeout) = do nothing this pass. It is
# marked reaped only on a real ABSENT row afterwards. A launch-failed cluster that sky status (read successfully)
# does not list is marked reaped: it never ran.
#
# HOLD instead of a silent autodown: a pod that cannot be reaped safely (pull incomplete, verification failed, disk
# low or unreadable, a failed job whose files could not be verified here) has its autostop widened ONCE to HOLD_IDLE
# minutes (default 1440), confirmed in sky status, recorded in held/<cluster>, logged and notified (macOS
# notification). It is then for a human to inspect and `sky down <cluster> -y` by hand; the reaper keeps retrying the
# pull and reaps it if a later pass succeeds.
# A JOB_FAILED pod (job_failed/<cluster>) is pulled once (failed_pulled/<cluster>): units whose directory never existed
# on the pod are skipped (ssh `test -d` exit 1; exit 255 = unreadable = retry), so no empty unit dir is made here to
# block a relaunch. The pull is then checked with `sky/v2_verify.sh --partial` (every file of the pod's manifest here
# with the same size and sha256; result in failed_verify/<cluster>). If it passes, nothing would be lost by losing the
# pod, so its autostop is set to FAILED_IDLE minutes (default 90: time to inspect it live), not HOLD_IDLE, confirmed
# in sky status (failed_safe/<cluster>) and notified; once sky status no longer lists it, it is marked reaped. If the
# pull is incomplete or --partial fails, it is HELD as above.
# ALERT: when ALERT_N (default 3) or more clusters of the round are held or failed (held/, job_failed/,
# launch_failed/, setup_failed/), a summary line is logged and a notification with a sound is sent, again each time
# the count grows; the summary line is repeated every 10 passes. A systematic failure across 20 pods should be seen
# before it costs a day of 20 held pods.
# The result trees are pulled with --ignore-existing (a file on this machine is never overwritten; a differing one
# fails verification); the pod-owned metadata dir $ROUND/pods/<cluster>/ is pulled without it, so the manifest is
# never stale.
#
# Each pass reads `sky status` ONCE for every cluster (sky_status_all in sky/v2_lib.sh; a failed or unrecognised call
# makes every cluster UNREADABLE for that pass, so nothing is done); a teardown or an autostop change is confirmed
# with a fresh per-cluster `sky status <cluster>`.
# Other duties each pass: heartbeat ($ROUND/reaper.heartbeat; sky/v2_status.sh reports REAPER NOT RUNNING when it is
# older than 15 min); apply autostop (-i IDLE --down) to an UP row without one; every 10 passes, for a pod with no
# pods dir SETUP_WARN_MIN minutes after it was first seen UP, check `sky queue` for FAILED_SETUP (then the teardown
# above) or warn.
# One reaper per round: $ROUND/reaper.lock (mkdir; holds the pid). A lock whose pid is dead is moved aside.
#
# Pull root: the unit trees land under the root recorded by the launcher in $ROUND/pull_root (absolute; default the
# repo), at the same relative paths as on the pod. The round dir itself (actions/, pods/, logs) stays in the repo.
#
# Usage: nohup caffeinate -i bash sky/v2_reap.sh <round dir> > /dev/null 2>&1 &   (log: <round dir>/reap.log)
#        (sky/v2_launch.sh starts it itself.)
# Environment: INTERVAL (s, default 180), MAX_PASSES (default 2000), IDLE (autostop minutes applied to a row without
# one, default 240), HOLD_IDLE (default 1440), FAILED_IDLE (default 90), ALERT_N (default 3), MIN_FREE_GB (default
# 20), SETUP_WARN_MIN (default 30), SKY_TIMEOUT (s, default 120), DRY=1 (everything except sky down).
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
export PATH="${REAP_TEST_BIN:+$REAP_TEST_BIN:}$HOME/bin:$PATH"   # REAP_TEST_BIN: fakes, sky/test_v2_verify.sh only
# shellcheck source=sky/v2_lib.sh
. sky/v2_lib.sh || exit 2
ROUND=${1:?round dir}
case "$ROUND" in /*|*..*) echo "round dir must be relative to the repo, without '..'" >&2; exit 2 ;; esac
[ -d "$ROUND/actions" ] || { echo "no $ROUND/actions: not a round dir" >&2; exit 2; }
LOG=$ROUND/reap.log
INTERVAL=${INTERVAL:-180}; IDLE=${IDLE:-240}; HOLD_IDLE=${HOLD_IDLE:-1440}; MIN_FREE_GB=${MIN_FREE_GB:-20}
FAILED_IDLE=${FAILED_IDLE:-90}; ALERT_N=${ALERT_N:-3}
SETUP_WARN_MIN=${SETUP_WARN_MIN:-30}; DRY=${DRY:-0}
for v in INTERVAL IDLE HOLD_IDLE FAILED_IDLE ALERT_N MIN_FREE_GB SETUP_WARN_MIN; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v must be a whole number, not '${!v}'" >&2; exit 2; }
done
mkdir -p "$ROUND/reaped" "$ROUND/failed_pulled" "$ROUND/held" "$ROUND/verified" "$ROUND/first_up" "$ROUND/setup_failed" \
  "$ROUND/job_failed" "$ROUND/failed_verify" "$ROUND/failed_safe"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
notify() { local m=${1//\"/}; osascript -e "display notification \"$m\" with title \"v2 reaper: $ROUND\"${2:+ sound name \"$2\"}" > /dev/null 2>&1 || true; }
SSHO="-o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no"
remote_test() { tmo 60 ssh $SSHO "$1" "test $2 ~/sky_workdir/$3" < /dev/null 2>/dev/null; }   # exit: 0 yes, 1 no, else unreadable

# --- pull root ---------------------------------------------------------------------------------------
PR=$REPO
[ -f "$ROUND/pull_root" ] && PR=$(cat "$ROUND/pull_root")
if [ -n "${PULL_ROOT:-}" ] && [ "$PULL_ROOT" != "$PR" ]; then
  echo "PULL_ROOT=$PULL_ROOT differs from the round's pull root $PR ($ROUND/pull_root, written by the launcher)" >&2; exit 2
fi
case "$PR" in /*) ;; *) echo "pull root must be absolute: $PR" >&2; exit 2 ;; esac

# --- one reaper per round ----------------------------------------------------------------------------
L=$ROUND/reaper.lock
if ! mkdir "$L" 2>/dev/null; then
  p=$(reaper_pid "$ROUND")
  case "$p" in
    "?") echo "$L exists without a readable pid: check that no reaper runs on $ROUND, then remove it by hand" | tee -a "$LOG" >&2; exit 3 ;;
    "") stale=$L.stale.$$.$(date +%s)
        mv "$L" "$stale" 2>/dev/null && say "moved a stale reaper lock (dead pid) aside to $stale"
        mkdir "$L" 2>/dev/null || { echo "cannot take $L" >&2; exit 3; } ;;
    *) echo "a reaper (pid $p) already runs on $ROUND; exiting" >&2; exit 3 ;;
  esac
fi
echo $$ > "$L/pid"
trap '[ "$(cat "$L/pid" 2>/dev/null)" = $$ ] && rm -rf "$L"' EXIT

beat() { echo "$$ $(date '+%F %T') pass ${pass:-0}" > "$ROUND/reaper.heartbeat"; }
every10() { [ $((pass % 10)) = 1 ]; }
fmt_idle() { [ $(($1 % 60)) = 0 ] && [ "$1" -gt 0 ] && echo "$(($1 / 60))h" || echo "${1}m"; }   # as sky status prints it

set_autostop() {  # $1 = cluster, $2 = minutes: sky autostop -i <m> --down, then confirmed in a fresh sky status row.
  # 0 = confirmed; 1 = the call failed or timed out; 2 = not confirmed (SKY_STATE/SKY_LINE say why).
  tmo "$SKY_TIMEOUT" sky autostop "$1" -i "$2" --down -y < /dev/null >> "$LOG" 2>&1 || return 1
  sky_row "$1"
  [ "$SKY_STATE" = ROW ] && printf '%s\n' "$SKY_LINE" | grep -qF "$(fmt_idle "$2") (down)" && return 0
  return 2
}

hold() {  # $1 = cluster, $2 = reason. Widen autostop to HOLD_IDLE once, confirmed in sky status; never destroys.
  local c=$1 why=$2 r
  if [ -e "$ROUND/held/$c" ]; then every10 && say "$c HELD since $(cat "$ROUND/held/$c"); now: $why"; return 0; fi
  set_autostop "$c" "$HOLD_IDLE"; r=$?
  case $r in
    0) echo "$(date '+%F %T') $why" > "$ROUND/held/$c"
       say "$c HELD: autostop widened to ${HOLD_IDLE}m ($why). Inspect, then sky down $c -y by hand once its results are safe"
       notify "$c HELD: $why" ;;
    1) say "$c HOLD FAILED: sky autostop -i $HOLD_IDLE returned non-zero or timed out; retrying next pass ($why)"
       every10 && notify "$c HOLD FAILED: $why" ;;
    *) say "$c HOLD NOT CONFIRMED (sky status $SKY_STATE: ${SKY_LINE:-no row}); retrying next pass ($why)"
       every10 && notify "$c HOLD NOT CONFIRMED: $why" ;;
  esac
  return 0
}

failed_short() {  # $1 = cluster: a JOB_FAILED pod whose pull passed v2_verify.sh --partial. Autostop FAILED_IDLE
  # (narrowed from a 24 h HOLD if one was applied before the pull succeeded), confirmed; never destroys.
  local c=$1 r
  if [ -e "$ROUND/failed_safe/$c" ]; then
    every10 && say "$c JOB_FAILED, files verified here; autostop ${FAILED_IDLE}m since $(cat "$ROUND/failed_safe/$c") (inspect $ROUND/pods/$c/progress.log; sky down $c -y to end it sooner)"
    return 0
  fi
  set_autostop "$c" "$FAILED_IDLE"; r=$?
  if [ $r = 0 ]; then
    date '+%F %T' > "$ROUND/failed_safe/$c"
    rm -f "$ROUND/held/$c"
    say "$c JOB_FAILED: every file of its manifest is here (v2_verify.sh --partial); autostop set to ${FAILED_IDLE}m, not a ${HOLD_IDLE}m HOLD. Inspect $ROUND/pods/$c/progress.log"
    notify "$c JOB_FAILED: files safe here, pod autodowns in ${FAILED_IDLE}m"
  else
    say "$c: autostop -i $FAILED_IDLE for the failed job $([ $r = 1 ] && echo 'returned non-zero or timed out' || echo "NOT CONFIRMED (sky status $SKY_STATE)"); retrying next pass"
  fi
}

down_confirmed() {  # $1 = cluster, $2 = reason for reaped/: sky down, then reaped/ only on a real ABSENT row
  local c=$1 why=$2
  if [ "$DRY" = 1 ]; then say "DRY: would sky down $c ($why)"; touch "$ROUND/reaped/$c.dry"; return 0; fi
  tmo 600 sky down "$c" -y < /dev/null >> "$LOG" 2>&1
  sky_row "$c"
  case "$SKY_STATE" in
    ABSENT) echo "$(date '+%F %T') $why" > "$ROUND/reaped/$c"; say "$c DOWN (absent from sky status): $why" ;;
    ROW) say "$c sky down did not remove it: $SKY_LINE" ;;
    *) say "$c sky down issued; sky status UNREADABLE, so not marked reaped (re-checked next pass)" ;;
  esac
}

teardown_if_nothing_ran() {  # $1 = cluster, $2 = why (launch failed / setup failed). Returns 0 if it handled the
  # cluster this pass (torn down, or could not tell), 1 if the job did start (the caller goes on as for any pod).
  local c=$1 why=$2 r
  remote_test "$c" -d "$ROUND/pods/$c"; r=$?
  case $r in
    1) say "$c $why and its job never started (no $ROUND/pods/$c on the pod): tearing it down"
       down_confirmed "$c" "$why; nothing ran; torn down"; return 0 ;;
    0) every10 && say "$c $why, but its job did start ($ROUND/pods/$c exists on the pod): handled as a normal pod"; return 1 ;;
    *) say "$c $why; cannot tell whether its job started (ssh exit $r): doing nothing this pass"; return 0 ;;
  esac
}

alert_check() {  # ALERT_N or more clusters held or failed in the round: summary line + loud notification when it grows
  local names n last
  names=$(failed_or_held "$ROUND")
  n=$(printf '%s' "$names" | grep -c .)
  [ "$n" -ge "$ALERT_N" ] || return 0
  last=$(cat "$ROUND/alert_count" 2>/dev/null); [[ "$last" =~ ^[0-9]+$ ]] || last=0
  if [ "$n" -gt "$last" ]; then
    say "ALERT: $n clusters of this round are held or failed: $(echo $names). A systematic failure? Check before more pods bill"
    notify "ALERT: $n pods held or failed in this round" Basso
    echo "$n" > "$ROUND/alert_count"
  elif every10; then
    say "summary: $n clusters held or failed: $(echo $names)"
  fi
}

disk_ok() {  # free space on the pull root > MIN_FREE_GB + the size of the pod's manifest (if pulled); sets DISK_MSG
  local avail need pull=0 m=$ROUND/pods/$1/manifest.tsv
  avail=$(df -Pk "$PR" 2>/dev/null | awk 'NR == 2 { print $4 }')
  [[ "$avail" =~ ^[0-9]+$ ]] || { DISK_MSG="free space on $PR UNREADABLE"; return 1; }
  [ -s "$m" ] && pull=$(awk -F'\t' '{ s += $1 } END { printf "%d", s / 1024 }' "$m")
  [[ "$pull" =~ ^[0-9]+$ ]] || { DISK_MSG="manifest size UNREADABLE: $m"; return 1; }
  need=$((MIN_FREE_GB * 1048576 + pull))
  DISK_MSG="disk: $((avail / 1048576)) GB free on $PR, need > $MIN_FREE_GB GB + $((pull / 1048576)) GB to pull"
  [ "$avail" -gt "$need" ]
}

pull_meta() {  # $1 = cluster: the pod-owned metadata dir, overwritten (never --ignore-existing: no stale manifest)
  local c=$1
  mkdir -p "$ROUND/pods/$c" || return 1
  tmo 900 rsync -az --timeout=120 -e "ssh $SSHO" "$c:~/sky_workdir/$ROUND/pods/$c/" "$ROUND/pods/$c/" < /dev/null >> "$LOG" 2>&1 \
    || { say "$c rsync FAILED: pod metadata $ROUND/pods/$c"; return 1; }
}

pull_units() {  # $1 = cluster, $2 = complete|failed. Never makes an empty unit dir here: only units that exist on the pod.
  local c=$1 mode=$2 tree env args unit flag r rc=0
  while IFS='|' read -r tree env args unit flag || [ -n "$tree" ]; do
    [ -z "$tree" ] && continue
    beat
    remote_test "$c" -d "$unit"; r=$?
    if [ $r -eq 1 ]; then
      if [ "$mode" = failed ]; then say "$c: $unit never existed on the pod; skipped"; else say "$c: $unit MISSING on the pod"; rc=1; fi
      continue
    elif [ $r -ne 0 ]; then
      say "$c: cannot tell whether $unit exists on the pod (ssh exit $r)"; rc=1; continue
    fi
    mkdir -p "$PR/$(dirname "$unit")" || { rc=1; continue; }
    tmo 7200 rsync -az --ignore-existing --timeout=120 -e "ssh $SSHO" "$c:~/sky_workdir/$unit/" "$PR/$unit/" < /dev/null >> "$LOG" 2>&1 \
      || { say "$c rsync FAILED: $unit"; rc=1; }
  done < "$ROUND/actions/$c.txt"
  return $rc
}

setup_check() {  # $1 = cluster, UP, no sentinel yet: warn when no pods dir appears SETUP_WARN_MIN after first seen UP
  local c=$1 t0 r q
  [ -s "$ROUND/first_up/$c" ] || date +%s > "$ROUND/first_up/$c"
  t0=$(cat "$ROUND/first_up/$c")
  every10 || return 0
  [ $(( $(date +%s) - t0 )) -ge $((SETUP_WARN_MIN * 60)) ] || return 0
  remote_test "$c" -d "$ROUND/pods/$c"; r=$?
  case $r in
    0) return 0 ;;
    1) q=$(tmo "$SKY_TIMEOUT" sky queue "$c" < /dev/null 2>&1 | strip)
       if printf '%s\n' "$q" | grep -qw FAILED_SETUP; then
         [ -e "$ROUND/setup_failed/$c" ] || { date '+%F %T' > "$ROUND/setup_failed/$c"; notify "$c SETUP FAILED"; }
         say "$c SETUP FAILED (sky queue shows FAILED_SETUP; see sky logs $c)"
         teardown_if_nothing_ran "$c" "setup failed (FAILED_SETUP)" || true
       else
         say "WARNING $c UP for $(( ($(date +%s) - t0) / 60 )) min and its job has not started (no $ROUND/pods/$c on the pod): setup still running or failed silently; see sky queue $c"
       fi ;;
    *) say "$c: cannot read whether the job started (ssh exit $r)" ;;
  esac
}

say "reaper starting on $ROUND (pid $$, INTERVAL=${INTERVAL}s, HOLD_IDLE=${HOLD_IDLE}m, FAILED_IDLE=${FAILED_IDLE}m, MIN_FREE_GB=$MIN_FREE_GB, pull root $PR$([ "$DRY" = 1 ] && echo ', DRY: no teardown'))"
for pass in $(seq 1 "${MAX_PASSES:-2000}"); do
  beat
  left=0
  sky_status_all   # ONE sky status per pass (N4); unreadable = every cluster UNREADABLE = nothing done this pass
  [ "$SKY_ALL_OK" = 1 ] || say "sky status UNREADABLE (sky failed, timed out or printed something unexpected); doing nothing this pass"
  for af in "$ROUND"/actions/*.txt; do
    [ -f "$af" ] || continue
    c=$(basename "$af" .txt)
    [ -e "$ROUND/reaped/$c" ] && continue
    left=$((left + 1))
    beat
    sky_row_all "$c"
    lf=""; [ -e "$ROUND/launch_failed/$c" ] && lf="launch failed ($(cat "$ROUND/launch_failed/$c" 2>/dev/null))"
    case "$SKY_STATE" in
      UNREADABLE) say "$c sky status UNREADABLE; doing nothing this pass"; continue ;;
      ABSENT)
        if [ -e "$ROUND/verified/$c" ]; then
          echo "$(date '+%F %T') verified, then absent from sky status" > "$ROUND/reaped/$c"; say "$c DOWN (verified earlier; now absent from sky status)"
        elif [ -e "$ROUND/failed_safe/$c" ]; then
          echo "$(date '+%F %T') JOB_FAILED, files verified here (--partial), then absent from sky status" > "$ROUND/reaped/$c"; say "$c gone (failed job, files verified here earlier; now absent from sky status)"
        elif [ -n "$lf" ]; then
          echo "$(date '+%F %T') $lf; absent from sky status: nothing ran, nothing to pull" > "$ROUND/reaped/$c"; say "$c $lf and absent from sky status: marked reaped (nothing ran)"
        else
          every10 && say "$c NOT IN sky status (failed launch, or gone: CHECK $ROUND/pods/$c and the RunPod console; once handled, touch $ROUND/reaped/$c)"
        fi
        continue ;;
    esac
    st=$(row_status "$SKY_LINE")
    [ "$st" = UP ] || { every10 && say "$c not UP${lf:+ ($lf)}: $SKY_LINE"; continue; }
    if ! has_autostop "$SKY_LINE"; then
      say "WARNING $c has no autostop: applying -i $IDLE --down"
      tmo "$SKY_TIMEOUT" sky autostop "$c" -i "$IDLE" --down -y < /dev/null >> "$LOG" 2>&1 || say "$c: sky autostop failed; retrying next pass"
    fi
    if [ -n "$lf" ]; then teardown_if_nothing_ran "$c" "$lf" && continue; fi
    remote_test "$c" -f "$ROUND/pods/$c/JOB_COMPLETE"; r=$?
    if [ $r -eq 0 ]; then
      say "$c reports JOB_COMPLETE; pulling"
      pull_meta "$c" || { hold "$c" "pull of the pod metadata failed"; continue; }
      disk_ok "$c" || { say "$c NOT PULLED: $DISK_MSG"; hold "$c" "$DISK_MSG"; continue; }
      pull_units "$c" complete || { say "$c PULL INCOMPLETE; will retry"; hold "$c" "pull incomplete"; continue; }
      if bash sky/v2_verify.sh "$REPO/$ROUND" "$c" "$PR" >> "$LOG" 2>&1; then
        say "$c verified locally ($(tail -1 "$LOG"))"
        date '+%F %T' > "$ROUND/verified/$c"
        down_confirmed "$c" "verified and down"
      else
        say "$c pulled but local verification FAILED (details above)"
        hold "$c" "verification failed"
      fi
      continue
    elif [ $r -ne 1 ]; then
      every10 && say "$c: cannot read the sentinel (ssh exit $r); doing nothing"; continue
    fi
    remote_test "$c" -f "$ROUND/pods/$c/JOB_FAILED"; r=$?
    if [ $r -eq 0 ]; then
      [ -e "$ROUND/job_failed/$c" ] || date '+%F %T' > "$ROUND/job_failed/$c"
      if [ ! -e "$ROUND/failed_pulled/$c" ]; then
        say "$c JOB_FAILED: pulling what there is, once"
        if ! pull_meta "$c"; then say "$c: metadata pull failed; will retry"
        elif ! disk_ok "$c"; then say "$c NOT PULLED: $DISK_MSG"
        elif pull_units "$c" failed; then
          touch "$ROUND/failed_pulled/$c"; say "$c failed job pulled"
          if bash sky/v2_verify.sh --partial "$REPO/$ROUND" "$c" "$PR" >> "$LOG" 2>&1; then echo OK > "$ROUND/failed_verify/$c"
          else echo FAIL > "$ROUND/failed_verify/$c"; say "$c: the failed job's pull did NOT pass v2_verify.sh --partial (details above)"; fi
        else say "$c: pull of the failed job incomplete; will retry"
        fi
      fi
      if [ "$(cat "$ROUND/failed_verify/$c" 2>/dev/null)" = OK ]; then
        failed_short "$c"
      elif [ -e "$ROUND/failed_pulled/$c" ]; then
        hold "$c" "JOB_FAILED; its pull failed v2_verify.sh --partial"
      else
        hold "$c" "JOB_FAILED; pull incomplete or not possible yet"
      fi
    elif [ $r -eq 1 ]; then
      setup_check "$c"
    else
      every10 && say "$c: cannot read the sentinel (ssh exit $r); doing nothing"
    fi
  done
  alert_check
  [ "$left" = 0 ] && { say "every cluster of the round is reaped; reaper exiting"; exit 0; }
  [ "$DRY" = 1 ] && [ "$(ls "$ROUND/reaped/" | grep -c '\.dry$')" -ge "$left" ] && { say "DRY: all verified; exiting"; exit 0; }
  sleep "$INTERVAL"
done
say "reaper hit MAX_PASSES; clusters may remain"
