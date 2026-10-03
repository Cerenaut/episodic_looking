#!/bin/bash
# Launch K v2 pod jobs from K actions files, one pod each (sky/v2_pod.yaml -> sky/v2_pod_job.sh), apply autostop
# separately (sky launch -d ignores -i/--down: running_experiments_kb.md section 6), and start the reaper.
#
# Autostop here is a MONEY BACKSTOP ONLY, with a long idle window (IDLE minutes, default 240): nothing is mirrored
# off a pod while it runs, so an autodown that fires on a finished job destroys its results (section 5). The reaper
# (sky/v2_reap.sh) pulls, verifies locally and only then tears a pod down, and widens the autostop of a pod it cannot
# reap (HOLD). This script starts it under `caffeinate -i` unless one already runs on the round (reaper.lock).
#
# Usage: UNIT_TIMEOUT=<duration> bash sky/v2_launch.sh <round dir> <actions file> [<actions file> ...]
#   round dir     relative to the repo, e.g. runs_local/v2_round/cont_rl_20261004; holds actions/<cluster>.txt (the
#                 launcher's copies, which the reaper verifies against), launch logs, pull_root, and later the pulled
#                 pod logs.
#   actions file  basename = cluster name (lowercase letters, digits, '-'), e.g. v2c-rl-pb-s1.txt. Format: header of
#                 sky/v2_pod_job.sh. Validated here: unit dir = run_v2.sh's layout (PRINT_UNIT=1), run_v2.sh accepts
#                 the line (DRY=1), no reserved key (RUNS, PY, DRY, PRINT_UNIT, UNIT_TIMEOUT) in the environment
#                 field and SAVE_STM only 0 or 1, an STM unit's pre-training is in the same file, and no unit dir
#                 exists under the pull root yet (results are never merged or overwritten).
# Environment:
#   UNIT_TIMEOUT  REQUIRED. Per-unit cap on the pod (timeout(1) syntax: 90m, 6h). One value per launch, so it must
#                 cover the slowest unit type in the files. Suggested: about 3x the expected unit time with NPROC=3
#                 (running_experiments_kb.md section 9): pre-training only ~17 min -> 1h; continual order ~1h50m -> 6h;
#                 single-stream at the draft budget ~4h -> 12h; few-shot: 3x a measured unit (no cloud measurement yet).
#   IDLE (autostop minutes, default 240), NPROC (per pod, default 3), ALLOW_DIRTY=1 (launch from a tree with
#   uncommitted changes or untracked non-ignored files; the commit is then recorded as +dirty), AUTOSTOP_WAIT (s,
#   default 3600), LAUNCH_TIMEOUT (s per sky launch, default 5400), SKY_TIMEOUT (s per other sky call, default 120),
#   PULL_ROOT (absolute dir the reaper pulls the unit trees into, at the same relative paths; default the repo;
#   recorded in <round dir>/pull_root and fixed for the round), DRY=1 (validate and print the sky commands only; writes
#   no actions copies and starts nothing).
set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
export PATH="${REAP_TEST_BIN:+$REAP_TEST_BIN:}$HOME/bin:$PATH"   # REAP_TEST_BIN: fakes, sky/test_v2_verify.sh only
# shellcheck source=sky/v2_lib.sh
. sky/v2_lib.sh || exit 2
ROUND=${1:?round dir (relative to the repo)}
shift
[ $# -ge 1 ] || { echo "usage: UNIT_TIMEOUT=6h $0 <round dir> <actions file> [...]" >&2; exit 2; }
case "$ROUND" in /*|*..*) echo "round dir must be relative to the repo, without '..'" >&2; exit 2 ;; esac
IDLE=${IDLE:-240}
NPROC=${NPROC:-3}
DRY=${DRY:-0}
UNIT_TIMEOUT=${UNIT_TIMEOUT:-}
mkdir -p "$ROUND"
LOG=$ROUND/launch.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { say "LAUNCH REFUSED: $*"; exit 2; }

command -v sky >/dev/null || die "sky not on PATH (expected ~/bin/sky -> ~/.venvs/sky)"
[[ "$UNIT_TIMEOUT" =~ ^[1-9][0-9]*[smhd]?$ ]] || die "UNIT_TIMEOUT must be set to a positive duration, e.g. UNIT_TIMEOUT=6h (suggested values: header of $0), not '$UNIT_TIMEOUT'"
[[ "$IDLE" =~ ^[1-9][0-9]*$ ]] || die "IDLE must be a positive whole number of minutes"

# --- pull root (fixed per round) ------------------------------------------------------------------
PR=${PULL_ROOT:-$REPO}
case "$PR" in /*) ;; *) die "PULL_ROOT must be absolute: $PR" ;; esac
[ -d "$PR" ] || die "PULL_ROOT $PR is not a directory"
PR=$(cd "$PR" && pwd -P)
if [ -f "$ROUND/pull_root" ]; then
  [ "$(cat "$ROUND/pull_root")" = "$PR" ] || die "$ROUND/pull_root says $(cat "$ROUND/pull_root"), this launch $PR (one pull root per round)"
fi

# --- provenance ----------------------------------------------------------------------------------
COMMIT=$(git rev-parse --short HEAD) || die "not a git repo"
DIRTY=$(git status --porcelain)   # tracked changes and untracked non-ignored files: both are synced to the pod
if [ -n "$DIRTY" ]; then
  [ "${ALLOW_DIRTY:-0}" = 1 ] || die "uncommitted changes or untracked files (the working tree is what gets synced); commit, or ALLOW_DIRTY=1:
$DIRTY"
  COMMIT="$COMMIT+dirty"
fi
say "round $ROUND, code $COMMIT on $(git rev-parse --abbrev-ref HEAD), IDLE=${IDLE}m, NPROC=$NPROC, UNIT_TIMEOUT=$UNIT_TIMEOUT, pull root $PR$([ "$DRY" = 1 ] && echo ', DRY')"
[ -n "$DIRTY" ] && say "dirty: $(echo "$DIRTY" | tr '\n' ';')"

# --- validation ----------------------------------------------------------------------------------
print_unit() {  # tree, env, run_v2.sh args... -> run_v2.sh's unit dir (PRINT_UNIT=1: the layout lives in run_v2.sh only)
  local tree=$1 env=$2; shift 2
  local -a envw; eval "envw=($env)"
  env ${envw[@]+"${envw[@]}"} RUNS="$tree" PRINT_UNIT=1 bash run_v2.sh "$@" 2>/dev/null
}

CLUSTERS=()
for f in "$@"; do
  [ -s "$f" ] || die "missing or empty actions file $f"
  c=$(basename "$f" .txt)
  [[ "$c" =~ ^[a-z][a-z0-9-]{0,30}$ ]] || die "cluster name '$c' (from $f): lowercase letters, digits and '-', starting with a letter"
  units=" "; n=0
  while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    n=$((n + 1))
    [ "$(printf '%s' "$line" | tr -cd '|' | wc -c | tr -d ' ')" = 4 ] || die "$f line $n: not 5 |-separated fields"
    IFS='|' read -r tree env args unit flag <<< "$line"
    case "$tree" in ''|/*|*..*) die "$f line $n: tree must be relative without '..': $tree" ;; esac
    eval "envw=($env)" 2>/dev/null || die "$f line $n: environment field is not shell words: $env"
    for w in ${envw[@]+"${envw[@]}"}; do
      case "$w" in
        RUNS=*|PY=*|DRY=*|PRINT_UNIT=*|UNIT_TIMEOUT=*) die "$f line $n: reserved key in the environment field: $w" ;;
        SAVE_STM=*) case "$w" in SAVE_STM=0|SAVE_STM=1) ;; *) die "$f line $n: SAVE_STM must be 0 or 1: $w" ;; esac ;;
      esac
    done
    eval "argw=($args)" 2>/dev/null || die "$f line $n: arguments are not shell words: $args"
    exp=$(print_unit "$tree" "$env" ${argw[@]+"${argw[@]}"})
    [ -n "$exp" ] && [ "$exp" = "$unit" ] || die "$f line $n: unit dir '$unit' is not run_v2.sh's '${exp:-(run_v2.sh refused)}'"
    [ -e "$PR/$unit" ] && die "$f line $n: $PR/$unit already exists on this machine (results are never merged; archive it or drop the line)"
    [ "$PR" != "$REPO" ] && [ -e "$REPO/$unit" ] && die "$f line $n: $REPO/$unit already exists in the repo (results are never merged)"
    case "$units" in *" $unit "*) die "$f line $n: unit $unit listed twice" ;; esac
    units="$units$unit "
    ( env ${envw[@]+"${envw[@]}"} RUNS="$tree" DRY=1 bash run_v2.sh ${argw[@]+"${argw[@]}"} > /dev/null ) || die "$f line $n: run_v2.sh rejects: $env | $args"
  done < "$f"
  [ "$n" -gt 0 ] || die "$f has no units"
  # An STM unit (rl, actor) other than pre-training needs its pre-training unit in the same file (the pod has no other
  # copy): run_v2.sh's own layout for "pretrain" with the same model, pair, seed and environment.
  while IFS='|' read -r tree env args unit flag || [ -n "$tree" ]; do
    [ -z "$tree" ] && continue
    eval "argw=($args)"
    case "${argw[1]:-}" in rl|actor) ;; *) continue ;; esac
    [ "${argw[0]}" = pretrain ] && continue
    pu=$(print_unit "$tree" "$env" pretrain "${argw[1]}" "${argw[2]:-}" "${argw[3]:-}")
    [ -n "$pu" ] || die "$f: cannot get the pre-training unit of $unit from run_v2.sh"
    case "$units" in *" $pu "*) ;; *) die "$f: $unit needs pre-training $pu, which is not in the file (the pod has no other copy)" ;; esac
  done < "$f"
  if [ -e "$ROUND/actions/$c.txt" ]; then
    cmp -s "$f" "$ROUND/actions/$c.txt" || die "$ROUND/actions/$c.txt exists with different content (cluster name reused?)"
  fi
  for o in ${CLUSTERS[@]+"${CLUSTERS[@]}"}; do [ "$o" = "$c" ] && die "cluster $c given twice"; done
  CLUSTERS+=("$c")
  say "validated $f: $n units -> cluster $c"
done

# --- every cluster's sky state must be readable before anything is launched -------------------------
# (an UNREADABLE state is never taken for "absent": launching onto an existing cluster would re-run its job)
TOLAUNCH=()
for c in "${CLUSTERS[@]}"; do
  sky_row "$c"
  case "$SKY_STATE" in
    ROW) say "$c already exists in sky status; not launching it again: $SKY_LINE" ;;
    ABSENT) TOLAUNCH+=("$c") ;;
    *) die "sky status for $c is UNREADABLE (sky failed, timed out or printed something unexpected); nothing launched" ;;
  esac
done

# --- launch (backgrounded: sky launch blocks through provisioning, section 7) -----------------------
if [ "$DRY" = 1 ]; then
  for f in "$@"; do
    c=$(basename "$f" .txt); B64=$(base64 < "$f" | tr -d '\n')
    say "DRY: sky launch -c $c sky/v2_pod.yaml --env JOB=$c --env ROUND=$ROUND --env COMMIT=$COMMIT --env NPROC=$NPROC --env UNIT_TIMEOUT=$UNIT_TIMEOUT --env ACTIONS_B64=<${#B64} chars> -y -d"
  done
  say "DRY: nothing launched, no actions copies written, no reaper started"; exit 0
fi
mkdir -p "$ROUND/actions" || die "cannot create $ROUND/actions"
[ -f "$ROUND/pull_root" ] || echo "$PR" > "$ROUND/pull_root"
PIDS=(); LRC=()
i=0
for f in "$@"; do
  c=$(basename "$f" .txt)
  PIDS[$i]=0; LRC[$i]=""
  case " ${TOLAUNCH[*]-} " in *" $c "*) ;; *) i=$((i + 1)); continue ;; esac
  cp "$f" "$ROUND/actions/$c.txt" || die "cannot copy $f to $ROUND/actions"
  B64=$(base64 < "$f" | tr -d '\n')
  say "launching $c"
  nohup perl -e 'alarm shift; exec @ARGV or exit 127' "${LAUNCH_TIMEOUT:-5400}" \
      sky launch -c "$c" sky/v2_pod.yaml --env JOB="$c" --env ROUND="$ROUND" --env COMMIT="$COMMIT" \
      --env NPROC="$NPROC" --env UNIT_TIMEOUT="$UNIT_TIMEOUT" --env ACTIONS_B64="$B64" -y -d \
      < /dev/null > "$ROUND/$c.launch.log" 2>&1 &
  PIDS[$i]=$!
  i=$((i + 1))
  sleep 3
done

# --- the reaper (one per round) -------------------------------------------------------------------
p=$(reaper_pid "$ROUND")
if [ "$p" = "?" ]; then
  say "REAPER LOCK $ROUND/reaper.lock has no readable pid: no reaper started. Check, remove it, and start the reaper by hand"
elif [ -n "$p" ]; then
  say "reaper already running on $ROUND (pid $p)"
else
  IDLE=$IDLE nohup caffeinate -i bash sky/v2_reap.sh "$ROUND" > /dev/null 2>&1 &
  sleep 3
  p=$(reaper_pid "$ROUND")
  case "$p" in ""|"?") say "REAPER DID NOT START: see $ROUND/reap.log; start it by hand: nohup caffeinate -i bash sky/v2_reap.sh $ROUND > /dev/null 2>&1 &" ;;
    *) say "reaper started (pid $p, under caffeinate -i; log $ROUND/reap.log)" ;; esac
fi

# --- launch exit codes, and autostop applied separately and confirmed in sky status --------------------
collect() {  # $1 = index: record the exit code of an ended launch (wait), once
  local i=$1 pid=${PIDS[$1]} c=${CLUSTERS[$1]} r
  [ "$pid" = 0 ] || [ -n "${LRC[$i]}" ] && return 0
  kill -0 "$pid" 2>/dev/null && return 0
  wait "$pid"; r=$?; LRC[$i]=$r
  if [ "$r" = 0 ]; then say "$c sky launch exited 0"
  else say "$c LAUNCH/SETUP FAILED: sky launch exited $r$([ "$r" = 142 ] && echo " (LAUNCH_TIMEOUT)"): see $ROUND/$c.launch.log"; fi
}
say "waiting for clusters to come UP to apply autostop -i $IDLE --down (money backstop only)"
deadline=$(( $(date +%s) + ${AUTOSTOP_WAIT:-3600} ))
pending=" ${TOLAUNCH[*]-} "
while [ "$(echo "$pending" | tr -d ' ')" != "" ] && [ "$(date +%s)" -lt "$deadline" ]; do
  i=0
  for c in "${CLUSTERS[@]}"; do
    collect $i; idx=$i; i=$((i + 1))
    case "$pending" in *" $c "*) ;; *) continue ;; esac
    sky_row "$c"
    case "$SKY_STATE" in
      UNREADABLE) say "$c sky status UNREADABLE; retrying"; continue ;;
      ABSENT)
        if [ -n "${LRC[$idx]}" ]; then
          say "$c LAUNCH FAILED (sky status lists no such cluster and the launch ended, exit ${LRC[$idx]}): see $ROUND/$c.launch.log"
          pending=${pending/ $c / }
        fi
        continue ;;
    esac
    if has_autostop "$SKY_LINE"; then say "$c autostop confirmed: $SKY_LINE"; pending=${pending/ $c / }; continue; fi
    case "$(row_status "$SKY_LINE")" in
      UP) tmo "$SKY_TIMEOUT" sky autostop "$c" -i "$IDLE" --down -y < /dev/null >> "$ROUND/$c.launch.log" 2>&1 \
            || say "$c: sky autostop returned non-zero or timed out; retrying" ;;
    esac
  done
  [ "$(echo "$pending" | tr -d ' ')" = "" ] || sleep 30
done
# Autostop is on, so the money backstop holds; now wait for every sky launch to return (each is capped by
# LAUNCH_TIMEOUT), so that its exit code is logged: a launch can still be syncing files or running setup here.
i=0
for c in "${CLUSTERS[@]}"; do
  if [ "${PIDS[$i]}" != 0 ] && [ -z "${LRC[$i]}" ] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
    say "$c: waiting for sky launch to return (file sync / setup; capped at ${LAUNCH_TIMEOUT:-5400}s)"
    wait "${PIDS[$i]}" 2>/dev/null
  fi
  collect $i; i=$((i + 1))
done
say "--- sky status ---"
tmo "$SKY_TIMEOUT" sky status < /dev/null 2>&1 | strip | tee -a "$LOG"
if [ "$(echo "$pending" | tr -d ' ')" != "" ]; then
  say "AUTOSTOP NOT CONFIRMED for:$pending -- set it by hand: sky autostop <cluster> -i $IDLE --down -y (the reaper also applies it to an UP row without one)"
  exit 1
fi
say "launch finished; check the reaper with: bash sky/v2_status.sh $ROUND"
