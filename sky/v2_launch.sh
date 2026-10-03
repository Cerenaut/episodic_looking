#!/bin/bash
# Launch K v2 pod jobs from K actions files, one pod each (sky/v2_pod.yaml -> sky/v2_pod_job.sh), then apply
# autostop separately (sky launch -d ignores -i/--down: running_experiments_kb.md section 6).
#
# Autostop here is a MONEY BACKSTOP ONLY, with a long idle window (IDLE minutes, default 240): nothing is mirrored
# off a pod while it runs, so an autodown that fires on a finished job destroys its results (section 5). Run
# sky/v2_reap.sh alongside every round; it pulls, verifies locally and only then tears a pod down.
#
# Usage: bash sky/v2_launch.sh <round dir> <actions file> [<actions file> ...]
#   round dir     relative to the repo, e.g. runs_local/v2_round/cont_rl_20261004; holds actions/<cluster>.txt (the
#                 launcher's copies, which the reaper verifies against), launch logs, and later the pulled pod logs.
#   actions file  basename = cluster name (lowercase letters, digits, '-'), e.g. v2c-rl-pb-s1.txt. Format: header of
#                 sky/v2_pod_job.sh. Validated here: unit dir = run_v2.sh's layout, an STM unit's pre-training is in
#                 the same file, and no unit dir exists on this machine yet (results are never merged or overwritten).
# Environment: IDLE (autostop minutes, default 240), NPROC (per pod, default 3), ALLOW_DIRTY=1 (launch from a tree with
# uncommitted tracked changes; the commit is then recorded as +dirty), AUTOSTOP_WAIT (s, default 3600), DRY=1
# (validate and print the sky commands only).
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
ROUND=${1:?round dir (relative to the repo)}
shift
[ $# -ge 1 ] || { echo "usage: $0 <round dir> <actions file> [...]" >&2; exit 2; }
case "$ROUND" in /*|*..*) echo "round dir must be relative to the repo, without '..'" >&2; exit 2 ;; esac
IDLE=${IDLE:-240}
NPROC=${NPROC:-3}
mkdir -p "$ROUND/actions"
LOG=$ROUND/launch.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { say "LAUNCH REFUSED: $*"; exit 2; }
strip() { perl -pe 's/\e\[[0-9;]*[mK]//g'; }
status_line() { sky status "$1" 2>/dev/null | strip | awk -v c="$1" '$1==c'; }

command -v sky >/dev/null || die "sky not on PATH (expected ~/bin/sky -> ~/.venvs/sky)"

# --- provenance ----------------------------------------------------------------------------------
COMMIT=$(git rev-parse --short HEAD) || die "not a git repo"
DIRTY=$(git status --porcelain --untracked-files=no)
if [ -n "$DIRTY" ]; then
  [ "${ALLOW_DIRTY:-0}" = 1 ] || die "uncommitted tracked changes (the working tree is what gets synced); commit, or ALLOW_DIRTY=1:
$DIRTY"
  COMMIT="$COMMIT+dirty"
fi
say "round $ROUND, code $COMMIT on $(git rev-parse --abbrev-ref HEAD), IDLE=${IDLE}m, NPROC=$NPROC"
[ -n "$DIRTY" ] && say "dirty: $(echo "$DIRTY" | tr '\n' ';')"

# --- validation ----------------------------------------------------------------------------------
expected_unit() {  # tree, env, args -> run_v2.sh's UNIT_DIR (mirrors run_v2.sh; a mismatch refuses the launch)
  local tree=$1 env=$2 args=$3 suffix="" setting model cc seed a5 n pair base
  local -a envw; eval "envw=($env)"
  local w; for w in ${envw[@]+"${envw[@]}"}; do [ "$w" = LTM=e40 ] && suffix=_e40; done
  eval "set -- $args"
  setting=${1:-} model=${2:-} cc=${3:-} seed=${4:-} a5=${5:-} n=${6:-}
  pair=pair$(echo $cc | tr ' ' '_')
  base=$tree/$setting$suffix/$model/$pair/seed$seed
  case "$setting" in
    pretrain) echo "$base" ;;
    baseline) [ "$model" = ltm ] && echo "$tree/baseline$suffix/ltm/$pair" || echo "$base" ;;
    continual) echo "$base/order$(echo ${a5:-3 4 5} | tr ' ' '_')" ;;
    stream) echo "$base/fine$a5" ;;
    fewshot) echo "$base/fine${a5}_n$n" ;;
    *) echo "?" ;;
  esac
}
pretrain_unit() {  # tree, env, args -> the pre-training unit an STM unit needs (empty if none)
  local tree=$1 env=$2 args=$3 suffix=""
  local -a envw; eval "envw=($env)"
  local w; for w in ${envw[@]+"${envw[@]}"}; do [ "$w" = LTM=e40 ] && suffix=_e40; done
  eval "set -- $args"
  case "${2:-}" in rl|actor) ;; *) return ;; esac
  [ "${1:-}" = pretrain ] && return
  echo "$tree/pretrain$suffix/$2/pair$(echo ${3:-} | tr ' ' '_')/seed${4:-}"
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
    exp=$(expected_unit "$tree" "$env" "$args")
    [ "$exp" = "$unit" ] || die "$f line $n: unit dir '$unit' is not run_v2.sh's '$exp'"
    [ -e "$unit" ] && die "$f line $n: $unit already exists on this machine (results are never merged; archive it or drop the line)"
    case "$units" in *" $unit "*) die "$f line $n: unit $unit listed twice" ;; esac
    units="$units$unit "
    ( eval "envw=($env)"; eval "set -- $args"
      env ${envw[@]+"${envw[@]}"} RUNS="$tree" DRY=1 bash run_v2.sh "$@" > /dev/null ) || die "$f line $n: run_v2.sh rejects: $env | $args"
  done < "$f"
  [ "$n" -gt 0 ] || die "$f has no units"
  while IFS='|' read -r tree env args unit flag || [ -n "$tree" ]; do
    [ -z "$tree" ] && continue
    pu=$(pretrain_unit "$tree" "$env" "$args")
    [ -z "$pu" ] || case "$units" in *" $pu "*) ;; *) die "$f: $unit needs pre-training $pu, which is not in the file (the pod has no other copy)" ;; esac
  done < "$f"
  if [ -e "$ROUND/actions/$c.txt" ]; then
    cmp -s "$f" "$ROUND/actions/$c.txt" || die "$ROUND/actions/$c.txt exists with different content (cluster name reused?)"
  fi
  for o in ${CLUSTERS[@]+"${CLUSTERS[@]}"}; do [ "$o" = "$c" ] && die "cluster $c given twice"; done
  CLUSTERS+=("$c")
  say "validated $f: $n units -> cluster $c"
done

# --- launch (backgrounded: sky launch blocks through provisioning, section 7) -----------------------
PIDS=()
for f in "$@"; do
  c=$(basename "$f" .txt)
  if [ -n "$(status_line "$c")" ]; then say "$c already exists in sky status; not launching it again"; PIDS+=(0); continue; fi
  cp "$f" "$ROUND/actions/$c.txt"
  B64=$(base64 < "$f" | tr -d '\n')
  if [ "${DRY:-0}" = 1 ]; then
    say "DRY: sky launch -c $c sky/v2_pod.yaml --env JOB=$c --env ROUND=$ROUND --env COMMIT=$COMMIT --env NPROC=$NPROC --env ACTIONS_B64=<${#B64} chars> -y -d"
    PIDS+=(0); continue
  fi
  say "launching $c"
  nohup sky launch -c "$c" sky/v2_pod.yaml --env JOB="$c" --env ROUND="$ROUND" --env COMMIT="$COMMIT" \
      --env NPROC="$NPROC" --env ACTIONS_B64="$B64" -y -d > "$ROUND/$c.launch.log" 2>&1 &
  PIDS+=($!)
  sleep 3
done
[ "${DRY:-0}" = 1 ] && { say "DRY: nothing launched"; exit 0; }

# --- autostop, applied separately and confirmed in sky status ----------------------------------------
say "waiting for clusters to come UP to apply autostop -i $IDLE --down (money backstop only)"
deadline=$(( $(date +%s) + ${AUTOSTOP_WAIT:-3600} ))
pending=" ${CLUSTERS[*]} "
while [ "$pending" != " " ] && [ "$(date +%s)" -lt "$deadline" ]; do
  i=0
  for c in "${CLUSTERS[@]}"; do
    pid=${PIDS[$i]}; i=$((i + 1))
    case "$pending" in *" $c "*) ;; *) continue ;; esac
    line=$(status_line "$c")
    if echo "$line" | grep -qE '[0-9]+[hm] \(down\)'; then
      say "$c autostop confirmed: $line"; pending=${pending/ $c / }; continue
    fi
    if echo "$line" | grep -qw UP; then
      sky autostop "$c" -i "$IDLE" --down -y >> "$ROUND/$c.launch.log" 2>&1 || say "$c: sky autostop returned non-zero; retrying"
      continue
    fi
    if [ -z "$line" ] && [ "$pid" != 0 ] && ! kill -0 "$pid" 2>/dev/null; then
      say "$c LAUNCH FAILED (no cluster, launch process ended): see $ROUND/$c.launch.log"; pending=${pending/ $c / }
    fi
  done
  [ "$pending" = " " ] || sleep 30
done
say "--- sky status ---"
sky status 2>&1 | strip | tee -a "$LOG"
if [ "$pending" != " " ]; then
  say "AUTOSTOP NOT CONFIRMED for:$pending -- set it by hand: sky autostop <cluster> -i $IDLE --down -y"
  exit 1
fi
say "launch finished. Now run the reaper: nohup bash sky/v2_reap.sh $ROUND > /dev/null 2>&1 &"
