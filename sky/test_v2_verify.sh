#!/bin/bash
# Tests of the v2 cloud path on fake trees, with a fake run_v2.sh and fake sky, ssh, rsync, timeout and osascript: no
# GPU, no pod, no real results. Covers the destructive predicate (sky/v2_verify.sh), the pod job's sentinels, unit
# timeout and validation (sky/v2_pod_job.sh), the reaper's decisions (sky/v2_reap.sh: teardown, HOLD, disk guard,
# UNREADABLE sky state, failed-pod pull, lock, setup check), the status script (sky/v2_status.sh) and the launcher's
# refusals, DRY and launch-failure paths (sky/v2_launch.sh). Run it after any change to these scripts, before a live
# round. Every known-good case must pass and every known-bad one must be refused.
# Usage: bash sky/test_v2_verify.sh <empty scratch dir>     (never a results directory)
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd)
W=${1:?scratch dir}
mkdir -p "$W" && W=$(cd "$W" && pwd)
[ -z "$(ls -A "$W")" ] || { echo "scratch dir $W is not empty" >&2; exit 2; }
ROUND=runs_local/test_round
J=v2t-job
fails=0; passes=0
ok() { echo "pass  $*"; passes=$((passes + 1)); }
wrong() { echo "WRONG $*"; fails=$((fails + 1)); }
expect() {  # $1 = label; the rest = a test command that must succeed
  local l=$1; shift
  if "$@"; then ok "$l"; else wrong "$l"; fi
}
check() {  # $1 = expected (OK|FAIL), $2 = label, $3 = mac root
  local out rc got
  out=$(bash "$SRC/sky/v2_verify.sh" "$ROUND" "$J" "$3" 2>&1); rc=$?
  got=FAIL; [ $rc -eq 0 ] && got=OK
  if [ "$got" = "$1" ]; then ok "verify expect $1  $2"; else wrong "verify expect $1 got $got  $2"; echo "$out" | sed 's/^/      /'; fi
}

# --- fakes ---------------------------------------------------------------------------------------------
# A fake run_v2.sh that writes what the real one writes (layout, results, checkpoints by SAVE_STM), with PRINT_UNIT
# and DRY like the real one, FAKE_FAIL=1 (fail after making the unit dir) and FAKE_SLEEP=<s> (a hung unit).
make_repo() {  # $1 = dir
  mkdir -p "$1/sky"
  cp "$SRC/sky/v2_pod_job.sh" "$1/sky/"
  cat > "$1/run_v2.sh" <<'EOF'
#!/bin/bash
set -u
cd "$(dirname "$0")"
S=$1 M=$2 CC=$3 SEED=$4 A5=${5:-} N=${6:-}
P=pair$(echo $CC | tr ' ' '_')
D=${RUNS:-runs_v2}/$S/$M/$P/seed$SEED
case "$S" in pretrain) ;; continual) D=$D/order$(echo $A5 | tr ' ' '_') ;; stream) D=$D/fine$A5 ;; fewshot) D=$D/fine${A5}_n$N ;; *) echo "unknown setting" >&2; exit 2 ;; esac
[ "${PRINT_UNIT:-0}" = 1 ] && { echo "$D"; exit 0; }
[ "${DRY:-0}" = 1 ] && { echo "python fake $*"; exit 0; }
[ "${FAKE_FAIL:-0}" = 1 ] && { mkdir -p "$D"; echo boom > "$D/job.log"; exit 1; }
mkdir -p "$D/.lock"
[ -n "${FAKE_SLEEP:-}" ] && sleep "$FAKE_SLEEP"
R="$D/cifar_100/${S}_[3, 4]_500/2026/10/03/12-00-00"
mkdir -p "$R"
printf '[0, 1], [3], evaluate, 0, 0.5\n' > "$R/results_$S.txt"
printf '[0, 1], [3], evaluate, 0, 0.6\n' > "$R/results_${S}_val.txt"
{ echo "log $*"; echo "EXTRA=[${EXTRA:-}] SAVE_STM=[${SAVE_STM:-}]"; } > "$D/job.log"; echo "abc123" > "$D/code_commit.txt"
case "$S" in
  pretrain) head -c 5000 /dev/zero > "$D/stm_pretrain.pth" ;;
  continual) [ "${SAVE_STM:-0}" = 1 ] && head -c 4000 /dev/zero > "$D/stm_phase3.pth" ;;
  *) [ "${SAVE_STM:-0}" = 1 ] && head -c 3000 /dev/zero > "$D/stm_final.pth" ;;
esac
rmdir "$D/.lock"
touch "$D/job.done"
EOF
}
# A minimal GNU timeout for the Mac (the pod has the real one): timeout [--kill-after=K] DURATION cmd...; the command
# runs in its own process group, which gets TERM at DURATION and KILL K seconds later; exit 124 on a timeout.
PB=$W/podbin; mkdir -p "$PB"
cat > "$PB/timeout" <<'EOF'
#!/usr/bin/perl
use POSIX ":sys_wait_h";
my $k = 0; if ($ARGV[0] =~ /^--kill-after=(\d+)$/) { $k = $1; shift; }
my $d = shift; my %m = (s => 1, m => 60, h => 3600, d => 86400);
$d =~ /^(\d+)([smhd]?)$/ or die "bad duration"; $d = $1 * ($2 ? $m{$2} : 1);
my $pid = fork; die "fork" unless defined $pid;
if ($pid == 0) { setpgrp(0, 0); exec @ARGV; exit 127; }
my $timed = 0;
$SIG{ALRM} = sub { if (!$timed) { $timed = 1; kill 'TERM', -$pid; alarm($k) if $k; } else { kill 'KILL', -$pid; } };
alarm $d;
my $r; do { $r = waitpid($pid, 0) } while ($r == -1 && $!{EINTR});
my $st = $?; alarm 0;
exit 124 if $timed;
exit($st & 127 ? 128 + ($st & 127) : $st >> 8);
EOF
chmod +x "$PB/timeout"

T=runs_local/fake_tree
cat > "$W/actions.txt" <<EOF
$T|PRETRAIN_EPOCHS=1|pretrain rl "0 1" 1|$T/pretrain/rl/pair0_1/seed1|
$T|EPOCHS=1 EXTRA="--ltm-obs-cache check"|continual rl "0 1" 1 "3 4"|$T/continual/rl/pair0_1/seed1/order3_4|
$T|EPOCHS=2|stream rl "0 1" 1 3|$T/stream/rl/pair0_1/seed1/fine3|
$T|EPOCHS=1 SAVE_STM=0|fewshot rl "0 1" 1 3 4|$T/fewshot/rl/pair0_1/seed1/fine3_n4|
EOF
U_CONT=$T/continual/rl/pair0_1/seed1/order3_4
U_FEW=$T/fewshot/rl/pair0_1/seed1/fine3_n4

run_pod() {  # $1 = pod repo dir, $2 = actions file (default the main one); extra env in the caller
  ( cd "$1" && PATH="$PB:$PATH" ROUND=$ROUND JOB=$J PY=python3 REQUIRE_CUDA=0 NPROC=${NPROC:-2} UNIT_TIMEOUT=${UT-60} \
      bash sky/v2_pod_job.sh "${2:-$W/actions.txt}" > pod_stdout.txt 2>&1 )
}
pull() {  # $1 = pod repo, $2 = mac root: the reaper's pull, local to local
  mkdir -p "$2/$ROUND/actions" && cp "$W/actions.txt" "$2/$ROUND/actions/$J.txt"
  while IFS='|' read -r tree env args unit flag; do
    [ -z "$tree" ] && continue
    mkdir -p "$2/$unit" && rsync -a --ignore-existing "$1/$unit/" "$2/$unit/"
  done < "$W/actions.txt"
  mkdir -p "$2/$ROUND/pods/$J" && rsync -a "$1/$ROUND/pods/$J/" "$2/$ROUND/pods/$J/"
}
fresh() { rm -rf "$W/mac" && cp -R "$W/mac_good" "$W/mac"; }
M=$W/mac/$ROUND/pods/$J/manifest.tsv
sed_manifest() {  # $1 = perl expression applied to each line of the manifest of $W/mac; REL in the environment
  REL=${REL:-} perl -pi -e "$1" "$M"
}

# === the pod job ==========================================================================================
make_repo "$W/pod"
run_pod "$W/pod"
[ -f "$W/pod/$ROUND/pods/$J/JOB_COMPLETE" ] && [ ! -e "$W/pod/$ROUND/pods/$J/JOB_FAILED" ] \
  && ok "pod job leaves JOB_COMPLETE only" || { wrong "pod job sentinels"; cat "$W/pod/pod_stdout.txt"; }
expect "quoted EXTRA reaches run_v2.sh as one word; SAVE_STM=1 by default" \
  grep -qF 'EXTRA=[--ltm-obs-cache check] SAVE_STM=[1]' "$W/pod/$U_CONT/job.log"
expect "a line's SAVE_STM=0 reaches run_v2.sh" grep -qF 'SAVE_STM=[0]' "$W/pod/$U_FEW/job.log"
expect "manifest lines are size<TAB>sha256<TAB>path" \
  awk -F'\t' 'NF != 3 || $1 !~ /^[0-9]+$/ || $2 !~ /^[0-9a-f]{64}$/ { bad = 1 } END { exit bad || NR == 0 }' "$W/pod/$ROUND/pods/$J/manifest.tsv"
expect "manifest leaves out dot paths" bash -c "! grep -q '/\.' '$W/pod/$ROUND/pods/$J/manifest.tsv'"
pull "$W/pod" "$W/mac_good"
check OK "complete pull (one unit with SAVE_STM=0 and no checkpoint)" "$W/mac_good"

# --- verification: known-bad copies ----------------------------------------------------------------------
check FAIL "nothing pulled" "$W/empty_mac"
fresh; rm "$W/mac/$ROUND/pods/$J/JOB_COMPLETE";            check FAIL "no JOB_COMPLETE" "$W/mac"
fresh; touch "$W/mac/$ROUND/pods/$J/JOB_FAILED";           check FAIL "JOB_FAILED present" "$W/mac"
fresh; rm "$W/mac/$ROUND/actions/$J.txt";                  check FAIL "launcher actions copy missing" "$W/mac"
fresh; echo "x|y|pretrain a|x/u|" >> "$W/mac/$ROUND/actions/$J.txt"; check FAIL "actions differ" "$W/mac"
fresh; rm "$M";                                            check FAIL "manifest missing" "$W/mac"
fresh; : > "$M";                                           check FAIL "manifest empty" "$W/mac"
fresh; echo "garbage" >> "$M";                             check FAIL "manifest garbage line" "$W/mac"
fresh; printf '12\t/etc/passwd\n' >> "$M";                 check FAIL "manifest line in the old two-field format" "$W/mac"
fresh; printf '12\t%064d\t/etc/passwd\n' 0 >> "$M";        check FAIL "manifest path outside units" "$W/mac"
fresh; f=$(find "$W/mac/$U_CONT" -name 'results_continual.txt' | head -1); printf 'x' >> "$f"; check FAIL "a results file differs in size" "$W/mac"
fresh; f=$(find "$W/mac/$U_CONT" -name 'results_continual.txt' | head -1); perl -pi -e 's/0\.5/0.7/' "$f"
       check FAIL "a results file: same size, different content (sha256)" "$W/mac"
fresh; echo extra > "$W/mac/$U_CONT/extra.txt";            check FAIL "an extra file in a unit (not in the manifest)" "$W/mac"
fresh; f=$(find "$W/mac/$U_CONT" -name 'results_continual.txt' | head -1); rel=${f#"$W/mac/"}
       REL=$rel sed_manifest 's/^(\d+)\t[0-9a-f]{64}\t(\Q$ENV{REL}\E)\n/$1\t${\("7" x 64)}\t$2\n/' 
       check FAIL "stale manifest (lists another version of a file)" "$W/mac"
fresh; REL=$U_CONT/code_commit.txt sed_manifest '$_ = "" if $_ eq "\t$ENV{REL}\n" || index($_, "\t$ENV{REL}\n") >= 0' 
       check FAIL "manifest without a file that is here (stale or extra file)" "$W/mac"
fresh; touch "$W/mac/$U_CONT/.DS_Store"; mkdir "$W/mac/$U_CONT/.lock"; check OK "dot paths here are ignored" "$W/mac"
fresh; find "$W/mac/$T/stream" -name 'results_stream_val.txt' -delete; check FAIL "a _val results file missing" "$W/mac"
fresh; rm "$W/mac/$U_CONT/job.done";                       check FAIL "a job.done missing" "$W/mac"
fresh; rm "$W/mac/$T/pretrain/rl/pair0_1/seed1/stm_pretrain.pth"; check FAIL "pre-training checkpoint missing" "$W/mac"
fresh; rm "$W/mac/$U_CONT/stm_phase3.pth"; sed_manifest '$_ = "" if /stm_phase3\.pth\n/' 
       check FAIL "continual checkpoint missing (and out of the manifest) with SAVE_STM=1" "$W/mac"
fresh; rm -r "$W/mac/$T/stream";                           check FAIL "a whole unit missing" "$W/mac"
fresh; check OK "fresh copy is still good (the cases above did not leak)" "$W/mac"

# --- pod job: failures --------------------------------------------------------------------------------------
pod_failed() {  # $1 = pod dir: JOB_FAILED and no JOB_COMPLETE
  [ -f "$1/$ROUND/pods/$J/JOB_FAILED" ] && [ ! -e "$1/$ROUND/pods/$J/JOB_COMPLETE" ]
}
make_repo "$W/podbad"; FAKE_FAIL=1 run_pod "$W/podbad"
expect "failing pod job leaves JOB_FAILED only" pod_failed "$W/podbad"
pull "$W/podbad" "$W/mac_bad"
check FAIL "pull of a failed job" "$W/mac_bad"

make_repo "$W/podmal"; printf 'only|three|fields\n' > "$W/mal.txt"; run_pod "$W/podmal" "$W/mal.txt"
expect "malformed actions file leaves JOB_FAILED only" pod_failed "$W/podmal"

for k in 'RUNS=x' 'PY=python' 'DRY=1' 'PRINT_UNIT=1' 'UNIT_TIMEOUT=5' 'SAVE_STM=2'; do
  d=$W/podres_${k%%=*}; make_repo "$d"
  printf '%s\n' "$T|$k|pretrain rl \"0 1\" 1|$T/pretrain/rl/pair0_1/seed1|" > "$d/act.txt"; run_pod "$d" "$d/act.txt"
  expect "pod job refuses '$k' in the environment field (JOB_FAILED, nothing run)" \
    bash -c "[ -f '$d/$ROUND/pods/$J/JOB_FAILED' ] && [ ! -e '$d/$T' ] && grep -q 'reserved key\|SAVE_STM must be' '$d/$ROUND/pods/$J/progress.log'"
done
make_repo "$W/podnot"; UT= run_pod "$W/podnot"
expect "pod job without UNIT_TIMEOUT: JOB_FAILED, nothing run" \
  bash -c "[ -f '$W/podnot/$ROUND/pods/$J/JOB_FAILED' ] && [ ! -e '$W/podnot/$T' ]"

# A hung unit: FAKE_SLEEP far beyond UNIT_TIMEOUT. The unit is killed with its process group, the job ends JOB_FAILED.
make_repo "$W/podslow"
cat > "$W/slow.txt" <<EOF
$T|PRETRAIN_EPOCHS=1|pretrain rl "0 1" 1|$T/pretrain/rl/pair0_1/seed1|
$T|EPOCHS=2 FAKE_SLEEP=37|stream rl "0 1" 1 3|$T/stream/rl/pair0_1/seed1/fine3|
EOF
t0=$(date +%s); UT=3 run_pod "$W/podslow" "$W/slow.txt"; el=$(( $(date +%s) - t0 ))
expect "unit timeout: JOB_FAILED, TIMEOUT logged, the fast unit done (${el}s)" \
  bash -c "[ -f '$W/podslow/$ROUND/pods/$J/JOB_FAILED' ] && grep -q 'TIMEOUT' '$W/podslow/$ROUND/pods/$J/progress.log' && [ -f '$W/podslow/$T/pretrain/rl/pair0_1/seed1/job.done' ] && [ $el -lt 30 ]"
expect "unit timeout: the unit's children were killed too (no 'sleep 37' left)" bash -c "! pgrep -f 'sleep 37' > /dev/null"

# === the reaper, with fake sky, ssh, rsync and osascript (REAP_TEST_BIN) ==============================================
# Fake pod = a local dir standing for ~/sky_workdir. Fake sky: a cluster is UP iff $FAKE_STATE/<c>.up exists; its
# autostop column is $FAKE_STATE/<c>.autostop (default "4h (down)"); `sky autostop` writes it and logs to autostops;
# `sky down` logs to downs. Flags: skyfail (every sky call exits 1), skygarbage (status prints no table),
# statusfail_after_down, <c>.failed_setup (sky queue), launchfail_<c>. Fake ssh: test -f, test -d, bash -s.
B=$W/bin; mkdir -p "$B"
cat > "$B/sky" <<'EOF'
#!/bin/bash
S=$FAKE_STATE
echo "$*" >> "$S/sky_calls"
[ -f "$S/skyfail" ] && { echo "sky: internal error" >&2; exit 1; }
case "$1" in
  status)
    [ -f "$S/skygarbage" ] && { echo "Something unexpected"; exit 0; }
    [ -f "$S/statusfail_after_down" ] && [ -s "$S/downs" ] && exit 1
    c=${2:-}
    if [ -n "$c" ] && [ -f "$S/$c.up" ]; then
      au="4h (down)"; [ -f "$S/$c.autostop" ] && au=$(cat "$S/$c.autostop")
      printf 'Enabled Infra: runpod\n\nClusters\nNAME  INFRA  RESOURCES  STATUS  AUTOSTOP  LAUNCHED\n%s  RunPod  1x(L4)  UP  %s  1m ago\n' "$c" "$au"
    elif [ -n "$c" ]; then
      printf 'Cluster(s) not found: \033[1m%s\033[0m.\nEnabled Infra: runpod\n\nClusters\nCluster %s not found.\n' "$c" "'$c'"
    else
      printf 'Enabled Infra: runpod\n\nClusters\nNo existing clusters.\n'
    fi ;;
  autostop)
    c=$2; i=$4; echo "$c $i" >> "$S/autostops"
    if [ $((i % 60)) = 0 ]; then echo "$((i / 60))h (down)" > "$S/$c.autostop"; else echo "${i}m (down)" > "$S/$c.autostop"; fi ;;
  down) rm -f "$S/$2.up"; echo "$2" >> "$S/downs" ;;
  queue) if [ -f "$S/$2.failed_setup" ]; then echo " ID  NAME  STATUS"; echo " 1   v2-pod  FAILED_SETUP"; else echo " 1  v2-pod  RUNNING"; fi ;;
  launch) c=$3; [ -f "$S/launchfail_$c" ] && { echo "launch failed" ; exit 3; }; touch "$S/$c.up"; echo "-" > "$S/$c.autostop" ;;
esac
exit 0
EOF
cat > "$B/ssh" <<'EOF'
#!/bin/bash
a=("$@"); n=${#a[@]}; cmd=${a[$((n-1))]}; host=${a[$((n-2))]}
[ -f "$FAKE_STATE/$host.sshfail" ] && exit 255
case "$cmd" in
  "test -f ~/sky_workdir/"*) test -f "$FAKE_POD/${cmd#"test -f ~/sky_workdir/"}" ;;
  "test -d ~/sky_workdir/"*) test -d "$FAKE_POD/${cmd#"test -d ~/sky_workdir/"}" ;;
  "bash -s") mkdir -p "$FAKE_STATE/home"; ln -sfn "$FAKE_POD" "$FAKE_STATE/home/sky_workdir"; HOME=$FAKE_STATE/home exec bash -s ;;
  *) exit 255 ;;
esac
EOF
cat > "$B/rsync" <<'EOF'
#!/bin/bash
out=(); skip=0
for a in "$@"; do
  [ $skip = 1 ] && { skip=0; continue; }
  case "$a" in -e) skip=1; continue ;; *":~/sky_workdir/"*) a="$FAKE_POD/${a#*":~/sky_workdir/"}" ;; esac
  out+=("$a")
done
echo "${out[*]}" >> "$FAKE_STATE/rsync_calls"   # (a local rsync also runs its --server side through this fake)
[ -f "$FAKE_STATE/rsyncfail" ] && exit 12
exec /usr/bin/rsync "${out[@]}"
EOF
cat > "$B/osascript" <<'EOF'
#!/bin/bash
echo "$*" >> "$FAKE_STATE/notify"
EOF
chmod +x "$B/sky" "$B/ssh" "$B/rsync" "$B/osascript"

export FAKE_STATE=$W/state
st() { rm -rf "$FAKE_STATE"; mkdir -p "$FAKE_STATE"; local x; for x in "$@"; do touch "$FAKE_STATE/$x"; done; }
MR=$W/macr
mac_reaper_root() {  # a fresh Mac repo with the scripts and the round's actions copy
  rm -rf "$MR"; mkdir -p "$MR/sky" "$MR/$ROUND/actions"
  cp "$SRC/sky/v2_reap.sh" "$SRC/sky/v2_verify.sh" "$SRC/sky/v2_lib.sh" "$SRC/sky/v2_status.sh" "$MR/sky/"
  cp "${ACTIONS:-$W/actions.txt}" "$MR/$ROUND/actions/$J.txt"
}
reap_case() {  # $1 = expect (DOWN|UP), $2 = label, $3 = fake pod dir; set up $FAKE_STATE (st) first.
  # Env: PREEXIST (a file already on the Mac), PREP (a command run on the fresh Mac root), RENV (env for the reaper)
  local got
  mac_reaper_root
  [ -n "${PREEXIST:-}" ] && { mkdir -p "$(dirname "$MR/$PREEXIST")"; echo "mac's own file" > "$MR/$PREEXIST"; }
  [ -n "${PREP:-}" ] && eval "$PREP"
  env FAKE_POD="$3" REAP_TEST_BIN="$B" MAX_PASSES=${PASSES:-1} INTERVAL=0 MIN_FREE_GB=0 ${RENV:-} bash "$MR/sky/v2_reap.sh" "$ROUND" > "$W/reap_out.txt" 2>&1
  got=UP; grep -qx "$J" "$FAKE_STATE/downs" 2>/dev/null && got=DOWN
  if [ "$got" = "$1" ]; then ok "reaper expect $1  $2"; else wrong "reaper expect $1 got $got  $2"; sed 's/^/      /' "$W/reap_out.txt"; fi
  if [ "$got" = DOWN ] && [ ! -e "$MR/$ROUND/reaped/$J" ] && [ -z "${NOREAPED:-}" ]; then wrong "reaper downed without a reaped marker ($2)"; fi
  if [ "$got" = UP ] && [ -e "$MR/$ROUND/reaped/$J" ]; then wrong "reaper wrote a reaped marker for a pod it left up ($2)"; fi
  if [ -n "${PREEXIST:-}" ] && ! grep -q "mac's own file" "$MR/$PREEXIST"; then wrong "reaper overwrote a file on the Mac ($2)"; fi
  [ -s "$MR/$ROUND/reaper.heartbeat" ] || wrong "reaper wrote no heartbeat ($2)"
  [ -e "$MR/$ROUND/reaper.lock" ] && wrong "reaper left its lock behind ($2)"
  return 0
}
held() {  # held marker, autostop widened once to 1440 (24h), notification sent
  [ -e "$MR/$ROUND/held/$J" ] && [ "$(grep -c "^$J 1440\$" "$FAKE_STATE/autostops" 2>/dev/null)" = 1 ] \
    && grep -q 'HELD' "$FAKE_STATE/notify" 2>/dev/null && grep -q "$J HELD" "$MR/$ROUND/reap.log"
}
not_held() { [ ! -e "$MR/$ROUND/held/$J" ] && ! grep -q "^$J 1440\$" "$FAKE_STATE/autostops" 2>/dev/null; }

cp -R "$W/pod" "$W/pod_running"; rm "$W/pod_running/$ROUND/pods/$J/JOB_COMPLETE"
cp -R "$W/pod" "$W/pod_lost"; find "$W/pod_lost/$T/stream" -name 'results_stream.txt' -delete
cp -R "$W/podbad" "$W/podbad_part"; rm -rf "$W/podbad_part/$T/stream" "$W/podbad_part/$U_FEW"
mkdir -p "$W/pod_empty"

st;                      reap_case UP   "cluster absent from sky status" "$W/pod"
expect "  absent: no hold, nothing pulled" bash -c "[ ! -e '$MR/$ROUND/held/$J' ] && [ ! -e '$MR/$T' ]"
st "$J.up" skyfail;      reap_case UP   "sky status fails (exit 1): UNREADABLE" "$W/pod"
expect "  sky failure logged as UNREADABLE, no reaped marker, nothing pulled" \
  bash -c "grep -q 'UNREADABLE' '$MR/$ROUND/reap.log' && [ ! -e '$MR/$ROUND/reaped/$J' ] && [ ! -e '$MR/$T' ]"
st "$J.up" skygarbage;   reap_case UP   "sky status prints no table: UNREADABLE" "$W/pod"
expect "  garbage logged as UNREADABLE" grep -q 'UNREADABLE' "$MR/$ROUND/reap.log"
st "$J.up" "$J.sshfail"; reap_case UP   "ssh fails" "$W/pod"
expect "  ssh failure: no hold, nothing pulled" bash -c "[ ! -e '$MR/$ROUND/held/$J' ] && [ ! -e '$MR/$T' ]"
st "$J.up";              reap_case UP   "job still running (no sentinel)" "$W/pod_running"
expect "  running: not held" not_held
st "$J.up" rsyncfail;    reap_case UP   "rsync fails" "$W/pod"
expect "  rsync failure: HELD (autostop 24h once, notified)" held
st "$J.up";              reap_case UP   "sentinel, but a manifest file is gone from the pod" "$W/pod_lost"
expect "  verification failure: HELD" held
st "$J.up"; PASSES=3 reap_case UP "verification failure over 3 passes" "$W/pod_lost"
expect "  hold applied once only over 3 passes" held
st "$J.up"; RENV="MIN_FREE_GB=99999999" reap_case UP "disk low" "$W/pod"
expect "  disk low: HELD, no unit pulled, logged" bash -c "[ -e '$MR/$ROUND/held/$J' ] && [ ! -e '$MR/$T' ] && grep -q 'NOT PULLED: disk' '$MR/$ROUND/reap.log'"
st "$J.up"; PREP='echo /nonexistent/pull_root_zz > "$MR/$ROUND/pull_root"' reap_case UP "pull root missing: free space UNREADABLE" "$W/pod"
expect "  unreadable free space: HELD, nothing pulled" bash -c "[ -e '$MR/$ROUND/held/$J' ] && grep -q 'UNREADABLE' '$MR/$ROUND/reap.log' && [ ! -e /nonexistent/pull_root_zz ]"
st "$J.up";              reap_case UP   "JOB_FAILED" "$W/podbad"
expect "  JOB_FAILED: pulled once (failed_pulled) and HELD" bash -c "[ -e '$MR/$ROUND/failed_pulled/$J' ] && [ -e '$MR/$ROUND/held/$J' ]"
st "$J.up"; PASSES=3 reap_case UP "JOB_FAILED, some units never started, 3 passes" "$W/podbad_part"
expect "  failed pod: unit dirs that never existed on the pod are not created here" \
  bash -c "[ ! -e '$MR/$T/stream/rl/pair0_1/seed1/fine3' ] && [ ! -e '$MR/$U_FEW' ] && [ -d '$MR/$U_CONT' ] && grep -q 'never existed on the pod' '$MR/$ROUND/reap.log'"
expect "  failed pod: pulled once only over 3 passes (1 metadata + 2 units rsyncs)" \
  bash -c "[ \"\$(grep -vc '^--server' '$FAKE_STATE/rsync_calls')\" = 3 ] || { cat '$FAKE_STATE/rsync_calls'; false; }"
expect "  failed pod: held once" held
st "$J.up"; PREEXIST=$T/stream/rl/pair0_1/seed1/fine3/job.log reap_case UP "a different file already on the Mac (kept, not overwritten)" "$W/pod"
expect "  differing file: HELD" held
st "$J.up"; PREP='mkdir -p "$MR/$ROUND/pods/$J"; printf "1\t%064d\told/stale\n" 0 > "$MR/$ROUND/pods/$J/manifest.tsv"'
            reap_case DOWN "a stale manifest on the Mac is replaced by the pod's" "$W/pod"
st "$J.up" "$J.noauto"; echo "-" > "$FAKE_STATE/$J.autostop"; reap_case UP "row without autostop (job running)" "$W/pod_running"
expect "  reaper applied autostop -i 240" grep -qx "$J 240" "$FAKE_STATE/autostops"
st "$J.up" statusfail_after_down; NOREAPED=1 reap_case DOWN "sky down issued, then sky status UNREADABLE" "$W/pod"
expect "  no reaped marker while sky status is unreadable after down (verified marker kept)" \
  bash -c "[ ! -e '$MR/$ROUND/reaped/$J' ] && [ -e '$MR/$ROUND/verified/$J' ]"
st "$J.up" "$J.failed_setup"; PREP='mkdir -p "$MR/$ROUND/first_up"; echo 1000 > "$MR/$ROUND/first_up/$J"'
            reap_case UP "UP for long, no pods dir, sky queue FAILED_SETUP" "$W/pod_empty"
expect "  setup failure logged and notified once" bash -c "grep -q 'SETUP FAILED' '$MR/$ROUND/reap.log' && [ -e '$MR/$ROUND/setup_failed/$J' ] && grep -q 'SETUP FAILED' '$FAKE_STATE/notify'"
st "$J.up"; PREP='mkdir -p "$MR/$ROUND/first_up"; echo 1000 > "$MR/$ROUND/first_up/$J"'
            reap_case UP "UP for long, no pods dir, setup still running" "$W/pod_empty"
expect "  warning logged" grep -q 'job has not started' "$MR/$ROUND/reap.log"
st "$J.up"; PREP='mkdir -p "$MR/$ROUND/reaper.lock"; sleep 30 > /dev/null 2>&1 & LOCKPID=$!; echo $LOCKPID > "$MR/$ROUND/reaper.lock/pid"'
            reap_case UP "lock held by a live process that is not a reaper (stale): taken over" "$W/pod_running"
kill "$LOCKPID" 2>/dev/null; wait "$LOCKPID" 2>/dev/null
expect "  stale lock moved aside" bash -c "ls -d '$MR/$ROUND'/reaper.lock.stale.* > /dev/null 2>&1"
# a live reaper holds the lock: a second one must exit without doing anything
mac_reaper_root; st "$J.up"
FAKE_POD=$W/pod_running REAP_TEST_BIN=$B MAX_PASSES=50 INTERVAL=1 MIN_FREE_GB=0 bash "$MR/sky/v2_reap.sh" "$ROUND" > /dev/null 2>&1 & RP=$!
sleep 2
FAKE_POD=$W/pod REAP_TEST_BIN=$B MAX_PASSES=1 INTERVAL=0 MIN_FREE_GB=0 bash "$MR/sky/v2_reap.sh" "$ROUND" > "$W/reap2.txt" 2>&1; r2=$?
expect "second reaper on a round exits (rc 3) and downs nothing" bash -c "[ $r2 = 3 ] && grep -q 'already runs' '$W/reap2.txt' && ! grep -qx '$J' '$FAKE_STATE/downs' 2>/dev/null"
kill $RP 2>/dev/null; wait $RP 2>/dev/null
expect "killed reaper's lock is released (EXIT trap)" bash -c "[ ! -e '$MR/$ROUND/reaper.lock' ]"
ACTIONS=$W/actions.txt
st "$J.up"; reap_case DOWN "complete job" "$W/pod"
expect "  complete: not held, verified marker, no notification" bash -c "[ ! -e '$MR/$ROUND/held/$J' ] && [ -e '$MR/$ROUND/verified/$J' ] && [ ! -s '$FAKE_STATE/notify' ]"

# === the status script ==================================================================================
status_case() {  # $1 = expected rc, $2 = label, $3 = grep pattern that must appear; FAKE_POD and st set by caller
  local out r
  out=$(env REAP_TEST_BIN="$B" bash "$MR/sky/v2_status.sh" "$ROUND" 2>&1); r=$?
  if [ "$r" = "$1" ] && printf '%s\n' "$out" | grep -q -- "$3"; then ok "status rc $1  $2"
  else wrong "status expect rc $1 and '$3', got rc $r  $2"; printf '%s\n' "$out" | sed 's/^/      /'; fi
}
cp -R "$W/pod" "$W/pod_nonl"; perl -pi -e 'chomp if eof' "$W/pod_nonl/$ROUND/pods/$J/actions.txt"
mac_reaper_root; st "$J.up"; export FAKE_POD=$W/pod_nonl
date > "$MR/$ROUND/reaper.heartbeat"
status_case 0 "UP, fresh heartbeat, actions copy without a final newline counts every unit" "4/4 *JOB_COMPLETE"
touch -t 202001010000 "$MR/$ROUND/reaper.heartbeat"
status_case 1 "heartbeat older than 15 min" "REAPER NOT RUNNING"
rm "$MR/$ROUND/reaper.heartbeat"
status_case 1 "no heartbeat file" "REAPER NOT RUNNING"
date > "$MR/$ROUND/reaper.heartbeat"
st "$J.up" skyfail;     status_case 1 "sky status fails" "UNREADABLE"
st "$J.up" skygarbage;  status_case 1 "sky status prints no table" "UNREADABLE"
st;                     status_case 1 "cluster absent" "ABSENT"
st "$J.up"; echo "-" > "$FAKE_STATE/$J.autostop"; status_case 1 "UP row without autostop" "NO-AUTOSTOP"
st "$J.up" "$J.sshfail"; status_case 1 "ssh fails" "UNREADABLE (ssh failed"
st "$J.up"; mkdir -p "$MR/$ROUND/held"; echo "2026-10-04 x" > "$MR/$ROUND/held/$J"; status_case 0 "held pod shows HELD" "HELD("
unset FAKE_POD

# === the launcher (a fake git repo with the fake run_v2.sh) =================================================
LR=$W/lrepo
mkdir -p "$LR/sky"; make_repo "$LR"; rm "$LR/sky/v2_pod_job.sh"
cp "$SRC/sky/v2_launch.sh" "$SRC/sky/v2_lib.sh" "$SRC/sky/v2_reap.sh" "$SRC/sky/v2_verify.sh" "$SRC/sky/v2_pod.yaml" "$LR/sky/"
echo "runs_local/" > "$LR/.gitignore"
( cd "$LR" && git init -q && git add -A && git -c user.name=t -c user.email=t@t commit -q -m fake ) || wrong "fake launcher repo"
LA=$W/la; mkdir -p "$LA"
cp "$W/actions.txt" "$LA/v2t-a.txt"
printf '%s\n' "$T|EPOCHS=1|pretrain rl \"0 1\" 2|$T/pretrain/rl/pair0_1/seed2|" > "$LA/v2t-b.txt"
LROUND=runs_local/lround
launch() {  # env in the caller; prints the log
  ( cd "$LR" && env REAP_TEST_BIN="$B" FAKE_POD="$W/pod_empty" AUTOSTOP_WAIT=${AW:-5} SKY_TIMEOUT=20 INTERVAL=2 MAX_PASSES=4 \
      bash sky/v2_launch.sh "$LROUND" "$@" ) > "$W/launch_out.txt" 2>&1
}
st; launch "$LA/v2t-a.txt"; r=$?
expect "launcher refuses without UNIT_TIMEOUT" bash -c "[ $r = 2 ] && grep -q 'UNIT_TIMEOUT must be set' '$W/launch_out.txt'"
st; UNIT_TIMEOUT=1h DRY=1 launch "$LA/v2t-a.txt" "$LA/v2t-b.txt"; r=$?
expect "DRY launch: exit 0, prints the commands with UNIT_TIMEOUT, writes no actions copies, launches nothing" \
  bash -c "[ $r = 0 ] && grep -q 'UNIT_TIMEOUT=1h' '$W/launch_out.txt' && [ ! -e '$LR/$LROUND/actions' ] && ! grep -q '^launch' '$FAKE_STATE/sky_calls'"
for k in 'RUNS=x' 'PY=p' 'DRY=1' 'PRINT_UNIT=1' 'UNIT_TIMEOUT=1' 'SAVE_STM=yes'; do
  printf '%s\n' "$T|$k|pretrain rl \"0 1\" 3|$T/pretrain/rl/pair0_1/seed3|" > "$LA/v2t-res.txt"
  st; UNIT_TIMEOUT=1h DRY=1 launch "$LA/v2t-res.txt"; r=$?
  expect "launcher refuses '$k' in the environment field" bash -c "[ $r = 2 ] && grep -q 'reserved key\|SAVE_STM must be' '$W/launch_out.txt'"
done
printf '%s\n' "$T|EPOCHS=1|pretrain rl \"0 1\" 3|$T/pretrain/rl/pair0_1/seed9|" > "$LA/v2t-lay.txt"
st; UNIT_TIMEOUT=1h DRY=1 launch "$LA/v2t-lay.txt"; r=$?
expect "launcher refuses a unit dir that is not run_v2.sh's (PRINT_UNIT)" bash -c "[ $r = 2 ] && grep -q \"is not run_v2.sh's\" '$W/launch_out.txt'"
printf '%s\n' "$T|EPOCHS=1|continual rl \"0 1\" 4 \"3 4\"|$T/continual/rl/pair0_1/seed4/order3_4|" > "$LA/v2t-nopt.txt"
st; UNIT_TIMEOUT=1h DRY=1 launch "$LA/v2t-nopt.txt"; r=$?
expect "launcher refuses an STM unit without its pre-training in the file" bash -c "[ $r = 2 ] && grep -q 'needs pre-training' '$W/launch_out.txt'"
st skyfail; UNIT_TIMEOUT=1h launch "$LA/v2t-a.txt"; r=$?
expect "launcher: sky status UNREADABLE refuses the launch, nothing launched, no actions copy" \
  bash -c "[ $r = 2 ] && grep -q 'UNREADABLE' '$W/launch_out.txt' && ! grep -q '^launch' '$FAKE_STATE/sky_calls' && [ ! -e '$LR/$LROUND/actions' ]"
touch "$LR/untracked.txt"; st; UNIT_TIMEOUT=1h DRY=1 launch "$LA/v2t-a.txt"; r=$?
expect "launcher refuses an untracked non-ignored file (dirty)" bash -c "[ $r = 2 ] && grep -q 'untracked' '$W/launch_out.txt'"
rm "$LR/untracked.txt"
# a real (fake-sky) launch: v2t-b fails to launch; v2t-a comes UP without autostop, which the launcher applies
st launchfail_v2t-b; AW=60 UNIT_TIMEOUT=1h launch "$LA/v2t-a.txt" "$LA/v2t-b.txt"; r=$?
LL=$LR/$LROUND/launch.log
expect "launch: sky launch exit code collected and logged for the failed cluster" grep -q 'v2t-b LAUNCH/SETUP FAILED: sky launch exited 3' "$LL"
expect "launch: failed cluster marked LAUNCH FAILED only on a real empty row" grep -q 'v2t-b LAUNCH FAILED (sky status lists no such cluster' "$LL"
expect "launch: autostop applied and confirmed on the launched cluster" bash -c "grep -qx 'v2t-a 240' '$FAKE_STATE/autostops' && grep -q 'v2t-a autostop confirmed' '$LL'"
expect "launch: UNIT_TIMEOUT passed to sky launch" grep -q 'launch -c v2t-a sky/v2_pod.yaml .*--env UNIT_TIMEOUT=1h' "$FAKE_STATE/sky_calls"
expect "launch: pull_root recorded" bash -c "[ \"\$(cat '$LR/$LROUND/pull_root')\" = \"\$(cd '$LR' && pwd -P)\" ]"
expect "launch: reaper started under caffeinate, holding the round's lock" grep -q 'reaper started (pid' "$LL"
RPID=$(cat "$LR/$LROUND/reaper.lock/pid" 2>/dev/null)
expect "launch: the reaper process is alive" bash -c "[ -n '$RPID' ] && kill -0 '$RPID' 2>/dev/null"
UNIT_TIMEOUT=1h launch "$LA/v2t-a.txt"   # a second launch must not start a second reaper
expect "second launch: the existing reaper is reused and the UP cluster not relaunched" \
  bash -c "grep -q 'reaper already running' '$W/launch_out.txt' && grep -q 'already exists in sky status' '$W/launch_out.txt' && [ \"\$(grep -c '^launch -c v2t-a' '$FAKE_STATE/sky_calls')\" = 1 ]"
[ -n "$RPID" ] && kill "$RPID" 2>/dev/null
sleep 1

echo "---"; echo "$passes passed, $fails wrong"
[ "$fails" = 0 ] && { echo "ALL TESTS PASS"; exit 0; } || { echo "$fails TEST(S) WRONG"; exit 1; }
