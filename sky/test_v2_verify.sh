#!/bin/bash
# Tests of the v2 cloud path on fake trees, with a fake run_v2.sh and fake sky, ssh, rsync, timeout and osascript: no
# GPU, no pod, no real results. Covers the destructive predicate (sky/v2_verify.sh), the pod job's sentinels, unit
# timeout and validation (sky/v2_pod_job.sh), the reaper's decisions (sky/v2_reap.sh: teardown, HOLD, disk guard,
# UNREADABLE sky state, failed-pod pull, --partial verification and short autostop, launch-failed and FAILED_SETUP
# teardown only when no job ever ran (job marker, run dirs, sky queue; across rounds; exit-142 grace), a job killed
# without a sentinel (HOLD and pull), held/failed alert, one sky status per pass, lock, setup check), the status script
# (sky/v2_status.sh), the launcher's refusals (incl. a cluster name live in another round), DRY, regions, capacity-only
# fallback, e40 mount, LTM sha256s and launch-failure paths (sky/v2_launch.sh), and the dataset and LTM phases of the
# pod setup with a fake download (sky/v2_pod_setup.sh). Run it after any change to these scripts, before a live
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
check() {  # $1 = expected (OK|FAIL), $2 = label, $3 = mac root; PARTIAL=1: v2_verify.sh --partial
  local out rc got
  out=$(bash "$SRC/sky/v2_verify.sh" ${PARTIAL:+--partial} "$ROUND" "$J" "$3" 2>&1); rc=$?
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
      JOB_MARKER=${JOB_MARKER:-$W/marker_$(basename "$1")} bash sky/v2_pod_job.sh "${2:-$W/actions.txt}" > pod_stdout.txt 2>&1 )
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
expect "pod job appends the round-independent job marker (ROUND, JOB, COMMIT) as its first action" \
  grep -qE "^[0-9TZ:-]+ ROUND=$ROUND JOB=$J COMMIT=unknown PID=[0-9]+\$" "$W/marker_pod"
make_repo "$W/podnomark"; JOB_MARKER=$W/nonexistent_dir/v2_job_started run_pod "$W/podnomark"; r=$?
expect "pod job that cannot write its job marker: fails, nothing done (no pods dir, no unit)" \
  bash -c "[ $r != 0 ] && [ ! -e '$W/podnomark/$ROUND' ] && [ ! -e '$W/podnomark/$T' ] && grep -q 'cannot append to the job marker' '$W/podnomark/pod_stdout.txt'"
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
# --partial: every manifest file here with the same size and sha256, nothing else required
PARTIAL=1 check OK "--partial: pull of a failed job (JOB_FAILED, no job.done)" "$W/mac_bad"
PARTIAL=1 check OK "--partial: a complete pull passes too" "$W/mac_good"
rm -rf "$W/mac_bad2"; cp -R "$W/mac_bad" "$W/mac_bad2"; f=$(find "$W/mac_bad2/$T" -name job.log | head -1); printf 'x' >> "$f"
PARTIAL=1 check FAIL "--partial: a file differs in size" "$W/mac_bad2"
rm -rf "$W/mac_bad2"; cp -R "$W/mac_bad" "$W/mac_bad2"; f=$(find "$W/mac_bad2/$T" -name job.log | head -1); perl -pi -e 's/boom/BOOM/' "$f"
PARTIAL=1 check FAIL "--partial: a file differs in content (sha256)" "$W/mac_bad2"
rm -rf "$W/mac_bad2"; cp -R "$W/mac_bad" "$W/mac_bad2"; f=$(find "$W/mac_bad2/$T" -name job.log | head -1); rm "$f"
PARTIAL=1 check FAIL "--partial: a manifest file missing here" "$W/mac_bad2"
rm -rf "$W/mac_bad2"; cp -R "$W/mac_bad" "$W/mac_bad2"; rm "$W/mac_bad2/$ROUND/pods/$J/manifest.tsv"
PARTIAL=1 check FAIL "--partial: manifest missing" "$W/mac_bad2"
rm -rf "$W/mac_bad2"; cp -R "$W/mac_bad" "$W/mac_bad2"; echo "x|y|pretrain a|x/u|" >> "$W/mac_bad2/$ROUND/actions/$J.txt"
PARTIAL=1 check FAIL "--partial: pulled actions differ from the launcher's copy" "$W/mac_bad2"
rm -rf "$W/mac_bad2"; cp -R "$W/mac_bad" "$W/mac_bad2"; echo garbage >> "$W/mac_bad2/$ROUND/pods/$J/manifest.tsv"
PARTIAL=1 check FAIL "--partial: unparsable manifest line" "$W/mac_bad2"

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
expect "pod job that failed in its pre-flight: the EXIT trap writes an empty manifest" \
  bash -c "[ -f '$W/podnot/$ROUND/pods/$J/manifest.tsv' ] && [ ! -s '$W/podnot/$ROUND/pods/$J/manifest.tsv' ]"
rm -rf "$W/mac_not"; mkdir -p "$W/mac_not/$ROUND/actions" "$W/mac_not/$ROUND/pods/$J"; cp "$W/actions.txt" "$W/mac_not/$ROUND/actions/$J.txt"
rsync -a "$W/podnot/$ROUND/pods/$J/" "$W/mac_not/$ROUND/pods/$J/"
PARTIAL=1 check OK "--partial: job failed before copying its actions (empty manifest)" "$W/mac_not"
check FAIL "full verify of that pull" "$W/mac_not"

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
expect "unit timeout: a TIMEOUT line in the unit's own job.log" grep -q 'TIMEOUT: killed by the pod job' "$W/podslow/$T/stream/rl/pair0_1/seed1/fine3/job.log"
expect "unit timeout: the manifest (written after) lists the job.log with its TIMEOUT line" \
  bash -c "f='$T/stream/rl/pair0_1/seed1/fine3/job.log'; grep -q \"\$(cd '$W/podslow' && shasum -a 256 < \"\$f\" | cut -d' ' -f1)\" '$W/podslow/$ROUND/pods/$J/manifest.tsv'"

# === the reaper, with fake sky, ssh, rsync and osascript (REAP_TEST_BIN) ==============================================
# Fake pod = a local dir standing for ~/sky_workdir. Fake sky: a cluster is UP iff $FAKE_STATE/<c>.up exists; its
# autostop column is $FAKE_STATE/<c>.autostop (default "4h (down)"); `sky autostop` writes it and logs to autostops;
# `sky down` logs to downs. Flags: skyfail (every sky call exits 1), skygarbage (status prints no table),
# statusfail_after_down, <c>.failed_setup or <c>.queue (sky queue: the file's words are the job statuses, empty = no
# job; default one RUNNING job), queuefail, queuegarbage, launchfail_<c> (capacity), launchfail1_<c> (capacity once),
# launcherr_<c> (not capacity), launchupcap_<c> (capacity, then provisioned, then failed), launchhang_<c> (sleeps: with
# LAUNCH_TIMEOUT=3, exit 142). Fake ssh: test -f, test -d, bash -s (HOME = $FAKE_STATE/home: the job marker is
# $FAKE_STATE/home/v2_job_started). Fake pgrep: alive, pgrepfail.
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
    if [ -z "$c" ]; then   # every cluster, as the real sky status prints it
      printf 'Enabled Infra: runpod\n\nClusters\n'
      n=0; for u in "$S"/*.up; do [ -f "$u" ] || continue; n=$((n + 1)); done
      if [ $n = 0 ]; then printf 'No existing clusters.\n'
      else
        printf 'NAME  INFRA  RESOURCES  STATUS  AUTOSTOP  LAUNCHED\n'
        for u in "$S"/*.up; do x=$(basename "$u" .up); au="4h (down)"; [ -f "$S/$x.autostop" ] && au=$(cat "$S/$x.autostop")
          printf '%s  RunPod  1x(L4)  UP  %s  1m ago\n' "$x" "$au"; done
      fi
      printf '\nManaged jobs\nNo in-progress managed jobs.\n'
    elif [ -f "$S/$c.up" ]; then
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
  queue)   # as SkyPilot 0.13 prints it: no table at all when there is no job; "Failed to get..." with exit 0
    c=$2
    [ -f "$S/queuefail" ] && { echo "sky: error" >&2; exit 1; }
    printf 'Fetching and parsing job queue...\nFetching job queue for: %s\n\n' "$c"
    [ -f "$S/queuegarbage" ] && { printf "\033[33mFailed to get the job queue for cluster '%s'.\033[0m\n  ClusterNotUpError: x\n" "$c"; exit 0; }
    printf 'Job queue of current user on cluster %s\n' "$c"
    if [ -f "$S/$c.failed_setup" ]; then sts=FAILED_SETUP; elif [ -f "$S/$c.queue" ]; then sts=$(cat "$S/$c.queue"); else sts=RUNNING; fi
    if [ -n "$sts" ]; then
      printf 'ID  NAME    USER    SUBMITTED   STARTED     DURATION  RESOURCES          STATUS        LOG                        GIT COMMIT  \n'
      i=0; for x in $sts; do i=$((i + 1))
        printf '%s   v2-pod  gideon  5 mins ago  4 mins ago  4m 2s     1x[L4:1, cpus=4+]  \033[1m%s\033[0m  ~/sky_logs/sky-x/run.log  -  \n' "$i" "$x"; done
    fi ;;
  launch) c=$3; echo "$4" >> "$S/yaml_$c"
          [ -f "$S/launchfail_$c" ] && { echo "sky.exceptions.ResourcesUnavailableError: Failed to provision all possible launchable resources. Relax the task's resource requirements: 1x RunPod(...)"; exit 3; }
          [ -f "$S/launchfail1_$c" ] && { rm "$S/launchfail1_$c"; echo "sky.exceptions.ResourcesUnavailableError: Failed to acquire resources in all zones in RO for {RunPod(cpus=4+, mem=16+, {'L4': 1}, disk_size=60)}"; exit 3; }
          [ -f "$S/launcherr_$c" ] && { echo "sky.exceptions.CommandError: Command rsync -Pavz ... failed with return code 255."; exit 1; }
          [ -f "$S/launchupcap_$c" ] && { echo "sky.exceptions.ResourcesUnavailableError: Failed to acquire resources in all zones in AU for {...}"; echo "✓ Cluster launched: $c.  View logs: sky logs --provision $c"; echo "sky.exceptions.CommandError: rsync failed"; exit 1; }
          [ -f "$S/launchhang_$c" ] && { sleep 8; exit 0; }
          [ -f "$S/launchup_fail_$c" ] && { touch "$S/$c.up"; echo "-" > "$S/$c.autostop"; echo "setup failed"; exit 5; }
          touch "$S/$c.up"; echo "-" > "$S/$c.autostop"
          [ -f "$S/launchslow_$c" ] && { sleep 40; exit 4; } ;;
esac
exit 0
EOF
cat > "$B/ssh" <<'EOF'
#!/bin/bash
a=("$@"); n=${#a[@]}; cmd=${a[$((n-1))]}; host=${a[$((n-2))]}
echo "$host $cmd" >> "$FAKE_STATE/ssh_calls"
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
# fake pgrep, reached by the reaper's probe through the fake ssh's bash -s: alive = a pod job process exists
cat > "$B/pgrep" <<'EOF'
#!/bin/bash
echo "$*" >> "$FAKE_STATE/pgrep_calls"
[ -f "$FAKE_STATE/pgrepfail" ] && exit 3
[ -f "$FAKE_STATE/alive" ] && exit 0
exit 1
EOF
chmod +x "$B/sky" "$B/ssh" "$B/rsync" "$B/osascript" "$B/pgrep"

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
  if [ "$got" = UP ] && [ -e "$MR/$ROUND/reaped/$J" ] && [ -z "${REAPED_OK:-}" ]; then wrong "reaper wrote a reaped marker for a pod it left up ($2)"; fi
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
st "$J.up";              reap_case UP   "JOB_FAILED, pull verified with --partial" "$W/podbad"
expect "  JOB_FAILED + verified partial pull: autostop FAILED_IDLE (90m) confirmed, NOT held (no 24h), notified" \
  bash -c "[ -e '$MR/$ROUND/failed_pulled/$J' ] && [ -e '$MR/$ROUND/failed_safe/$J' ] && [ \"\$(cat '$MR/$ROUND/failed_verify/$J')\" = OK ] && grep -qx '$J 90' '$FAKE_STATE/autostops' && ! grep -q '^$J 1440\$' '$FAKE_STATE/autostops' && [ ! -e '$MR/$ROUND/held/$J' ] && grep -q 'files safe' '$FAKE_STATE/notify' && [ -e '$MR/$ROUND/job_failed/$J' ]"
st "$J.up"; PASSES=3 reap_case UP "JOB_FAILED, verified partial pull, 3 passes" "$W/podbad"
expect "  short autostop applied once over 3 passes" bash -c "[ \"\$(grep -c '^$J 90\$' '$FAKE_STATE/autostops')\" = 1 ]"
st "$J.up" rsyncfail;    reap_case UP   "JOB_FAILED, pull incomplete (rsync fails)" "$W/podbad"
held_not_short() { held && ! grep -q "^$J 90\$" "$FAKE_STATE/autostops" && [ ! -e "$MR/$ROUND/failed_safe/$J" ]; }
expect "  JOB_FAILED + incomplete pull: HELD 24h, no short autostop" held_not_short
st "$J.up"; PREP='mkdir -p "$MR/$ROUND/held"; touch "$MR/$ROUND/held/other-a"'
            reap_case UP   "JOB_FAILED with 1 other held pod (2 in all)" "$W/podbad"
expect "  2 held or failed: no ALERT" bash -c "! grep -q ALERT '$MR/$ROUND/reap.log'"
st "$J.up"; PREP='mkdir -p "$MR/$ROUND/held" "$MR/$ROUND/launch_failed"; touch "$MR/$ROUND/held/other-a" "$MR/$ROUND/launch_failed/other-b"'
            reap_case UP   "JOB_FAILED with 2 other failed/held pods (3 in all)" "$W/podbad"
expect "  3 held or failed: ALERT line logged and a notification with a sound" \
  bash -c "grep -q 'ALERT: 3 clusters of this round are held or failed: other-a other-b $J' '$MR/$ROUND/reap.log' && grep -q 'ALERT.*sound name' '$FAKE_STATE/notify'"
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
            reap_case DOWN "UP for long, no pods dir (test -d exit 1), sky queue FAILED_SETUP: torn down" "$W/pod_empty"
expect "  setup failure logged and notified, reaped as 'nothing ran'" bash -c "grep -q 'SETUP FAILED' '$MR/$ROUND/reap.log' && [ -e '$MR/$ROUND/setup_failed/$J' ] && grep -q 'SETUP FAILED' '$FAKE_STATE/notify' && grep -q 'nothing ran' '$MR/$ROUND/reaped/$J'"
# launch failed (launch_failed/<c>, written by the launcher): torn down only when test -d of the pods dir exits exactly 1
LFPREP='mkdir -p "$MR/$ROUND/launch_failed"; echo "exit 1 at x" > "$MR/$ROUND/launch_failed/$J"'
st "$J.up" "$J.queue"; PREP=$LFPREP reap_case DOWN "launch failed, UP, nothing on the pod, no marker, sky queue: no job" "$W/pod_empty"
expect "  launch-failed pod: reaped marker says nothing ran" grep -q 'nothing ran (no job marker, no run dirs, sky queue: no jobs); torn down' "$MR/$ROUND/reaped/$J"
st "$J.up" "$J.sshfail"; PREP=$LFPREP reap_case UP "launch failed, UP, ssh fails (test -d exit 255)" "$W/pod_empty"
expect "  ssh failure: nothing done, logged" grep -q 'cannot tell whether a job ever ran on it (ssh probe failed or unreadable): doing nothing' "$MR/$ROUND/reap.log"
st "$J.up"; PREP=$LFPREP reap_case UP "launch failed, but the job started (test -d exit 0) and is running" "$W/pod_running"
expect "  started job: handled as a normal pod (no teardown, not held)" not_held
st "$J.up"; PREP=$LFPREP reap_case DOWN "launch failed, but the job started and completed: pulled, verified, down" "$W/pod"
expect "  completed despite the failed launch: verified marker" test -e "$MR/$ROUND/verified/$J"
st;         PREP=$LFPREP REAPED_OK=1 reap_case UP "launch failed, cluster ABSENT from a readable sky status (no down: nothing to down)" "$W/pod_empty"
expect "  absent launch-failed cluster: reaped marker, no sky down" bash -c "grep -q 'absent from sky status: nothing ran' '$MR/$ROUND/reaped/$J' && [ ! -s '$FAKE_STATE/downs' ]"
st skyfail; PREP=$LFPREP reap_case UP "launch failed, sky status UNREADABLE" "$W/pod_empty"
expect "  unreadable sky status: no reaped marker, no ssh" bash -c "[ ! -e '$MR/$ROUND/reaped/$J' ] && [ ! -s '$FAKE_STATE/ssh_calls' ]"
# F1: "nothing ran" must hold across rounds: the job marker (~/v2_job_started), run dirs, and sky queue
mark_pod() { mkdir -p "$FAKE_STATE/home"; echo "2026-10-04T00:00:00Z ROUND=${1:-runs_local/other_round} JOB=$J COMMIT=abc PID=1" >> "$FAKE_STATE/home/v2_job_started"; }
cp -R "$W/pod" "$W/pod_other"; mv "$W/pod_other/$ROUND" "$W/pod_other/runs_local/other_round"
st "$J.up"; echo SUCCEEDED > "$FAKE_STATE/$J.queue"; mark_pod
PREP=$LFPREP reap_case UP "two rounds: this round's launch failed; the pod of the same name carries the OTHER round's marker, sentinel and results" "$W/pod_other"
expect "  other round's pod: logged as 'a job ran on this pod', not reaped, nothing pulled here" \
  bash -c "grep -q 'a job ran on this pod (~/v2_job_started: .*ROUND=runs_local/other_round' '$MR/$ROUND/reap.log' && [ ! -e '$MR/$ROUND/reaped/$J' ] && [ ! -e '$MR/$T' ]"
st "$J.up"; echo SUCCEEDED > "$FAKE_STATE/$J.queue"
PREP=$LFPREP reap_case UP "two rounds, no marker (older pod job), the other round's run dirs on the pod" "$W/pod_other"
expect "  run dirs alone keep it up" grep -q 'run dirs under' "$MR/$ROUND/reap.log"
st "$J.up" "$J.queue"; mark_pod "$ROUND"
PREP=$LFPREP reap_case UP "launch failed, marker present, no run dirs, sky queue: no job" "$W/pod_empty"
for q in RUNNING SUCCEEDED FAILED CANCELLED FAILED_DRIVER "FAILED_SETUP FAILED"; do
  st "$J.up"; echo "$q" > "$FAKE_STATE/$J.queue"
  PREP=$LFPREP reap_case UP "launch failed, no marker, no run dirs, sky queue shows '$q' (past setup)" "$W/pod_empty"
done
expect "  queue past setup: logged" grep -q 'sky queue shows a job past setup' "$MR/$ROUND/reap.log"
for q in SETTING_UP PENDING INIT; do
  st "$J.up"; echo "$q" > "$FAKE_STATE/$J.queue"
  PREP=$LFPREP reap_case UP "launch failed, no marker, no run dirs, sky queue shows '$q' (setup may still run)" "$W/pod_empty"
done
st "$J.up"; echo FAILED_SETUP > "$FAKE_STATE/$J.queue"
PREP=$LFPREP reap_case DOWN "launch failed, no marker, no run dirs, sky queue FAILED_SETUP only (not read as FAILED)" "$W/pod_empty"
st "$J.up" queuefail; PREP=$LFPREP reap_case UP "launch failed, nothing on the pod, sky queue fails" "$W/pod_empty"
expect "  sky queue failure: logged UNREADABLE" grep -q 'sky queue is UNREADABLE' "$MR/$ROUND/reap.log"
st "$J.up" queuegarbage; PREP=$LFPREP reap_case UP "launch failed, nothing on the pod, sky queue prints 'Failed to get the job queue' (exit 0)" "$W/pod_empty"
st "$J.up" "$J.queue" pgrepfail; PREP=$LFPREP reap_case DOWN "launch failed, nothing on the pod, pgrep unusable (does not block the teardown)" "$W/pod_empty"
LF142='mkdir -p "$MR/$ROUND/launch_failed"; echo "exit 142 at x" > "$MR/$ROUND/launch_failed/$J"'
st "$J.up" "$J.queue"; PREP=$LF142 reap_case UP "launch exit 142 (LAUNCH_TIMEOUT) a moment ago, nothing on the pod yet: grace" "$W/pod_empty"
expect "  exit 142 within LF142_GRACE_MIN: logged, no probe, no sky queue" \
  bash -c "grep -q 'may still be launching' '$MR/$ROUND/reap.log' && ! grep -q 'bash -s' '$FAKE_STATE/ssh_calls' && ! grep -q '^queue' '$FAKE_STATE/sky_calls'"
st "$J.up" "$J.failed_setup"; PREP="$LF142; mkdir -p \"\$MR/\$ROUND/first_up\"; echo 1000 > \"\$MR/\$ROUND/first_up/\$J\""
            reap_case UP "launch exit 142 a moment ago, sky queue FAILED_SETUP, nothing on the pod: still within the grace" "$W/pod_empty"
st "$J.up" "$J.queue"; PREP="$LF142; touch -t 202001010000 \"\$MR/\$ROUND/launch_failed/\$J\""
            reap_case DOWN "launch exit 142 long ago (past LF142_GRACE_MIN), nothing on the pod, sky queue: no job" "$W/pod_empty"
# one sky status per pass, for every cluster
st "$J.up"; PASSES=3 reap_case UP "job running, 3 passes" "$W/pod_running"
expect "  exactly one sky status call per pass (3), none per cluster" bash -c "[ \"\$(grep -cx 'status' '$FAKE_STATE/sky_calls')\" = 3 ] && ! grep -q '^status .' '$FAKE_STATE/sky_calls'"
st "$J.up" skyfail; PASSES=2 reap_case UP "sky status fails on every pass" "$W/pod"
expect "  whole-call failure: no ssh, no autostop, no down, nothing pulled" \
  bash -c "[ ! -s '$FAKE_STATE/ssh_calls' ] && [ ! -s '$FAKE_STATE/autostops' ] && [ ! -s '$FAKE_STATE/downs' ] && [ ! -e '$MR/$T' ] && [ \"\$(grep -c '^status' '$FAKE_STATE/sky_calls')\" = 2 ]"
st "$J.up"; PREP='mkdir -p "$MR/$ROUND/first_up"; echo 1000 > "$MR/$ROUND/first_up/$J"'
            reap_case UP "UP for long, no pods dir, setup still running" "$W/pod_empty"
expect "  warning logged" grep -q 'job has not started' "$MR/$ROUND/reap.log"
# F3: a job killed without its EXIT trap (SIGKILL, OOM): pods dir, no sentinel, no manifest, no pod job process
cp -R "$W/podbad" "$W/pod_killed"; rm -f "$W/pod_killed/$ROUND/pods/$J/JOB_FAILED" "$W/pod_killed/$ROUND/pods/$J/manifest.tsv"
ns_held() { held && [ -e "$MR/$ROUND/no_sentinel/$J" ] && grep -q 'without a sentinel' "$FAKE_STATE/notify" && grep -q 'JOB ENDED WITHOUT A SENTINEL' "$MR/$ROUND/reap.log"; }
not_ns() { not_held && [ ! -e "$MR/$ROUND/no_sentinel/$J" ]; }
ns_pulled() { ns_held && [ -e "$MR/$ROUND/no_sentinel_pulled/$J" ] && [ -f "$MR/$U_CONT/job.log" ] && [ ! -e "$MR/$ROUND/reaped/$J" ]; }
ns_unreadable() { not_ns && grep -q 'sky queue is UNREADABLE' "$MR/$ROUND/reap.log"; }
ns_noqueue() { not_ns && ! grep -q '^queue' "$FAKE_STATE/sky_calls"; }
st "$J.up"; echo FAILED > "$FAKE_STATE/$J.queue"
            reap_case UP "no sentinel, no pod job process, sky queue FAILED (killed job)" "$W/pod_killed"
expect "  killed job: HELD (24h once), no_sentinel marker, notified; what there is pulled once" \
  ns_pulled
st "$J.up"; echo SUCCEEDED > "$FAKE_STATE/$J.queue"; PASSES=3 reap_case UP "no sentinel, sky queue SUCCEEDED, 3 passes" "$W/pod_killed"
expect "  killed job over 3 passes: held once, pulled once (1 metadata + 4 units rsyncs)" \
  bash -c "[ \"\$(grep -c '^$J 1440\$' '$FAKE_STATE/autostops')\" = 1 ] && [ \"\$(grep -vc '^--server' '$FAKE_STATE/rsync_calls')\" = 5 ] || { cat '$FAKE_STATE/rsync_calls'; false; }"
st "$J.up" pgrepfail; echo CANCELLED > "$FAKE_STATE/$J.queue"; reap_case UP "no sentinel, pgrep unusable, sky queue CANCELLED: falls through to the queue" "$W/pod_killed"
expect "  pgrep unusable: still HELD on the queue's word" ns_held
for q in RUNNING SETTING_UP PENDING ""; do
  st "$J.up"; printf '%s' "$q" > "$FAKE_STATE/$J.queue"
  reap_case UP "no sentinel, no pod job process, sky queue shows '${q:-no job}'" "$W/pod_killed"
  expect "  queue '${q:-no job}': not held, no no_sentinel marker" not_ns
done
st "$J.up" queuefail; reap_case UP "no sentinel, no pod job process, sky queue fails" "$W/pod_killed"
expect "  unreadable queue: not held, logged" ns_unreadable
st "$J.up" queuegarbage; reap_case UP "no sentinel, sky queue prints 'Failed to get the job queue'" "$W/pod_killed"
expect "  queue garbage: not held" not_ns
st "$J.up" alive; echo FAILED > "$FAKE_STATE/$J.queue"; reap_case UP "no sentinel, a pod job process alive (queue would say FAILED)" "$W/pod_killed"
expect "  alive: not held, sky queue not even read" ns_noqueue
st "$J.up" "$J.sshfail"; echo FAILED > "$FAKE_STATE/$J.queue"; reap_case UP "no sentinel, ssh fails" "$W/pod_killed"
expect "  ssh failure: not held" not_ns
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
export FAKE_POD=$W/podnot; st "$J.up"; status_case 0 "job failed before copying its actions file shows JOB_FAILED" "JOB_FAILED *(before the units"
export FAKE_POD=$W/pod_empty; rm -f "$MR/$ROUND/held/$J"; mkdir -p "$MR/$ROUND/launch_failed"; echo "exit 1 at x" > "$MR/$ROUND/launch_failed/$J"
st "$J.up"; status_case 1 "launch failed, pod UP with nothing run: LAUNCH_FAILED, exit 1" "LAUNCH_FAILED"
st;         status_case 1 "launch failed, cluster absent: LAUNCH_FAILED, exit 1" "LAUNCH_FAILED(exit 1"
rm -f "$MR/$ROUND/launch_failed/$J"; mkdir -p "$MR/$ROUND/setup_failed"; touch "$MR/$ROUND/setup_failed/$J"
st "$J.up"; status_case 1 "FAILED_SETUP: SETUP_FAILED, exit 1" "SETUP_FAILED"
rm -f "$MR/$ROUND/setup_failed/$J"
mkdir -p "$MR/$ROUND/no_sentinel"; echo "2026-10-04 x sky queue: FAILED" > "$MR/$ROUND/no_sentinel/$J"
st "$J.up"; status_case 1 "job ended without a sentinel: NO_SENTINEL, exit 1" "NO_SENTINEL("
rm -f "$MR/$ROUND/no_sentinel/$J"
st "$J.up"; status_case 0 "status reads sky status once" "NOT-STARTED"
expect "  status: one sky status call, for every cluster" bash -c "[ \"\$(grep -c '^status' '$FAKE_STATE/sky_calls')\" = 1 ] && grep -qx status '$FAKE_STATE/sky_calls'"
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
mkdir -p "$W/cifar" "$W/ltm"
for f in train test meta; do echo "fake cifar $f" > "$W/cifar/$f"; done
echo e13 > "$W/ltm/cifar_100_subclasses_12_e13.pth"; echo e40 > "$W/ltm/cifar_100_subclasses_12_e40.pth"
launch() {  # env in the caller; prints the log
  ( cd "$LR" && env REAP_TEST_BIN="$B" FAKE_POD="$W/pod_empty" AUTOSTOP_WAIT=${AW:-5} SKY_TIMEOUT=20 INTERVAL=2 MAX_PASSES=4 \
      CIFAR_SRC="${CIFAR_SRC:-$W/cifar}" LTM_SRC="$W/ltm" bash sky/v2_launch.sh "$LROUND" "$@" ) > "$W/launch_out.txt" 2>&1
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
expect "launch: UNIT_TIMEOUT passed to sky launch" grep -q "launch -c v2t-a $LROUND/yaml/v2t-a.yaml .*--env UNIT_TIMEOUT=1h" "$FAKE_STATE/sky_calls"
expect "launch: the Mac's CIFAR-100 sha256s passed to sky launch" \
  grep -q "launch -c v2t-a .*--env CIFAR_SHA256_TRAIN=$(shasum -a 256 < "$W/cifar/train" | cut -d' ' -f1) --env CIFAR_SHA256_TEST=$(shasum -a 256 < "$W/cifar/test" | cut -d' ' -f1) --env CIFAR_SHA256_META=$(shasum -a 256 < "$W/cifar/meta" | cut -d' ' -f1) " "$FAKE_STATE/sky_calls"
expect "launch: per-cluster yaml restricted to the preferred region (default RO) by any_of" \
  bash -c "grep -qx '  any_of:' '$LR/$LROUND/yaml/v2t-a.yaml' && grep -qx '    - infra: runpod/RO' '$LR/$LROUND/yaml/v2t-a.yaml' && ! grep -q '^  infra: runpod\$' '$LR/$LROUND/yaml/v2t-a.yaml' && [ \"\$(grep -c 'infra: runpod/' '$LR/$LROUND/yaml/v2t-a.yaml')\" = 1 ]"
expect "launch: fallback yaml has every other region except IS (avoided) and RO (preferred)" \
  bash -c "y='$LR/$LROUND/yaml/v2t-a.fallback.yaml'; [ \"\$(grep -c 'infra: runpod/' \"\$y\")\" = 12 ] && ! grep -q 'runpod/IS\$' \"\$y\" && ! grep -q 'runpod/RO\$' \"\$y\" && grep -q 'runpod/CZ\$' \"\$y\""
expect "launch: no LTM=e40 unit, so no e40 mount (still commented), e13 mounted, no CIFAR mount" \
  bash -c "y='$LR/$LROUND/yaml/v2t-a.yaml'; grep -q '^  #E40 ' \"\$y\" && ! grep -q '^  ~/cifar_100_pretrain/cifar_100_subclasses_12_e40.pth:' \"\$y\" && grep -q '^  ~/cifar_100_pretrain/cifar_100_subclasses_12_e13.pth:' \"\$y\" && ! grep -q '^  ~/cifar-100-python' \"\$y\""
expect "launch: v2t-b failed in the preferred region, was absent, and was retried with the fallback yaml (then failed: launch_failed marker)" \
  bash -c "grep -qx '$LROUND/yaml/v2t-b.yaml' '$FAKE_STATE/yaml_v2t-b' && grep -qx '$LROUND/yaml/v2t-b.fallback.yaml' '$FAKE_STATE/yaml_v2t-b' && grep -q 'trying the fallback regions' '$LL' && grep -q '^exit 3' '$LR/$LROUND/launch_failed/v2t-b' && [ ! -e '$LR/$LROUND/launch_failed/v2t-a' ]"
cp "$FAKE_STATE/sky_calls" "$FAKE_STATE.sky_calls.keep"
expect "launch: pull_root recorded" bash -c "[ \"\$(cat '$LR/$LROUND/pull_root')\" = \"\$(cd '$LR' && pwd -P)\" ]"
expect "launch: reaper started under caffeinate, holding the round's lock" grep -q 'reaper started (pid' "$LL"
RPID=$(cat "$LR/$LROUND/reaper.lock/pid" 2>/dev/null)
expect "launch: the reaper process is alive" bash -c "[ -n '$RPID' ] && kill -0 '$RPID' 2>/dev/null"
UNIT_TIMEOUT=1h launch "$LA/v2t-a.txt"   # a second launch must not start a second reaper
expect "second launch: the existing reaper is reused and the UP cluster not relaunched" \
  bash -c "grep -q 'reaper already running' '$W/launch_out.txt' && grep -q 'already exists in sky status' '$W/launch_out.txt' && [ \"\$(grep -c '^launch -c v2t-a' '$FAKE_STATE/sky_calls')\" = 1 ]"
[ -n "$RPID" ] && kill "$RPID" 2>/dev/null
# a launch that is still running (file sync, setup) after autostop is confirmed: its exit code is still collected
printf '%s\n' "$T|EPOCHS=1|pretrain rl \"0 1\" 5|$T/pretrain/rl/pair0_1/seed5|" > "$LA/v2t-slow.txt"
st launchslow_v2t-slow; AW=120 UNIT_TIMEOUT=1h launch "$LA/v2t-slow.txt"
expect "launch: waits for a launch still running after autostop, and logs its exit code" \
  bash -c "grep -q 'v2t-slow autostop confirmed' '$LL' && grep -q 'v2t-slow: waiting for sky launch to return' '$LL' && grep -q 'v2t-slow LAUNCH/SETUP FAILED: sky launch exited 4' '$LL'"
RPID=$(cat "$LR/$LROUND/reaper.lock/pid" 2>/dev/null); [ -n "$RPID" ] && kill "$RPID" 2>/dev/null
sleep 1
# an LTM=e40 unit: the e40 checkpoint mounted; preferred region full, fallback succeeds
printf '%s\n' "$T|LTM=e40 EPOCHS=1|pretrain rl \"0 1\" 6|$T/pretrain/rl/pair0_1/seed6|" > "$LA/v2t-e.txt"
st launchfail1_v2t-e; AW=60 REGIONS="RO CZ" UNIT_TIMEOUT=1h launch "$LA/v2t-e.txt"
RPID=$(cat "$LR/$LROUND/reaper.lock/pid" 2>/dev/null); [ -n "$RPID" ] && kill "$RPID" 2>/dev/null
cp "$FAKE_STATE/sky_calls" "$FAKE_STATE.sky_calls.e"
expect "launch: LTM=e40 unit -> e40 checkpoint mounted in both yamls; REGIONS='RO CZ' in the preferred yaml" \
  bash -c "for y in '$LR/$LROUND/yaml/v2t-e.yaml' '$LR/$LROUND/yaml/v2t-e.fallback.yaml'; do grep -q '^  ~/cifar_100_pretrain/cifar_100_subclasses_12_e40.pth: ~/Dev/cifar_100_pretrain/cifar_100_subclasses_12_e40.pth' \"\$y\" || exit 1; done; grep -qx '    - infra: runpod/CZ' '$LR/$LROUND/yaml/v2t-e.yaml' && [ \"\$(grep -c 'infra: runpod/' '$LR/$LROUND/yaml/v2t-e.fallback.yaml')\" = 11 ]"
expect "launch: preferred region without capacity -> fallback launch exit 0, no launch_failed marker" \
  bash -c "grep -q 'v2t-e sky launch exited 0' '$LL' && [ ! -e '$LR/$LROUND/launch_failed/v2t-e' ] && grep -qx '$LROUND/yaml/v2t-e.fallback.yaml' '$FAKE_STATE/yaml_v2t-e'"
# STRICT_REGIONS=1: no fallback yaml, one attempt only
printf '%s\n' "$T|EPOCHS=1|pretrain rl \"0 1\" 7|$T/pretrain/rl/pair0_1/seed7|" > "$LA/v2t-s.txt"
st launchfail_v2t-s; AW=30 STRICT_REGIONS=1 UNIT_TIMEOUT=1h launch "$LA/v2t-s.txt"
RPID=$(cat "$LR/$LROUND/reaper.lock/pid" 2>/dev/null); [ -n "$RPID" ] && kill "$RPID" 2>/dev/null
expect "launch: STRICT_REGIONS=1 -> one attempt, no fallback yaml, launch_failed recorded" \
  bash -c "[ \"\$(grep -c . '$FAKE_STATE/yaml_v2t-s')\" = 1 ] && [ ! -e '$LR/$LROUND/yaml/v2t-s.fallback.yaml' ] && [ -e '$LR/$LROUND/launch_failed/v2t-s' ]"
# a launch that brings the pod UP and then fails (setup): not retried elsewhere (the cluster exists), launch_failed
printf '%s\n' "$T|EPOCHS=1|pretrain rl \"0 1\" 8|$T/pretrain/rl/pair0_1/seed8|" > "$LA/v2t-u.txt"
st launchup_fail_v2t-u; AW=60 UNIT_TIMEOUT=1h launch "$LA/v2t-u.txt"
RPID=$(cat "$LR/$LROUND/reaper.lock/pid" 2>/dev/null); [ -n "$RPID" ] && kill "$RPID" 2>/dev/null
expect "launch: pod UP but sky launch failed (setup) -> NOT relaunched in other regions; launch_failed with exit 5" \
  bash -c "[ \"\$(grep -c . '$FAKE_STATE/yaml_v2t-u')\" = 1 ] && grep -q 'NOT retrying in other regions' '$LL' && grep -q '^exit 5' '$LR/$LROUND/launch_failed/v2t-u'"
# F4a: the Mac's LTM checkpoint sha256s are passed (e40 only to a pod with LTM=e40 units)
SHA13=$(shasum -a 256 < "$W/ltm/cifar_100_subclasses_12_e13.pth" | cut -d' ' -f1); SHA40=$(shasum -a 256 < "$W/ltm/cifar_100_subclasses_12_e40.pth" | cut -d' ' -f1)
expect "launch: LTM_SHA256_E13 = the Mac's e13 and LTM_SHA256_E40 empty for a pod without LTM=e40 units" \
  grep -q "launch -c v2t-a .*--env LTM_SHA256_E13=$SHA13 --env LTM_SHA256_E40= -y -d" "$FAKE_STATE.sky_calls.keep"
expect "launch: LTM_SHA256_E40 = the Mac's e40 for the pod with LTM=e40 units" \
  grep -q "launch -c v2t-e .*--env LTM_SHA256_E13=$SHA13 --env LTM_SHA256_E40=$SHA40 -y -d" "$FAKE_STATE.sky_calls.e"
# F2: fall back to other regions only on a capacity failure
lone() {  # $1 = cluster, $2 = fake flag; launches one fresh cluster with a fallback yaml (default REGIONS); env in the caller
  printf '%s\n' "$T|EPOCHS=1|pretrain rl \"0 1\" $3|$T/pretrain/rl/pair0_1/seed$3|" > "$LA/$1.txt"
  st "$2_$1"; AW=${AW:-60} UNIT_TIMEOUT=1h launch "$LA/$1.txt"
  local p; p=$(cat "$LR/$LROUND/reaper.lock/pid" 2>/dev/null); [ -n "$p" ] && kill "$p" 2>/dev/null
  sleep 1
}
LAUNCH_TIMEOUT=3 lone v2t-h launchhang 11
expect "launch: exit 142 (LAUNCH_TIMEOUT) -> NOT retried in other regions, logged why; launch_failed exit 142" \
  bash -c "[ \"\$(grep -c . '$FAKE_STATE/yaml_v2t-h')\" = 1 ] && grep -q 'v2t-h: launch in the preferred regions killed by LAUNCH_TIMEOUT (exit 142)' '$LL' && grep -q '^exit 142' '$LR/$LROUND/launch_failed/v2t-h'"
lone v2t-n launcherr 12
expect "launch: a non-capacity error (CommandError), cluster absent -> NOT retried in other regions, logged why" \
  bash -c "[ \"\$(grep -c . '$FAKE_STATE/yaml_v2t-n')\" = 1 ] && grep -q 'v2t-n: launch in the preferred regions exited 1 without a capacity failure' '$LL' && grep -q '^exit 1' '$LR/$LROUND/launch_failed/v2t-n'"
lone v2t-p launchupcap 13
expect "launch: capacity errors, then 'Cluster launched', then a failure -> NOT retried in other regions" \
  bash -c "[ \"\$(grep -c . '$FAKE_STATE/yaml_v2t-p')\" = 1 ] && grep -q 'v2t-p: launch in the preferred regions exited 1 after the cluster was launched' '$LL'"
lone v2t-c launchfail1 14
expect "launch: a capacity failure (ResourcesUnavailableError), cluster absent -> retried with the fallback yaml, exit 0" \
  bash -c "[ \"\$(grep -c . '$FAKE_STATE/yaml_v2t-c')\" = 2 ] && grep -q 'v2t-c: launch in the preferred regions (RO) exited 3 with a capacity failure' '$LL' && grep -q 'v2t-c sky launch exited 0' '$LL' && [ ! -e '$LR/$LROUND/launch_failed/v2t-c' ]"
# F1b: a cluster name still live in another round dir is refused
mkdir -p "$LR/runs_local/otherround/actions"; echo x > "$LR/runs_local/otherround/actions/v2t-a.txt"
st; UNIT_TIMEOUT=1h DRY=1 launch "$LA/v2t-a.txt"; r=$?
expect "launcher refuses a cluster name still live in another round (actions/<c>.txt, no reaped/<c>)" \
  bash -c "[ $r = 2 ] && grep -q 'cluster name v2t-a is still live in another round: runs_local/otherround' '$W/launch_out.txt'"
mkdir -p "$LR/runs_local/otherround/reaped"; touch "$LR/runs_local/otherround/reaped/v2t-a.dry"
st; UNIT_TIMEOUT=1h DRY=1 launch "$LA/v2t-a.txt"; r=$?
expect "  a DRY reaper's reaped/<c>.dry does not count as reaped" bash -c "[ $r = 2 ] && grep -q 'still live in another round' '$W/launch_out.txt'"
touch "$LR/runs_local/otherround/reaped/v2t-a"
st; UNIT_TIMEOUT=1h DRY=1 launch "$LA/v2t-a.txt"; r=$?
expect "  once reaped there, the name is accepted (DRY exit 0)" bash -c "[ $r = 0 ]"
st; UNIT_TIMEOUT=1h REGIONS="RO XX" DRY=1 launch "$LA/v2t-a.txt"; r=$?
expect "launcher refuses an unknown region" bash -c "[ $r = 2 ] && grep -q \"unknown region 'XX'\" '$W/launch_out.txt'"
st; UNIT_TIMEOUT=1h CIFAR_SRC=$W/nonexistent DRY=1 launch "$LA/v2t-a.txt"; r=$?
expect "launcher refuses without the Mac's CIFAR-100 copy" bash -c "[ $r = 2 ] && grep -q 'no $W/nonexistent/train' '$W/launch_out.txt'"
sleep 1

# === the pod setup's dataset phase, with a fake download (curl on PATH) ===============================================
SD=$W/setup; mkdir -p "$SD/src/cifar-100-python" "$SD/bin" "$SD/repo/sky"
cp "$SRC/sky/v2_pod_setup.sh" "$SD/repo/sky/"
for f in train test meta; do head -c 3000 /dev/urandom > "$SD/src/cifar-100-python/$f"; done
touch "$SD/src/cifar-100-python/file.txt~"
( cd "$SD/src" && COPYFILE_DISABLE=1 tar -czf "$SD/good.tar.gz" cifar-100-python )
cp -R "$SD/src" "$SD/src_bad"; printf 'x' >> "$SD/src_bad/cifar-100-python/test"
( cd "$SD/src_bad" && COPYFILE_DISABLE=1 tar -czf "$SD/bad.tar.gz" cifar-100-python )
cat > "$SD/bin/curl" <<'EOF2'
#!/bin/bash
# fake curl: -o FILE URL; copies $FAKE_TGZ, or fails with FAKE_CURL_FAIL=1; logs each call
o=""; while [ $# -gt 0 ]; do case "$1" in -o) o=$2; shift ;; esac; shift; done
echo call >> "$SETUP_STATE/curl_calls"
[ "${FAKE_CURL_FAIL:-0}" = 1 ] && exit 7
cp "$FAKE_TGZ" "$o"
EOF2
chmod +x "$SD/bin/curl"
md5of() { md5 -q "$1" 2>/dev/null || md5sum < "$1" | cut -d' ' -f1; }
shaof() { shasum -a 256 < "$1" | cut -d' ' -f1; }
GOODMD5=$(md5of "$SD/good.tar.gz"); BADMD5=$(md5of "$SD/bad.tar.gz")
setup_run() {  # $1 = tarball, $2 = md5 to expect; env in the caller (e.g. SHA_TEST); prints the exit code
  rm -rf "$SD/state"; mkdir -p "$SD/state"
  ( cd "$SD/repo" && env PATH="$SD/bin:$PATH" SETUP_STATE="$SD/state" FAKE_TGZ="$1" CIFAR_MD5="$2" DATA_DIR="$SD/home/cifar-100-python" \
      SETUP_LOG="$SD/state/setup.log" FETCH_TRIES=2 RETRY_SLEEP=0 \
      CIFAR_SHA256_TRAIN="${SHA_TRAIN-$(shaof "$SD/src/cifar-100-python/train")}" \
      CIFAR_SHA256_TEST="${SHA_TEST-$(shaof "$SD/src/cifar-100-python/test")}" \
      CIFAR_SHA256_META="${SHA_META-$(shaof "$SD/src/cifar-100-python/meta")}" \
      bash sky/v2_pod_setup.sh dataset > "$SD/state/out.txt" 2>&1 ); echo $?
}
mkdir -p "$SD/home"
r=$(setup_run "$SD/good.tar.gz" "$GOODMD5")
expect "setup: good download -> exit 0, dataset in place, byte-identical, no leftovers" \
  bash -c "[ $r = 0 ] && cmp -s '$SD/src/cifar-100-python/train' '$SD/home/cifar-100-python/train' && grep -q 'SETUP OK' '$SD/state/setup.log' && [ -z \"\$(ls -A '$SD/home' | grep -v '^cifar-100-python\$')\" ]"
r=$(setup_run "$SD/good.tar.gz" "$GOODMD5")
expect "setup: dataset already present and matching -> not fetched again" bash -c "[ $r = 0 ] && [ ! -e '$SD/state/curl_calls' ] && grep -q 'not fetched' '$SD/state/setup.log'"
rm -rf "$SD/home"; mkdir -p "$SD/home"
r=$(setup_run "$SD/bad.tar.gz" "$BADMD5")
expect "setup: extracted test file differs from the Mac's (md5 of the tarball fine) -> exit 1, loud, no dataset dir" \
  bash -c "[ $r = 1 ] && grep -q 'SETUP FAILED: extracted CIFAR-100 differs from the Mac.s copy: sha256 of .*/test is' '$SD/state/setup.log' && [ ! -e '$SD/home/cifar-100-python' ] && [ -z \"\$(ls -A '$SD/home')\" ]"
r=$(setup_run "$SD/bad.tar.gz" "$GOODMD5")
expect "setup: md5 of the download wrong on every attempt -> exit 1 after FETCH_TRIES (2) attempts, no dataset dir" \
  bash -c "[ $r = 1 ] && [ \"\$(grep -c . '$SD/state/curl_calls')\" = 2 ] && grep -q 'could not fetch' '$SD/state/setup.log' && [ ! -e '$SD/home/cifar-100-python' ]"
r=$(FAKE_CURL_FAIL=1 setup_run "$SD/good.tar.gz" "$GOODMD5")
expect "setup: download fails every time -> exit 1" bash -c "[ $r = 1 ] && grep -q 'download failed' '$SD/state/setup.log'"
r=$(SHA_META= setup_run "$SD/good.tar.gz" "$GOODMD5")
expect "setup: a sha256 not passed -> exit 1 before any download" bash -c "[ $r = 1 ] && [ ! -e '$SD/state/curl_calls' ] && grep -q 'CIFAR_SHA256_META not set' '$SD/state/setup.log'"
r=$(setup_run "$SD/good.tar.gz" "$GOODMD5"); mv "$SD/home/cifar-100-python" "$SD/home/keep"; cp -R "$SD/home/keep" "$SD/home/cifar-100-python"; printf 'x' >> "$SD/home/cifar-100-python/meta"
r=$(setup_run "$SD/good.tar.gz" "$GOODMD5")
expect "setup: an existing mismatching dataset is moved aside and fetched again" \
  bash -c "[ $r = 0 ] && ls -d '$SD/home'/cifar-100-python.mismatch.* > /dev/null 2>&1 && cmp -s '$SD/src/cifar-100-python/meta' '$SD/home/cifar-100-python/meta'"

# F4a: the pod setup's ltm phase checks the mounted LTM checkpoints against the Mac's sha256s
mkdir -p "$SD/ltm"; head -c 4000 /dev/urandom > "$SD/ltm/cifar_100_subclasses_12_e13.pth"; head -c 4500 /dev/urandom > "$SD/ltm/cifar_100_subclasses_12_e40.pth"
L13=$(shaof "$SD/ltm/cifar_100_subclasses_12_e13.pth"); L40=$(shaof "$SD/ltm/cifar_100_subclasses_12_e40.pth")
ltm_run() {  # env in the caller: E13, E40 (hashes to expect); prints the exit code
  rm -rf "$SD/state"; mkdir -p "$SD/state"
  ( cd "$SD/repo" && env LTM_DIR="$SD/ltm" SETUP_LOG="$SD/state/setup.log" LTM_SHA256_E13="${E13-$L13}" LTM_SHA256_E40="${E40-}" \
      bash sky/v2_pod_setup.sh ltm > "$SD/state/out.txt" 2>&1 ); echo $?
}
r=$(ltm_run);                      expect "setup ltm: e13 matches, no e40 asked -> exit 0" bash -c "[ $r = 0 ] && grep -q 'e13 checkpoint sha256 .* = the Mac' '$SD/state/setup.log'"
r=$(E40=$L40 ltm_run);             expect "setup ltm: e13 and e40 match -> exit 0" bash -c "[ $r = 0 ] && grep -q 'e40 checkpoint sha256' '$SD/state/setup.log'"
r=$(E13=$L40 ltm_run);             expect "setup ltm: e13 differs from the Mac's -> exit 1, loud" bash -c "[ $r = 1 ] && grep -q 'SETUP FAILED: sha256 of .*e13.pth is' '$SD/state/setup.log'"
r=$(E40=$L13 ltm_run);             expect "setup ltm: e40 differs from the Mac's -> exit 1, loud" bash -c "[ $r = 1 ] && grep -q 'SETUP FAILED: sha256 of .*e40.pth is' '$SD/state/setup.log'"
r=$(E13= ltm_run);                 expect "setup ltm: no e13 hash passed -> exit 1" bash -c "[ $r = 1 ] && grep -q 'LTM_SHA256_E13 not set' '$SD/state/setup.log'"
mv "$SD/ltm/cifar_100_subclasses_12_e40.pth" "$SD/ltm/e40.away"
r=$(E40=$L40 ltm_run);             expect "setup ltm: e40 asked but not mounted -> exit 1" bash -c "[ $r = 1 ] && grep -q 'no .*e40.pth (file mount' '$SD/state/setup.log'"
mv "$SD/ltm/e40.away" "$SD/ltm/cifar_100_subclasses_12_e40.pth"

echo "---"; echo "$passes passed, $fails wrong"
[ "$fails" = 0 ] && { echo "ALL TESTS PASS"; exit 0; } || { echo "$fails TEST(S) WRONG"; exit 1; }
