#!/bin/bash
# Tests of the destructive predicate (sky/v2_verify.sh) and of the pod job's sentinels (sky/v2_pod_job.sh) on fake
# trees, with a fake run_v2.sh: no GPU, no pod, no real results. Run it before using sky/v2_reap.sh after any change
# to the pod job, the verification or the pull. Every known-good case must print OK and every known-bad one FAIL.
# Usage: bash sky/test_v2_verify.sh <empty scratch dir>     (never a results directory)
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd)
W=${1:?scratch dir}
mkdir -p "$W" && W=$(cd "$W" && pwd)
[ -z "$(ls -A "$W")" ] || { echo "scratch dir $W is not empty" >&2; exit 2; }
ROUND=runs_local/test_round
J=v2t-job
fails=0
check() {  # $1 = expected (OK|FAIL), $2 = label, $3 = mac root
  local out rc got
  out=$(bash "$SRC/sky/v2_verify.sh" "$ROUND" "$J" "$3" 2>&1); rc=$?
  got=FAIL; [ $rc -eq 0 ] && got=OK
  if [ "$got" = "$1" ]; then echo "pass  expect $1  $2"; else echo "WRONG expect $1 got $got  $2"; echo "$out" | sed 's/^/      /'; fails=$((fails + 1)); fi
}

# --- a fake repo with a fake run_v2.sh that writes what the real one writes -------------------------
make_repo() {  # $1 = dir
  mkdir -p "$1/sky"
  cp "$SRC/sky/v2_pod_job.sh" "$1/sky/"
  cat > "$1/run_v2.sh" <<'EOF'
#!/bin/bash
set -u
cd "$(dirname "$0")"
S=$1 M=$2 CC=$3 SEED=$4 A5=${5:-} N=${6:-}
P=pair$(echo $CC | tr ' ' '_')
D=$RUNS/$S/$M/$P/seed$SEED
case "$S" in continual) D=$D/order$(echo $A5 | tr ' ' '_') ;; stream) D=$D/fine$A5 ;; fewshot) D=$D/fine${A5}_n$N ;; esac
[ "${FAKE_FAIL:-0}" = 1 ] && { mkdir -p "$D"; echo boom > "$D/job.log"; exit 1; }
R="$D/cifar_100/${S}_[3, 4]_500/2026/10/03/12-00-00"
mkdir -p "$R"
printf '[0, 1], [3], evaluate, 0, 0.5\n' > "$R/results_$S.txt"
printf '[0, 1], [3], evaluate, 0, 0.6\n' > "$R/results_${S}_val.txt"
{ echo "log $*"; echo "EXTRA=[${EXTRA:-}] SAVE_STM=[${SAVE_STM:-}]"; } > "$D/job.log"; echo "abc123" > "$D/code_commit.txt"
case "$S" in pretrain) head -c 5000 /dev/zero > "$D/stm_pretrain.pth" ;; continual) head -c 4000 /dev/zero > "$D/stm_phase3.pth" ;; *) head -c 3000 /dev/zero > "$D/stm_final.pth" ;; esac
touch "$D/job.done"
EOF
}

T=runs_local/fake_tree
cat > "$W/actions.txt" <<EOF
$T|PRETRAIN_EPOCHS=1|pretrain rl "0 1" 1|$T/pretrain/rl/pair0_1/seed1|
$T|EPOCHS=1 EXTRA="--ltm-obs-cache check"|continual rl "0 1" 1 "3 4"|$T/continual/rl/pair0_1/seed1/order3_4|
$T|EPOCHS=2|stream rl "0 1" 1 3|$T/stream/rl/pair0_1/seed1/fine3|
EOF

run_pod() {  # $1 = pod repo dir, extra env in the caller
  ( cd "$1" && ROUND=$ROUND JOB=$J PY=python3 REQUIRE_CUDA=0 NPROC=2 bash sky/v2_pod_job.sh "$W/actions.txt" > pod_stdout.txt 2>&1 )
}
pull() {  # $1 = pod repo, $2 = mac root: the reaper's pull, local to local
  mkdir -p "$2/$ROUND/actions" && cp "$W/actions.txt" "$2/$ROUND/actions/$J.txt"
  while IFS='|' read -r tree env args unit flag; do
    [ -z "$tree" ] && continue
    mkdir -p "$2/$unit" && rsync -a --ignore-existing "$1/$unit/" "$2/$unit/"
  done < "$W/actions.txt"
  mkdir -p "$2/$ROUND/pods/$J" && rsync -a --ignore-existing "$1/$ROUND/pods/$J/" "$2/$ROUND/pods/$J/"
}
fresh() { rm -rf "$W/mac" && cp -R "$W/mac_good" "$W/mac"; }

# --- good pod job ------------------------------------------------------------------------------
make_repo "$W/pod"
run_pod "$W/pod"
[ -f "$W/pod/$ROUND/pods/$J/JOB_COMPLETE" ] && [ ! -e "$W/pod/$ROUND/pods/$J/JOB_FAILED" ] \
  && echo "pass  pod job leaves JOB_COMPLETE only" || { echo "WRONG pod job sentinels"; cat "$W/pod/pod_stdout.txt"; fails=$((fails + 1)); }
grep -qF 'EXTRA=[--ltm-obs-cache check] SAVE_STM=[1]' "$W/pod/$T/continual/rl/pair0_1/seed1/order3_4/job.log" \
  && echo "pass  quoted EXTRA reaches run_v2.sh as one word" || { echo "WRONG quoted EXTRA"; fails=$((fails + 1)); }
pull "$W/pod" "$W/mac_good"
check OK "complete pull" "$W/mac_good"

check FAIL "nothing pulled" "$W/empty_mac"
fresh; rm "$W/mac/$ROUND/pods/$J/JOB_COMPLETE";            check FAIL "no JOB_COMPLETE" "$W/mac"
fresh; touch "$W/mac/$ROUND/pods/$J/JOB_FAILED";           check FAIL "JOB_FAILED present" "$W/mac"
fresh; rm "$W/mac/$ROUND/actions/$J.txt";                  check FAIL "launcher actions copy missing" "$W/mac"
fresh; echo "x|y|pretrain a|x/u|" >> "$W/mac/$ROUND/actions/$J.txt"; check FAIL "actions differ" "$W/mac"
fresh; rm "$W/mac/$ROUND/pods/$J/manifest.tsv";            check FAIL "manifest missing" "$W/mac"
fresh; : > "$W/mac/$ROUND/pods/$J/manifest.tsv";           check FAIL "manifest empty" "$W/mac"
fresh; echo "garbage" >> "$W/mac/$ROUND/pods/$J/manifest.tsv"; check FAIL "manifest garbage line" "$W/mac"
fresh; printf '12\t/etc/passwd\n' >> "$W/mac/$ROUND/pods/$J/manifest.tsv"; check FAIL "manifest path outside units" "$W/mac"
fresh; f=$(find "$W/mac/$T/continual" -name 'results_continual.txt' | head -1); printf 'x' >> "$f"; check FAIL "a results file differs in size" "$W/mac"
fresh; find "$W/mac/$T/stream" -name 'results_stream_val.txt' -delete; check FAIL "a _val results file missing" "$W/mac"
fresh; rm "$W/mac/$T/continual/rl/pair0_1/seed1/order3_4/job.done"; check FAIL "a job.done missing" "$W/mac"
fresh; rm "$W/mac/$T/pretrain/rl/pair0_1/seed1/stm_pretrain.pth"; check FAIL "pre-training checkpoint missing" "$W/mac"
fresh; rm -r "$W/mac/$T/stream";                           check FAIL "a whole unit missing" "$W/mac"
fresh; check OK "fresh copy is still good (the cases above did not leak)" "$W/mac"

# --- a pod job in which a unit fails ---------------------------------------------------------------
make_repo "$W/podbad"
( cd "$W/podbad" && FAKE_FAIL=1 ROUND=$ROUND JOB=$J PY=python3 REQUIRE_CUDA=0 bash sky/v2_pod_job.sh "$W/actions.txt" > pod_stdout.txt 2>&1 )
[ -f "$W/podbad/$ROUND/pods/$J/JOB_FAILED" ] && [ ! -e "$W/podbad/$ROUND/pods/$J/JOB_COMPLETE" ] \
  && echo "pass  failing pod job leaves JOB_FAILED only" || { echo "WRONG failing pod job sentinels"; fails=$((fails + 1)); }
pull "$W/podbad" "$W/mac_bad"
check FAIL "pull of a failed job" "$W/mac_bad"

# --- a pod job with a malformed actions file ---------------------------------------------------------
make_repo "$W/podmal"
printf 'only|three|fields\n' > "$W/mal.txt"
( cd "$W/podmal" && ROUND=$ROUND JOB=$J PY=python3 REQUIRE_CUDA=0 bash sky/v2_pod_job.sh "$W/mal.txt" > pod_stdout.txt 2>&1 )
[ -f "$W/podmal/$ROUND/pods/$J/JOB_FAILED" ] && [ ! -e "$W/podmal/$ROUND/pods/$J/JOB_COMPLETE" ] \
  && echo "pass  malformed actions file leaves JOB_FAILED only" || { echo "WRONG malformed actions sentinels"; fails=$((fails + 1)); }

# --- the reaper itself, with fake sky, ssh and rsync (sky/v2_reap.sh REAP_TEST_BIN) ----------------------
# Fake pod = a local dir standing for ~/sky_workdir. Fake sky: a cluster is UP iff $FAKE_STATE/<c>.up exists;
# `sky down` records the cluster in $FAKE_STATE/downs. Fake ssh: only `test -f ~/sky_workdir/<path>`.
B=$W/bin; mkdir -p "$B"
cat > "$B/sky" <<'EOF'
#!/bin/bash
case "$1" in
  status) echo "NAME  INFRA  RESOURCES  STATUS  AUTOSTOP  LAUNCHED"
          [ -n "${2:-}" ] && [ -f "$FAKE_STATE/$2.up" ] && echo "$2  RunPod  1x(L4)  UP  4h (down)  1m ago" ;;
  down)   rm -f "$FAKE_STATE/$2.up"; echo "$2" >> "$FAKE_STATE/downs" ;;
esac
exit 0
EOF
cat > "$B/ssh" <<'EOF'
#!/bin/bash
a=("$@"); n=${#a[@]}; cmd=${a[$((n-1))]}; host=${a[$((n-2))]}
[ -f "$FAKE_STATE/$host.sshfail" ] && exit 255
case "$cmd" in "test -f ~/sky_workdir/"*) test -f "$FAKE_POD/${cmd#"test -f ~/sky_workdir/"}" ;; *) exit 255 ;; esac
EOF
cat > "$B/rsync" <<'EOF'
#!/bin/bash
out=(); skip=0
for a in "$@"; do
  [ $skip = 1 ] && { skip=0; continue; }
  case "$a" in -e) skip=1; continue ;; *":~/sky_workdir/"*) a="$FAKE_POD/${a#*":~/sky_workdir/"}" ;; esac
  out+=("$a")
done
[ -f "$FAKE_STATE/rsyncfail" ] && exit 12
exec /usr/bin/rsync "${out[@]}"
EOF
chmod +x "$B/sky" "$B/ssh" "$B/rsync"

reap_case() {  # $1 = expect (DOWN|UP), $2 = label, $3 = fake pod dir; set up $FAKE_STATE (st) before the call
  local m=$W/macr got
  rm -rf "$m"; mkdir -p "$m/sky" "$m/$ROUND/actions"
  cp "$SRC/sky/v2_reap.sh" "$SRC/sky/v2_verify.sh" "$m/sky/"
  cp "$W/actions.txt" "$m/$ROUND/actions/$J.txt"
  [ -n "${PREEXIST:-}" ] && { mkdir -p "$(dirname "$m/$PREEXIST")"; echo "mac's own file" > "$m/$PREEXIST"; }
  FAKE_POD=$3 REAP_TEST_BIN=$B MAX_PASSES=1 INTERVAL=0 bash "$m/sky/v2_reap.sh" "$ROUND" > "$W/reap_out.txt" 2>&1
  got=UP; grep -qx "$J" "$FAKE_STATE/downs" 2>/dev/null && got=DOWN
  if [ "$got" = "$1" ]; then echo "pass  reaper expect $1  $2"; else echo "WRONG reaper expect $1 got $got  $2"; sed 's/^/      /' "$W/reap_out.txt"; fails=$((fails + 1)); fi
  if [ "$got" = DOWN ] && [ ! -e "$m/$ROUND/reaped/$J" ]; then echo "WRONG reaper downed without a reaped marker"; fails=$((fails + 1)); fi
  if [ -n "${PREEXIST:-}" ] && ! grep -q "mac's own file" "$m/$PREEXIST"; then echo "WRONG reaper overwrote a file on the Mac"; fails=$((fails + 1)); fi
  return 0
}
export FAKE_STATE=$W/state
st() { rm -rf "$FAKE_STATE"; mkdir -p "$FAKE_STATE"; local x; for x in "$@"; do touch "$FAKE_STATE/$x"; done; }
cp -R "$W/pod" "$W/pod_running"; rm "$W/pod_running/$ROUND/pods/$J/JOB_COMPLETE"
cp -R "$W/pod" "$W/pod_lost"; find "$W/pod_lost/$T/stream" -name 'results_stream.txt' -delete
st;                      reap_case UP   "cluster not in sky status" "$W/pod"
st "$J.up" "$J.sshfail"; reap_case UP   "ssh fails" "$W/pod"
st "$J.up";              reap_case UP   "job still running (no sentinel)" "$W/pod_running"
st "$J.up" rsyncfail;    reap_case UP   "rsync fails" "$W/pod"
st "$J.up";              reap_case UP   "sentinel, but a manifest file is gone from the pod" "$W/pod_lost"
st "$J.up";              reap_case UP   "JOB_FAILED" "$W/podbad"
[ -e "$W/macr/$ROUND/failed_pulled/$J" ] && echo "pass  JOB_FAILED pod was pulled" || { echo "WRONG JOB_FAILED pod not pulled"; fails=$((fails + 1)); }
st "$J.up"; PREEXIST=$T/stream/rl/pair0_1/seed1/fine3/job.log reap_case UP "a different file already on the Mac (kept, not overwritten)" "$W/pod"
st "$J.up";              reap_case DOWN "complete job" "$W/pod"

echo "---"; [ "$fails" = 0 ] && { echo "ALL TESTS PASS"; exit 0; } || { echo "$fails TEST(S) WRONG"; exit 1; }
