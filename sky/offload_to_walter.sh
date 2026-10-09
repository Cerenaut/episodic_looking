#!/bin/bash
# Move STM checkpoints (*.pth, ~44 MB each) of FINISHED run_v2.sh units from this Mac to Walter (storage only), at the
# same path relative to the repo: <repo>/<rel> -> g_walter:~/Dev/episodic_looking/<rel>. Results files, logs and every
# other file stay on the Mac. Storage policy (Gideon, 4 Oct 2026): Walter's ~/Dev is the main store; the Mac is the
# landing point for pod pulls (plan.md section 1b).
#
# Usage: [DRY=1] bash sky/offload_to_walter.sh <dir> [<dir> ...]
#   <dir>  result trees or unit dirs under the repo (relative to it or absolute), e.g. runs_v2/continual/rl,
#          runs_v2/pretrain/actor/pair0_1/seed2. Every *.pth file under them is a candidate (find -print0: paths with
#          brackets and spaces are fine; a path with a tab or newline is refused).
#
# A candidate is offloaded only if its UNIT (the nearest directory above it, up to the repo, holding job.done, job.log
# or a .lock dir: run_v2.sh's markers) is finished and safe to touch. Refused (reported, skipped, exit 3 at the end):
#   - no unit found (not a run_v2.sh unit: older trees have no job.done) or the unit has no job.done (unfinished);
#   - a .lock dir anywhere in the unit (a run is, or was killed, in progress);
#   - the unit is listed in a pod round's actions file (any <dir>/actions/<cluster>.txt in the repo, depth <= 6, as
#     sky/v2_launch.sh searches) whose pod is not reaped AND verified: <round>/reaped/<cluster> must exist, and
#     <round>/verified/<cluster> (JOB_COMPLETE, sky/v2_verify.sh passed) or <round>/failed_safe/<cluster> (JOB_FAILED,
#     --partial verification passed). So the reaper's verification is over before a file goes missing. A pod reaped by
#     hand (reaped/ without verified/ or failed_safe/) is refused: decide by hand. The one exception is an ABANDONED
#     pod (torn down or never started with nothing pulled, its units re-run by a relaunch of the same actions file under
#     another name): <round>/abandoned/<cluster>, written by sky/mark_abandoned.sh after its checks, is accepted in
#     place of verified/ or failed_safe/, only together with reaped/<cluster>, and only if it has a "reason:" line
#     with a non-blank reason, and never while pods/<cluster>, held/<cluster>, failed_verify/<cluster> or any
#     *_pulled/<cluster> exists in that round (else refused). If the round search fails, nothing is offloaded.
#
# Per file (each step must succeed, else the Mac copy is kept, the error reported, and the run STOPS, exit 1):
#   1. sha256 on the Mac (64 hex digits, else stop);
#   2. on Walter: absent -> mkdir -p its directory, then rsync -t --ignore-existing (exit 0 required); already there
#      with the SAME sha256 -> nothing sent (an earlier interrupted run); there with a DIFFERENT sha256, or not a
#      regular file -> STOP (never overwritten); an unreadable reply (ssh failure, unexpected output) -> STOP;
#   3. sha256 computed ON WALTER must equal the Mac's;
#   4. one line appended to the manifest and to the unit's pointer file (both before the delete; a failed write = stop);
#   5. only then the Mac copy is removed (rm, then checked gone).
# Nothing on Walter is ever deleted or overwritten. DRY=1: every check (eligibility, Mac hashes, Walter's state and
# hashes of existing files) runs, nothing is created, sent, written or removed.
#
# Records (on the Mac):
#   MANIFEST (default runs_local/offload_manifest.tsv): time<TAB>mac path<TAB>walter path<TAB>size<TAB>sha256 per file.
#   <unit>/OFFLOADED_TO_WALTER.txt: the same per moved file of that unit, with a restore command.
#
# AFTER OFFLOADING, A POD UNIT NO LONGER MATCHES ITS POD MANIFEST: the .pth files are gone and the pointer file is new,
# so sky/v2_verify.sh on that round would fail. That is expected: verification is done once, by the reaper, before
# the pod is torn down, and offloading is gated on it. Do not re-run sky/v2_verify.sh on an offloaded round; to check
# the checkpoints, compare the manifest's sha256 with `sha256sum` on Walter.
# PRE-TRAINING CHECKPOINTS (stm_pretrain.pth) are inputs: every other unit of the same tree, model, pair, seed and LTM
# loads it (continual, baseline, few-shot, and later single-stream units or relaunches). They are refused unless
# OFFLOAD_PRETRAIN=1, and even then while any existing unit that loads it is unfinished or locked (units not created
# yet cannot be seen: only set OFFLOAD_PRETRAIN=1 when nothing more will be run from that pre-training on this Mac or
# shipped from it to a pod).
# Paths outside the repo (a round with a PULL_ROOT elsewhere) are refused: Walter's path is relative to the repo.
# A unit whose checkpoint is needed again on the Mac (e.g. stm_pretrain.pth for a later local run) must be restored
# first (command in its pointer file); run_v2.sh does not know about Walter.
#
# Environment: DRY=1; OFFLOAD_PRETRAIN=1 (also move stm_pretrain.pth, see above); WALTER_HOST (ssh alias, default g_walter; always with -o ClearAllForwardings=yes, BatchMode);
# WALTER_ROOT (repo dir on Walter, relative to its home, default Dev/episodic_looking); MANIFEST.
# Exit: 0 all candidates moved (or, DRY, would be); 1 a transfer or check failed (stopped; that file and the rest kept);
# 2 usage or precondition (lock held, round search failed, Walter unreachable before anything was sent);
# 3 every eligible file moved but some candidates were refused.
# One run at a time: runs_local/offload.lock (mkdir; not taken under DRY). Bash 3.2 (macOS) compatible.
set -u
cd "$(dirname "$0")/.." || exit 2
REPO=$(pwd -P)
DRY=${DRY:-0}
HOST=${WALTER_HOST:-g_walter}
WROOT=${WALTER_ROOT:-Dev/episodic_looking}
MANIFEST=${MANIFEST:-$REPO/runs_local/offload_manifest.tsv}
[ $# -ge 1 ] || { echo "usage: [DRY=1] bash $0 <dir> [<dir> ...]" >&2; exit 2; }

say() { echo "[$(date '+%F %T')] $*"; }
die() { say "OFFLOAD REFUSED: $*" >&2; exit 2; }
sha256() { shasum -a 256 < "$1" | cut -d' ' -f1; }
fsize() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1" 2>/dev/null; }
is_hex64() { [[ "$1" =~ ^[0-9a-f]{64}$ ]]; }

CM="$HOME/.ssh/offload-%C"   # connection sharing: one ssh login for the whole run
SSH_OPTS="-o ClearAllForwardings=yes -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=120"
# shellcheck disable=SC2086  # SSH_OPTS is word-split on purpose (no spaces inside an option)
rssh() { ssh $SSH_OPTS "$HOST" "$@" < /dev/null; }

# --- lock ------------------------------------------------------------------------------------------------
LOCK=$REPO/runs_local/offload.lock
TMPL=""
cleanup() { [ -n "$TMPL" ] && rm -f "$TMPL" "$TMPL.ok"; [ "$DRY" = 1 ] || rmdir "$LOCK" 2>/dev/null; }
if [ "$DRY" != 1 ]; then
  mkdir -p "$REPO/runs_local" || die "cannot create runs_local"
  mkdir "$LOCK" 2>/dev/null || die "another offload holds $LOCK (or a killed one: check, then rmdir it)"
fi
trap cleanup EXIT
TMPL=$(mktemp -t offload_list) || die "mktemp failed"   # candidate list (paths only, not results)

# --- pod rounds: unit -> round, cluster -------------------------------------------------------------------
ROUNDIDX=""
hits=$(find "$REPO" -maxdepth 6 \( -name .git -o -name .venv -o -name node_modules \) -prune -o -type f -path '*/actions/*.txt' -print 2>/dev/null) \
  || die "cannot search the repo for pod rounds (find failed): nothing offloaded"
while IFS= read -r af; do
  [ -n "$af" ] || continue
  rd=$(dirname "$(dirname "$af")"); c=$(basename "$af" .txt)
  [ -r "$af" ] || die "cannot read $af: nothing offloaded"
  while IFS='|' read -r _tree _env _args unit _flag || [ -n "${_tree:-}" ]; do
    [ -n "${unit:-}" ] || continue
    ROUNDIDX="$ROUNDIDX$unit	$rd	$c
"
  done < "$af"
done <<< "$hits"

# unit_status <unit abs> <unit rel>: prints OK or the reason it is refused
unit_status() {
  local u=$1 rel=$2 lk m rd c ok=1 why=""
  [ -f "$u/job.done" ] || { echo "no job.done (unfinished, or not a run_v2.sh unit)"; return; }
  lk=$(find "$u" -name .lock -print 2>/dev/null) || { echo "cannot search the unit for .lock (find failed)"; return; }
  [ -z "$lk" ] || { echo "a .lock dir is present ($(echo "$lk" | head -1))"; return; }
  m=$(printf '%s' "$ROUNDIDX" | awk -F'\t' -v u="$rel" '$1 == u { print $2 "\t" $3 }')
  while IFS='	' read -r rd c; do
    [ -n "$rd" ] || continue
    if [ ! -f "$rd/reaped/$c" ]; then ok=0; why="$why pod $c of round ${rd#"$REPO/"} not reaped;"
    elif [ -e "$rd/verified/$c" ] || [ -e "$rd/failed_safe/$c" ]; then :
    elif [ -e "$rd/abandoned/$c" ]; then
      # accepted only with reaped/ (checked above) and a non-blank reason (sky/mark_abandoned.sh writes "reason: ...")
      # and never for a pod that pulled anything (defence in depth against a hand-written or stale marker)
      pulled=""
      for x in "$rd/pods/$c" "$rd/held/$c" "$rd/failed_verify/$c" "$rd"/*_pulled/"$c"; do [ -e "$x" ] && pulled="$pulled ${x#"$rd/"}"; done
      if [ ! -f "$rd/abandoned/$c" ] || ! grep -qE '^reason:.*[^[:space:]]' "$rd/abandoned/$c" 2>/dev/null; then
        ok=0; why="$why pod $c of round ${rd#"$REPO/"} has an abandoned/ marker without a non-blank reason: line;"
      elif [ -n "$pulled" ]; then
        ok=0; why="$why pod $c of round ${rd#"$REPO/"} has an abandoned/ marker but records of a pull or hold ($(echo $pulled)): decide by hand;"
      fi
    else
      ok=0; why="$why pod $c of round ${rd#"$REPO/"} reaped without verified/, failed_safe/ or abandoned/ (by hand? see sky/mark_abandoned.sh);"
    fi
  done <<< "$m"
  [ $ok = 1 ] && echo OK || echo "${why# }"
}

# dependents_status <pre-training unit abs>: OK, or why its checkpoint is still needed. The units that load it are
# <tree>/<setting><suffix>/<model>/<pair>/seed<k>/... for every setting other than pretrain (suffix _e40 for an e40
# pre-training, none otherwise: run_v2.sh's layout). Any of them unfinished (job.log without job.done) or locked = needed.
# Units not created yet cannot be seen: hence OFFLOAD_PRETRAIN=1 is required at all.
dependents_status() {
  local u=$1 seed pair model pdir tree suf S name D hits h ud
  seed=$(basename "$u"); pair=$(basename "$(dirname "$u")"); model=$(basename "$(dirname "$(dirname "$u")")")
  pdir=$(dirname "$(dirname "$(dirname "$u")")"); tree=$(dirname "$pdir")
  case "$(basename "$pdir")" in pretrain) suf="" ;; pretrain_e40) suf=_e40 ;; *) echo "not in run_v2.sh's pretrain layout"; return ;; esac
  for S in "$tree"/*; do
    name=$(basename "$S")
    case "$name" in pretrain*) continue ;; esac
    if [ -z "$suf" ]; then case "$name" in *_e40) continue ;; esac; else case "$name" in *_e40) ;; *) continue ;; esac; fi
    D=$S/$model/$pair/$seed
    [ -d "$D" ] || continue
    hits=$(find "$D" \( -name job.log -o -name .lock \) -print 2>/dev/null) || { echo "cannot search $D (find failed)"; return; }
    while IFS= read -r h; do
      [ -n "$h" ] || continue
      ud=$(dirname "$h")
      case "$h" in */.lock) echo "unit ${ud#"$REPO/"} is locked (running)"; return ;; esac
      [ -f "$ud/job.done" ] || { echo "unit ${ud#"$REPO/"} is unfinished"; return; }
    done <<< "$hits"
  done
  echo OK
}

# --- candidates ------------------------------------------------------------------------------------------
NREF=0; NCAND=0; UCACHE=""
for a in "$@"; do
  [ -d "$a" ] || die "not a directory: $a"
  A=$(cd "$a" && pwd -P) || die "cannot resolve $a"
  case "$A/" in "$REPO/"?*) ;; *) die "$a is not inside the repo $REPO" ;; esac
  find "$A" -type f -name '*.pth' -print0 > "$TMPL" || die "find failed under $A: nothing offloaded"
  while IFS= read -r -d '' f; do
    NCAND=$((NCAND + 1))
    case "$f" in *$'\t'*|*$'\n'*) say "REFUSED (tab or newline in the path): $f"; NREF=$((NREF + 1)); continue ;; esac
    # the unit: nearest dir up to the repo with job.done, job.log or .lock
    d=$(dirname "$f"); u=""
    while :; do
      if [ -e "$d/job.done" ] || [ -e "$d/job.log" ] || [ -d "$d/.lock" ]; then u=$d; break; fi
      [ "$d" = "$REPO" ] || [ "$d" = / ] && break
      d=$(dirname "$d")
    done
    if [ -z "$u" ]; then say "REFUSED (not inside a run_v2.sh unit: no job.done/job.log/.lock above it): ${f#"$REPO/"}"; NREF=$((NREF + 1)); continue; fi
    urel=${u#"$REPO/"}
    st=$(printf '%s' "$UCACHE" | awk -F'\t' -v u="$urel" '$1 == u { print $2; exit }')
    if [ -z "$st" ]; then st=$(unit_status "$u" "$urel"); UCACHE="$UCACHE$urel	$st
"; fi
    if [ "$st" != OK ]; then say "REFUSED ($st): ${f#"$REPO/"}"; NREF=$((NREF + 1)); continue; fi
    if [ "$(basename "$f")" = stm_pretrain.pth ]; then
      if [ "${OFFLOAD_PRETRAIN:-0}" != 1 ]; then
        say "REFUSED (pre-training checkpoint: an input of later units; kept unless OFFLOAD_PRETRAIN=1): ${f#"$REPO/"}"; NREF=$((NREF + 1)); continue
      fi
      dep=$(dependents_status "$u")
      if [ "$dep" != OK ]; then say "REFUSED (pre-training checkpoint still needed: $dep): ${f#"$REPO/"}"; NREF=$((NREF + 1)); continue; fi
    fi
    printf '%s\t%s\0' "$u" "$f" >> "$TMPL.ok" || die "cannot write the candidate list"
  done < "$TMPL"
  rm -f "$TMPL"
done
NOK=0; [ -s "$TMPL.ok" ] && NOK=$(tr -cd '\0' < "$TMPL.ok" | wc -c | tr -d ' ')
say "$NCAND candidate .pth file(s); $NOK eligible, $NREF refused$([ "$DRY" = 1 ] && echo '; DRY: nothing will be sent, written or removed')"
if [ "$NOK" = 0 ]; then rm -f "$TMPL.ok"; [ "$NREF" -gt 0 ] && exit 3; exit 0; fi

# --- Walter reachable, repo dir present ---------------------------------------------------------------------
r=$(rssh "test -d $(printf %q "$WROOT") && echo READY") || die "ssh to $HOST failed (exit $?): nothing sent"
[ "$r" = READY ] || die "Walter has no ~/$WROOT (reply: '$r'): nothing sent"

# --- move -------------------------------------------------------------------------------------------------
fail() { say "FAILED, Mac copy KEPT, stopping: $*" >&2; rm -f "$TMPL.ok"; exit 1; }
MOVED=0; SAME=0
while IFS= read -r -d '' rec; do
  u=${rec%%$'\t'*}; f=${rec#*$'\t'}
  rel=${f#"$REPO/"}; dest="$WROOT/$rel"; qd=$(printf %q "$dest")
  h=$(sha256 "$f"); is_hex64 "$h" || fail "cannot hash $rel on the Mac"
  sz=$(fsize "$f"); [[ "$sz" =~ ^[0-9]+$ ]] || fail "cannot read the size of $rel"
  # Walter's state of this path: ABSENT | EXISTS <sha256> | NOTFILE (strict one-line reply)
  st=$(rssh "if [ -e $qd ] || [ -L $qd ]; then if [ -f $qd ] && [ ! -L $qd ]; then h=\$(sha256sum < $qd | cut -d' ' -f1) && echo \"EXISTS \$h\"; else echo NOTFILE; fi; else echo ABSENT; fi")
  rc=$?
  [ $rc -eq 0 ] || fail "ssh to $HOST failed (exit $rc) checking $rel"
  case "$st" in
    ABSENT) pre=absent ;;
    "EXISTS $h") pre=same ;;
    EXISTS\ *) is_hex64 "${st#EXISTS }" || fail "unreadable sha256 on Walter for $rel: '$st'"
               fail "Walter already has $dest with a DIFFERENT sha256 (${st#EXISTS }; Mac $h): never overwritten; decide by hand" ;;
    NOTFILE) fail "Walter's $dest exists but is not a regular file" ;;
    *) fail "unreadable reply from Walter for $rel: '$st'" ;;
  esac
  if [ "$DRY" = 1 ]; then
    say "DRY: would $([ $pre = same ] && echo 'keep the identical copy on Walter and remove' || echo 'send, verify and remove') $rel ($sz bytes, sha256 ${h:0:12}..)"
    continue
  fi
  if [ $pre = absent ]; then
    rssh "mkdir -p $(printf %q "$(dirname "$dest")")" || fail "cannot create the directory of $dest on Walter"
    rsync -t --ignore-existing -e "ssh $SSH_OPTS" -- "$f" "$HOST:$qd" < /dev/null; rc=$?
    [ $rc -eq 0 ] || fail "rsync of $rel exited $rc"
  fi
  wh=$(rssh "sha256sum < $qd | cut -d' ' -f1"); rc=$?
  [ $rc -eq 0 ] && is_hex64 "$wh" || fail "cannot read the sha256 of $dest on Walter (exit $rc, '$wh')"
  [ "$wh" = "$h" ] || fail "sha256 on Walter ($wh) differs from the Mac's ($h) for $rel"
  t=$(date '+%F %T')
  [ -f "$MANIFEST" ] || printf 'time\tmac path\twalter path\tsize\tsha256\n' > "$MANIFEST" || fail "cannot create $MANIFEST"
  printf '%s\t%s\t%s\t%s\t%s\n' "$t" "$f" "$HOST:~/$dest" "$sz" "$h" >> "$MANIFEST" || fail "cannot append to $MANIFEST"
  P=$u/OFFLOADED_TO_WALTER.txt
  if [ ! -f "$P" ]; then
    { echo "# STM checkpoints of this unit were moved to Walter (storage) by sky/offload_to_walter.sh, each after its"
      echo "# sha256 was computed on Walter and matched this Mac's. Walter path = $HOST:~/$WROOT/<path relative to the repo>."
      echo "# Restore one (from the repo root): rsync -t -e 'ssh -o ClearAllForwardings=yes' $HOST:'~/$WROOT/<rel>' '<rel>'"
      echo "# This unit no longer matches its pod manifest; sky/v2_verify.sh was run before offloading and is not re-run."
      printf '# time\tfile (relative to the repo)\tsize\tsha256\twalter path\n'; } > "$P" || fail "cannot write $P"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$t" "$rel" "$sz" "$h" "$HOST:~/$dest" >> "$P" || fail "cannot append to $P"
  rm -f -- "$f" && [ ! -e "$f" ] || fail "verified on Walter and recorded, but could not remove the Mac copy $rel"
  MOVED=$((MOVED + 1)); [ $pre = same ] && SAME=$((SAME + 1))
  say "moved $rel ($sz bytes, sha256 ${h:0:12}..$([ $pre = same ] && echo ', identical copy already on Walter'))"
done < "$TMPL.ok"
rm -f "$TMPL.ok"
[ "$DRY" = 1 ] && say "DRY: $NOK file(s) would be moved; $NREF refused" || say "done: $MOVED moved ($SAME already on Walter), $NREF refused; manifest $MANIFEST"
[ "$NREF" -gt 0 ] && exit 3
exit 0
