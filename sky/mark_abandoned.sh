#!/bin/bash
# Mark a pod of a v2 round as ABANDONED: torn down (or never started) without its files being pulled, its units re-run
# by a relaunch of the same actions file under another cluster name. sky/offload_to_walter.sh accepts
# <round>/abandoned/<cluster> in place of verified/ or failed_safe/, but only together with reaped/<cluster> and only
# when the marker has a non-empty "reason:" line. Without it, the abandoned pod's round (which lists the same unit
# dirs as the relaunch) blocks the offload of those units forever.
#
# Usage: bash sky/mark_abandoned.sh <round dir> <cluster> "<reason>"
#   <round dir>  e.g. runs_local/v2_round_main2a (relative to the repo, or absolute)
#   <reason>     one line, not blank: why the pod was abandoned and where its units were re-run
#
# Refuses (exit 1, nothing written) unless ALL hold:
#   1. <round>/actions/<cluster>.txt exists (the pod belongs to this round);
#   2. <round>/reaped/<cluster> is a regular file (the reaper or a person confirmed it gone);
#   3. `sky status`, read successfully (sky/v2_lib.sh sky_status_all: exit 0 and a recognised Clusters table), has no
#      row for <cluster>; an unreadable sky status refuses;
#   4. nothing was pulled from that pod: none of verified/, failed_safe/, failed_verify/, held/ or any *_pulled/ dir
#      of the round has an entry for it; there is no <round>/pods/<cluster> (the pod metadata and manifest.tsv the
#      reaper pulls before any unit; a pulled manifest means its units may be here); and the round's reap.log has no
#      line of the reaper pulling from it (JOB_COMPLETE pulling, JOB_FAILED pulling, failed job pulled, no sentinel
#      pulled);
#   5. <round>/abandoned/<cluster> does not exist yet (never overwritten), and the reason is one non-blank line.
# Writes <round>/abandoned/<cluster>: "reason: <reason>", the date, who ran it, and the result of each check.
# A partial pull archived by hand (moved out of pods/ and the result trees) is the person's call: record it in the reason.
# Environment: SKY_TIMEOUT (default 120 s); REAP_TEST_BIN (fakes, sky/test_v2_verify.sh only). Bash 3.2 compatible.
set -u
cd "$(dirname "$0")/.." || exit 2
REPO=$(pwd -P)
export PATH="${REAP_TEST_BIN:+$REAP_TEST_BIN:}$HOME/bin:$PATH"
# shellcheck source=sky/v2_lib.sh
. sky/v2_lib.sh
[ $# -eq 3 ] || { echo "usage: bash $0 <round dir> <cluster> \"<reason>\"" >&2; exit 2; }
R=$1; C=$2; WHY=$3
no() { echo "MARK ABANDONED REFUSED [$C]: $*" >&2; exit 1; }
case "$C" in ''|*/*|.*|*[[:space:]]*|*[]*?[]*) no "bad cluster name '$C'" ;; esac
[ -d "$R" ] || no "no round dir $R"
RD=$(cd "$R" && pwd -P) || no "cannot resolve $R"
case "$WHY" in *$'\n'*|*$'\r'*) no "the reason must be one line" ;; esac
[[ "$WHY" =~ [^[:space:]] ]] || no "the reason is empty"

[ -f "$RD/actions/$C.txt" ] || no "no $R/actions/$C.txt: not a pod of this round"
[ -f "$RD/reaped/$C" ] || no "no $R/reaped/$C: the pod is not marked reaped"
[ -e "$RD/abandoned/$C" ] && no "$R/abandoned/$C already exists (never overwritten)"

sky_status_all
sky_row_all "$C"
case "$SKY_STATE" in
  ABSENT) ;;
  ROW) no "the cluster is still in sky status: $SKY_LINE" ;;
  *) no "sky status UNREADABLE (failed, timed out or unrecognised output): cannot confirm the cluster is gone" ;;
esac

for d in verified failed_safe failed_verify held; do
  [ -e "$RD/$d/$C" ] && no "$R/$d/$C exists: the pod was pulled or held; not abandoned"
done
for d in "$RD"/*_pulled; do
  [ -d "$d" ] || continue
  [ -e "$d/$C" ] && no "${d#"$RD/"}/$C exists: files were pulled from this pod"
done
[ -e "$RD/pods/$C" ] && no "$R/pods/$C exists: the reaper pulled this pod's metadata (and manifest), so its units may have been pulled"
LOGCHK="no reap.log"
if [ -e "$RD/reap.log" ]; then
  [ -r "$RD/reap.log" ] || no "$R/reap.log is unreadable"
  hit=$(grep -F -e "] $C reports JOB_COMPLETE; pulling" -e "] $C JOB_FAILED: pulling" -e "] $C failed job pulled" \
          -e "] $C (no sentinel): what there is pulled" "$RD/reap.log"); rc=$?
  [ $rc -le 1 ] || no "cannot read $R/reap.log (grep exit $rc)"
  [ -z "$hit" ] || no "reap.log shows a pull from this pod: $(echo "$hit" | head -1)"
  LOGCHK="reap.log has no pull of it"
fi

mkdir -p "$RD/abandoned" || no "cannot create $R/abandoned"
T=$(date '+%F %T')
{ echo "reason: $WHY"
  echo "marked: $T by $(id -un 2>/dev/null) with sky/mark_abandoned.sh"
  echo "reaped: $(head -1 "$RD/reaped/$C")"
  echo "checks: actions/$C.txt present; reaped/$C present; absent from a readable sky status; no verified/, failed_safe/, failed_verify/, held/ or *_pulled/ entry; no pods/$C; $LOGCHK"
} > "$RD/abandoned/$C.tmp" && mv "$RD/abandoned/$C.tmp" "$RD/abandoned/$C" || { rm -f "$RD/abandoned/$C.tmp"; no "cannot write $R/abandoned/$C"; }
echo "marked $R/abandoned/$C ($T): $WHY"
