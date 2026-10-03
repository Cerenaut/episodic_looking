# Shared helpers of the v2 cloud path (sourced by sky/v2_launch.sh, sky/v2_reap.sh, sky/v2_status.sh; bash 3.2+).
# `source` reads the whole file before running it, so editing this file never disturbs a running script.
#
# Rule (running_experiments_kb.md sections 1 and 5b): an unreadable state is never read as absent or zero.

strip() { perl -pe 's/\e\[[0-9;]*[mK]//g'; }

# tmo N cmd...: run cmd with an N-second cap (macOS has no timeout(1)); exit 142 (SIGALRM) when it fires.
tmo() { local n=$1; shift; perl -e 'alarm shift; exec @ARGV or exit 127' "$n" "$@"; }

SKY_TIMEOUT=${SKY_TIMEOUT:-120}

# sky_row <cluster>: sets SKY_STATE to ROW, ABSENT or UNREADABLE and SKY_LINE to the cluster's row (ROW only).
#   ROW         sky status exited 0, printed the table header (NAME ... STATUS) and a row whose first field is <cluster>
#   ABSENT      sky status exited 0 and said so in words ("Cluster '<c>' not found." or "No existing clusters."),
#               with no row for it: the only state that may be read as "the cluster does not exist"
#   UNREADABLE  anything else: non-zero exit, timeout, no header, unrecognised output
# Not a $(...) function, so that it can set two variables.
sky_row() {
  local c=$1 out rc
  SKY_LINE=""; SKY_STATE=UNREADABLE
  out=$(tmo "$SKY_TIMEOUT" sky status "$c" 2>&1 < /dev/null); rc=$?
  [ $rc -eq 0 ] || return 0
  out=$(printf '%s\n' "$out" | strip)
  SKY_LINE=$(printf '%s\n' "$out" | awk -v c="$c" '$1==c' | head -1)
  if [ -n "$SKY_LINE" ] && printf '%s\n' "$out" | grep -qE '^NAME[[:space:]].*STATUS'; then SKY_STATE=ROW; return 0; fi
  SKY_LINE=""
  if printf '%s\n' "$out" | grep -qF "Cluster '$c' not found." || printf '%s\n' "$out" | grep -qx 'No existing clusters\.'; then
    SKY_STATE=ABSENT
  fi
  return 0
}
# sky_status_all: ONE `sky status` call for every cluster (sky/v2_reap.sh calls it once per pass, not once per cluster:
# with 20 pods and a slow API server per-cluster calls made a pass take tens of minutes). Sets SKY_ALL_OK=1 and
# SKY_ROWS (the rows of the Clusters table, possibly empty) only when sky exited 0 and printed the Clusters section
# in a recognised form: a line "Clusters" followed by the table header (NAME ... STATUS) or "No existing clusters.".
# Anything else (non-zero exit, timeout, no Clusters section, another header) leaves SKY_ALL_OK=0: every cluster
# then reads UNREADABLE from sky_row_all. Rows are taken from the Clusters table only (it ends at the first blank
# line), so a managed job or service with the same name can never stand in for a cluster.
sky_status_all() {
  local out rc
  SKY_ALL_OK=0; SKY_ROWS=""
  out=$(tmo "$SKY_TIMEOUT" sky status 2>&1 < /dev/null); rc=$?
  [ $rc -eq 0 ] || return 0
  out=$(printf '%s\n' "$out" | strip)
  # the Clusters section: its first non-empty line, then its rows up to the first blank line
  local first
  first=$(printf '%s\n' "$out" | awk 'p && NF { print; exit } /^Clusters[[:space:]]*$/ { p = 1 }')
  if printf '%s\n' "$first" | grep -qx 'No existing clusters\.'; then SKY_ALL_OK=1; return 0; fi
  printf '%s\n' "$first" | grep -qE '^NAME[[:space:]].*STATUS' || return 0
  SKY_ROWS=$(printf '%s\n' "$out" | awk 'h && !NF { exit } h { print } p && NF && !h { h = 1 } /^Clusters[[:space:]]*$/ { p = 1 }')
  SKY_ALL_OK=1
}
# sky_row_all <cluster>: SKY_STATE/SKY_LINE as sky_row sets them, from the last sky_status_all:
#   ROW (a row whose first field is <cluster>), ABSENT (the Clusters table was read and has no such row),
#   UNREADABLE (the last sky_status_all could not be read).
sky_row_all() {
  SKY_LINE=""; SKY_STATE=UNREADABLE
  [ "${SKY_ALL_OK:-0}" = 1 ] || return 0
  SKY_LINE=$(printf '%s\n' "$SKY_ROWS" | awk -v c="$1" '$1==c' | head -1)
  if [ -n "$SKY_LINE" ]; then SKY_STATE=ROW; else SKY_STATE=ABSENT; fi
}
has_autostop() { printf '%s\n' "$1" | grep -qE '[0-9]+[hm] \(down\)'; }
row_status() { printf '%s\n' "$1" | grep -owE 'UP|INIT|STOPPED' | head -1; }

# reaper_pid <round>: prints the pid of the reaper holding <round>/reaper.lock if it is alive and is a v2_reap.sh;
# prints nothing if there is no lock or its pid is dead; prints "?" if the lock exists but its pid is unreadable
# (a reaper between mkdir and writing its pid, or a crashed one: check by hand).
reaper_pid() {
  local L=$1/reaper.lock p
  [ -d "$L" ] || return 0
  p=$(cat "$L/pid" 2>/dev/null)
  [[ "$p" =~ ^[0-9]+$ ]] || { echo "?"; return 0; }
  if kill -0 "$p" 2>/dev/null && ps -p "$p" -o command= 2>/dev/null | grep -q 'v2_reap\.sh'; then echo "$p"; fi
}

# mtime <file>: seconds since the epoch (BSD stat, then GNU stat); empty if unreadable.
mtime() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null; }

# Words of an actions line's environment field (shell words) -> the value of KEY (last assignment wins, as with env);
# empty if unset.
env_value() {  # $1 = environment field, $2 = KEY
  local -a w; local x v=""
  eval "w=($1)" 2>/dev/null || return 1
  for x in ${w[@]+"${w[@]}"}; do case "$x" in "$2="*) v=${x#"$2="} ;; esac; done
  printf '%s' "$v"
}

# sha256 <file>: hex digest, read on stdin (no filename escaping). sha256sum (Linux, macOS 15+) or shasum.
sha256() {
  if command -v sha256sum >/dev/null; then sha256sum < "$1" | cut -d' ' -f1; else shasum -a 256 < "$1" | cut -d' ' -f1; fi
}

# Markers of clusters that failed or were held in a round (sky/v2_reap.sh writes held/, job_failed/, setup_failed/;
# sky/v2_launch.sh writes launch_failed/). failed_or_held <round>: prints the distinct cluster names, one per line.
failed_or_held() {
  local d f
  for d in held job_failed launch_failed setup_failed; do
    for f in "$1/$d"/*; do [ -e "$f" ] && basename "$f"; done
  done | LC_ALL=C sort -u
}
