#!/bin/bash
# Setup of a v2 pod (run by sky/v2_pod.yaml's setup, from ~/sky_workdir). Fails loudly (non-zero exit, a "SETUP FAILED"
# line) on anything it cannot verify, so the job never starts. SkyPilot 0.13 detaches setup under `sky launch -d`
# ("Setup detached"), so a failed setup shows as FAILED_SETUP in sky queue rather than as a failed sky launch; the
# reaper checks sky queue every pass for a pod without its pods dir and tears it down once it has proved nothing ran
# (sky/v2_reap.sh: setup_failed/).
#
# Phase "dataset": CIFAR-100 is fetched ON THE POD from the official URL instead of being synced from the Mac
# (the file-mount sync crawled at 4-90 KB/s to some regions and died: pod-path re-review of 2026-10-04, N1):
#   1. download with retries (curl, else wget), aborting a transfer that stays below 100 KB/s for a minute;
#   2. md5 of the tarball = CIFAR_MD5 (the value published on https://www.cs.toronto.edu/~kriz/cifar.html);
#   3. extract into a temporary dir, then sha256 of train, test and meta = CIFAR_SHA256_TRAIN/_TEST/_META, which
#      sky/v2_launch.sh computes from the Mac's copy (~/Dev/datasets/cifar-100-python) at launch time and passes in,
#      so the pod's files are byte-for-byte the Mac's; only then is the dir moved to $DATA_DIR.
#   An existing $DATA_DIR whose three files already match is kept (no download); one that does not match is moved
#   aside to $DATA_DIR.mismatch.<time> (pod-local input data, never results) and fetched again.
# Phase "env": .venv from /usr/bin/python3.12 (the image's python3 is 3.10: section 11); requirements installed with
#   uv (pip install uv, then uv pip install ... --index-strategy unsafe-best-match; pip once took 53 min); then the
#   installed torch must be exactly requirements.txt's pin (2.13.0+cu126) and, unless REQUIRE_CUDA=0, see a GPU.
# Timings of each step go to $SETUP_LOG (default ~/v2_setup.log); sky/v2_pod_job.sh copies it into the round's pods
# dir, so it is pulled with the results.
# Usage: bash sky/v2_pod_setup.sh [dataset|env|all]     (default all: the two phases run in parallel)
# Environment: CIFAR_SHA256_TRAIN, CIFAR_SHA256_TEST, CIFAR_SHA256_META (required for "dataset"); DATA_DIR (default
# ~/cifar-100-python, where ../cifar-100-python resolves from ~/sky_workdir: section 12); CIFAR_URL, CIFAR_MD5,
# FETCH_TRIES (default 4), RETRY_SLEEP (s, default 20), PYBIN (default /usr/bin/python3.12), UV_TIMEOUT (s per uv
# install attempt, default 1200; two attempts), REQUIRE_CUDA (default 1), SETUP_LOG.
# Written for bash 3.2 as well as 5, so the dataset phase can be tested on the Mac (sky/test_v2_verify.sh).
set -u
cd "$(dirname "$0")/.." || exit 1
PHASE=${1:-all}
DATA_DIR=${DATA_DIR:-$HOME/cifar-100-python}
CIFAR_URL=${CIFAR_URL:-https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz}
CIFAR_MD5=${CIFAR_MD5:-eb9058c3a382ffc7106e4002c42a8d85}
FETCH_TRIES=${FETCH_TRIES:-4}; RETRY_SLEEP=${RETRY_SLEEP:-20}
PYBIN=${PYBIN:-/usr/bin/python3.12}
UV_TIMEOUT=${UV_TIMEOUT:-1200}
SETUP_LOG=${SETUP_LOG:-$HOME/v2_setup.log}
T0=$(date +%s)
log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] +$(( $(date +%s) - T0 ))s $*" | tee -a "$SETUP_LOG"; }
die() { log "SETUP FAILED: $*"; exit 1; }
sha256() { if command -v sha256sum >/dev/null; then sha256sum < "$1" | cut -d' ' -f1; else shasum -a 256 < "$1" | cut -d' ' -f1; fi; }
md5() { if command -v md5sum >/dev/null; then md5sum < "$1" | cut -d' ' -f1; else command md5 -q "$1"; fi; }

expected() {  # $1 = train|test|meta -> the Mac's sha256
  case "$1" in train) echo "${CIFAR_SHA256_TRAIN:-}" ;; test) echo "${CIFAR_SHA256_TEST:-}" ;; meta) echo "${CIFAR_SHA256_META:-}" ;; esac
}
files_match() {  # $1 = dir: 0 iff train, test and meta all have the Mac's sha256; prints the first mismatch on stdout
  local f h
  for f in train test meta; do
    [ -f "$1/$f" ] || { echo "$f missing in $1"; return 1; }
    h=$(sha256 "$1/$f")
    [ "$h" = "$(expected $f)" ] || { echo "sha256 of $1/$f is ${h:-unreadable}, the Mac's copy is $(expected $f)"; return 1; }
  done
}

fetch_once() {  # $1 = output file
  if command -v curl >/dev/null; then
    curl -fsSL --connect-timeout 30 --speed-limit 102400 --speed-time 60 -o "$1" "$CIFAR_URL"
  elif command -v wget >/dev/null; then
    wget -q --timeout=60 --tries=1 -O "$1" "$CIFAR_URL"
  else
    echo "no curl or wget" >&2; return 1
  fi
}

dataset() {
  local f why tgz tmpd i m
  for f in train test meta; do
    [[ "$(expected $f)" =~ ^[0-9a-f]{64}$ ]] || die "CIFAR_SHA256_$(echo $f | tr a-z A-Z) not set to a sha256 (sky/v2_launch.sh passes the Mac's): '$(expected $f)'"
  done
  if [ -d "$DATA_DIR" ]; then
    if why=$(files_match "$DATA_DIR"); then log "dataset: $DATA_DIR already present, sha256 of train, test, meta = the Mac's; not fetched"; return 0; fi
    log "dataset: existing $DATA_DIR does not match ($why); moving it aside"
    mv "$DATA_DIR" "$DATA_DIR.mismatch.$(date +%s)" || die "cannot move $DATA_DIR aside"
  elif [ -e "$DATA_DIR" ]; then
    die "$DATA_DIR exists and is not a directory"
  fi
  tgz=$DATA_DIR.tar.gz.part
  i=1
  while :; do
    rm -f "$tgz"
    log "dataset: fetching $CIFAR_URL (attempt $i of $FETCH_TRIES)"
    if fetch_once "$tgz"; then
      m=$(md5 "$tgz")
      [ "$m" = "$CIFAR_MD5" ] && break
      log "dataset: md5 of the download is ${m:-unreadable}, expected $CIFAR_MD5"
    else
      log "dataset: download failed (exit $?)"
    fi
    [ "$i" -ge "$FETCH_TRIES" ] && { rm -f "$tgz"; die "could not fetch CIFAR-100 with md5 $CIFAR_MD5 in $FETCH_TRIES attempts"; }
    i=$((i + 1)); sleep "$RETRY_SLEEP"
  done
  log "dataset: downloaded $(wc -c < "$tgz" | tr -d ' ') bytes, md5 $CIFAR_MD5 OK"
  tmpd=$DATA_DIR.extract.$$
  rm -rf "$tmpd"; mkdir -p "$tmpd" || die "cannot create $tmpd"
  tar -xzf "$tgz" -C "$tmpd" || { rm -rf "$tmpd"; die "tar could not extract $tgz"; }
  if ! why=$(files_match "$tmpd/cifar-100-python"); then
    rm -rf "$tmpd"; rm -f "$tgz"
    die "extracted CIFAR-100 differs from the Mac's copy: $why"
  fi
  mv "$tmpd/cifar-100-python" "$DATA_DIR" || die "cannot move the extracted dataset to $DATA_DIR"
  rm -rf "$tmpd"; rm -f "$tgz"
  log "dataset: $DATA_DIR in place; sha256 train $CIFAR_SHA256_TRAIN test $CIFAR_SHA256_TEST meta $CIFAR_SHA256_META = the Mac's"
}

env_setup() {
  local want got r i
  want=$(sed -n 's/^torch==//p' requirements.txt)
  [ -n "$want" ] || die "no torch== pin in requirements.txt"
  test -x .venv/bin/python || "$PYBIN" -m venv .venv || die "cannot create .venv with $PYBIN"
  log "env: venv $(.venv/bin/python --version 2>&1)"
  .venv/bin/python -m pip install --quiet --disable-pip-version-check uv || die "pip install uv failed"
  log "env: $(.venv/bin/uv --version 2>&1)"
  for i in 1 2; do
    timeout "$UV_TIMEOUT" .venv/bin/uv pip install --python .venv/bin/python -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match; r=$?
    [ $r -eq 0 ] && break
    log "env: uv pip install attempt $i exited $r$([ $r = 124 ] && echo " (timeout ${UV_TIMEOUT}s)")"
    [ $i = 2 ] && die "uv pip install failed twice"
  done
  log "env: requirements installed"
  got=$(.venv/bin/python -c 'import torch; print(torch.__version__)' 2>&1 | tail -1)
  [ "$got" = "$want" ] || die "installed torch is '$got', requirements.txt pins '$want'"
  log "env: torch $got = the pinned version"
  if [ "${REQUIRE_CUDA:-1}" = 1 ]; then
    got=$(.venv/bin/python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO-CUDA")' 2>&1 | tail -1)
    [ "$got" != NO-CUDA ] && [ -n "$got" ] || die "torch sees no CUDA device ($got)"
    log "env: CUDA device $got"
  fi
}

log "setup starting ($PHASE) on $(hostname 2>/dev/null)"
case "$PHASE" in
  dataset) dataset ;;
  env) env_setup ;;
  all) ( dataset ) & dp=$!   # the download runs while uv installs; both must succeed
       env_setup
       wait $dp || die "dataset phase failed (see its lines above)" ;;
  *) die "unknown phase $PHASE" ;;
esac
log "SETUP OK ($PHASE) in $(( $(date +%s) - T0 )) s"
