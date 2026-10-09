#!/bin/bash
# Sub-$1 smoke test, run once on one pod before committing to the round. Answers four
# questions, each of which would otherwise be discovered expensively mid-round:
#   1. Is /dev/shm big enough? The dataset uses multiprocessing.shared_memory (~80 MB per
#      process) and Docker's default is 64 MB.
#   2. How many processes per pod? The M3/Walter profiling says the script is batch-16
#      bound and parallel processes buy nothing on a saturated GPU. That was measured on a
#      GTX 1060; a newer card may not saturate, in which case parallelism is most of the
#      round's speed. Measured here, not assumed.
#   3. Does --seed reproduce on CUDA? It was only ever verified on MPS. A seeded round
#      whose seeds do not reproduce is worse than an unseeded one.
#   4. Does the environment build and does the LTM checkpoint load?
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CK=../cifar_100_pretrain
E13=$CK/cifar_100_subclasses_12_e13.pth
OUT=runs_local/smoke
STEPS=${STEPS:-400}
mkdir -p "$OUT" "$CK/variants"
say() { echo "[smoke] $*"; }

say "=== 1. shared memory ==="
df -h /dev/shm | tail -1
SHM_KB=$(df -k /dev/shm | tail -1 | awk '{print $2}')
if [ "$SHM_KB" -lt 1000000 ]; then say "WARNING: /dev/shm is under 1 GB ($SHM_KB kB). Multi-process runs may fail."; fi

say "=== 2. device ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

one_run() {  # $1 = run-root suffix, $2 = seed
  $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --coarse-classes 0 1 \
      --epochs 1 --training-steps "$STEPS" --evaluate-steps 8 --seed "$2" --eval-bias mean \
      --ltm-checkpoint $E13 --stm-checkpoint "$CK/variants/smoke_$1.pth" \
      --run-root "$OUT/r_$1" > "$OUT/$1.log" 2>&1
}

say "=== 3. throughput at 1, 2, 4 processes ($STEPS steps each) ==="
printf "%-6s %-10s %-14s %-10s\n" "procs" "seconds" "agg steps/s" "per-proc"
for n in 1 2 4; do
  rm -rf "$OUT"/r_p* ; t0=$(date +%s)
  for i in $(seq 1 "$n"); do one_run "p${n}_$i" 1 & done
  if ! wait; then say "FATAL: a training process failed at n=$n; see $OUT/p${n}_*.log"; tail -15 "$OUT/p${n}_1.log"; exit 1; fi
  t1=$(date +%s); el=$((t1-t0)); [ "$el" -eq 0 ] && el=1
  agg=$(awk -v n="$n" -v s="$STEPS" -v e="$el" 'BEGIN{printf "%.1f", n*s/e}')
  per=$(awk -v s="$STEPS" -v e="$el" 'BEGIN{printf "%.1f", s/e}')
  printf "%-6s %-10s %-14s %-10s\n" "$n" "$el" "$agg" "$per"
  echo "$n $el $agg $per" >> "$OUT/throughput.txt"
done

say "=== 4. does --seed reproduce on CUDA? ==="
one_run seedA 7 || { say "FATAL: seedA run failed"; tail -15 "$OUT/seedA.log"; exit 1; }
one_run seedB 7 || { say "FATAL: seedB run failed"; exit 1; }
one_run seedC 8 || { say "FATAL: seedC run failed"; exit 1; }
A=$(find "$OUT/r_seedA" -name 'results_*.txt' | head -1)
B=$(find "$OUT/r_seedB" -name 'results_*.txt' | head -1)
C=$(find "$OUT/r_seedC" -name 'results_*.txt' | head -1)
echo "--- same seed (7 vs 7) ---"; diff "$A" "$B" && say "IDENTICAL: seeding reproduces on CUDA" || say "DIFFER: seeding does NOT reproduce on CUDA"
echo "--- different seed (7 vs 8) ---"; diff "$A" "$C" > /dev/null && say "WARNING: different seeds gave identical results" || say "OK: different seeds differ"
say "=== smoke complete ==="
