#!/bin/bash
# Held-out digit comparison heads (Dave 14 Nov 2026): the snap9 harness (stm_frozen.py, snapshot, 75% nines, data seed
# 0, 5 seeds) with `head` in place of the RL STM, the same sleep as the best RL STM ("STM + sleep + reset": 1e-4, 100
# steps, priority 1). Phase-1 runs (w17_*) sleep at the 16th nine as w16_stm_sl16_1e-4_k100_reset; phase-2 runs (w17b_*)
# sleep at the 64th nine then 200 samples of 75% sevens as w16b_stm_middle_reset. Heads knn, fly, sdmlp, each reset at
# sleep or kept. The RL STM, LTM ladder and ER (= "LTM only, balanced": live + 15 seen replay) are reused from snap9.
# TUNED=1 (Dave 16 Nov 2026): the head tuning winners (report_tune, stage 2): knn defaults (kept: replay 0); fly lr 2;
# sdmlp repeats 64, lr .5.
tuned() { [ -z "$TUNED" ] && return; case $1_$2 in knn_keep) echo " --head-replay 0";; fly_*) echo " --fly-lr 2";;
  sdmlp_*) echo " --head-repeats 64 --sdmlp-lr 0.5";; esac; }
cd /home/incubator/dev/claude/episodic_looking
export PY=/home/incubator/dev/claude/.venv/bin/python D=${D:-runs_ss/stm_frozen_20261101/snap9} W=${TUNED:+w18}
W=${W:-w17}
export B="--preproc runs_ss/preproc/omniglot_conv_c32_s0/preproc.pt --new-digit 9 --new-shots 16 --data-seed 0 --episodes 200 --new-fraction 0.75 --new-order shuffle --phase1-nines 64 --first-checkpoint 2 --checkpoints 16 --nine-checkpoints 64 --eval-images 2000 --threads 1 --sleep-steps 100 --sleep-lr 1e-4 --sleep-batch 16 --sleep-priority 1 --sleep-floor 0.05 --head-replay 1"
run() { name=$1; shift; if $PY -m single_stream.stm_frozen head --out $D --name $name $B "$@" > $D/$name.stdout 2>&1; then
  echo "$(date '+%F %T') done $name" >> $D/queue_hd.log; else echo "$(date '+%F %T') FAILED $name" >> $D/queue_hd.log; fi; }
export -f run
echo "$(date '+%F %T') hd $W start" >> $D/queue_hd.log
{
for s in ${SEEDS:-0 1 2 3 4}; do
  for h in knn fly sdmlp; do
    for r in "reset --sleep-reset-stm" "keep --seed $s"; do  # keep repeats --seed so no line ends in a blank
      read t f <<< "$r"
      echo "run ${W}_head_${h}_${t}_s$s --seed $s --head $h$(tuned $h $t) --sleep-at-nines 16 $f"
      echo "run ${W}b_head_${h}_${t}_s$s --seed $s --head $h$(tuned $h $t) --sleep-at-nines 64 --phase2-digit 7 $f"
    done
  done
done
} | xargs -P ${PAR:-8} -L 1 bash -c 'eval "$@"' _
echo "$(date '+%F %T') hd $W done" >> $D/queue_hd.log
