#!/bin/bash
# Four-task comparison models (Dave 14 Nov 2026: comparison models in both experiments; reuse four1's RL-STM and
# LTM-alone runs, same base config and seeds). Task 1 in full, tasks 2-4 cut to 5,120 exposures; lp and dih; c 512 and
# 128; seeds 0 and 1. Models, all 4 sample-gradients per exposure as the LTM alone (64 x 16 per 256):
#   ladder1   --model ltm: the LTM trained on each live sample (1 step, batch 1)
#   ladder2   buffered, replay spread evenly, no sleep: 1 step of 4 buffer samples every exposure
#   er        ladder2 + the live sample: 1 step of live + 3 buffer samples every exposure
#   <head>_reset / <head>_keep  knn, fly, sdmlp heads on the sleep LTM (P 256 K 64), online after task 1 (as the
#             STM), 1 seen-replay sample per exposure; reset at each sleep or kept
# Every run line ends in a flag (xargs -L continuation).
# TUNED=1 (Dave 16 Nov 2026): the head tuning winners (report_tune, stage 2): knn defaults (kept: replay 0); fly lr 2;
# sdmlp repeats 64, lr .5.
tuned() { [ -z "$TUNED" ] && return; case $1_$2 in knn_keep) echo " --head-replay 0";; fly_*) echo " --fly-lr 2";;
  sdmlp_*) echo " --head-repeats 64 --sdmlp-lr 0.5";; esac; }
# Tuned: heads only, into comparisons/cmp4_tuned.
cd /home/incubator/dev/claude/episodic_looking
export PY=/home/incubator/dev/claude/.venv/bin/python D=${D:-runs_ss/mnist_clustered_single_stream/comparisons/cmp4${TUNED:+_tuned}}
mkdir -p $D
export B="--dataset mnist --val-holdout 600 --epochs 1 --max-exposures ${MAXEXP:-5120} ${FULLFIRST---full-first-task} --preproc runs_ss/preproc/omniglot_conv_c32_s0/preproc.pt --threads 1 --net transformer --blocks 1 --optimizer adamw --lr 0.0003 --replay-batch 16 --alpha-evict 3 --alpha-replay 1 --n-log-checkpoints 20 --grid-per-epoch 20 --feature-cache runs_ss/mnist_clustered_single_stream/feature_cache --replay-period 256 --replay-steps 64 --buffer priority --replay-capacity 500 --bias-mode what"
export SPREAD="--model buffered --replay-period 1 --replay-steps 1"
export HEAD="--model head --head-delay ${HEADDELAY:-53500} --head-replay 1"
run() { name=$1; shift; if $PY -m single_stream.run_tasks $B "$@" --out $D/$name > $D/$name.stdout 2>&1; then
  echo "$(date '+%F %T') done $name $(grep -o 'final.*' $D/$name/log.txt | tail -1)" >> $D/queue.log; else
  echo "$(date '+%F %T') FAILED $name" >> $D/queue.log; fi; }
export -f run
echo "$(date '+%F %T') cmp4 start" >> $D/queue.log
{
for s in ${SEEDS:-0 1}; do
  for c in ${CS:-512 128}; do
    for cfg in "lp id,lperm1,lperm2,lperm3" "dih id,rot90,hflip,rot180"; do
      read fam tasks <<< "$cfg"
      [ -n "$FAMS" ] && [[ " $FAMS " != *" $fam "* ]] && continue
      T="--tasks $tasks --cluster-size $c --seed $s"
      if [ -z "$TUNED" ]; then
        echo "run ladder1_${fam}_c${c}_s$s $T --model ltm"
        echo "run ladder2_${fam}_c${c}_s$s $T \$SPREAD --replay-batch 4"
        echo "run er_${fam}_c${c}_s$s $T \$SPREAD --replay-batch 3 --replay-live"
      fi
      for h in knn fly sdmlp; do
        echo "run ${h}_reset_${fam}_c${c}_s$s $T \$HEAD --head $h$(tuned $h reset)"
        echo "run ${h}_keep_${fam}_c${c}_s$s $T \$HEAD --head $h$(tuned $h keep) --head-no-reset"
      done
    done
  done
done
} | xargs -P ${PAR:-8} -L 1 bash -c 'eval "$@"' _
echo "$(date '+%F %T') cmp4 done" >> $D/queue.log
