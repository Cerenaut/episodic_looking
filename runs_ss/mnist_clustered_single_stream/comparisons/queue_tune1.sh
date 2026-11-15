#!/bin/bash
# Head tuning, stage 1 (Dave 15 Nov 2026: tune FlyModel and SDMLP, and kNN too for fairness; "launch when ready").
# Rules as the RL-STM search: exposure counts fixed; repeats (--head-repeats) and replays (--head-replay) free; lr and
# other hparams open. Setup as hp1: task 1 in full (heads online at 53,500), task 2 to 2,048; lp c 512 and dih c 512;
# seed 0; reset and kept arms; same LTM, sleep (P 256, K 64) and val-holdout as cmp4. One factor at a time around the
# ported defaults (fly: kc 2000, response 16, k-frac .05, lr .2; sdmlp: 1000 neurons, k 32, gaba 100, lr .05; knn:
# k 1, capacity 4096; replay 1, repeats 1). Not varied: fly repeats (W += lr KC with W <= 1, so repeats ~ lr) and knn
# repeats (duplicates). Score: single_stream.score on task 2.
# STAGE=1b (15 Nov, after stage 1): beyond stage 1's best ends (fly lr; sdmlp repeats) and their combinations with
# the other helpful settings; no knn (defaults best); output comparisons/tune1b. Every run line ends in a flag (xargs -L continuation).
cd /home/incubator/dev/claude/episodic_looking
export PY=/home/incubator/dev/claude/.venv/bin/python STAGE=${STAGE:-1} D=${D:-runs_ss/mnist_clustered_single_stream/comparisons/tune$STAGE}
mkdir -p $D
export B="--dataset mnist --val-holdout 600 --epochs 1 --max-exposures ${MAXEXP:-2048} ${FULLFIRST---full-first-task} --preproc runs_ss/preproc/omniglot_conv_c32_s0/preproc.pt --threads 1 --net transformer --blocks 1 --optimizer adamw --lr 0.0003 --replay-batch 16 --alpha-evict 3 --alpha-replay 1 --n-log-checkpoints 20 --grid-per-epoch 20 --feature-cache runs_ss/mnist_clustered_single_stream/feature_cache --replay-period 256 --replay-steps 64 --buffer priority --replay-capacity 500 --bias-mode what --model head --head-delay ${HEADDELAY:-53500}"
run() { name=$1; shift; if $PY -m single_stream.run_tasks $B "$@" --out $D/$name > $D/$name.stdout 2>&1; then
  echo "$(date '+%F %T') done $name $(grep -o 'final.*' $D/$name/log.txt | tail -1)" >> $D/queue.log; else
  echo "$(date '+%F %T') FAILED $name" >> $D/queue.log; fi; }
export -f run
# (name, flags); every setting runs reset and kept
settings() {
  echo "knn_default --head knn"
  for k in 3 10; do echo "knn_k$k --head knn --knn-k $k"; done
  for r in 0 4 16; do echo "knn_replay$r --head knn --head-replay $r"; done
  echo "fly_default --head fly"
  for v in 0.05 0.5 1.0; do echo "fly_lr$v --head fly --fly-lr $v"; done
  for v in 500 8000; do echo "fly_kc$v --head fly --fly-kc $v"; done
  for v in 4 32; do echo "fly_resp$v --head fly --fly-response $v"; done
  for v in 0.01 0.2; do echo "fly_kfrac$v --head fly --fly-k-frac $v"; done
  for r in 4 16; do echo "fly_replay$r --head fly --head-replay $r"; done
  echo "sdmlp_default --head sdmlp"
  for v in 0.01 0.2 0.5; do echo "sdmlp_lr$v --head sdmlp --sdmlp-lr $v"; done
  for v in 250 4000; do echo "sdmlp_n$v --head sdmlp --sdmlp-neurons $v"; done
  for v in 8 128; do echo "sdmlp_k$v --head sdmlp --sdmlp-k $v"; done
  for v in 10 1000; do echo "sdmlp_gaba$v --head sdmlp --sdmlp-gaba $v"; done
  for r in 4 16; do echo "sdmlp_replay$r --head sdmlp --head-replay $r"; done
  for v in 4 16; do echo "sdmlp_rep$v --head sdmlp --head-repeats $v"; done
}
settings_1b() {
  for v in 2 4; do echo "fly_lr$v --head fly --fly-lr $v"; done
  echo "fly_lr1_kfrac0.01 --head fly --fly-lr 1 --fly-k-frac 0.01"
  echo "fly_lr1_replay16 --head fly --fly-lr 1 --head-replay 16"
  echo "fly_lr1_kc8000 --head fly --fly-lr 1 --fly-kc 8000"
  echo "fly_lr1_kfrac0.01_replay16 --head fly --fly-lr 1 --fly-k-frac 0.01 --head-replay 16"
  echo "sdmlp_rep64 --head sdmlp --head-repeats 64"
  for v in 0.2 0.5; do echo "sdmlp_rep16_lr$v --head sdmlp --head-repeats 16 --sdmlp-lr $v"; done
  echo "sdmlp_rep64_lr0.5 --head sdmlp --head-repeats 64 --sdmlp-lr 0.5"
  echo "sdmlp_rep16_gaba1000 --head sdmlp --head-repeats 16 --sdmlp-gaba 1000"
  echo "sdmlp_rep16_replay16 --head sdmlp --head-repeats 16 --head-replay 16"
}
echo "$(date '+%F %T') tune$STAGE start" >> $D/queue.log
{
for s in ${SEEDS:-0}; do
  for cfg in "lp 512 id,lperm1" "dih 512 id,rot90"; do
    read fam c tasks <<< "$cfg"
    [ -n "$FAMS" ] && [[ " $FAMS " != *" $fam "* ]] && continue
    T="--tasks $tasks --cluster-size $c --seed $s"
    [ $STAGE = 1 ] && S=settings || S=settings_$STAGE
    $S | while read name flags; do
      [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && continue
      echo "run ${name}_reset_${fam}_c${c}_s$s $T $flags"
      echo "run ${name}_keep_${fam}_c${c}_s$s $T $flags --head-no-reset"
    done
    # knn kept: capacity bounds what a never-reset head holds
    [ $STAGE = 1 ] && for v in 1024 16384; do
      [ -n "$ONLY" ] && [[ " $ONLY " != *" knn_cap$v "* ]] && continue
      echo "run knn_cap${v}_keep_${fam}_c${c}_s$s $T --head knn --knn-capacity $v --head-no-reset"
    done
  done
done
} | xargs -P ${PAR:-8} -L 1 bash -c 'eval "$@"' _
echo "$(date '+%F %T') tune$STAGE done" >> $D/queue.log
