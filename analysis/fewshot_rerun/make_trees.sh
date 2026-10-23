#!/bin/bash
# Symlink trees that combine the few-shot and continual final results for the tables and figures (15 Oct 2026). The
# results come from three places, kept where they landed (never merged in place, except the few-shot heads below):
#   runs_v2/                               Rodney: few-shot seeds 6-10 of the heads and LTM-only, the supervised STM
#                                          head (all seeds), LTM baseline pair A; plus seeds 1-5 copied from Walter
#                                          (cp -rn from runs_local/walter_fewshot_20261015, 0 overlapping units)
#   runs_v2_fs_rl/rl_lr0.1/                RL STM few-shot finals (pods) and their baselines (Rodney)
#   runs_v2_stmhead/                       supervised STM head, continual (Rodney)
#   runs_local/walter_continual_20261015/  copy of Walter's runs_v2/continual/{rl,linear,ncm,flymodel,sdmlp,ltm} and
#                                          runs_v2/baseline/{rl,ltm} (original RL STM, the draft's main row)
# Trees (relative symlinks, so they survive a move of the repo):
#   runs_local/fewshot_final_20261015/tree/fewshot/<model>
#   runs_local/continual_final_20261015/tree/{continual,baseline}/<model>
# Re-running replaces the links only. Usage (repo root): bash analysis/fewshot_rerun/make_trees.sh
set -eu
cd "$(dirname "$0")/../.."
F=runs_local/fewshot_final_20261015/tree/fewshot
mkdir -p $F
for m in linear ncm flymodel sdmlp ltm stm; do ln -sfn ../../../../runs_v2/fewshot/$m $F/$m; done
ln -sfn ../../../../runs_v2_fs_rl/rl_lr0.1/fewshot/rl $F/rl
C=runs_local/continual_final_20261015/tree
W=../../../../runs_local/walter_continual_20261015/runs_v2
mkdir -p $C/continual $C/baseline
ln -sfn ../../../../runs_v2_stmhead/continual/stm $C/continual/stm
for m in rl linear ncm flymodel sdmlp ltm; do ln -sfn $W/continual/$m $C/continual/$m; done
for m in rl ltm; do ln -sfn $W/baseline/$m $C/baseline/$m; done
for d in $F/* $C/continual/* $C/baseline/*; do [ -d "$d/" ] || { echo "broken link: $d" >&2; exit 1; }; done
echo "trees ok: $F, $C"
