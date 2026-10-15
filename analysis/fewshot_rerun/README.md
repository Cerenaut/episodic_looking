# Few-shot rerun and final-figure analysis (14-15 Oct 2026)

Version-controlled copies of the scripts behind the few-shot rerun and the v3 few-shot and continual figures. They
were first run from `runs_local/` (gitignored), where the originals stay beside their outputs as the record of what
ran; these copies differ only in their usage lines. Run everything from the repo root. Each run is logged in the
paper repo's `Notes/experiments/ledger.md` (14-15 Oct rows); the protocol is `plan.md` section 3, decision 10.

| Script | What it does | Writes to |
|---|---|---|
| `make_units.sh`, `launch.sh` | Rodney queue: supervised STM head budget selection; seeds 6-10 of the heads and LTM-only at N = 1, 4 | `runs_local/fewshot_rodney_20261014/` (units, log); results in `runs_v2_pilot_fewshot/`, `runs_v2/` |
| `make_stm_final_units.sh`, `launch_stm_final.sh` | Rodney queue: supervised STM head few-shot finals (budgets 1/8/42/6/1), chained after `launch.sh` | `runs_local/fewshot_rodney_20261014/`; results in `runs_v2/fewshot/stm` |
| `check.py` | Stop rule (Dave): trained-class accuracy, first / best / last per unit, flags N > 1 below 0.60 | stdout |
| `sel_curves.py`, `sel_fine12.py`, `sel_e24_vs_e48.py` | RL STM budget selection (validation): trained class, fine 1-2, 24- vs 48-epoch rounds | `runs_local/fsel_rl_curves_20261014/` |
| `rl_summary.py` | RL STM few-shot finals (test) against the draft's row | `runs_local/fsfinal_rl_20261015/` |
| `baselines.sh` | Starting points: each RL STM final seed's pre-trained STM, and the LTM, on the four sets | `runs_v2_fs_rl/rl_lr0.1/baseline/`, `runs_v2/baseline/ltm/` |
| `make_trees.sh` | Symlink trees combining Rodney, pod and Walter results (see its header for where each part comes from) | `runs_local/{fewshot,continual}_final_20261015/tree` |
| `table.py` | Few-shot table, every model (test) | `runs_local/fewshot_final_20261015/table_test.md` |

The RL STM pod rounds themselves are in `sky/rounds/fsel/` (selection) and `sky/rounds/fsfinal/` (finals).

The figures are drawn by the paper repo's scripts, from the trees above, with one shared style per model
(`scripts/model_styles.py`):

    python3 ~/dev/claude/episodic_looking_full/scripts/fewshot_figs_v3.py  --root .   # figures/v3/fewshot_panels_v3.png
    python3 ~/dev/claude/episodic_looking_full/scripts/continual_figs_v3.py --root . --runs runs_local/continual_final_20261015/tree
                                                                                      # figures/v3/cl_summary_v3.png, cl_per_set_{A,B,C,D}_v3.png
