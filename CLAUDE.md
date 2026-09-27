# Episodic looking: code and experiments

Code for the "Episodic looking" paper. The paper, notes and plans live in `~/Dev/writeups/episodic_looking_full`
(its `CLAUDE.md` has the project background and the comparison protocol). Before running experiments, read
`Gideon Notes/HANDOFF.md`, `Gideon Notes/RESULTS_INDEX.md` and `Gideon Notes/running_experiments_kb.md` there.

## Experiment ledger and raw results (rule, 27 Sep 2026)

- **Log every experiment in `~/Dev/writeups/episodic_looking_full/Gideon Notes/EXPERIMENT_LEDGER.md`**: one row
  per launch (a sweep or round is one row), written when it is launched and completed when it ends. This includes
  pilots, sweeps, sanity and regression checks, diagnostics on existing checkpoints, and runs that fail or are
  aborted. Each row gives start and end date-time, machine, what and why, the command or launcher, the code commit
  (`+dirty` and what, if uncommitted changes were used), the results path, the status, and where it is written up.
- **Keep all raw results.** Write outputs under this repo (`runs_*/` or `runs_local/`), never to `/tmp` or a
  scratch directory. Never delete or overwrite a results directory: a superseded or broken run is moved to an
  archive directory and the ledger row says so. Copy results back from pods or Walter before the machine is
  released. These directories are gitignored, so the Mac's disk is the only copy: do not clean them up.
