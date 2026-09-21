#!/usr/bin/env python3
"""
Status of the running STM experiments on the M3 and on Walter, as one markdown table with ETAs.

Walter's files are mirrored first with rsync into walter_mirror/ (progress logs + results files only), then
both trees are read locally. Rates are measured from each run's own results-file timestamps; queued stages
are chained after the running one with nominal per-epoch times (M3_MIN, WALTER_MIN).

    python status_experiments.py            # mirror Walter, print table
    python status_experiments.py --no-sync  # local only (Walter from the last mirror)
"""
import argparse
import datetime as dt
import glob
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
MIRROR = os.path.join(ROOT, "walter_mirror")
SSH = ["-e", "ssh -o ClearAllForwardings=yes -o ConnectTimeout=20"]
M3_MIN, WALTER_MIN = 13.0, 4.5  # nominal minutes per pre-training epoch for not-yet-started stages (M3: 3 parallel processes; Walter: 3 parallel orders); x CONT_FACTOR for continual
PRETRAIN_EPOCHS, CONT_EPOCHS = 12, 36
CONT_FACTOR = 1.5  # a continual epoch evaluates 4 test sets (800 steps each): 7200 vs 4800 agent steps
ORDERS = ["3, 4, 5", "4, 5, 3", "5, 3, 4"]

# name, machine, run root (relative), pretrain epochs, list of (stage label, results glob, expected epochs)
JOBS = [
    ("A.1 differentiable actor, repeat 1 (lr 0.1; orders stopped: diverged)", "M3", "runs_a1_diff", 12),
    ("A.1 e40 screen (pre-training only, lr 0.1)", "M3", "runs_a1_diff_e40", 12),
    ("A.1 differentiable actor lr 0.01, repeat 1", "M3", "runs_a1_diff_lr0.01", 12),
    ("A.1 lr 0.01 e40 screen (pre-training only)", "M3", "runs_a1_diff_lr0.01_e40", 12),
    ("sweep e9 (fixed RL)", "M3", "runs_p0_sweep/e9", 12),
    ("sweep e20 (fixed RL)", "M3", "runs_p0_sweep/e20", 12),
    ("sweep e30 (fixed RL)", "M3", "runs_p0_sweep/e30", 12),
    ("sweep e5 (fixed RL)", "Walter", "runs_p0_sweep/e5", 12),
    ("fixed RL reference, repeat 1", "Walter", "runs_ref_fixed", 12),
    ("A.1 differentiable actor lr 0.01, repeat 2", "Walter", "runs_a1_diff_s2", 12),
    ("fixed RL reference, repeat 2 (overnight)", "M3", "runs_ref_fixed_s2", 12),
    ("40-epoch STM pre-training + 3 orders (all on the M3 from 14:05; Walter freed)", "M3", "runs_p0_pt40", 40),
    ("streaming (batch 1) fixed RL at lr 0.1: classes 3 and 5 diverged (NaN actor)", "M3", "runs_stream_ref_fixed", 0),
    ("streaming (batch 1) fixed RL at lr 0.01 (rerun of 3, 4, 5)", "M3", "runs_stream_ref_fixed_lr0.01", 0),
    ("streaming (batch 1) A.1 lr 0.01, classes 3/4/5", "M3", "runs_stream_a1_diff_lr0.01", 0),
    ("seed round (run_m3_seeds.sh): CANCELLED by Gideon 2026-09-21 12:58 two minutes in, nothing produced", "M3", "runs_seed/ref_s1", 12),
]
# The remaining seed jobs (runs_seed/<ref|a1>_s<k>) are not listed; run_m3_seeds.sh is resumable if the round is revived.
SEED_JOBS = ["runs_seed/ref_s1"]
CANCELLED = {"runs_seed/ref_s1"}  # rows shown as cancelled, never queued/running
DIVERGED = {("runs_stream_ref_fixed", "class 3"), ("runs_stream_ref_fixed", "class 5")}  # stopped; shown as diverged, not running/queued
# Jobs with their own stage list: (label, results glob, expected epochs). run_m3_streaming.sh: 6 batch-1 runs in parallel,
# 12 epochs of 8,000 steps + 4 x 1,600 evaluation steps each.
STREAM_JOBS = ("runs_stream_ref_fixed", "runs_stream_ref_fixed_lr0.01", "runs_stream_a1_diff_lr0.01")
CUSTOM_STAGES = {sub: [(f"class {c}", f"few-shot_[[]{c}[]]_500", 12) for c in (3, 4, 5)] for sub in STREAM_JOBS}
NOMINAL_MIN = {sub: 25.0 for sub in STREAM_JOBS}  # min/epoch per run with 6 in parallel
NOMINAL_MIN.update({sub: 10.5 for sub in SEED_JOBS})  # seed round: 3 orders in parallel, M3 to itself (pt40 measured 15.8 min/epoch continual)
PRETRAIN_MIN = {sub: 6.0 for sub in SEED_JOBS}  # seeded pre-training runs alone (pt40 measured 5.8 min/epoch)
AFTER_JOB = {sub: "runs_p0_pt40" for sub in STREAM_JOBS}  # starts when that job's script exits
# Hooks for splitting a job's orders across machines (used 2026-09-20 10:09-14:05, then reverted: Walter freed on Gideon's instruction).
ORDER_SUBSET = {}
SKIP_PRETRAIN = set()
PRETRAIN_ONLY = {"runs_a1_diff", "runs_a1_diff_e40", "runs_a1_diff_lr0.01_e40", "runs_p0_sweep/e9", "runs_p0_sweep/e20", "runs_p0_sweep/e30", "runs_p0_sweep/e5"}
WALTER_SEQUENTIAL = ["runs_ref_fixed", "runs_a1_diff_s2"]  # queue v2 order
M3_AFTER_A1 = ["runs_p0_pt40"]  # run_m3_night.sh: starts when runs_a1_diff orders finish
M3_SEQUENTIAL = ["runs_p0_sweep/e9", "runs_p0_sweep/e20", "runs_p0_sweep/e30"] + SEED_JOBS  # run_m3_sweep.sh order, then run_m3_seeds.sh order


def sync_walter():
    os.makedirs(MIRROR, exist_ok=True)
    cmd = ["rsync", "-az", "--prune-empty-dirs", "--include=*/", "--include=results_*.txt", "--include=progress.log",
           "--exclude=*"] + SSH + ["g_walter:~/Dev/episodic_looking/runs_*", "g_walter:~/Dev/episodic_looking/runs_local/queue",
                                    MIRROR + "/"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.returncode == 0, (r.stderr or "").strip().splitlines()[-1:]


def read_results(path):
    evals = []
    with open(path) as f:
        for line in f:
            m = re.match(r"\[.*?\], \[?'?([\d, ]+)'?\]?, (training|evaluate), (\d+), ([\d.]+)", line.strip())
            if m and m.group(2) == "evaluate":
                evals.append((m.group(1).replace(" ", ""), int(m.group(3)), float(m.group(4))))
    return evals


def stage_status(base, sub, expected_epochs, glob_name):
    """Return (epochs_done, latest per-testset acc dict, start mtime, last mtime, path) for the newest results file."""
    files = sorted(glob.glob(os.path.join(base, sub, "cifar_100", glob_name, "*", "*", "*", "*", "results_*.txt")))
    if not files:
        return None
    path = files[-1]
    evals = read_results(path)
    if not evals:
        return None
    epochs = max(e for _, e, _ in evals) + 1
    latest = {}
    for k, e, a in evals:
        if e == epochs - 1:
            latest[k] = a
    # start: the run directory timestamp HH-MM-SS from the path
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/(\d{2})-(\d{2})-(\d{2})/", path)
    start = dt.datetime(*map(int, m.groups())) if m else None
    last = dt.datetime.fromtimestamp(os.path.getmtime(path))
    return epochs, latest, start, last, path


def fmt_acc(latest):
    order = ["12", "1,2", "3", "4", "5"]
    parts = [f"{latest[k]:.3f}" for k in order if k in latest]
    return " / ".join(parts) if parts else "-"


def curfew(t, machine, minutes):
    """Walter is idle 21:00-07:00: a stage of `minutes` that would end after 21:00 starts at 07:00 instead."""
    if machine != "Walter" or t is None:
        return t
    start = t - dt.timedelta(minutes=minutes)
    end_limit = start.replace(hour=21, minute=0, second=0, microsecond=0)
    if start.hour < 7:
        start = start.replace(hour=7, minute=0, second=0, microsecond=0)
    elif t > end_limit:
        start = (start + dt.timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
    else:
        return t
    return start + dt.timedelta(minutes=minutes)


def fmt_eta(t):
    if t is None:
        return "-"
    now = dt.datetime.now()
    day = "" if t.date() == now.date() else t.strftime(" %a")
    return t.strftime("%H:%M") + day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    args = ap.parse_args()
    now = dt.datetime.now()
    notes = []
    if not args.no_sync:
        ok, err = sync_walter()
        if not ok:
            notes.append("Walter mirror FAILED: " + " ".join(err))
    rows = []
    machine_end = {"M3": None, "Walter": None}
    walter_chain = None  # ETA of the previous sequential Walter stage
    m3_chain = None
    a1_end = None
    m3_free = None
    pt40_pre_end = None  # the Walter pt40 orders start after the M3 pre-training AND A.1 repeat 2
    job_ends = {}  # run root -> ETA/finish of the whole job (for AFTER_JOB chaining)
    for name, machine, sub, pre_epochs in JOBS:
        base = ROOT if machine == "M3" else MIRROR
        nominal = NOMINAL_MIN.get(sub, M3_MIN if machine == "M3" else WALTER_MIN)
        if sub in CUSTOM_STAGES:
            stages = list(CUSTOM_STAGES[sub])
        else:
            stages = [] if (machine, sub) in SKIP_PRETRAIN else [("pre-training", "pretrain_[[]1, 2[]]_500", pre_epochs)]
            if sub not in PRETRAIN_ONLY:
                stages += [(f"order {o}", f"continual_[[]{o}[]]_500", CONT_EPOCHS) for o in ORDER_SUBSET.get((machine, sub), ORDERS)]
        chain = walter_chain if machine == "Walter" and sub in WALTER_SEQUENTIAL else (m3_chain if sub in M3_SEQUENTIAL else None)
        if sub in M3_AFTER_A1:
            chain = m3_free  # run_m3_pt40_split.sh starts when the A.1 lr0.01 and ref repeat 2 pipelines finish
        if (machine, sub) in SKIP_PRETRAIN and pt40_pre_end and (chain is None or pt40_pre_end > chain):
            chain = pt40_pre_end
        if sub in AFTER_JOB:
            chain = job_ends.get(AFTER_JOB[sub])
        parallel_all = sub in CUSTOM_STAGES  # all custom stages run concurrently
        parallel_orders = True  # both machines now run a variant's 3 orders concurrently
        order_etas = []
        for label, g, expected in stages:
            st = stage_status(base, sub, expected, g)
            if sub in CANCELLED:
                rows.append((machine, name, label, f"{(st[0] if st else 0)}/{expected}", "-", "-", "-", "cancelled"))
                continue
            if st is None and (sub, label) in DIVERGED:
                rows.append((machine, name, label, f"0/{expected}", "-", "-", "-", "diverged"))
                continue
            if st is None:
                # not started: chain after previous stage
                per_epoch = PRETRAIN_MIN.get(sub, nominal) if label == "pre-training" else nominal
                stage_min = per_epoch * expected * (CONT_FACTOR if label.startswith("order") else 1.0)
                dur = dt.timedelta(minutes=stage_min)
                if (parallel_orders and label.startswith("order") or parallel_all) and order_etas:
                    eta = order_etas[0]  # same as the sibling orders
                else:
                    eta = curfew((max(chain, now) if chain else now) + dur, machine, stage_min)  # a chain end in the past means: starts now
                    if label.startswith("order") or parallel_all:
                        order_etas.append(eta)
                    chain = eta
                rows.append((machine, name, label, f"0/{expected}", "-", "-", fmt_eta(eta), "queued"))
                if label == "pre-training" and (machine, sub) == ("M3", "runs_p0_pt40"):
                    pt40_pre_end = eta
                continue
            epochs, latest, start, last, path = st
            if (sub, label) in DIVERGED:
                rows.append((machine, name, label, f"{epochs}/{expected}", fmt_acc(latest), "-", last.strftime("%H:%M"), "diverged"))
                continue
            rate = None
            if start and epochs > 0:
                rate = (last - start).total_seconds() / 60.0 / epochs
            if epochs >= expected:
                rows.append((machine, name, label, f"{epochs}/{expected}", fmt_acc(latest), f"{rate:.1f}" if rate else "-",
                             last.strftime("%H:%M"), "done"))
                chain = max(chain, last) if chain else last
                if label.startswith("order"):
                    order_etas.append(last)
                if label == "pre-training" and (machine, sub) == ("M3", "runs_p0_pt40"):
                    pt40_pre_end = last
                continue
            r = rate or nominal * (CONT_FACTOR if label.startswith("order") else 1.0)
            eta = now + dt.timedelta(minutes=r * (expected - epochs))
            rows.append((machine, name, label, f"{epochs}/{expected}", fmt_acc(latest), f"{r:.1f}", fmt_eta(eta), "running"))
            if label.startswith("order"):
                order_etas.append(eta)
            if label == "pre-training" and (machine, sub) == ("M3", "runs_p0_pt40"):
                pt40_pre_end = eta
            chain = max(chain, eta) if chain else eta
        job_end = chain
        job_ends[sub] = job_end
        if machine == "Walter" and sub in WALTER_SEQUENTIAL:
            walter_chain = chain
        if sub in M3_SEQUENTIAL:
            m3_chain = chain
        if sub in ("runs_a1_diff_lr0.01", "runs_ref_fixed_s2") and chain and (m3_free is None or chain > m3_free):
            m3_free = chain
        if job_end and (machine_end[machine] is None or job_end > machine_end[machine]):
            machine_end[machine] = job_end
    print(f"### Experiment status {now.strftime('%a %H:%M')}")
    print()
    print(f"Estimated all-done: M3 {fmt_eta(machine_end['M3'])}, Walter {fmt_eta(machine_end['Walter'])}")
    for n in notes:
        print(n)
    print()
    print("| machine | job | stage | epochs | latest acc (1,2 / 3 / 4 / 5) | min/epoch | ETA / finished | state |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
