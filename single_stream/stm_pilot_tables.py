"""Tables for the STM pilot report (8 Nov 2026): each STM run of batches 1-4 against the LTM alone with the same LTM
settings, on task 2. Few-shot (Memorisation, Generalisation, Protection) at a class's n-th exposure, mean over the
classes that reached n; held-out accuracy at the checkpoints both runs share; the STM's effect on identical LTM
weights (stm_step_acc last - first, 1,000 held-out images), mean over task 2's checkpoints. Markdown to stdout.

    python -m single_stream.stm_pilot_tables --run runs_ss/mnist_clustered_single_stream/stm_pilot > tables.md
"""
import argparse
import json
import os

import numpy as np

NS = (1, 8, 32, 64, 256)
FAMS = {"lp": "id,lperm1", "dih": "id,rot90"}

# (batch, STM run, baseline run, label); cut = task 2 exposures compared (the shorter run's length)
PAIRS = []
for b, cut in (("batch1", 10000), ("batch2", 10000)):
    for fam in ("lp", "dih"):
        for c in (128, 512):
            for P in (256, 1024):
                PAIRS.append((b, f"{b}/cls_{fam}_c{c}_P{P}", f"{b}/ltm_{fam}_c{c}_P{P}", f"{fam} c{c} P{P}", "cap 16", cut))
for fam, c in (("lp", 512), ("dih", 512), ("lp", 128)):
    base = f"batch2/ltm_{fam}_c{c}_P256"
    PAIRS += [("batch3", f"batch3/A_{fam}_c{c}", base, f"{fam} c{c} P256", "A", 2048),
              ("batch3", f"batch3/B_{fam}_c{c}", base, f"{fam} c{c} P256", "B", 2048),
              ("batch3", f"batch3/D_{fam}_c{c}", f"batch3/ltmfifo_{fam}_c{c}", f"{fam} c{c} P256", "D (FIFO 2048)", 2048),
              ("batch4", f"batch4/B500_{fam}_c{c}", f"batch4/ltmfifo500_{fam}_c{c}", f"{fam} c{c} P256",
               "B, LTM FIFO 500", 2048)]


def load(run, d):
    path = os.path.join(run, d)
    if not os.path.exists(os.path.join(path, "summary.json")):
        return None
    return {"fs": [json.loads(l) for l in open(os.path.join(path, "fewshot.jsonl"))],
            "ck": [json.loads(l) for l in open(os.path.join(path, "checkpoints.jsonl"))]}


def fewshot(rd, key, n, cut):
    v = [r[key] for r in rd["fs"] if r["task"] == 1 and r["threshold"] == n and r["t_task"] <= cut]
    return np.mean(v) if v else np.nan


def f(v):
    return "-" if v is None or np.isnan(v) else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    for key, title in (("memorisation", "Memorisation"), ("recall", "Generalisation"), ("protection", "Protection")):
        print(f"\n### {title} at the class's n-th exposure in task 2: STM / LTM alone\n")
        print("| batch | condition | STM | " + " | ".join(f"n {n}" for n in NS) + " |")
        print("|---|---|---|" + "---|" * len(NS))
        for b, d, base, cond, label, cut in PAIRS:
            s, l = load(args.run, d), load(args.run, base)
            if s is None or l is None:
                continue
            print(f"| {b} | {cond} | {label} | " +
                  " | ".join(f"{f(fewshot(s, key, n, cut))} / {f(fewshot(l, key, n, cut))}" for n in NS) + " |")
    print("\n### Held-out accuracy on task 2 (mean over shared checkpoints) and the STM's effect on the same weights\n")
    print("| batch | condition | STM | CLS | LTM alone | difference | STM effect, same weights | end CLS / LTM |")
    print("|---|---|---|---|---|---|---|---|")
    for b, d, base, cond, label, cut in PAIRS:
        s, l = load(args.run, d), load(args.run, base)
        if s is None or l is None:
            continue
        sk = {r["t_task"]: r for r in s["ck"] if r["task"] == 1 and r["t_task"] <= cut}
        lk = {r["t_task"]: r for r in l["ck"] if r["task"] == 1 and r["t_task"] <= cut}
        xs = sorted(set(sk) & set(lk))
        a, la = np.mean([sk[x]["acc"] for x in xs]), np.mean([lk[x]["acc"] for x in xs])
        eff = np.mean([sk[x]["stm_step_acc"][-1] - sk[x]["stm_step_acc"][0] for x in sk if sk[x].get("stm_online")])
        print(f"| {b} | {cond} | {label} | {a:.3f} | {la:.3f} | {a - la:+.3f} | {eff:+.3f} | "
              f"{sk[xs[-1]]['acc']:.2f} / {lk[xs[-1]]['acc']:.2f} (at {xs[-1]}) |")


if __name__ == "__main__":
    main()
