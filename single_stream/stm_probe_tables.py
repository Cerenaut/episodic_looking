"""Tables for the factorial STM probe (9 Nov 2026): per probe run, point group and condition, the mean p(y) before and
after the STM's loop, how many points reach p(y) >= .5, and the median repeats to first reach .5 (blocks of the trace).
Points are grouped by exposure within task 2: before the first sleep (1-15), just after it (17-23), between sleeps
(64-256), and the first exposures of the next class (513-527). "LTM wrong" keeps the points whose p(y) under the live
LTM is below .05. Markdown to stdout.

    python -m single_stream.stm_probe_tables --run runs_ss/mnist_clustered_single_stream/stm_pilot/probe_what
"""
import argparse
import glob
import json
import os

import numpy as np

GROUPS = (("before the first sleep", 1, 15), ("just after it", 17, 23), ("between sleeps", 64, 256),
          ("next class", 513, 527))


def label(r):
    name = ("STM as is" if r["stm"] == "as_is" else "fresh STM") + (", replay" if r["replay"] else ", no replay")
    if r["temperature"] != 1.0:
        name += f", logits / {r['temperature']:g}"
    return name + f", cap {r['repeats_cap']}"


def load(path):
    """Records with the condition's repeat cap. The first probe (before aeb91de) overwrote the cap with the repeats
    used; there the ceiling is the last of each point's records."""
    recs = [json.loads(l) for l in open(path)]
    if "repeats_cap" in recs[0]:
        for r in recs:
            r["repeats_cap"] = r["repeats_cap"] or 256
        return recs
    per = 9
    for i, r in enumerate(recs):
        r["repeats_cap"] = 1024 if i % per == per - 1 else 256
    return recs


def first_reach(r, p=0.5):
    for b in r["blocks"]:
        if max(b[1:]) >= p:
            return b[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    for path in sorted(glob.glob(os.path.join(args.run, "*", "stm_probe.jsonl"))):
        recs = load(path)
        print(f"\n### {os.path.basename(os.path.dirname(path))}\n")
        print("| points | n | condition | p(y) before | p(y) after | reach .5 | median repeats to .5 |")
        print("|---|---|---|---|---|---|---|")
        for gname, lo, hi in GROUPS:
            for wrong in (False, True):
                rs = [r for r in recs if lo <= r["t_task"] <= hi]
                live = {r["t_task"]: r["p0"] for r in rs if r["stm"] == "as_is" and r["replay"]
                        and r["temperature"] == 1.0}
                if wrong:
                    rs = [r for r in rs if live.get(r["t_task"], 1) < 0.05]
                if not rs:
                    continue
                conds = list(dict.fromkeys(label(r) for r in rs))
                for c in conds:
                    g = [r for r in rs if label(r) == c]
                    reach = [first_reach(r) for r in g]
                    ok = [k for k in reach if k is not None]
                    med = f"{np.median(ok):.0f}" if ok else "-"
                    pts = gname + (", LTM wrong" if wrong else "")
                    print(f"| {pts} | {len(g)} | {c} | {np.mean([r['p0'] for r in g]):.2f} | "
                          f"{np.mean([r['p_end'] for r in g]):.2f} | {len(ok)}/{len(g)} | {med} |")


if __name__ == "__main__":
    main()
