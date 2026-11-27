"""
CIFAR-100 coarse-pair stream (single_stream_paper plan.md, "Next: CIFAR-100, then the preprint"; Dave 26 Nov 2026).

Binary classification between the two coarse classes of a pair (label 0 = side a, 1 = side b); samples arrive by fine
class (5 per coarse class). One run (pair, seed, cluster size c):
  pretrain     the first fine class of each side, `pretrain_epochs` epochs of the clustered stream (the LTM's
               pre-training; the STM variants are offline until it ends)
  intro 1-8    the other 8 fine classes one at a time, alternating sides (a, b, a, b, ...): each of the new class's
               training images once, mixed with clusters of the fine classes seen so far so that the new class is
               `share` (75%) of the exposures
  resilience   the first introduced fine class again, in the same way (the run's extension)
Every fine class's order within its side is drawn from the seed; the validation split (`val_per_class` of each fine
class's 500 training images) is fixed by `split_seed`, the same for every seed and model. The schedule (classes and
the exact sample order, as indices into CIFAR-100's training file) is saved with each run, and its hash goes into the
run's summary, so every model of a seed sees the same samples.
Paired (`paired=True`, Dave 27 Nov 2026; run_cifar --paired, from probe2): the 8 classes arrive as 4 pairs, one fine class
of each side (a_k, b_k), each phase only the pair's images, every one once, in alternating clusters of c (a, b, a,
...): the labels are 50/50 within a phase and the classes seen so far are always balanced across the sides, so
always answering one label scores 0.5 everywhere. The resilience extension re-presents the first pair.
Clustering: within a phase, the new class's images and the others' are cut into clusters of c samples of one fine
class and interleaved as ClusteredStream does (a group is not picked twice in a row while another remains). The
others' clusters come from seen fine classes in a random cyclic order, c samples each (the last cut to fit).
"""
import hashlib
import os
import pickle

import numpy as np
import torch

from single_stream.stream import ClusteredStream

FINE_PER_COARSE = 5
POINTS = (1, 2, 4, 8, 16, 32, 64)  # a new class's n-th exposure: the few-shot points (as the held-out digit)


def load_cifar100(root):
    """{'train': (x, fine, coarse), 'test': (...)}: x (N, 3072) float32 in [0, 1], channel-major as the files; plus
    'fine_names', 'coarse_names'."""
    out = {}
    for split in ("train", "test"):
        d = pickle.load(open(os.path.join(root, split), "rb"), encoding="latin1")
        out[split] = (torch.from_numpy(d["data"]).float() / 255.0, np.array(d["fine_labels"]),
                      np.array(d["coarse_labels"]))
    meta = pickle.load(open(os.path.join(root, "meta"), "rb"), encoding="latin1")
    out["fine_names"], out["coarse_names"] = meta["fine_label_names"], meta["coarse_label_names"]
    return out


def fine_of(coarse, fine, c):
    """The fine classes of coarse class c, ascending."""
    return sorted(set(fine[coarse == c].tolist()))


def validation_split(fine, per_class, split_seed=0):
    """(train, validation) indices into the training file: `per_class` validation images of every fine class."""
    rng = np.random.default_rng(split_seed)
    val = np.sort(np.concatenate([rng.choice(np.flatnonzero(fine == f), per_class, replace=False)
                                  for f in np.unique(fine)]))
    return np.setdiff1d(np.arange(len(fine)), val), val


def _phase(new_idx, by_class, seen, groups_of, c, share, rng):
    """One introduction: the new class's images once plus clusters of seen classes up to (1 - share) of the phase.
    by_class: {fine: its training indices}; seen: the classes the others come from (no image twice in a phase).
    Returns the sample order (indices into the training file)."""
    n_other = int(round(len(new_idx) * (1 - share) / share))
    pools = {f: list(rng.permutation(by_class[f])) for f in seen} if n_other > 0 else {}
    chosen, cycle = [], []
    while n_other > 0:
        if not cycle:
            cycle = list(rng.permutation(seen))
        f = int(cycle.pop(0))
        k = min(c, n_other, len(pools[f]))
        chosen.append(np.array(pools[f][:k], dtype=np.int64))
        del pools[f][:k]
        n_other -= k
    idx = np.concatenate([new_idx] + chosen) if chosen else np.asarray(new_idx)
    # ClusteredStream cuts each group into clusters of c and interleaves them; the others drawn from one class in two
    # visits of the cycle merge into one group, which it re-cuts into clusters of c
    epoch = ClusteredStream(groups_of[idx], c, rng).epoch()
    return idx[epoch.order]


def make_schedule(pair, fine, coarse, train_idx, seed, cluster_size, pretrain_epochs, share=0.75, paired=False):
    """The run's classes and sample order. fine, coarse: labels of the training file; train_idx: its training split.
    Returns a dict (JSON-able) with 'stream' (np.int64 indices into the training file) and 'phases'. Paired: a phase's
    'new' is the pair [a_k, b_k] and share is not used."""
    rng = np.random.default_rng(seed)
    a, b = pair
    order = {side: [int(f) for f in rng.permutation(fine_of(coarse, fine, c))] for side, c in (("a", a), ("b", b))}
    pretrain = [order["a"][0], order["b"][0]]
    intros = [f for k in range(1, FINE_PER_COARSE) for f in (order["a"][k], order["b"][k])]
    by_class = {f: train_idx[fine[train_idx] == f] for f in pretrain + intros}

    stream, phases = [], []
    pre = np.concatenate([by_class[f] for f in pretrain])
    cs = ClusteredStream(fine[pre], cluster_size, rng)
    for _ in range(pretrain_epochs):
        stream.append(pre[cs.epoch().order])
    phases.append({"kind": "pretrain", "new": None, "start": 0, "length": int(sum(len(s) for s in stream))})
    seen = list(pretrain)
    if paired:
        pairs = [[order["a"][k], order["b"][k]] for k in range(1, FINE_PER_COARSE)]
        for kind, pr in [("intro", pr) for pr in pairs] + [("resilience", pairs[0])]:
            idx = np.concatenate([by_class[f] for f in pr])
            start = int(sum(len(s) for s in stream))
            stream.append(idx[ClusteredStream(fine[idx], cluster_size, rng).epoch().order])
            phases.append({"kind": kind, "new": [int(f) for f in pr], "start": start, "length": len(stream[-1])})
        stream = np.concatenate(stream).astype(np.int64)
        return {"pair": [int(a), int(b)], "seed": int(seed), "cluster_size": int(cluster_size),
                "pretrain_epochs": int(pretrain_epochs), "paired": True, "order": order, "pretrain": pretrain,
                "introductions": intros, "pairs": pairs, "phases": phases, "stream": stream}
    for kind, f in [("intro", f) for f in intros] + [("resilience", intros[0])]:
        others = [g for g in seen if g != f]
        start = int(sum(len(s) for s in stream))
        stream.append(_phase(rng.permutation(by_class[f]), by_class, others, fine, cluster_size, share, rng))
        phases.append({"kind": kind, "new": int(f), "start": start, "length": len(stream[-1])})
        if f not in seen:
            seen.append(f)
    stream = np.concatenate(stream).astype(np.int64)
    return {"pair": [int(a), int(b)], "seed": int(seed), "cluster_size": int(cluster_size),
            "pretrain_epochs": int(pretrain_epochs), "share": float(share), "order": order, "pretrain": pretrain,
            "introductions": intros, "phases": phases, "stream": stream}


def stream_hash(stream):
    return hashlib.sha256(np.ascontiguousarray(stream, dtype=np.int64).tobytes()).hexdigest()


def label(coarse, pair):
    """Binary target: 0 for side a's coarse class, 1 for side b's."""
    return torch.from_numpy((np.asarray(coarse) == pair[1]).astype(np.int64))


def score(records, phases, points=POINTS):
    if phases and any(isinstance(ph["new"], list) for ph in phases):
        return score_paired(records, phases, points)
    return score_single(records, phases, points)


def score_paired(records, phases, points=POINTS):
    """
    Paired introductions (records: n and mem are {fine: value} over the pair). For each class f of a pair, at its
    n-th exposure (the first checkpoint there) for n in points, with its partner g as the control (Dave 27 Nov 2026:
    a cluster is all one label, so a class scored alone mid-cluster rewards answering the current label):
      Memorisation    mean(mem[f], acc[g]);  Generalisation  mean(acc[f], acc[g]);  zero_shot: acc[f] at n = 0
      Protection      worst pooled accuracy on the classes seen before the phase over its checkpoints
      Retention       pooled accuracy on the seen classes and the pair at the phase's end
      Resilience      Generalisation in the resilience phase
    Each averaged over n, the pair's two classes and the phases; Overall, Overall - worst, rescaled as score_single.
    """
    def pooled(r, classes):
        return float(np.mean([r["acc"][str(f)] for f in classes])) if classes else float("nan")

    per = {"Memorisation": [], "Generalisation": [], "zero_shot": [], "Protection": [], "Retention": []}
    res = []
    for i, ph in enumerate(phases):
        rs = [r for r in records if r["phase"] == i]
        if ph["kind"] == "pretrain" or not rs:
            continue
        mem, gen, zero = [], [], []
        for f, g in (ph["new"], ph["new"][::-1]):
            f, g = str(f), str(g)
            at = {}
            for r in rs:
                at.setdefault(r["n"][f], r)
            for n in points:
                if n in at:
                    mem.append(0.5 * (at[n]["mem"][f] + at[n]["acc"][g]))
                    gen.append(0.5 * (at[n]["acc"][f] + at[n]["acc"][g]))
            zero.append(rs[0]["acc"][f])
        if ph["kind"] == "resilience":
            res.append(np.mean(gen))
            continue
        others = [c for c in rs[0]["seen"] if c not in ph["new"]]
        per["Memorisation"].append(np.mean(mem))
        per["Generalisation"].append(np.mean(gen))
        per["zero_shot"].append(np.mean(zero))
        per["Protection"].append(min(pooled(r, others) for r in rs))
        per["Retention"].append(pooled(rs[-1], others + list(ph["new"])))
    return _summarise(per, res)


def score_single(records, phases, points=POINTS):
    """
    The metrics from a run's eval records (run_cifar.py, one per checkpoint: phase index, kind, new class, n = the new
    class's exposures so far in the phase, acc = {fine: accuracy on its evaluation images}, mem = accuracy on the new
    class's training images shown so far, seen = the fine classes seen before the phase).
      Memorisation    mean over the introductions of the mean over n in points of mem
      Generalisation  ... of acc[new]; zero_shot: acc[new] at n = 0 (logged; Dave 27 Nov 2026: no gain)
      Protection      mean over the introductions of the worst pooled accuracy on the seen classes (excluding the
                      new one) over the introduction's checkpoints
      Retention       mean over the introductions of the pooled accuracy on the seen classes and the new one at the
                      introduction's end
      Resilience      mean over n in points of acc[new] in the resilience phase
    Overall = the mean of the five (of those present: _summarise); Overall - worst = their minimum. Pooled =
    equal weight per class.
    """
    def pooled(r, classes):
        return float(np.mean([r["acc"][str(f)] for f in classes])) if classes else float("nan")

    per = {"Memorisation": [], "Generalisation": [], "zero_shot": [], "Protection": [], "Retention": []}
    res = []
    for i, ph in enumerate(phases):
        rs = [r for r in records if r["phase"] == i]
        if ph["kind"] == "pretrain" or not rs:
            continue
        at = {}
        for r in rs:  # the first checkpoint at each n: the one right after the n-th exposure (n stays put while
            at.setdefault(r["n"], r)  # other classes' clusters are shown)
        f = str(ph["new"])
        gen = [at[n]["acc"][f] for n in points if n in at]
        if ph["kind"] == "resilience":
            res.append(np.mean(gen))
            continue
        per["Memorisation"].append(np.mean([at[n]["mem"] for n in points if n in at]))
        per["Generalisation"].append(np.mean(gen))
        per["zero_shot"].append(at[0]["acc"][f] if 0 in at else float("nan"))
        others = [g for g in rs[0]["seen"] if g != ph["new"]]
        per["Protection"].append(min(pooled(r, others) for r in rs))
        per["Retention"].append(pooled(rs[-1], others + [ph["new"]]))
    return _summarise(per, res)


def _summarise(per, res):
    """Overall and Overall - worst over the core metrics the run has: a stream cut before the resilience phase
    (--max-exposures; Dave 27 Nov 2026: the RL STM's tuning probes skip it) has no Resilience, and its Overall is the
    mean of the other four ("core" lists the metrics used)."""
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in per.items()}
    out["Resilience"] = float(np.mean(res)) if res else float("nan")
    names = [k for k in ("Memorisation", "Generalisation", "Protection", "Retention", "Resilience")
             if not np.isnan(out[k])]
    core = [out[k] for k in names]
    out["core"] = names
    out["Overall"] = float(np.mean(core)) if core else float("nan")
    out["Overall - worst"] = float(np.min(core)) if core else float("nan")
    out["rescaled"] = {k: 2 * out[k] - 1 for k in ("Memorisation", "Generalisation", "Protection", "Retention",
                                                    "Resilience", "Overall", "Overall - worst")}
    return out
