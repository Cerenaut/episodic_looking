"""
Oracle bias test on CIFAR (Dave 28 Nov 2026: does a "where" bias add headroom over "what" on CIFAR, whose images
have far more distraction than MNIST? A fast check before any RL runs; method as the 9 Nov MNIST oracle,
stm_probe.oracle). Replays an LTM run (run_cifar's args.json; the stream hash is checked) to the given exposures. At
each, on the validation images of the fine classes seen so far, the bias is optimised per image by gradient (Adam on
the raw action, cross-entropy on the label, no RL), within the STM's range (gain exp(s tanh(a)), s = --bias-scale) and narrower ones (--scales: a family that
still flips with less range is more efficient; at the full range "what" alone may already flip everything),
for each family of bias, all expressed through one what+where copy of the LTM (same weights) and a fixed basis:

    what         64 per-feature gains (the RL STM's current bias)
    where        a per-token attention term (8 x 8 tokens)
    where4/2     where on a coarse 4 x 4 / 2 x 2 grid, upsampled to the tokens
    what+where   both, full;  what+where4: what + coarse 4 x 4 where

Per family, on the images the unbiased LTM gets wrong: the fraction flipped (argmax = label) after 10, 25, 50, 100,
200 steps (speed: a proxy for how easily a search can find it), mean p(y), and the sparsity: the fraction still
flipped when only the m largest action elements (by |tanh(a)|) are kept, m in 2, 4, 8, 16, 32. Batched over the
images; each image has its own bias, so the optimisations are independent.

    python -m single_stream.bias_oracle --run runs_ss/cifar/probe2/ltm_sleep_c64 --stops 4000,4600,5600,6400,8000 \
        --out runs_ss/cifar/oracle1/ltm_sleep_c64
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from model.transformer import BiasableTransformer
from single_stream import cifar, run_cifar
from single_stream.datasets import StreamDataset
from single_stream.preprocessor import featurize, load_checkpoint
from single_stream.run_stream import build_model
from single_stream.stm import grid_basis

RECORD = (0, 10, 25, 50, 100, 200)
KEEP = (2, 4, 8, 16, 32)


def bases(d, side):
    """Action -> what+where bias (d + side^2) for each family."""
    n = side * side
    z = lambda r, c: torch.zeros(r, c)
    out = {"what": torch.cat([torch.eye(d), z(d, n)], 1), "where": torch.cat([z(n, d), torch.eye(n)], 1)}
    for g in (4, 2):
        out[f"where{g}"] = torch.cat([z(g * g, d), grid_basis(g, side)], 1)
    out["what+where"] = torch.eye(d + n)
    out["what+where4"] = torch.block_diag(torch.eye(d), grid_basis(4, side))
    return out


def optimise(net, x, y, basis, steps=200, lr=0.1):
    a = torch.zeros(len(y), basis.shape[0], requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    curve = {}
    for t in range(steps + 1):
        logits = net(x, a @ basis)
        if t in RECORD or t == steps:
            with torch.no_grad():
                curve[t] = {"flip": float((logits.argmax(1) == y).float().mean()),
                            "p": float(F.softmax(logits, 1).gather(1, y[:, None]).mean())}
        if t == steps:
            break
        loss = F.cross_entropy(logits, y, reduction="sum")
        opt.zero_grad()
        loss.backward()
        opt.step()
    a = a.detach()
    sparse = {}
    with torch.no_grad():
        mag = torch.tanh(a).abs()
        for m in KEEP:
            if m >= a.shape[1]:
                continue
            idx = mag.topk(m, dim=1).indices
            kept = torch.zeros_like(a).scatter(1, idx, a.gather(1, idx))
            sparse[m] = float((net(x, kept @ basis).argmax(1) == y).float().mean())
        sat = float((mag > 0.95).float().mean())
    return {"curve": curve, "keep_top": sparse, "saturated": sat, "action_size": int(basis.shape[0])}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a run_cifar output directory (LTM model: buffered)")
    p.add_argument("--stops", required=True, help="exposures at which to test, comma-separated")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--scales", default="0.25,0.5,1", help="bias ranges as fractions of the STM's (--bias-scale)")
    p.add_argument("--out", required=True)
    o = p.parse_args()
    os.makedirs(o.out, exist_ok=True)
    stops = sorted(int(v) for v in o.stops.split(","))
    a = json.load(open(f"{o.run}/args.json"))
    args = run_cifar.parse_args(["--pair", a["pair"], "--out", o.out])
    vars(args).update(a)
    if args.model != "buffered":
        raise SystemExit("replays the LTM alone (--model buffered)")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    pair = tuple(int(v) for v in args.pair.split(","))
    data = cifar.load_cifar100(args.cifar_root)
    x_file, fine, coarse = data["train"]
    train_idx, val_idx = cifar.validation_split(fine, args.val_per_class, args.split_seed)
    sched = cifar.make_schedule(pair, fine, coarse, train_idx, args.seed, args.cluster_size, args.pretrain_epochs,
                                args.share, args.paired)
    stream = sched["stream"]
    if cifar.stream_hash(stream) != json.load(open(f"{o.run}/summary.json"))["stream_sha256"]:
        raise SystemExit("stream hash differs from the run's")
    pre = load_checkpoint(args.preproc)
    classes = sched["pretrain"] + sched["introductions"]
    tr_rows = train_idx[np.isin(fine[train_idx], classes)]
    row_of = {int(i): k for k, i in enumerate(tr_rows)}
    x_tr, y_tr = featurize(pre, x_file[tr_rows]), cifar.label(coarse[tr_rows], pair)
    ev = val_idx[np.isin(fine[val_idx], classes)]
    x_ev, f_ev, y_ev = featurize(pre, x_file[ev]), fine[ev], cifar.label(coarse[ev], pair)
    model = build_model(args, StreamDataset(x_tr, y_tr, fine[tr_rows], 2, "train"), (pre.num_tokens, pre.channels))
    evals = [json.loads(l) for l in open(f"{o.run}/evals.jsonl")]
    report = []
    for t, idx in enumerate(stream[:max(stops)], 1):
        r = row_of[int(idx)]
        model.observe(x_tr[r:r + 1], y_tr[r:r + 1])
        if t not in stops:
            continue
        with torch.random.fork_rng(devices=[]):  # the test must not move the LTM's later random draws
            report.append(test_point(args, o, model.net, pre, t, stream, fine, x_ev, f_ev, y_ev, evals))
    json.dump(report, open(f"{o.out}/oracle.json", "w"), indent=1)


def test_point(args, o, ltm, pre, t, stream, fine, x_ev, f_ev, y_ev, evals):
    """One test point: the LTM at exposure t, every bias family and range on its wrong validation images."""
    net = BiasableTransformer(pre.num_tokens, pre.channels, 2, d_model=args.d_model, heads=args.heads,
                              blocks=args.blocks, mlp_ratio=args.mlp_ratio, bias_block=args.bias_block,
                              bias_mode="what+where", bias_scale=args.bias_scale)
    net.load_state_dict(ltm.state_dict())
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    seen = sorted({int(fine[i]) for i in stream[:t]})
    m = torch.from_numpy(np.isin(f_ev, seen))
    x, y, f = x_ev[m], y_ev[m], f_ev[m.numpy()]
    with torch.no_grad():
        pred = net(x).argmax(1)
    acc = float((pred == y).float().mean())
    logged = [e for e in evals if e["t"] == t and e.get("phase", 0) >= 0]
    check = None
    if logged:  # the run's own record at t: the replay must reproduce it
        e = logged[-1]
        check = max(abs(e["acc"][str(c)] - float((pred[torch.from_numpy(f == c)] == y[torch.from_numpy(f == c)])
                                                   .float().mean())) for c in seen)
    wrong = pred != y
    rec = {"t": t, "seen": seen, "images": int(len(y)), "ltm_acc": acc, "wrong": int(wrong.sum()),
           "replay_check_max_diff": check, "families": {}}
    side = int(round(pre.num_tokens ** 0.5))
    for k in (float(v) for v in o.scales.split(",")):
        net.bias_scale = args.bias_scale * k
        for name, b in bases(args.d_model, side).items():
            rec["families"][f"{name}@{k:g}"] = optimise(net, x[wrong], y[wrong], b, steps=o.steps)
    line = f"t {t}: {len(seen)} classes, {len(y)} images, LTM acc {acc:.3f}, wrong {int(wrong.sum())}" + \
           (f", replay check {check:.4f}" if check is not None else "")
    print(line, flush=True)
    for name, fam in rec["families"].items():
        c = fam["curve"]
        print(f"   {name:16s} ({fam['action_size']:3d}) flipped " +
              " ".join(f"{s}:{c[s]['flip']:.2f}" for s in RECORD[1:] if s in c) +
              f"  p(y) {c[max(c)]['p']:.2f}  keep-top " +
              " ".join(f"{k}:{v:.2f}" for k, v in fam["keep_top"].items()) + f"  sat {fam['saturated']:.2f}",
              flush=True)
    return rec


if __name__ == "__main__":
    main()
