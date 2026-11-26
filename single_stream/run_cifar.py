"""
CIFAR-100 coarse-pair runs (single_stream/cifar.py; Dave 26 Nov 2026): any stream model (run_stream's --model and
its flags) on one pair's stream: LTM pre-training on the first fine class of each side, 8 fine-class introductions,
the resilience extension. Phase-blind: the model sees only (x, y), one sample at a time; the STM variants (--model
cls, head) come online when the pre-training ends (--stm-delay and --head-delay are set to its length).

Evaluation (no learning, random state untouched) per phase: at n = 0 (before the new class's first exposure), at
its n-th exposure for n in --points, every --check-every exposures and at the phase's end: accuracy on the
evaluation images of each of the pair's 10 fine classes, and on the new class's training images shown so far
(mem); for the STM variants also the LTM's own accuracy per fine class (acc_ltm: its output without the STM or
head; Dave 26 Nov 2026: does the LTM come to separate look-alike fine classes of the two sides after sleep?). One
JSON line per checkpoint in evals.jsonl; metrics (cifar.score) in summary.json.

    python -m single_stream.run_cifar --pair 0,1 --cluster-size 64 --preproc runs_ss/preproc/tinyimagenet_conv_c32_s0/preproc.pt \
        --net transformer --model buffered ... --out <dir>
"""
import json
import os
import sys
import time

import numpy as np
import torch

from single_stream import cifar, metrics
from single_stream.datasets import StreamDataset
from single_stream.preprocessor import featurize, load_checkpoint
from single_stream.run_stream import Tee, build_model, build_parser, rng_neutral


def parse_args(argv=None):
    p = build_parser(__doc__)
    p.add_argument("--pair", required=True, help="two coarse class indices, a,b (label 0 = a, 1 = b)")
    p.add_argument("--cifar-root", default=os.path.expanduser("~/dev/claude/cifar-100-python"))
    p.add_argument("--pretrain-epochs", type=int, default=10, help="epochs of the pre-training classes' stream")
    p.add_argument("--share", type=float, default=0.75, help="the new fine class's share of an introduction")
    p.add_argument("--val-per-class", type=int, default=100, help="validation images cut from each fine class")
    p.add_argument("--eval-split", default="val", choices=("val", "test"),
                   help="val: the validation images (selection, the development pair); test: the test set")
    p.add_argument("--points", default=",".join(map(str, cifar.POINTS)), help="the new class's exposures to score at")
    p.add_argument("--check-every", type=int, default=64, help="extra checkpoints every this many phase exposures")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if os.path.exists(os.path.join(args.out, "summary.json")):
        sys.exit(f"{args.out} already has results; refusing to overwrite")
    if not args.preproc:
        sys.exit("--preproc is required (the frozen CIFAR preprocessor)")
    os.makedirs(args.out, exist_ok=True)
    log = Tee(os.path.join(args.out, "log.txt"))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    pair = tuple(int(v) for v in args.pair.split(","))
    points = sorted(int(v) for v in args.points.split(","))

    data = cifar.load_cifar100(args.cifar_root)
    x_file, fine, coarse = data["train"]
    train_idx, val_idx = cifar.validation_split(fine, args.val_per_class, args.split_seed)
    sched = cifar.make_schedule(pair, fine, coarse, train_idx, args.seed, args.cluster_size, args.pretrain_epochs,
                                args.share)
    pretrain_len = sched["phases"][0]["length"]
    args.stm_delay = args.head_delay = pretrain_len  # the STM variants come online when the pre-training ends
    stream = sched.pop("stream")
    np.save(os.path.join(args.out, "stream.npy"), stream)
    sched["stream_sha256"] = cifar.stream_hash(stream)
    with open(os.path.join(args.out, "schedule.json"), "w") as f:
        json.dump(sched, f, indent=1)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    # Frozen tokens of the pair's images only: training (stream and mem) and evaluation
    pre = load_checkpoint(args.preproc)
    classes = sched["pretrain"] + sched["introductions"]
    tr_rows = train_idx[np.isin(fine[train_idx], classes)]
    row_of = {int(i): k for k, i in enumerate(tr_rows)}
    x_tr = featurize(pre, x_file[tr_rows])
    y_tr = cifar.label(coarse[tr_rows], pair)
    if args.eval_split == "val":
        ev = val_idx[np.isin(fine[val_idx], classes)]
        x_ev, f_ev, c_ev = featurize(pre, x_file[ev]), fine[ev], coarse[ev]
    else:
        x_te, f_te, c_te = data["test"]
        ev = np.flatnonzero(np.isin(f_te, classes))
        x_ev, f_ev, c_ev = featurize(pre, x_te[ev]), f_te[ev], c_te[ev]
    evals = StreamDataset(x_ev, cifar.label(c_ev, pair), f_ev, 2, "eval")
    train = StreamDataset(x_tr, y_tr, fine[tr_rows], 2, "train")
    model = build_model(args, train, (pre.num_tokens, pre.channels))
    log(f"pair {pair} ({data['coarse_names'][pair[0]]} vs {data['coarse_names'][pair[1]]})  seed {args.seed}  "
        f"c {args.cluster_size}  order {sched['order']}  stream {len(stream)} (pre-training {pretrain_len})  "
        f"eval {args.eval_split} {len(evals)}  sha256 {sched['stream_sha256'][:12]}  memory {model.memory_values() / 1e6:.2f}M")

    ev_file = open(os.path.join(args.out, "evals.jsonl"), "w")
    records = []
    t = 0
    train_time = eval_time = 0.0
    seen = []
    for i, ph in enumerate(sched["phases"]):
        seg = stream[ph["start"]:ph["start"] + ph["length"]]
        if args.max_exposures:  # smoke tests and timing only: cut the stream
            seg = seg[:max(0, args.max_exposures - ph["start"])]
            if len(seg) == 0:
                break
        new = ph["new"]
        shown = []  # rows of the new class's images shown so far in this phase
        n = 0
        online_correct = 0

        def checkpoint(t_phase):
            nonlocal eval_time
            t0 = time.perf_counter()
            with rng_neutral():
                p = metrics.evaluate_variants(model, evals)["main"]
                mem = None
                if shown:
                    rows = torch.tensor(shown)
                    mem = float((model.predict(x_tr[rows]).argmax(1) == y_tr[rows]).float().mean())
                # the LTM's own answer (no STM or head): does the LTM itself come to separate look-alike classes?
                p_ltm = metrics.evaluate(model.ltm, evals) if hasattr(model, "ltm") else None
            correct = p == evals.y.numpy()
            rec = {"t": t, "phase": i, "kind": ph["kind"], "new": new, "n": n, "t_phase": t_phase,
                   "seen": list(seen), "mem": mem,
                   "acc": {str(f): float(correct[f_ev == f].mean()) for f in classes},
                   "acc_ltm": None if p_ltm is None else
                   {str(f): float((p_ltm == evals.y.numpy())[f_ev == f].mean()) for f in classes},
                   "online_acc": online_correct / t_phase if t_phase else None}
            eval_time += time.perf_counter() - t0
            records.append(rec)
            ev_file.write(json.dumps(rec) + "\n")
            ev_file.flush()
            return rec

        if ph["kind"] != "pretrain":
            checkpoint(0)
        for j, idx in enumerate(seg):
            r = row_of[int(idx)]
            t0 = time.perf_counter()
            pred = model.observe(x_tr[r:r + 1], y_tr[r:r + 1])
            train_time += time.perf_counter() - t0
            online_correct += int(pred[0] == y_tr[r])
            t += 1
            hit = False
            if new is not None and fine[idx] == new:
                n += 1
                shown.append(r)
                hit = n in points
            if hit or (j + 1) % args.check_every == 0 or j + 1 == len(seg):
                if ph["kind"] == "pretrain" and j + 1 < len(seg) and (j + 1) % (args.check_every * 8):
                    continue  # pre-training: sparser checkpoints
                rec = checkpoint(j + 1)
        shown_classes = sched["pretrain"] if new is None else seen + ([new] if new not in seen else [])
        acc_seen = np.mean([rec["acc"][str(f)] for f in shown_classes])
        log(f"phase {i} {ph['kind']} new {new} ({data['fine_names'][new] if new is not None else '-'}): "
            f"end acc on seen classes {acc_seen:.3f}  online {online_correct / len(seg):.3f}  "
            f"train {train_time:.0f}s eval {eval_time:.0f}s")
        if ph["kind"] == "pretrain":
            seen.extend(sched["pretrain"])
        elif new not in seen:
            seen.append(new)

    ev_file.close()
    summary = {"metrics": cifar.score(records, sched["phases"], points), "stream_sha256": sched["stream_sha256"],
               "exposures": t, "train_time": train_time, "eval_time": eval_time,
               "ms_per_step": 1000 * train_time / max(t, 1)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("metrics " + json.dumps({k: round(v, 4) for k, v in summary["metrics"].items() if isinstance(v, float)}))
    log(f"done  {summary['ms_per_step']:.2f} ms/step")


if __name__ == "__main__":
    main()
