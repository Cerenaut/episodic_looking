"""
Task 1 runner: train a stream model on the correlated single stream and evaluate at fixed exposure checkpoints.

Example (selection run on the validation split):
  python -m single_stream.run_stream --dataset mnist --cluster-size 16 --epochs 3 --lr 0.003 \
      --val-holdout 600 --out runs_ss/pilot/c16_s0

Outputs in --out: args.json, checkpoints.jsonl (one record per checkpoint), fewshot.jsonl (one record per
few-shot trigger, below), stream_stats.json (run lengths per epoch), summary.json, log.txt.

Few-shot triggers: besides the fixed checkpoints, the model is evaluated whenever a group's own exposure count
first reaches one of --fewshot-thresholds (default 1, 2, 4, ..., 512), so every group is scored at every
threshold however the stream is clustered. The trigger is the harness's bookkeeping; the model sees nothing.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from single_stream import datasets, metrics
from single_stream.buffers import NeuralBuffer, NoveltyGate, PriorityBuffer, RingBuffer, part_specs
from single_stream.models import BufferedReplayLTM, ContinualBackprop, LTMOnly, ReplayLTM, ReplayOnlyLTM, build_net
from single_stream.stream import ClusteredStream, run_lengths


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=sorted(datasets.LOADERS))
    p.add_argument("--data-root", default=os.path.expanduser("~/dev/claude/mnist"))
    p.add_argument("--permutation-seed", type=int, default=None, help="fixed pixel permutation (permuted MNIST)")
    p.add_argument("--val-holdout", type=int, default=0,
                   help="samples per group held out of training as the evaluation set; 0 = full train, test set")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--cluster-size", type=int, default=1, help="c")
    p.add_argument("--batch-size", type=int, default=1, help="b")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--model", default="ltm", choices=["ltm", "replay", "replayonly", "buffered"],
                   help="ltm: trained on the stream; replay: stream + reservoir replay mixed into each step; "
                        "replayonly: trained only from the replay buffer; buffered: replayonly with a pluggable, "
                        "novelty-gated buffer (--buffer)")
    p.add_argument("--net", default="mlp", choices=["mlp", "resmlp"])
    p.add_argument("--widths", default="2000,2000,2000")
    p.add_argument("--blocks", type=int, default=3, help="residual blocks (resmlp)")
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=1, help="gradient steps per stream step")
    p.add_argument("--cbp-rate", type=float, default=0.0, help="continual backprop replacement rate; 0 = off")
    p.add_argument("--cbp-decay", type=float, default=0.99)
    p.add_argument("--cbp-maturity", type=int, default=100)
    p.add_argument("--replay-capacity", type=int, default=1000)
    p.add_argument("--replay-batch", type=int, default=10)
    p.add_argument("--replay-period", type=int, default=16, help="replayonly: exposures between replay updates")
    p.add_argument("--replay-steps", type=int, default=1, help="replayonly: gradient steps per replay update")
    p.add_argument("--replay-policy", default="fifo", choices=["fifo", "reservoir"], help="replayonly")
    p.add_argument("--replay-float", action="store_true", help="replayonly: store float32 instead of uint8")
    p.add_argument("--buffer", default="ring", choices=["ring", "priority", "neural"],
                   help="buffered: ring = FIFO of --replay-capacity; priority = PriorityBuffer of --replay-capacity; "
                        "neural = NeuralBuffer of --nb-cells")
    p.add_argument("--alpha-evict", type=float, default=0.0, help="priority: eviction 1/rank^a, rank 1 least novel")
    p.add_argument("--alpha-replay", type=float, default=0.0, help="priority: replay 1/rank^a, rank 1 most novel")
    p.add_argument("--gate-floor", type=float, default=1.0,
                   help="buffered: store with p = floor + (1 - floor) x novelty; 1 = store everything")
    p.add_argument("--nb-cells", type=int, default=1000)
    p.add_argument("--nb-fan-in", type=int, default=64, help="key (input) values seen by each cell")
    p.add_argument("--nb-util-decay", type=float, default=0.999)
    p.add_argument("--nb-match-threshold", type=float, default=None, help="recognition threshold; default off")
    p.add_argument("--n-log-checkpoints", type=int, default=12)
    p.add_argument("--fewshot-thresholds", default="1,2,4,8,16,32,64,128,256,512",
                   help="evaluate when a group's exposure count first reaches each of these ('' = off)")
    p.add_argument("--grid-per-epoch", type=int, default=10)
    p.add_argument("--max-exposures", type=int, default=0, help="stop early (smoke tests and timing only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def build_model(args, train):
    widths = [int(w) for w in args.widths.split(",")]
    net = build_net(args.net, train.x.shape[1], train.num_classes, widths, args.blocks)
    cbp = None
    if args.cbp_rate > 0:
        if args.net != "mlp":
            raise ValueError("continual backprop is implemented for --net mlp only")
        cbp = ContinualBackprop(net, args.cbp_rate, args.cbp_decay, args.cbp_maturity)
    common = dict(lr=args.lr, steps=args.steps, momentum=args.momentum, weight_decay=args.weight_decay, cbp=cbp)
    if args.model == "ltm":
        return LTMOnly(net, **common)
    if args.model == "replayonly":
        if args.steps != 1:
            raise ValueError("replayonly uses --replay-steps, not --steps")
        del common["steps"]
        return ReplayOnlyLTM(net, capacity=args.replay_capacity, batch=args.replay_batch, period=args.replay_period,
                             steps=args.replay_steps, policy=args.replay_policy, store_uint8=not args.replay_float,
                             input_shape=train.x.shape[1:], seed=args.seed, **common)
    if args.model == "buffered":
        if args.steps != 1:
            raise ValueError("buffered uses --replay-steps, not --steps")
        del common["steps"]
        gate = NoveltyGate(args.gate_floor, seed=args.seed + 1)
        parts = part_specs(train.x[:1], train.y[:1])  # (image, label)
        if args.buffer == "ring":
            buffer = RingBuffer(args.replay_capacity, parts, gate, seed=args.seed)
        elif args.buffer == "priority":
            buffer = PriorityBuffer(args.replay_capacity, parts, alpha_evict=args.alpha_evict,
                                    alpha_replay=args.alpha_replay, gate=gate, seed=args.seed)
        else:
            buffer = NeuralBuffer(args.nb_cells, parts, fan_in=args.nb_fan_in, key_parts=(0,),
                                  util_decay=args.nb_util_decay, match_threshold=args.nb_match_threshold, gate=gate,
                                  seed=args.seed)
        return BufferedReplayLTM(net, buffer=buffer, batch=args.replay_batch, period=args.replay_period,
                                 steps=args.replay_steps, **common)
    return ReplayLTM(net, capacity=args.replay_capacity, replay_batch=args.replay_batch,
                     input_shape=train.x.shape[1:], seed=args.seed, **common)


def buffer_diagnostics(buffer, num_classes):
    """Diagnostics only (the model never sees them): stored labels per class, written count, and for the priority
    buffer the mean age and score of stored samples."""
    z = buffer.buf[:buffer.size] if hasattr(buffer, "buf") else buffer.C[buffer.occupied]
    labels = buffer.unpack(z)[1] if len(z) else torch.zeros(0, dtype=torch.long)
    out = {"buffer_labels": torch.bincount(labels, minlength=num_classes).tolist(), "buffer_written": buffer.written}
    if hasattr(buffer, "ages") and len(z):
        out.update({"buffer_mean_age": float(buffer.ages().float().mean()),
                    "buffer_mean_score": float(buffer.score[:buffer.size].mean())})
    return out


class Tee:

    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, msg):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def main(argv=None):
    args = parse_args(argv)
    if os.path.exists(os.path.join(args.out, "summary.json")):
        sys.exit(f"{args.out} already has results; refusing to overwrite")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    log = Tee(os.path.join(args.out, "log.txt"))

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train, eval_set = datasets.load(args.dataset, args.data_root, args.val_holdout, args.split_seed,
                                    permutation_seed=args.permutation_seed)
    model = build_model(args, train)
    stream = ClusteredStream(train.groups, args.cluster_size, rng)
    group_ids = stream.group_ids
    group_index = {g: i for i, g in enumerate(group_ids)}
    group_label = metrics.group_labels(train.groups, train.y.numpy(), group_ids)
    b = args.batch_size
    epoch_len = len(train)
    total = epoch_len * args.epochs
    if args.max_exposures:
        total = min(total, args.max_exposures)
    checkpoints, grid = metrics.checkpoint_schedule(epoch_len, args.epochs, args.n_log_checkpoints,
                                                    args.grid_per_epoch, b)
    checkpoints = [t for t in checkpoints if t <= total]
    if not checkpoints or checkpoints[-1] != total:
        checkpoints.append(total)
    log(f"train {len(train)}  eval {len(eval_set)}  groups {len(group_ids)}  c={args.cluster_size} b={b}  "
        f"epochs {args.epochs}  exposures {total}  checkpoints {len(checkpoints)}  "
        f"memory {model.memory_values() / 1e6:.2f}M values")

    thresholds = np.array([int(v) for v in args.fewshot_thresholds.split(",") if v.strip()], dtype=np.int64)
    fewshot_file = open(os.path.join(args.out, "fewshot.jsonl"), "w")
    n = np.zeros(len(group_ids), dtype=np.int64)
    q = np.zeros(len(group_ids), dtype=np.int64)
    last_seen = np.full(len(group_ids), -1, dtype=np.int64)
    online_correct = online_count = 0
    t = 0
    next_ckpt = 0
    stream_stats = []
    train_time = eval_time = 0.0
    ckpt_file = open(os.path.join(args.out, "checkpoints.jsonl"), "w")

    for epoch in range(args.epochs):
        if t >= total:
            break
        ep = stream.epoch()
        sample_groups = train.groups[ep.order]
        stream_stats.append({
            "epoch": epoch,
            "group_run_lengths": np.bincount(run_lengths(sample_groups)).tolist(),
            "label_run_lengths": np.bincount(run_lengths(train.y[ep.order].numpy())).tolist(),
            "cluster_lengths": np.bincount(ep.cluster_lengths).tolist(),
        })
        cluster_start = np.zeros(epoch_len, dtype=bool)
        cluster_start[np.concatenate([[0], np.cumsum(ep.cluster_lengths)[:-1]])] = True

        for i in range(0, epoch_len, b):
            if t >= total:
                break
            idx = ep.order[i:i + b]
            t0 = time.perf_counter()
            pred = model.observe(train.x[idx], train.y[idx])
            train_time += time.perf_counter() - t0
            online_correct += int((pred == train.y[idx]).sum())
            online_count += len(idx)
            n_before = n.copy()
            for j, k in enumerate(idx):
                gi = group_index[train.groups[k]]
                n[gi] += 1
                q[gi] += int(cluster_start[i + j])
                last_seen[gi] = t + j + 1
            t += len(idx)

            crossed = [(gi, int(v)) for gi in np.flatnonzero(n > n_before)
                       for v in thresholds if n_before[gi] < v <= n[gi]]
            if crossed:
                t0 = time.perf_counter()
                preds = metrics.evaluate(model, eval_set)
                eval_time += time.perf_counter() - t0
                confusion = metrics.group_confusion(preds, eval_set.groups, group_ids, eval_set.num_classes)
                balanced = metrics.balanced_group_score(confusion, group_label)
                acc = float((preds == eval_set.y.numpy()).mean())
                for gi, v in crossed:
                    row = confusion[gi]
                    fewshot_file.write(json.dumps({
                        "t": t, "group": int(group_ids[gi]), "threshold": v, "n": int(n[gi]), "q": int(q[gi]),
                        "balanced": float(balanced[gi]), "recall": float(row[group_label[gi]] / row.sum()),
                        "acc": acc, "confusion": confusion.tolist()}) + "\n")
                fewshot_file.flush()

            if next_ckpt < len(checkpoints) and t >= checkpoints[next_ckpt]:
                t0 = time.perf_counter()
                preds = metrics.evaluate(model, eval_set)
                eval_time += time.perf_counter() - t0
                correct = preds == eval_set.y.numpy()
                gacc = metrics.per_group_accuracy(correct, eval_set.groups, group_ids)
                confusion = metrics.group_confusion(preds, eval_set.groups, group_ids, eval_set.num_classes)
                age = np.where(last_seen >= 0, t - last_seen, -1)
                rec = {"t": t, "epoch": epoch, "grid": t in grid, "acc": float(correct.mean()),
                       "group_acc": gacc.tolist(), "confusion": confusion.tolist(), "n": n.tolist(), "q": q.tolist(), "age": age.tolist(),
                       "online_acc": online_correct / max(online_count, 1),
                       "train_time": train_time, "eval_time": eval_time}
                if hasattr(model, "buffer"):
                    rec.update(buffer_diagnostics(model.buffer, eval_set.num_classes))
                ckpt_file.write(json.dumps(rec) + "\n")
                ckpt_file.flush()
                log(f"t {t:>8d}  ep {epoch}  acc {rec['acc']:.4f}  online {rec['online_acc']:.4f}  "
                    f"train {train_time:.0f}s  eval {eval_time:.0f}s")
                online_correct = online_count = 0
                while next_ckpt < len(checkpoints) and t >= checkpoints[next_ckpt]:
                    next_ckpt += 1

    ckpt_file.close()
    fewshot_file.close()
    with open(os.path.join(args.out, "stream_stats.json"), "w") as f:
        json.dump(stream_stats, f)
    with open(os.path.join(args.out, "checkpoints.jsonl")) as f:
        records = [json.loads(line) for line in f]
    summary = metrics.summarize(records, group_label)
    with open(os.path.join(args.out, "fewshot.jsonl")) as f:
        summary["fewshot_triggered"] = metrics.summarize_fewshot([json.loads(line) for line in f])
    if hasattr(model, "updates"):
        summary["replay_updates"] = model.updates
    if hasattr(model, "buffer"):
        summary.update({f"buffer_{k}": getattr(model.buffer, k, None)
                        for k in ("offered", "written", "recognised", "evicted")})
    if getattr(model, "cbp", None) is not None:
        summary["cbp_replaced_units"] = model.cbp.replaced
    if hasattr(model, "memory_bytes"):
        summary["memory_bytes"] = model.memory_bytes()
    summary.update({"exposures": t, "memory_values": model.memory_values(), "train_time": train_time,
                    "eval_time": eval_time, "ms_per_step": 1000 * train_time / max(t // b, 1)})
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"final {summary['final_acc']:.4f}  anytime {summary['anytime_acc']}  "
        f"{summary['ms_per_step']:.2f} ms/step")


if __name__ == "__main__":
    main()
