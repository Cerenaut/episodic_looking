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
import copy
import json
import os
import sys
import time

import numpy as np
import torch

from single_stream import datasets, metrics
from single_stream.buffers import NeuralBuffer, NoveltyGate, PriorityBuffer, RingBuffer, part_specs
from single_stream.models import (BufferedReplayLTM, CLSModel, ContinualBackprop, LTMOnly, ReplayLTM, ReplayOnlyLTM,
                                  build_net)
from single_stream.stream import ClusteredStream, run_lengths


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def build_parser(description=__doc__):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=sorted(datasets.LOADERS))
    p.add_argument("--data-root", default=os.path.expanduser("~/dev/claude/mnist"))
    p.add_argument("--permutation-seed", type=int, default=None, help="fixed pixel permutation (permuted MNIST)")
    p.add_argument("--val-holdout", type=int, default=0,
                   help="samples per group held out of training as the evaluation set; 0 = full train, test set")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--cluster-size", type=int, default=1, help="c")
    p.add_argument("--batch-size", type=int, default=1, help="b")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--model", default="ltm", choices=["ltm", "replay", "replayonly", "buffered", "cls"],
                   help="ltm: trained on the stream; replay: stream + reservoir replay mixed into each step; "
                        "replayonly: trained only from the replay buffer; buffered: replayonly with a pluggable, "
                        "novelty-gated buffer (--buffer); cls: buffered LTM (--net transformer) + RL STM (--stm-*)")
    p.add_argument("--net", default="mlp", choices=["mlp", "resmlp", "transformer"])
    p.add_argument("--preproc", default="",
                   help="frozen preprocessor checkpoint (single_stream.preprocessor): inputs become its tokens")
    p.add_argument("--d-model", type=int, default=64, help="transformer")
    p.add_argument("--heads", type=int, default=4, help="transformer")
    p.add_argument("--mlp-ratio", type=int, default=2, help="transformer")
    p.add_argument("--bias-mode", default="where", choices=["where", "what"], help="transformer bias hook")
    p.add_argument("--bias-scale", type=float, default=1.3863, help="transformer bias bound s (default ln 4)")
    p.add_argument("--bias-block", type=int, default=0, help="transformer block the bias acts on")
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"], help="LTM optimizer")
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
    p.add_argument("--replay-live", action="store_true",
                   help="buffered: each update also trains on the live sample (batch = it + --replay-batch draws); "
                        "with --replay-period 1 this is the snap9 'balanced' LTM regime")
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
    p.add_argument("--stm-steps", type=int, default=4, help="T: bias steps per STM episode")
    p.add_argument("--stm-history", type=int, default=4, help="STM observation history window")
    p.add_argument("--stm-hidden", type=int, default=1000)
    p.add_argument("--stm-sparsity", type=int, default=32, help="k of the sparse clique; 0 = dense")
    p.add_argument("--stm-layers", type=int, default=3)
    p.add_argument("--stm-lr", type=float, default=0.01)
    p.add_argument("--stm-momentum", type=float, default=0.5)
    p.add_argument("--stm-reward", default="terminal-accuracy",
                   choices=["terminal-accuracy", "accuracy", "accuracy-improvement", "entropy", "entropy-improvement"])
    p.add_argument("--stm-returns", default="mc", choices=["mc", "td"])
    p.add_argument("--stm-discount", type=float, default=1.0)
    p.add_argument("--stm-policy-std", type=float, default=0.5)
    p.add_argument("--stm-actor-scale", type=float, default=0.01)
    p.add_argument("--stm-actor-updates", type=int, default=1)
    p.add_argument("--stm-mask-key", default="first", choices=["first", "current", "history"])
    p.add_argument("--stm-conditioning", default="none", choices=["none", "centre", "centre-scale", "layernorm"])
    p.add_argument("--stm-no-value-input", action="store_true", help="actor does not see the critic's values")
    p.add_argument("--stm-eval-bias", default="mean", choices=["mean", "sample"])
    p.add_argument("--stm-repeats", type=int, default=1, help="r: independent episodes per live sample")
    p.add_argument("--stm-replay-ratio", type=float, default=0.0, help="STM replay episodes per exposure")
    p.add_argument("--stm-no-live", action="store_true", help="STM learns from replay episodes only")
    p.add_argument("--stm-block", type=int, default=0,
                   help="adaptive repeats: probe the STM's answer every b episodes (0 = fixed --stm-repeats)")
    p.add_argument("--stm-stop-p", type=float, default=1.0, help="adaptive: stop once the probed p(y) reaches this")
    p.add_argument("--stm-patience", type=int, default=0,
                   help="adaptive: also stop once right and this many blocks in a row gained < --stm-gain-eps")
    p.add_argument("--stm-gain-eps", type=float, default=0.02)
    p.add_argument("--stm-interleave", type=int, default=0,
                   help="STM replay episodes (from the LTM's buffer) before each repeat after the first")
    p.add_argument("--stm-buffer", default="shared", choices=["shared", "ring", "priority"],
                   help="STM replay source: the LTM's buffer (shared), or its own FIFO (ring) or priority buffer "
                        "(--alpha-evict / --alpha-replay as the LTM's), never wiped")
    p.add_argument("--stm-buffer-capacity", type=int, default=256)
    p.add_argument("--stm-delay", type=int, default=0,
                   help="cls: exposures before the STM comes online (until then the model is its LTM, sleeps as usual)")
    p.add_argument("--sleep-reset-stm", action="store_true",
                   help="cls: the STM restarts from its initial state after each LTM replay update (a sleep: every "
                        "--replay-period exposures, --replay-steps steps)")
    p.add_argument("--stm-sparse-init", default="none", choices=["none", "critic", "actor", "both"],
                   help="sparsity-aware init (layers after the masked one scaled by sqrt(H/k), hidden biases 0)")
    p.add_argument("--stm-step-input", action="store_true", help="fraction of the episode remaining as an STM input")
    p.add_argument("--stm-class-bonus", type=float, default=0.0, help="STM reward per class newly reached")
    p.add_argument("--stm-incremental", action="store_true", help="STM actions are changes to a cumulative bias")
    p.add_argument("--stm-recruit-rate", type=float, default=0.0, help="idle-cell recruitment; 0 = off")
    p.add_argument("--stm-recruit-idle", type=int, default=10000)
    p.add_argument("--stm-recruit-prob", type=float, default=1.0)
    p.add_argument("--stm-diag-images", type=int, default=1000,
                   help="evaluation images for the STM diagnostics at each checkpoint (labels used by the harness only)")
    p.add_argument("--n-log-checkpoints", type=int, default=12)
    p.add_argument("--fewshot-thresholds", default="1,2,4,8,16,32,64,128,256,512",
                   help="evaluate when a group's exposure count first reaches each of these ('' = off)")
    p.add_argument("--mem-images", type=int, default=256,
                   help="few-shot triggers: Memorisation = accuracy on up to this many of the group's training "
                        "samples presented so far (the most recent)")
    p.add_argument("--grid-per-epoch", type=int, default=10)
    p.add_argument("--max-exposures", type=int, default=0, help="stop early (smoke tests and timing only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    return p


def build_model(args, train, token_shape=None):
    widths = [int(w) for w in args.widths.split(",")]
    transformer = None
    if args.net == "transformer":
        if token_shape is None:
            raise ValueError("--net transformer needs --preproc (tokens)")
        transformer = dict(num_tokens=token_shape[0], token_size=token_shape[1], d_model=args.d_model,
                           heads=args.heads, mlp_ratio=args.mlp_ratio, bias_block=args.bias_block,
                           bias_mode=args.bias_mode, bias_scale=args.bias_scale)
    net = build_net(args.net, train.x.shape[1], train.num_classes, widths, args.blocks, transformer)
    cbp = None
    if args.cbp_rate > 0:
        if args.net != "mlp":
            raise ValueError("continual backprop is implemented for --net mlp only")
        cbp = ContinualBackprop(net, args.cbp_rate, args.cbp_decay, args.cbp_maturity)
    common = dict(lr=args.lr, steps=args.steps, momentum=args.momentum, weight_decay=args.weight_decay, cbp=cbp)
    if args.model == "ltm":
        return LTMOnly(net, optimizer=args.optimizer, **common)
    if args.model == "replayonly":
        if args.steps != 1:
            raise ValueError("replayonly uses --replay-steps, not --steps")
        del common["steps"]
        return ReplayOnlyLTM(net, capacity=args.replay_capacity, batch=args.replay_batch, period=args.replay_period,
                             steps=args.replay_steps, policy=args.replay_policy, store_uint8=not args.replay_float,
                             input_shape=train.x.shape[1:], seed=args.seed, **common)
    if args.model in ("buffered", "cls"):
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
        ltm = BufferedReplayLTM(net, buffer=buffer, batch=args.replay_batch, period=args.replay_period,
                                steps=args.replay_steps, optimizer=args.optimizer, live=args.replay_live, **common)
        if args.model == "buffered":
            return ltm
        if args.replay_live:
            raise ValueError("cls: the LTM learns only in sleep (no --replay-live)")
        stm = build_stm(args, net, train.num_classes)
        initial = copy.deepcopy(stm.model)  # --sleep-reset-stm restarts from here
        stm_buffer = None
        if args.stm_buffer == "ring":
            stm_buffer = RingBuffer(args.stm_buffer_capacity, parts, seed=args.seed + 1)
        elif args.stm_buffer == "priority":
            stm_buffer = PriorityBuffer(args.stm_buffer_capacity, parts, alpha_evict=args.alpha_evict,
                                        alpha_replay=args.alpha_replay, seed=args.seed + 1)

        def stm_factory():
            from single_stream.stm import RLSTM
            return RLSTM(net, copy.deepcopy(initial), steps=args.stm_steps, lr=args.stm_lr, momentum=args.stm_momentum,
                         incremental=args.stm_incremental)
        return CLSModel(ltm, stm, repeats=args.stm_repeats, replay_ratio=args.stm_replay_ratio,
                        learn_live=not args.stm_no_live, block=args.stm_block, stop_p=args.stm_stop_p,
                        patience=args.stm_patience, gain_eps=args.stm_gain_eps, interleave=args.stm_interleave,
                        reset_on_sleep=args.sleep_reset_stm, stm_factory=stm_factory, stm_delay=args.stm_delay,
                        stm_buffer=stm_buffer)
    return ReplayLTM(net, capacity=args.replay_capacity, replay_batch=args.replay_batch,
                     input_shape=train.x.shape[1:], seed=args.seed, **common)


def build_stm(args, net, num_classes):
    from environment.mnist.mnist_model import MnistModel, MnistModelConfig
    from single_stream.stm import RLSTM

    if args.net != "transformer":
        raise ValueError("--model cls needs --net transformer (the biasable LTM)")
    config = MnistModelConfig(
        encoder_sparsity=args.stm_sparsity, history_size=args.stm_history, model_hidden_size=args.stm_hidden,
        model_layers=args.stm_layers, discount_factor=args.stm_discount, returns=args.stm_returns,
        reward_type=args.stm_reward, loss_actor_scale=args.stm_actor_scale, actor_updates=args.stm_actor_updates,
        num_classes=num_classes, policy_std=args.stm_policy_std, eval_bias=args.stm_eval_bias,
        actor_value_input=not args.stm_no_value_input, input_conditioning=args.stm_conditioning,
        mask_key=args.stm_mask_key, recruit_rate=args.stm_recruit_rate, recruit_idle_steps=args.stm_recruit_idle,
        recruit_prob=args.stm_recruit_prob, stm_critic_sparse_init=args.stm_sparse_init in ("critic", "both"),
        stm_actor_sparse_init=args.stm_sparse_init in ("actor", "both"), step_input=args.stm_step_input, class_bonus=args.stm_class_bonus, seed=args.seed)
    model = MnistModel(config, encoded_size=net.d_model, bias_size=net.bias_size)
    return RLSTM(net, model, steps=args.stm_steps, lr=args.stm_lr, momentum=args.stm_momentum,
                 incremental=args.stm_incremental)


@torch.no_grad()
def stm_diagnostics(model, data, n):
    """
    Harness-side diagnostics of the STM on the first n evaluation images (labels used here only, never by the
    model): per-step accuracy and p(y) of the LTM output (step 0 = unbiased: the within-episode curve), the mean
    critic value per step, its correlation with the episode's final score, the mean |bias| and cell utilisation.
    """
    x, y = data.x[:n], data.y[:n]
    res = model.stm.episode(x, y)
    acc = (res["logits"].argmax(2) == y[None]).float().mean(1)
    final = res["scores"][-1]
    v = res["values"]
    corr = [float(torch.corrcoef(torch.stack([v[t], final]))[0, 1]) if v[t].std() > 0 else 0.0
            for t in range(len(v))]
    out = {"stm_step_acc": acc.tolist(), "stm_step_score": res["scores"].mean(1).tolist(),
           "stm_step_value": v.mean(1).tolist(), "stm_value_final_corr": corr,
           "stm_bias_abs": res["bias"].abs().mean((1, 2)).tolist(),
           "stm_episodes": model.stm.episodes + model.stm_episodes_before,
           "stm_live_episodes": model.live_episodes, "stm_replay_episodes": model.replay_episodes,
           "stm_resets": model.resets, "stm_recruited": model.stm.recruited}
    m = model.stm.model
    if m.config.encoder_sparsity > 0:
        for name, net in (("actor", m.model_actor), ("critic", m.model_critic)):
            out.update({f"stm_{name}_{k}": val for k, val in net.utilisation().items()})
    return out


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


class rng_neutral:
    """Evaluation must not consume the global torch RNG (the STM's actor draws a policy sample on every forward pass,
    even when the mean is used), so the training trajectory does not depend on the evaluation schedule."""

    def __enter__(self):
        self.state = torch.get_rng_state()

    def __exit__(self, *exc):
        torch.set_rng_state(self.state)


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
    token_shape, preproc_values = None, 0
    if args.preproc:
        from single_stream.preprocessor import featurize, load_checkpoint
        pre = load_checkpoint(args.preproc)
        train.x, eval_set.x = featurize(pre, train.x), featurize(pre, eval_set.x)
        token_shape = (49, pre.channels)
        preproc_values = sum(p.numel() for p in pre.parameters())
    model = build_model(args, train, token_shape)
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
    seen = [[] for _ in group_ids]  # training samples presented, per group (Memorisation; harness bookkeeping)
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
                seen[gi].append(int(k))
            t += len(idx)

            crossed = [(gi, int(v)) for gi in np.flatnonzero(n > n_before)
                       for v in thresholds if n_before[gi] < v <= n[gi]]
            if crossed:
                t0 = time.perf_counter()
                rng = rng_neutral()
                rng.__enter__()
                variants = metrics.evaluate_variants(model, eval_set)
                scored = {}
                for name, vp in variants.items():
                    vc = metrics.group_confusion(vp, eval_set.groups, group_ids, eval_set.num_classes)
                    scored[name] = (vc, metrics.balanced_group_score(vc, group_label),
                                    float((vp == eval_set.y.numpy()).mean()))
                confusion, balanced, acc = scored["main"]
                hits = np.array([confusion[g, group_label[g]] for g in range(len(group_ids))], dtype=float)
                totals = confusion.sum(1).astype(float)
                for gi, v in crossed:
                    row = confusion[gi]
                    others = np.arange(len(group_ids)) != gi
                    mem_idx = torch.tensor(seen[gi][-args.mem_images:])
                    with torch.no_grad():
                        mem_pred = model.predict(train.x[mem_idx]).argmax(1)
                    rec = {"t": t, "group": int(group_ids[gi]), "threshold": v, "n": int(n[gi]), "q": int(q[gi]),
                           "balanced": float(balanced[gi]), "recall": float(row[group_label[gi]] / row.sum()),
                           # Memorisation: the group's own training samples presented so far (most recent first);
                           # Protection: accuracy on the other groups' evaluation images
                           "memorisation": float((mem_pred == train.y[mem_idx]).float().mean()),
                           "protection": float(hits[others].sum() / totals[others].sum()),
                           "acc": acc, "confusion": confusion.tolist()}
                    for name, (vc, vb, va) in scored.items():
                        if name != "main":
                            rec[f"{name}_balanced"] = float(vb[gi])
                            rec[f"{name}_recall"] = float(vc[gi, group_label[gi]] / vc[gi].sum())
                            rec[f"{name}_acc"] = va
                    fewshot_file.write(json.dumps(rec) + "\n")
                fewshot_file.flush()
                rng.__exit__()
                eval_time += time.perf_counter() - t0

            if next_ckpt < len(checkpoints) and t >= checkpoints[next_ckpt]:
                t0 = time.perf_counter()
                with rng_neutral():
                    variants = metrics.evaluate_variants(model, eval_set)
                preds = variants["main"]
                eval_time += time.perf_counter() - t0
                correct = preds == eval_set.y.numpy()
                gacc = metrics.per_group_accuracy(correct, eval_set.groups, group_ids)
                confusion = metrics.group_confusion(preds, eval_set.groups, group_ids, eval_set.num_classes)
                age = np.where(last_seen >= 0, t - last_seen, -1)
                rec = {"t": t, "epoch": epoch, "grid": t in grid, "acc": float(correct.mean()),
                       "group_acc": gacc.tolist(), "confusion": confusion.tolist(), "n": n.tolist(), "q": q.tolist(), "age": age.tolist(),
                       "online_acc": online_correct / max(online_count, 1),
                       "train_time": train_time, "eval_time": eval_time}
                for name, vp in variants.items():
                    if name != "main":
                        rec[f"{name}_acc"] = float((vp == eval_set.y.numpy()).mean())
                if hasattr(model, "buffer"):
                    rec.update(buffer_diagnostics(model.buffer, eval_set.num_classes))
                if isinstance(model, CLSModel):
                    rec["stm_online"] = model.online
                    with rng_neutral():
                        rec.update(stm_diagnostics(model, eval_set, args.stm_diag_images))
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
    if isinstance(model, CLSModel):
        summary.update({"stm_episodes": model.stm.episodes + model.stm_episodes_before,
                        "stm_live_episodes": model.live_episodes, "stm_replay_episodes": model.replay_episodes,
                        "stm_resets": model.resets, "stm_values": model.stm.num_values()})
    summary.update({"exposures": t, "memory_values": model.memory_values(), "preprocessor_values": preproc_values,
                    "train_time": train_time,
                    "eval_time": eval_time, "ms_per_step": 1000 * train_time / max(t // b, 1)})
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"final {summary['final_acc']:.4f}  anytime {summary['anytime_acc']}  "
        f"{summary['ms_per_step']:.2f} ms/step")


if __name__ == "__main__":
    main()
