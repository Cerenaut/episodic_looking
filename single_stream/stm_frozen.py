"""
STM unit experiments on a frozen LTM gap snapshot (plan.md section 5, draft STM validation order, step 1 and a
first RL check). Freezing the LTM removes the moving target, so these test the STM alone.

  snapshot: a BiasableTransformer trained iid on the frozen preprocessor's tokens of MNIST digits except `new_digit`,
            plus only `new_shots` images of `new_digit` mixed in (a few-shot gap). Saved as snapshot.pt.
  bound:    the label-supervised optimal per-image bias (Adam on the bias through the frozen LTM, bounded by the bias
            hook) on validation images: how much of the gap biasing can close (recall of the new digit, accuracy of
            the others).
  rl:       the RL STM (single_stream/stm.py) trained on an iid stream of the snapshot's own training images (with
            --new-fraction, that fraction of episodes drawn from the few-shot images, as a prioritised buffer would),
            evaluated (no label, policy mean) on validation images: per-step accuracy (within-episode curve), new-digit
            recall, others' accuracy, critic value vs outcome, cell utilisation.

Thinking time per sample (wave 9, Dave 3 Nov 2026): with --block b the repeats of a sample run in blocks of b, and
after each block the STM's answer on the sample is probed (policy mean, no learning; the label is the training
reward's, so this is allowed). --stop-p: stop the sample's repeats once the probed p(y) reaches it (adaptive
persistence; --repeats is then the cap); --patience n: also stop once the answer is right (p(y) >= 0.5) and n
blocks in a row brought no gain. --sigma-boost f: after a block with no gain (probed p(y) up by less than
--gain-eps) while p(y) < --boost-below, the policy std for this sample's next block is multiplied by f (up to
--sigma-max); a block with gain resets it. The interleaved replay episodes always use the base std.
--replay-priority a: seen-replay draws weighted by ((1 - p(y) at the image's last learning episode) + --replay-floor)^a
(0: uniform). --first-stop-p / --first-min-blocks: on an image's first presentation, stop at this p(y) instead of
--stop-p, and do not stop (by stop-p or patience) before this many blocks (wave 10: few-shot speed; uses only whether
this image has been seen before, as a replay buffer would).

Consolidation and a second phase (wave 16, Dave 5 Nov 2026; exploratory, so the sleep points are placed by the
experimenter, not by the model): --sleep-at-nines n1,n2 runs a sleep right after the n-th presentation of the new digit:
--sleep-steps Adam steps on the LTM (all weights, fresh optimiser, --sleep-lr), minibatches of --sleep-batch drawn from
the buffer (every distinct image presented so far) with weights ((1 - the LTM's own p(y)) + --sleep-floor)^a
(--sleep-priority a; 0: uniform), the drawn images' p(y) recomputed after each step. rl --sleep-reset-stm: the STM
restarts from its initial state after each sleep. --phase1-nines n ends the stream right after the n-th new-digit
sample (a prefix of the usual stream, so it reproduces earlier runs); --phase2-digit d appends --phase2-samples samples,
--phase2-fraction of them from --new-shots fixed training images of d (without replacement per pass), the rest iid
from the digits other than d and the new digit (which stops). Phase-2 metrics: mem2_seen (the shown images of d),
new2_recall (held-out d), others2_acc (digits other than d and the new digit); the new digit's keys measure retention.

Example:
  python -m single_stream.stm_frozen --out runs_ss/stm_frozen_20261101/snap9 snapshot
  python -m single_stream.stm_frozen --out runs_ss/stm_frozen_20261101/snap9 bound
  python -m single_stream.stm_frozen --out runs_ss/stm_frozen_20261101/snap9 rl --name where_term --episodes 20000
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from environment.mnist.mnist_model import MnistModel, MnistModelConfig
from model.transformer import BiasableTransformer
from single_stream import datasets
from single_stream.preprocessor import featurize, load_checkpoint
from single_stream.stm import RLSTM, grid_basis


def rng_neutral(fn):
    """Run a measurement without consuming the global torch RNG: the actor draws a policy sample on every forward
    pass, even when the mean is used, so without this the checkpoint schedule and --diag changed the training
    trajectory (found 3 Nov 2026; runs before then are one random realisation each)."""
    def wrapped(*a, **k):
        state = torch.get_rng_state()
        try:
            return fn(*a, **k)
        finally:
            torch.set_rng_state(state)
    return wrapped


def load_data(args):
    train, val = datasets.load("mnist", args.data_root, val_holdout=600, split_seed=0)
    pre = load_checkpoint(args.preproc)
    data_seed = getattr(args, "data_seed", None)
    rng = np.random.default_rng(args.seed if data_seed is None else data_seed)  # the few-shot images
    y = train.y.numpy()
    new = np.flatnonzero(y == args.new_digit)
    keep = np.concatenate([np.flatnonzero(y != args.new_digit), rng.choice(new, args.new_shots, replace=False)])
    keep = np.sort(keep)
    return featurize(pre, train.x[keep]), train.y[keep], featurize(pre, val.x), val.y


def per_digit(pred, y):
    return [float((pred[y == d] == d).float().mean()) for d in range(10)]


def summary(pred, y, new_digit):
    pd = per_digit(pred, y)
    others = (y != new_digit)
    return {"acc": float((pred == y).float().mean()), "new_recall": pd[new_digit],
            "others_acc": float((pred[others] == y[others]).float().mean()),
            "new_predicted_frac": float((pred == new_digit).float().mean()), "per_digit": pd,
            # digits 0-8 answered as the new digit: the share of the damage on them that is a false "9"
            "others_as_new": float((pred[others] == new_digit).float().mean())}


def make_net(args, mode):
    return BiasableTransformer(num_tokens=49, token_size=32, num_classes=10, d_model=args.d_model, heads=4,
                               bias_mode=mode, bias_scale=args.bias_scale)


def load_snapshot(args, mode):
    net = make_net(args, mode)
    state = torch.load(os.path.join(args.out, "snapshot.pt"), weights_only=True)
    net.load_state_dict(state)
    for p in net.parameters():
        p.requires_grad_(False)
    return net.eval()


def cmd_snapshot(args, xt, yt, xv, yv, log):
    torch.manual_seed(args.seed)
    net = make_net(args, "where")  # parameters do not depend on the bias mode
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.snapshot_epochs):
        net.train()
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]
            loss = F.cross_entropy(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            s = summary(net(xv).argmax(1), yv, args.new_digit)
        log(f"snapshot epoch {epoch}: {json.dumps({k: v for k, v in s.items() if k != 'per_digit'})}")
    torch.save(net.state_dict(), os.path.join(args.out, "snapshot.pt"))
    return s


def cmd_bound(args, xt, yt, xv, yv, log):
    out = {}
    for mode in ("where", "what"):
        net = load_snapshot(args, mode)
        a = torch.zeros(len(yv), net.bias_size, requires_grad=True)
        opt = torch.optim.Adam([a], lr=0.1)
        for _ in range(args.bound_steps):
            loss = F.cross_entropy(net(xv, a), yv)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            s0 = summary(net(xv).argmax(1), yv, args.new_digit)
            s = summary(net(xv, a).argmax(1), yv, args.new_digit)
        out[mode] = {"unbiased": s0, "optimal": s}
        log(f"bound {mode}: unbiased acc {s0['acc']:.4f} new {s0['new_recall']:.3f} | optimal acc {s['acc']:.4f} "
            f"new {s['new_recall']:.3f} others {s['others_acc']:.4f}")
    return out


def stream_order(args, yt):
    """The training samples presented, one per episode (rl) or per sample (ltm): iid over the snapshot's training
    images, with --new-fraction of them drawn from the few-shot images (a stand-in for a prioritised buffer).
    Identical for rl and ltm with the same seed, so their curves share the x-axis."""
    stream_seed = getattr(args, "stream_seed", None)
    g = torch.Generator().manual_seed((args.seed if stream_seed is None else stream_seed) + 7)
    shuffle = getattr(args, "new_order", "replace") == "shuffle"
    if shuffle and args.new_fraction > 0:
        # the other slots draw iid from the other digits only, so few-shot images arrive only via their own slots
        rest = torch.nonzero(yt != args.new_digit).flatten()
        order = rest[torch.randint(len(rest), (args.episodes,), generator=g)]
    else:
        order = torch.randint(len(yt), (args.episodes,), generator=g)
    if args.new_fraction > 0:
        new_idx = torch.nonzero(yt == args.new_digit).flatten()
        pick_new = torch.rand(args.episodes, generator=g) < args.new_fraction
        n_new = int(pick_new.sum())
        if shuffle:
            # without replacement: each few-shot slot gets a new image until all have been shown, then a reshuffled pass
            passes = (n_new + len(new_idx) - 1) // len(new_idx)
            seq = torch.cat([new_idx[torch.randperm(len(new_idx), generator=g)] for _ in range(passes)])[:n_new]
        else:
            seq = new_idx[torch.randint(len(new_idx), (n_new,), generator=g)]
        order[pick_new] = seq
    return phase_stream(args, yt, order)


def phase2_shots(args, yt):
    """The phase-2 digit's fixed training images (memorisation), chosen with the data seed (as the new digit's)."""
    if getattr(args, "phase2_digit", -1) < 0:
        return None
    data_seed = args.seed if args.data_seed is None else args.data_seed
    idx = torch.nonzero(yt == args.phase2_digit).flatten()
    rng = np.random.default_rng(data_seed + 1000)
    return idx[torch.from_numpy(np.sort(rng.choice(len(idx), args.new_shots, replace=False)))]


def phase_stream(args, yt, order):
    """--phase1-nines: the usual stream cut right after the n-th new-digit sample (a prefix: the same samples as the
    uncut runs). --phase2-digit: then --phase2-samples samples, --phase2-fraction from the phase-2 shots (without
    replacement per pass) and the rest iid from the digits other than it and the new digit."""
    if getattr(args, "phase1_nines", 0) > 0:
        nines = torch.nonzero(yt[order] == args.new_digit).flatten()
        if len(nines) < args.phase1_nines:
            raise ValueError(f"the stream has only {len(nines)} new-digit samples")
        order = order[:int(nines[args.phase1_nines - 1]) + 1]
    shots2 = phase2_shots(args, yt)
    if shots2 is None:
        return order
    stream_seed = args.seed if args.stream_seed is None else args.stream_seed
    g = torch.Generator().manual_seed(stream_seed + 17)
    m = args.phase2_samples
    rest = torch.nonzero((yt != args.new_digit) & (yt != args.phase2_digit)).flatten()
    order2 = rest[torch.randint(len(rest), (m,), generator=g)]
    pick = torch.rand(m, generator=g) < args.phase2_fraction
    n = int(pick.sum())
    passes = (n + len(shots2) - 1) // len(shots2)
    order2[pick] = torch.cat([shots2[torch.randperm(len(shots2), generator=g)] for _ in range(passes)])[:n]
    return torch.cat([order, order2])


def phase1_length(args, order, yt):
    """Samples in phase 1 (the whole stream without --phase2-digit)."""
    if getattr(args, "phase2_digit", -1) < 0:
        return len(order)
    return len(order) - args.phase2_samples


def sleep_points(args, order, yt):
    """Sample counts (1-based) right after the n-th new-digit presentation, for n in --sleep-at-nines."""
    if not getattr(args, "sleep_at_nines", ""):
        return set()
    nines = torch.nonzero(yt[order] == args.new_digit).flatten()
    return {int(nines[n - 1]) + 1 for n in map(int, args.sleep_at_nines.split(",")) if n <= len(nines)}


def sleep(args, net, buf, xt, yt, g):
    """Consolidation (Dave, 5 Nov 2026): --sleep-steps Adam steps on all LTM weights (a fresh optimiser), minibatches
    of --sleep-batch drawn with replacement from buf (the distinct images presented so far) with weights
    ((1 - p(y)) + floor)^a, p(y) the LTM's own (no STM bias), recomputed for the drawn images after each step, so
    learned images (mostly the many nines) give way to the rest."""
    params = list(net.parameters())
    req = [q.requires_grad for q in params]
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.sleep_lr)

    def p_y(idx):
        net.eval()
        with torch.no_grad():
            return torch.softmax(net(xt[idx]), -1)[torch.arange(len(idx)), yt[idx]]

    p = p_y(buf)
    p_start = float(p.mean())
    draws = torch.zeros(len(buf))
    for _ in range(args.sleep_steps):
        w = ((1 - p).clamp(min=0) + args.sleep_floor) ** args.sleep_priority
        k = torch.multinomial(w, args.sleep_batch, replacement=True, generator=g)
        net.train()
        loss = F.cross_entropy(net(xt[buf[k]]), yt[buf[k]])
        opt.zero_grad()
        loss.backward()
        opt.step()
        p[k] = p_y(buf[k])
        draws.index_add_(0, k, torch.ones(len(k)))
    for q, r in zip(params, req):
        q.requires_grad_(r)
    net.eval()
    new = yt[buf] == args.new_digit
    return {"buffer": len(buf), "buffer_new": int(new.sum()), "draws_new_frac": float(draws[new].sum() / draws.sum())
            if args.sleep_steps else None, "p_y_before": p_start, "p_y_after": float(p_y(buf).mean())}


def sleep_settings(args):
    """--sleep-sweep "lr:steps:priority,...": the sleep options for each setting (batch and floor from args)."""
    out = []
    for item in args.sleep_sweep.split(","):
        lr, steps, prio = item.split(":")
        out.append(argparse.Namespace(sleep_lr=float(lr), sleep_steps=int(steps), sleep_priority=float(prio),
                                      sleep_batch=args.sleep_batch, sleep_floor=args.sleep_floor,
                                      new_digit=args.new_digit))
    return out


def phase2_keys(args, pred, y, mem_pred, order, n, yt, pre=""):
    """Phase-2 metrics: held-out recall of the phase-2 digit, its shown training images, the digits other than it and
    the new digit."""
    d = args.phase2_digit
    shots2 = phase2_shots(args, yt)
    seen2 = torch.isin(shots2, order[:n])
    hit = (mem_pred == d).float()
    m = (y != args.new_digit) & (y != d)
    out = {pre + "new2_recall": float((pred[y == d] == d).float().mean()),
           pre + "mem2_seen": float(hit[seen2].mean()) if seen2.any() else None,
           pre + "others2_acc": float((pred[m] == y[m]).float().mean())}
    if not pre:
        p1 = phase1_length(args, order, yt)
        out.update({"phase2_samples": max(0, n - p1), "new2_samples": int((yt[order[p1:n]] == d).sum())})
    return out


def eval_checkpoints(args, order=None, yt=None):
    """Log-spaced sample counts, plus (--nine-checkpoints n) the sample count right after each of the first n
    presentations of the new digit in the stream, so recall is measured at every few-shot count from one shot."""
    c = set(np.unique(np.geomspace(args.first_checkpoint, args.episodes, args.checkpoints).astype(int)).tolist())
    if args.nine_checkpoints > 0 and order is not None:
        nines = torch.nonzero(yt[order] == args.new_digit).flatten()[:args.nine_checkpoints]
        c |= set((nines + 1).tolist())
        if getattr(args, "phase2_digit", -1) >= 0:
            p1 = phase1_length(args, order, yt)
            sevens = p1 + torch.nonzero(yt[order[p1:]] == args.phase2_digit).flatten()[:args.nine_checkpoints]
            c |= set((sevens + 1).tolist()) | {p1, len(order)}
            c = {x for x in c if x <= len(order)}
    return c


def evaluate_logits(args, logits, xt, yt, xe, ye, order, n):
    """The metrics of one checkpoint for a model given as a function x -> logits (ltm: the net; head: the head, or the
    net while the head is empty): the summary on the evaluation images, the new digit's training images (memorised =
    those shown so far, as in rl) and, with --phase2-digit, the phase-2 keys."""
    shots = torch.nonzero(yt == args.new_digit).flatten()
    with torch.no_grad():
        s = summary(logits(xe).argmax(1), ye, args.new_digit)
        hit = (logits(xt[shots]).argmax(1) == args.new_digit).float()
    s["new_train_recall"] = float(hit.mean())
    seen = torch.isin(shots, order[:n])  # training nines shown so far (memorised), as in rl
    s["shots_seen"] = int(seen.sum())
    s["mem_seen"] = float(hit[seen].mean()) if seen.any() else None
    s["mem_unseen"] = float(hit[~seen].mean()) if (~seen).any() else None
    s["samples"] = n
    s["new_samples"] = int((yt[order[:n]] == args.new_digit).sum())
    if args.phase2_digit >= 0:
        with torch.no_grad():
            s.update(phase2_keys(args, logits(xe).argmax(1), ye, logits(xt[phase2_shots(args, yt)]).argmax(1), order,
                                 n, yt))
    return s


def cmd_ltm(args, xt, yt, xv, yv, log):
    """The LTM's own learning from the same sample stream as `rl`: the snapshot trained on (AdamW, its lr), one
    minibatch of the last `ltm_batch` samples every `ltm_batch` samples (one sample-gradient per sample, as the
    replay LTM's batch 16 every 16). Evaluated at the same checkpoints on the same images.
    LTM-only regimes (Dave, 3 Nov 2026): --ltm-steps k optimiser steps per update; --ltm-replay m samples drawn
    uniformly from those presented so far added to each batch (the STM's seen replay); --ltm-optim sgd with
    --ltm-momentum; --ltm-train head: only the classifier head learns."""
    net = make_net(args, "where")
    net.load_state_dict(torch.load(os.path.join(args.out, "snapshot.pt"), weights_only=True))
    opt = ltm_optimiser(args, net)
    gr = torch.Generator().manual_seed(args.seed + 11)  # replay draws
    order = stream_order(args, yt)
    xe, ye = xv[:args.eval_images], yv[:args.eval_images]
    checkpoints = eval_checkpoints(args, order, yt)
    history = []

    def evaluate(n):
        net.eval()
        s = evaluate_logits(args, net, xt, yt, xe, ye, order, n)
        history.append(s)
        log(f"{args.name} samples {n:>6d}  acc {s['acc']:.4f}  mem {s['new_train_recall']:.3f}  new {s['new_recall']:.3f}  "
            f"others {s['others_acc']:.4f}")

    sleeps = sleep_points(args, order, yt)
    gs = torch.Generator().manual_seed(args.seed + 19)  # sleep draws
    evaluate(0)
    for n in range(1, len(order) + 1):
        ltm_update(args, net, opt, gr, order, xt, yt, n)
        if n in checkpoints or n in sleeps:
            evaluate(n)
        if n in sleeps:
            st = sleep(args, net, torch.unique(order[:n]), xt, yt, gs)
            evaluate(n)
            history[-1].update({"after_sleep": True, "sleep": st})
    sweep = []
    if args.sleep_sweep:  # sleeps at the end of the stream, each from the same state
        net_state = copy.deepcopy(net.state_dict())
        for setting in sleep_settings(args):
            st = sleep(setting, net, torch.unique(order), xt, yt, torch.Generator().manual_seed(args.seed + 19))
            evaluate(len(order))
            sweep.append({"setting": vars(setting), "sleep": st, "keep": history.pop()})
            net.load_state_dict(net_state)
    return {"history": history, "sleep_sweep": sweep}


def ltm_optimiser(args, net):
    params = net.head.parameters() if args.ltm_train == "head" else net.parameters()
    if args.ltm_optim == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.ltm_momentum)
    return torch.optim.AdamW(params, lr=args.lr)


def ltm_update(args, net, opt, gr, order, xt, yt, n):
    """The LTM's own learning after stream sample n (ltm, and rl --hybrid): every `ltm_every` samples (default
    `ltm_batch`), `ltm_steps` steps on the last `ltm_batch` stream samples plus `ltm_replay` drawn uniformly from those
    presented so far; unbiased forward pass, labels from the stream."""
    B, every = args.ltm_batch, args.ltm_every or args.ltm_batch
    if n % every:
        return
    idx = order[max(0, n - B):n]  # the last B samples (fewer at the start)
    if args.ltm_replay > 0:
        idx = torch.cat([idx, order[torch.randint(n, (args.ltm_replay,), generator=gr)]])
    net.train()
    for _ in range(args.ltm_steps):
        loss = F.cross_entropy(net(xt[idx]), yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()


def cmd_head(args, xt, yt, xv, yv, log):
    """
    Comparison heads (Dave 14 Nov 2026) on the snapshot's features, in place of the RL STM and with the same data and
    sleeps as rl: the LTM is frozen while awake; per stream sample the head learns the sample and --head-replay
    samples drawn uniformly from those presented so far (the seen replay); at --sleep-at-nines the LTM sleeps exactly
    as in rl and ltm (sleep()), and the head restarts from its initial state with --sleep-reset-stm (else it is kept,
    on the LTM's changed features). While the head is empty the model answers with the LTM. Same stream,
    checkpoints and metrics as ltm (evaluate_logits).
    """
    net = load_snapshot(args, "where")  # no bias is applied: the mode does not matter
    from single_stream.heads import make_head
    head = make_head(args, net.d_model, 10)
    initial = copy.deepcopy(head)
    gr = torch.Generator().manual_seed(args.seed + 11)  # replay draws
    order = stream_order(args, yt)
    xe, ye = xv[:args.eval_images], yv[:args.eval_images]
    checkpoints = eval_checkpoints(args, order, yt)
    history = []
    pending = [0.0]

    def features(x):
        net.eval()
        with torch.no_grad():
            return net.run(x)["features"]

    def logits(x):
        return net(x) if head.empty else head.logits(features(x))

    def evaluate(n):
        s = evaluate_logits(args, logits, xt, yt, xe, ye, order, n)
        s["head_empty"] = bool(head.empty)
        history.append(s)
        log(f"{args.name} samples {n:>6d}  acc {s['acc']:.4f}  mem {s['new_train_recall']:.3f}  new {s['new_recall']:.3f}  "
            f"others {s['others_acc']:.4f}")

    sleeps = sleep_points(args, order, yt)
    gs = torch.Generator().manual_seed(args.seed + 19)  # sleep draws, as ltm
    evaluate(0)
    for n in range(1, len(order) + 1):
        i = order[n - 1:n]
        head.fit(features(xt[i]), yt[i])
        pending[0] += args.head_replay
        while pending[0] >= 1:
            pending[0] -= 1
            j = order[torch.randint(n, (1,), generator=gr)]
            head.fit(features(xt[j]), yt[j])
        if n in checkpoints or n in sleeps:
            evaluate(n)
        if n in sleeps:
            st = sleep(args, net, torch.unique(order[:n]), xt, yt, gs)
            if args.sleep_reset_stm:
                head = copy.deepcopy(initial)
            evaluate(n)
            history[-1].update({"after_sleep": True, "sleep": st})
    return {"history": history}


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def episodes_per_sample(args):
    """Learning episodes per sample: the repeats, plus the interleaved episodes between bursts (fixed repeats; with
    --stop-p the actual count is recorded instead)."""
    return args.repeats + args.interleave * ((args.repeats - 1) // args.burst)


def cmd_rl(args, xt, yt, xv, yv, log):
    net = load_snapshot(args, args.bias_mode)
    if args.hybrid:
        # hybrid (Dave, 4 Nov 2026): the LTM also learns, slowly, from the stream (not from the STM's repeats or
        # replays): (2) the current input only (--ltm-batch 1), or (1) the current input plus --ltm-replay samples from
        # the stream so far (the STM's replay source; not consolidation, whose buffer is larger and persistent). It
        # updates on each new sample before the STM's repeats, so the STM works on the updated LTM. Its parameters need
        # gradients only inside ltm_update.
        ltm_params = list(net.head.parameters() if args.ltm_train == "head" else net.parameters())
        for p in ltm_params:
            p.requires_grad_(True)
        ltm_opt = ltm_optimiser(args, net)
        for p in ltm_params:
            p.requires_grad_(False)
        gr = torch.Generator().manual_seed(args.seed + 13)  # LTM replay draws, apart from the STM's
    torch.manual_seed(args.seed)
    config = MnistModelConfig(
        encoder_sparsity=args.sparsity, history_size=args.history, model_hidden_size=args.hidden,
        model_layers=args.layers, discount_factor=args.discount, returns=args.returns, reward_type=args.reward,
        loss_actor_scale=args.actor_scale, actor_updates=args.actor_updates, num_classes=10,
        policy_std=args.policy_std, reward_scale=args.reward_scale, actor_value_input=not args.no_value_input, input_conditioning=args.conditioning,
        mask_key=args.mask_key, recruit_rate=args.recruit_rate, recruit_idle_steps=args.recruit_idle,
        stm_critic_sparse_init=args.sparse_init in ("critic", "both"),
        stm_actor_sparse_init=args.sparse_init in ("actor", "both"), step_input=args.step_input, class_bonus=args.class_bonus, seed=args.seed)
    basis = None
    if args.bias_grid:
        if args.bias_mode != "where":
            raise ValueError("--bias-grid applies to the where bias (7x7 tokens)")
        basis = grid_basis(args.bias_grid)
    action_size = net.bias_size if basis is None else basis.shape[0]
    stm_model = MnistModel(config, encoded_size=net.d_model, bias_size=action_size)
    stm_initial = copy.deepcopy(stm_model)  # --sleep-reset-stm restarts from here
    stm = RLSTM(net, stm_model, steps=args.steps, lr=args.stm_lr, momentum=args.stm_momentum, bias_basis=basis,
                incremental=args.incremental)
    xe, ye = xv[:args.eval_images], yv[:args.eval_images]
    shots = torch.nonzero(yt == args.new_digit).flatten()  # the few-shot training images (memorisation)
    others_idx = torch.nonzero(yt != args.new_digit).flatten()
    gf = torch.Generator().manual_seed(args.seed + 11)
    history = []
    t0 = time.time()
    order = stream_order(args, yt)
    train_scores = []
    if args.group_size > 0 and args.block > 0:
        raise ValueError("--group-size and --block are not combined")
    probing = args.block > 0
    sigma0 = args.policy_std
    last_p = torch.ones(len(yt))   # p(y) at each image's last learning episode (replay priority)
    counts = {"learning_episodes": 0}
    repeats_done = []              # repeats run per sample

    def learn(i, sigma):
        """One learning episode on training image i with policy std sigma."""
        stm.model.policy_config.policy_std = sigma
        r = stm.episode(xt[i:i + 1], yt[i:i + 1], learn=True, defer=args.group_size > 0)
        stm.model.policy_config.policy_std = sigma0
        last_p[i] = float(torch.softmax(r["logits"][-1, 0], dim=-1)[yt[i]])
        counts["learning_episodes"] += 1
        return r

    def probe_p(i):
        with torch.no_grad():
            r = stm.episode(xt[i:i + 1], yt[i:i + 1])
        return float(torch.softmax(r["logits"][-1, 0], dim=-1)[yt[i]])

    @rng_neutral
    def evaluate(ep):
        with torch.no_grad():
            res = stm.episode(xe, ye)
        pred = res["logits"].argmax(2)
        step_acc = (pred == ye[None]).float().mean(1).tolist()
        s = summary(pred[-1], ye, args.new_digit)
        s0 = summary(pred[0], ye, args.new_digit)
        v = res["values"][-1]
        final = res["scores"][-1]
        with torch.no_grad():
            mem = stm.episode(xt[shots], yt[shots])["logits"][-1].argmax(1)
        rec_mem = float((mem == args.new_digit).float().mean())
        seen = torch.isin(shots, order[:ep])
        mem_hit = (mem == args.new_digit).float()
        rec = {"new_train_recall": rec_mem, "episodes": ep, "samples": ep, "learning_episodes": counts["learning_episodes"], "new_samples": int((yt[order[:ep]] == args.new_digit).sum()), "step_acc": step_acc, "new_recall_unbiased": s0["new_recall"],
               "new_recall": s["new_recall"], "others_acc": s["others_acc"], "acc": s["acc"],
               "per_digit": s["per_digit"], "others_as_new": s["others_as_new"],
               "step_score": res["scores"].mean(1).tolist(), "value_last": float(v.mean()),
               "value_final_corr": float(torch.corrcoef(torch.stack([v, final]))[0, 1]) if v.std() > 0 else 0.0,
               "bias_abs": float(res["bias"].abs().mean()),
               "train_final_score": float(np.mean(train_scores[-1000:])) if train_scores else None,
               "recruited": stm.recruited, "time": time.time() - t0,
               "shots_seen": int(seen.sum()),
               "mem_seen": float(mem_hit[seen].mean()) if seen.any() else None,
               "mem_unseen": float(mem_hit[~seen].mean()) if (~seen).any() else None,
               "actor_shared_norm": float(stm.param_vector("actor_shared").norm())}
        if args.phase2_digit >= 0:
            with torch.no_grad():
                mem2 = stm.episode(xt[phase2_shots(args, yt)], yt[phase2_shots(args, yt)])["logits"][-1].argmax(1)
            rec.update(phase2_keys(args, pred[-1], ye, mem2, order, ep, yt))
        if args.hybrid or args.sleep_at_nines or args.sleep_sweep:  # the LTM on its own (no bias): its contribution to the system's recall
            with torch.no_grad():
                lp = net(xe).argmax(1)
                lm = (net(xt[shots]).argmax(1) == args.new_digit).float()
            ls = summary(lp, ye, args.new_digit)
            rec.update({"ltm_new_recall": ls["new_recall"], "ltm_others_acc": ls["others_acc"], "ltm_acc": ls["acc"],
                        "ltm_per_digit": ls["per_digit"], "ltm_others_as_new": ls["others_as_new"],
                        "ltm_mem_seen": float(lm[seen].mean()) if seen.any() else None})
            if args.phase2_digit >= 0:
                with torch.no_grad():
                    lm2 = net(xt[phase2_shots(args, yt)]).argmax(1)
                rec.update(phase2_keys(args, lp, ye, lm2, order, ep, yt, pre="ltm_"))
        if config.encoder_sparsity > 0:
            for name, m in (("actor", stm.model.model_actor), ("critic", stm.model.model_critic)):
                rec.update({f"{name}_{k}": val for k, val in m.utilisation().items()})
        history.append(rec)
        log(f"{args.name} ep {ep:>6d}  step acc {' '.join(f'{a:.4f}' for a in step_acc)}  mem {rec_mem:.3f}  new {s0['new_recall']:.3f}"
            f"->{s['new_recall']:.3f}  others {s['others_acc']:.4f}  V {rec['value_last']:.3f} "
            f"corr {rec['value_final_corr']:.2f}  |b| {rec['bias_abs']:.2f}  seen {rec['shots_seen']} "
            f"mem seen/unseen {fmt(rec['mem_seen'])}/{fmt(rec['mem_unseen'])}  {rec['time']:.0f}s")

    @rng_neutral
    def probe(j):
        """The STM's current answer on training image j (policy mean, no learning): unbiased and final score p(y),
        final correctness, mean |bias|, the actor clique."""
        with torch.no_grad():
            r = stm.episode(xt[j:j + 1], yt[j:j + 1])
        p_y = torch.softmax(r["logits"][:, 0], dim=-1)[:, yt[j]]
        return {"score_0": float(r["scores"][0, 0]), "score_T": float(r["scores"][-1, 0]),
                "p_0": float(p_y[0]), "p_T": float(p_y[-1]),
                "correct_T": int(r["logits"][-1, 0].argmax() == yt[j]), "bias_abs": float(r["bias"][-1].abs().mean()),
                "clique_kept": float(torch.isin(r["active_actor"][-1][0], r["active_actor"][0][0]).float().mean())}, \
            r["active_actor"][0][0]

    def summarise(stats, episode_returns=None):
        """Mean of the per-update diagnostics over a block of episodes; return_std is the spread of the episode
        return over the block (the reward signal exploration produces), baseline_error the mean return - value."""
        out = {k: float(np.mean([u[k] for u in stats])) for k in stats[0]}
        g = np.array(episode_returns if episode_returns is not None else [u["return_0"] for u in stats])
        out["return_std"] = float(g.std())
        out["baseline_error"] = float(np.mean([u["return_0"] - u["value_0"] for u in stats]))
        return out

    # Per-sample diagnostics (the early-learning delay, Dave 1 Nov): one JSON line per sample presented
    diag = open(os.path.join(args.out, f"{args.name}_diag.jsonl"), "w") if args.diag else None
    actor_count = dict(stm.model.model_actor.named_buffers())["active_count"]
    presented = torch.zeros(len(yt), dtype=torch.long)

    checkpoints = eval_checkpoints(args, order, yt)
    def draw_fill(ep, n):
        """n replay images: between bursts of `burst` repeats, `interleave` learning episodes (replay of other
        material), so repeats are not all back to back."""
        if n <= 0:
            return order[:0]
        if args.interleave_source == "others":
            return others_idx[torch.randint(len(others_idx), (n,), generator=gf)]
        if args.interleave_source == "seen" and args.replay_priority > 0:  # prioritised by failure
            seen_idx = torch.unique(order[:ep])
            w = ((1 - last_p[seen_idx]).clamp(min=0) + args.replay_floor) ** args.replay_priority
            return seen_idx[torch.multinomial(w, n, replacement=True, generator=gf)]
        if args.interleave_source == "seen":  # replay: iid from the samples presented so far
            return order[torch.randint(ep, (n,), generator=gf)]
        # the replay mixture: iid training images, new_fraction of them few-shot images (any shot, including ones
        # not yet presented: a leak in the early region; use "seen")
        fill = torch.randint(len(yt), (n,), generator=gf)
        pick = torch.rand(n, generator=gf) < args.new_fraction
        fill[pick] = shots[torch.randint(len(shots), (int(pick.sum()),), generator=gf)]
        return fill

    sleeps = sleep_points(args, order, yt)
    gs = torch.Generator().manual_seed(args.seed + 19)  # sleep draws
    evaluate(0)
    for ep in range(1, len(order) + 1):
        j = order[ep - 1]
        if diag:
            pre, cells = probe(j)
            prior = actor_count[cells]
            w0 = {g: stm.param_vector(g) for g in stm.param_groups}
            target, other, target_returns = [], [], []
        if args.hybrid:  # the LTM learns from the new sample first, so the STM's repeats run on the updated LTM
            for p in ltm_params:
                p.requires_grad_(True)
            ltm_update(args, net, ltm_opt, gr, order, xt, yt, ep)
            for p in ltm_params:
                p.requires_grad_(False)
            net.eval()
        group = []  # deferred repeats of this sample (group-normalised advantage)
        sigma, sigmas = sigma0, []
        p_prev = probe_p(j) if probing else None
        k_done, flat = 0, 0   # repeats run; consecutive blocks without gain
        first = presented[j] == 0
        stop_p = args.first_stop_p if first and args.first_stop_p is not None else args.stop_p
        min_blocks = args.first_min_blocks if first else 0
        for k in range(args.repeats):  # up to r independent learning episodes per sample ("thinking time")
            if k > 0 and k % args.burst == 0:
                fill = draw_fill(ep, args.interleave)
                for f in fill:
                    learn(int(f), sigma0)
                    if diag:
                        other.append(stm.last_update)
            group_mode = args.group_size > 0
            res = learn(int(j), sigma)
            sigmas.append(sigma)
            k_done = k + 1
            if group_mode:
                group.append(res["deferred"])
                if len(group) == args.group_size or k == args.repeats - 1:
                    stm.group_update(group, std_floor=args.group_std_floor)
                    group = []
                    if diag:
                        target.append(stm.last_update)
            elif diag:
                target.append(stm.last_update)
            if diag:
                target_returns.append(float(res["returns"][0, 0]))
            if k == 0:
                train_scores.append(float(res["scores"][-1, 0]))
            if probing and (k + 1) % args.block == 0 and k + 1 < args.repeats:
                p_now = probe_p(j)
                no_gain = p_now - p_prev < args.gain_eps
                flat = flat + 1 if no_gain else 0
                if (k + 1) // args.block >= min_blocks and (
                        p_now >= stop_p or (args.patience and p_now >= 0.5 and flat >= args.patience)):
                    break
                if args.sigma_boost > 1:
                    stuck = no_gain and p_now < args.boost_below
                    sigma = min(sigma * args.sigma_boost, args.sigma_max) if stuck else sigma0
                p_prev = p_now
        if args.replay_per_sample > 0:  # top up to a fixed replay budget per sample, after the target repeats
            n_replayed = ((k_done - 1) // args.burst) * args.interleave
            for f in draw_fill(ep, max(0, args.replay_per_sample - n_replayed)):
                learn(int(f), sigma0)
                if diag:
                    other.append(stm.last_update)
        repeats_done.append(k_done)
        if diag:
            post, _ = probe(j)
            line = {"sample": ep, "index": int(j), "label": int(yt[j]), "new": int(yt[j] == args.new_digit),
                    "times_presented": int(presented[j]), "pre": pre, "post": post,
                    "repeats": k_done, "sigma_max_used": max(sigmas),
                    "clique_fresh": float((prior == 0).float().mean()), "clique_prior_mean": float(prior.float().mean()),
                    "target": summarise(target, target_returns), "interleaved": summarise(other) if other else None,
                    "weight_change": {g: float((stm.param_vector(g) - w0[g]).norm()) for g in w0}}
            diag.write(json.dumps(line) + "\n")
            diag.flush()
        presented[j] += 1
        if ep in checkpoints or ep in sleeps:
            evaluate(ep)
        if ep in sleeps:
            st = sleep(args, net, torch.unique(order[:ep]), xt, yt, gs)
            if args.sleep_reset_stm:  # the STM restarts; its replay priorities with it
                stm = RLSTM(net, copy.deepcopy(stm_initial), steps=args.steps, lr=args.stm_lr,
                            momentum=args.stm_momentum, bias_basis=basis, incremental=args.incremental)
                actor_count = dict(stm.model.model_actor.named_buffers())["active_count"]
                last_p.fill_(1.0)
            evaluate(ep)
            history[-1].update({"after_sleep": True, "sleep": st})
    sweep = []
    if args.sleep_sweep:  # sleeps at the end of the stream, each from the same state: with the STM kept, and reset
        net_state, stm_kept = copy.deepcopy(net.state_dict()), stm
        for setting in sleep_settings(args):
            st = sleep(setting, net, torch.unique(order), xt, yt, torch.Generator().manual_seed(args.seed + 19))
            out = {"setting": vars(setting), "sleep": st}
            for mode in ("keep", "reset"):
                stm = stm_kept if mode == "keep" else RLSTM(
                    net, copy.deepcopy(stm_initial), steps=args.steps, lr=args.stm_lr, momentum=args.stm_momentum,
                    bias_basis=basis, incremental=args.incremental)
                evaluate(len(order))
                out[mode] = history.pop()
            stm = stm_kept
            net.load_state_dict(net_state)
            sweep.append(out)
    n_episodes = counts["learning_episodes"]
    return {"history": history, "learning_episodes": n_episodes, "repeats_per_sample": repeats_done,
            "ms_per_episode": 1000 * (time.time() - t0) / n_episodes, "sleep_sweep": sweep}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["snapshot", "bound", "rl", "ltm", "head"])
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="rl")
    p.add_argument("--data-root", default=os.path.expanduser("~/dev/claude/mnist"))
    p.add_argument("--preproc", default="runs_ss/preproc/omniglot_conv_c32_s0/preproc.pt")
    p.add_argument("--new-digit", type=int, default=9)
    p.add_argument("--new-shots", type=int, default=16)
    p.add_argument("--snapshot-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--bias-scale", type=float, default=1.3863)
    p.add_argument("--bias-mode", default="where", choices=["where", "what"])
    p.add_argument("--step-input", action="store_true", help="rl: fraction of the episode remaining as an input")
    p.add_argument("--class-bonus", type=float, default=0.0, help="rl: reward per class newly reached in the episode")
    p.add_argument("--incremental", action="store_true", help="rl: actions are changes to a cumulative bias")
    p.add_argument("--bias-grid", type=int, default=0, help="rl: where-bias action on a coarse g x g grid (0 = 7x7)")
    p.add_argument("--bound-steps", type=int, default=200)
    p.add_argument("--ltm-batch", type=int, default=16)
    p.add_argument("--ltm-steps", type=int, default=1, help="ltm: optimiser steps per update")
    p.add_argument("--ltm-replay", type=int, default=0,
                   help="ltm: samples drawn uniformly from those presented so far, added to each update's batch")
    p.add_argument("--ltm-optim", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--ltm-momentum", type=float, default=0.5, help="ltm, sgd")
    p.add_argument("--ltm-train", choices=["all", "head"], default="all")
    p.add_argument("--episodes", type=int, default=20000, help="samples presented (rl: x repeats episodes)")
    p.add_argument("--repeats", type=int, default=1, help="rl: independent learning episodes per sample")
    p.add_argument("--interleave", type=int, default=0,
                   help="rl: learning episodes on iid other-digit images between consecutive repeats")
    p.add_argument("--burst", type=int, default=1,
                   help="rl: repeats back to back between interleaves (1: interleave between every two repeats)")
    p.add_argument("--interleave-source", default="others", choices=["others", "stream", "seen"],
                   help="rl: interleaved episodes on other digits only; on the replay mixture (new_fraction "
                        "few-shot images drawn from all shots, including ones not yet presented: leaks); or iid "
                        "replay of the samples presented so far (a replay buffer of the stream)")
    p.add_argument("--new-order", default="shuffle", choices=["shuffle", "replace"],
                   help="few-shot slots of the stream: without replacement (each slot a new image until all are shown; "
                        "default from wave 8) or with replacement (waves <= 7)")
    p.add_argument("--stream-seed", type=int, default=None, help="stream order seed (default: --seed)")
    p.add_argument("--hybrid", action="store_true",
                   help="rl: the LTM learns too, after each stream sample, with the ltm options (--lr, --ltm-batch, "
                        "--ltm-every, --ltm-steps, --ltm-replay, --ltm-optim, --ltm-train); the STM sees its current state")
    p.add_argument("--ltm-every", type=int, default=0,
                   help="ltm: one step every N samples on the last --ltm-batch samples (0: every --ltm-batch)")
    p.add_argument("--group-size", type=int, default=0,
                   help="rl: group-normalised advantage over this many consecutive repeats of a sample (0: critic "
                        "baseline, per-episode updates)")
    p.add_argument("--group-std-floor", type=float, default=1e-3)
    p.add_argument("--block", type=int, default=0,
                   help="rl: probe the STM's p(y) on the sample after every block of this many repeats (0: off)")
    p.add_argument("--stop-p", type=float, default=1.01,
                   help="rl, with --block: stop the sample's repeats once the probed p(y) reaches this (--repeats = cap)")
    p.add_argument("--sigma-boost", type=float, default=1.0,
                   help="rl, with --block: policy std x this after a block without gain (1: off)")
    p.add_argument("--sigma-max", type=float, default=2.0)
    p.add_argument("--gain-eps", type=float, default=0.02)
    p.add_argument("--boost-below", type=float, default=0.5,
                   help="rl: boost only while the probed p(y) is below this (default: while the answer is wrong)")
    p.add_argument("--patience", type=int, default=0,
                   help="rl, with --block: also stop once p(y) >= 0.5 and this many blocks in a row gained less than "
                        "--gain-eps (0: off)")
    p.add_argument("--first-stop-p", type=float, default=None,
                   help="rl, with --block: --stop-p for an image's first presentation (default: --stop-p)")
    p.add_argument("--first-min-blocks", type=int, default=0,
                   help="rl, with --block: on an image's first presentation, no early stop before this many blocks")
    p.add_argument("--replay-per-sample", type=int, default=0,
                   help="rl: replay episodes per sample regardless of repeats: those interleaved between repeats, "
                        "topped up after the sample's last repeat (0: off; decouples replay from adaptive repeats)")
    p.add_argument("--replay-priority", type=float, default=0.0,
                   help="rl, seen replay: weights ((1 - last p(y)) + floor)^a (0: uniform)")
    p.add_argument("--replay-floor", type=float, default=0.05)
    p.add_argument("--sleep-at-nines", default="", help="sleep (consolidation) right after these new-digit counts, e.g. 16,64")
    p.add_argument("--sleep-steps", type=int, default=0)
    p.add_argument("--sleep-lr", type=float, default=1e-4)
    p.add_argument("--sleep-batch", type=int, default=16)
    p.add_argument("--sleep-priority", type=float, default=1.0, help="sleep draw weights ((1 - p(y)) + floor)^a")
    p.add_argument("--sleep-floor", type=float, default=0.05)
    p.add_argument("--sleep-sweep", default="",
                   help="after the last sample, sleep with each of these settings from the same state (\"lr:steps:priority,...\"); "
                        "rl reads each with the STM kept and reset; the state is restored between settings")
    p.add_argument("--sleep-reset-stm", action="store_true",
                   help="rl: the STM restarts from its initial state after each sleep (head: the head does)")
    p.add_argument("--head", default="knn", choices=["knn", "fly", "sdmlp"], help="head: the comparison head")
    p.add_argument("--head-replay", type=float, default=1.0, help="head: seen-replay samples per stream sample")
    p.add_argument("--knn-k", type=int, default=1)
    p.add_argument("--knn-capacity", type=int, default=4096)
    p.add_argument("--fly-kc", type=int, default=2000)
    p.add_argument("--fly-response", type=int, default=16)
    p.add_argument("--fly-k-frac", type=float, default=0.05)
    p.add_argument("--fly-lr", type=float, default=0.2)
    p.add_argument("--sdmlp-neurons", type=int, default=1000)
    p.add_argument("--sdmlp-k", type=int, default=32)
    p.add_argument("--sdmlp-gaba", type=int, default=100)
    p.add_argument("--sdmlp-lr", type=float, default=0.05)
    p.add_argument("--phase1-nines", type=int, default=0, help="end phase 1 right after the n-th new-digit sample (0: off)")
    p.add_argument("--phase2-digit", type=int, default=-1, help="second phase: this digit at --phase2-fraction (-1: off)")
    p.add_argument("--phase2-samples", type=int, default=200)
    p.add_argument("--phase2-fraction", type=float, default=0.75)
    p.add_argument("--first-checkpoint", type=int, default=100)
    p.add_argument("--checkpoints", type=int, default=8)
    p.add_argument("--nine-checkpoints", type=int, default=0,
                   help="also evaluate right after each of the first n presentations of the new digit (0: off)")
    p.add_argument("--new-fraction", type=float, default=0.0,
                   help="rl: fraction of training episodes drawn from the new digit's few-shot images")
    p.add_argument("--eval-images", type=int, default=2000)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--hidden", type=int, default=1000)
    p.add_argument("--sparsity", type=int, default=32)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--stm-lr", type=float, default=0.01)
    p.add_argument("--stm-momentum", type=float, default=0.5)
    p.add_argument("--reward", default="terminal-accuracy")
    p.add_argument("--returns", default="mc")
    p.add_argument("--discount", type=float, default=1.0)
    p.add_argument("--policy-std", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="rl: accuracy score p(y)^s; s < 1 flattens it less where p(y) is small (s -> 0: ~log p(y))")
    p.add_argument("--actor-scale", type=float, default=0.01)
    p.add_argument("--actor-updates", type=int, default=1)
    p.add_argument("--no-value-input", action="store_true")
    p.add_argument("--conditioning", default="none")
    p.add_argument("--mask-key", default="first", choices=["first", "current", "history"])
    p.add_argument("--sparse-init", default="none", choices=["none", "critic", "actor", "both"])
    p.add_argument("--recruit-rate", type=float, default=0.0)
    p.add_argument("--recruit-idle", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=None,
                   help="which few-shot images (default: --seed); the snapshot was trained on data seed 0's, so STM "
                        "replicates on the same snapshot need --data-seed 0")
    p.add_argument("--diag", action="store_true", help="rl: per-sample diagnostics to <name>_diag.jsonl")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    result_path = os.path.join(args.out, f"{args.command if args.command in ('snapshot', 'bound') else args.name}.json")
    if os.path.exists(result_path):
        raise SystemExit(f"{result_path} exists; refusing to overwrite")
    logf = open(os.path.join(args.out, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"== {args.command} {json.dumps(vars(args))}")
    data = load_data(args)
    result = {"cmd_snapshot": cmd_snapshot, "cmd_bound": cmd_bound, "cmd_rl": cmd_rl, "cmd_ltm": cmd_ltm,
              "cmd_head": cmd_head}[f"cmd_{args.command}"](
        args, *data, log)
    with open(result_path, "w") as f:
        json.dump({"args": vars(args), "result": result}, f, indent=1)


if __name__ == "__main__":
    main()
