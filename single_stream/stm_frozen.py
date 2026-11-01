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

Example:
  python -m single_stream.stm_frozen --out runs_ss/stm_frozen_20261101/snap9 snapshot
  python -m single_stream.stm_frozen --out runs_ss/stm_frozen_20261101/snap9 bound
  python -m single_stream.stm_frozen --out runs_ss/stm_frozen_20261101/snap9 rl --name where_term --episodes 20000
"""
import argparse
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
from single_stream.stm import RLSTM


def load_data(args):
    train, val = datasets.load("mnist", args.data_root, val_holdout=600, split_seed=0)
    pre = load_checkpoint(args.preproc)
    rng = np.random.default_rng(args.seed)
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
            "new_predicted_frac": float((pred == new_digit).float().mean()), "per_digit": pd}


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


def cmd_rl(args, xt, yt, xv, yv, log):
    net = load_snapshot(args, args.bias_mode)
    torch.manual_seed(args.seed)
    config = MnistModelConfig(
        encoder_sparsity=args.sparsity, history_size=args.history, model_hidden_size=args.hidden,
        model_layers=args.layers, discount_factor=args.discount, returns=args.returns, reward_type=args.reward,
        loss_actor_scale=args.actor_scale, actor_updates=args.actor_updates, num_classes=10,
        policy_std=args.policy_std, actor_value_input=not args.no_value_input, input_conditioning=args.conditioning,
        mask_key=args.mask_key, recruit_rate=args.recruit_rate, recruit_idle_steps=args.recruit_idle,
        stm_critic_sparse_init=args.sparse_init in ("critic", "both"),
        stm_actor_sparse_init=args.sparse_init in ("actor", "both"), seed=args.seed)
    stm = RLSTM(net, MnistModel(config, encoded_size=net.d_model, bias_size=net.bias_size), steps=args.steps,
                lr=args.stm_lr, momentum=args.stm_momentum)
    g = torch.Generator().manual_seed(args.seed + 7)
    xe, ye = xv[:args.eval_images], yv[:args.eval_images]
    history = []
    t0 = time.time()
    order = torch.randint(len(yt), (args.episodes,), generator=g)
    if args.new_fraction > 0:
        # oversample the few-shot images (stand-in for a novelty-prioritised replay buffer)
        new_idx = torch.nonzero(yt == args.new_digit).flatten()
        pick_new = torch.rand(args.episodes, generator=g) < args.new_fraction
        order[pick_new] = new_idx[torch.randint(len(new_idx), (int(pick_new.sum()),), generator=g)]
    train_scores = []

    def evaluate(ep):
        with torch.no_grad():
            res = stm.episode(xe, ye)
        pred = res["logits"].argmax(2)
        step_acc = (pred == ye[None]).float().mean(1).tolist()
        s = summary(pred[-1], ye, args.new_digit)
        s0 = summary(pred[0], ye, args.new_digit)
        v = res["values"][-1]
        final = res["scores"][-1]
        rec = {"episodes": ep, "step_acc": step_acc, "new_recall_unbiased": s0["new_recall"],
               "new_recall": s["new_recall"], "others_acc": s["others_acc"], "acc": s["acc"],
               "step_score": res["scores"].mean(1).tolist(), "value_last": float(v.mean()),
               "value_final_corr": float(torch.corrcoef(torch.stack([v, final]))[0, 1]) if v.std() > 0 else 0.0,
               "bias_abs": float(res["bias"].abs().mean()),
               "train_final_score": float(np.mean(train_scores[-1000:])) if train_scores else None,
               "recruited": stm.recruited, "time": time.time() - t0}
        if config.encoder_sparsity > 0:
            for name, m in (("actor", stm.model.model_actor), ("critic", stm.model.model_critic)):
                rec.update({f"{name}_{k}": val for k, val in m.utilisation().items()})
        history.append(rec)
        log(f"{args.name} ep {ep:>6d}  step acc {' '.join(f'{a:.4f}' for a in step_acc)}  new {s0['new_recall']:.3f}"
            f"->{s['new_recall']:.3f}  others {s['others_acc']:.4f}  V {rec['value_last']:.3f} "
            f"corr {rec['value_final_corr']:.2f}  |b| {rec['bias_abs']:.2f}  {rec['time']:.0f}s")

    checkpoints = set(np.unique(np.geomspace(100, args.episodes, 8).astype(int)).tolist())
    evaluate(0)
    for ep in range(1, args.episodes + 1):
        j = order[ep - 1]
        res = stm.episode(xt[j:j + 1], yt[j:j + 1], learn=True)
        train_scores.append(float(res["scores"][-1, 0]))
        if ep in checkpoints:
            evaluate(ep)
    return {"history": history, "ms_per_episode": 1000 * (time.time() - t0) / args.episodes}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["snapshot", "bound", "rl"])
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
    p.add_argument("--bound-steps", type=int, default=200)
    p.add_argument("--episodes", type=int, default=20000)
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
    p.add_argument("--actor-scale", type=float, default=0.01)
    p.add_argument("--actor-updates", type=int, default=1)
    p.add_argument("--no-value-input", action="store_true")
    p.add_argument("--conditioning", default="none")
    p.add_argument("--mask-key", default="first")
    p.add_argument("--sparse-init", default="none", choices=["none", "critic", "actor", "both"])
    p.add_argument("--recruit-rate", type=float, default=0.0)
    p.add_argument("--recruit-idle", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    result_path = os.path.join(args.out, f"{args.command if args.command != 'rl' else args.name}.json")
    if os.path.exists(result_path):
        raise SystemExit(f"{result_path} exists; refusing to overwrite")
    logf = open(os.path.join(args.out, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"== {args.command} {json.dumps(vars(args))}")
    data = load_data(args)
    result = {"cmd_snapshot": cmd_snapshot, "cmd_bound": cmd_bound, "cmd_rl": cmd_rl}[f"cmd_{args.command}"](
        args, *data, log)
    with open(result_path, "w") as f:
        json.dump({"args": vars(args), "result": result}, f, indent=1)


if __name__ == "__main__":
    main()
