"""
How far is each task from the original, for an LTM trained on the original (Dave, 5 Nov 2026: steps (a) and (b))?

An LTM (the transformer of the clustered-stream runs, on the frozen preprocessor's tokens) is trained iid on the
original MNIST training split (AdamW, batch 16, --epochs). Then, for each task (tasks.py: the 8 dihedral transforms and
--lperms label permutations), on the task's validation split:

  zero_shot       accuracy of the original-trained LTM, unchanged;
  shared_bias     the best single "what" bias (one 64-vector for every image; Adam with the labels on --fit-images
                  training images of the task), evaluated on validation: what a fixed STM correction could recover;
  image_bias      the label-supervised optimal bias for each validation image on its own (as the snap9 bound): what
                  biasing can do with perfect knowledge, image by image (an upper bound, not a policy);
  head_probe      the LTM's classifier head retrained on the task (--fit-images training images), features frozen:
                  how much of the task the LTM's features still carry.

    python -m single_stream.transform_probe --out runs_ss/mnist_clustered_single_stream/transform_probe
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model.transformer import BiasableTransformer
from single_stream import datasets, tasks
from single_stream.preprocessor import load_checkpoint


def acc(logits, y):
    return float((logits.argmax(1) == y).float().mean())


def batched(net, x, bias=None, b=2000):
    with torch.no_grad():
        return torch.cat([net(x[i:i + b], None if bias is None else bias.expand(len(x[i:i + b]), -1))
                          for i in range(0, len(x), b)])


def fit_shared_bias(net, x, y, steps, lr=0.05, batch=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.zeros(1, net.bias_size, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    for _ in range(steps):
        idx = torch.randint(len(y), (batch,), generator=g)
        loss = F.cross_entropy(net(x[idx], a.expand(batch, -1)), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return a.detach()


def image_bias(net, x, y, steps=200, lr=0.1):
    a = torch.zeros(len(y), net.bias_size, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(net(x, a), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return acc(net(x, a), y)


def head_probe(net, xtr, ytr, xv, yv, epochs, lr=1e-3, seed=0):
    with torch.no_grad():
        ftr = torch.cat([net.run(xtr[i:i + 2000])["features"] for i in range(0, len(xtr), 2000)])
        fv = torch.cat([net.run(xv[i:i + 2000])["features"] for i in range(0, len(xv), 2000)])
    head = torch.nn.Linear(ftr.shape[1], 10)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(head(ftr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return acc(head(fv), yv)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default=os.path.expanduser("~/dev/claude/mnist"))
    p.add_argument("--preproc", default="runs_ss/preproc/omniglot_conv_c32_s0/preproc.pt")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lperms", type=int, default=3)
    p.add_argument("--fit-images", type=int, default=5000)
    p.add_argument("--bias-steps", type=int, default=500)
    p.add_argument("--image-bias-images", type=int, default=1000)
    p.add_argument("--probe-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train_raw, val_raw = datasets.load("mnist", args.data_root, val_holdout=600, split_seed=0)
    pre = load_checkpoint(args.preproc)
    t0 = time.time()

    net = BiasableTransformer(num_tokens=49, token_size=pre.channels, num_classes=10, d_model=64, heads=4,
                              bias_mode="what", bias_scale=1.3863)
    tr = tasks.featurized("id", train_raw, pre)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(tr.y), generator=g)
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]
            loss = F.cross_entropy(net(tr.x[idx]), tr.y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    torch.save(net.state_dict(), os.path.join(args.out, "ltm_original.pt"))

    names = list(tasks.DIHEDRAL) + [f"lperm{k}" for k in range(1, args.lperms + 1)]
    out = {"args": vars(args), "tasks": {}}
    rng = np.random.default_rng(args.seed)
    for name in names:
        trk = tr if name == "id" else tasks.featurized(name, train_raw, pre)
        vk = tasks.featurized(name, val_raw, pre)
        fit = torch.from_numpy(rng.choice(len(trk.y), args.fit_images, replace=False))
        r = {"zero_shot": acc(batched(net, vk.x), vk.y)}
        a = fit_shared_bias(net, trk.x[fit], trk.y[fit], args.bias_steps, seed=args.seed)
        r["shared_bias"] = acc(batched(net, vk.x, a), vk.y)
        r["image_bias"] = image_bias(net, vk.x[:args.image_bias_images], vk.y[:args.image_bias_images])
        r["head_probe"] = head_probe(net, trk.x[fit], trk.y[fit], vk.x, vk.y, args.probe_epochs, seed=args.seed)
        out["tasks"][name] = r
        print(f"{name:14s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()) + f"  ({time.time() - t0:.0f}s)",
              flush=True)
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
