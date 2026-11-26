"""
CIFAR-100 pair selection, second measure (Dave 26 Nov 2026: fine classes that resemble a fine class of the other coarse
class are a key cause of difficulty; the previous paper found sharks called aquatic mammals once whales were learnt).
Per coarse pair: a 10-way fine-class linear probe on the frozen tokens (training split), on the validation images:
each fine class's coarse error (images called a fine class of the other side), the worst fine class, and the most
confused cross-side fine pair. Output runs_ss/cifar/pair_selection/fine_confusion.json.
    PYTHONPATH=. python single_stream/pair_confusion.py
"""
import itertools, json, numpy as np, torch, torch.nn.functional as F
from single_stream import cifar
from single_stream.preprocessor import featurize, load_checkpoint
torch.set_num_threads(4); torch.manual_seed(0)
d = cifar.load_cifar100("/home/incubator/dev/claude/cifar-100-python")
x, fine, coarse = d["train"]; tr, va = cifar.validation_split(fine, 100)
feats = featurize(load_checkpoint("runs_ss/preproc/tinyimagenet_conv_c32_s0/preproc.pt"), x)
def probe(a, ya, b, k, l2=1e-3):
    mu, sd = a.mean(0), a.std(0) + 1e-6; a, b = (a - mu) / sd, (b - mu) / sd
    W = torch.zeros(a.shape[1], k, requires_grad=True); c = torch.zeros(k, requires_grad=True)
    opt = torch.optim.LBFGS([W, c], max_iter=100, line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad(); l = F.cross_entropy(a @ W + c, ya) + l2 * (W * W).sum(); l.backward(); return l
    opt.step(cl)
    with torch.no_grad(): return (b @ W + c).argmax(1)
rows = []
for A, B in itertools.combinations(range(20), 2):
    fs = cifar.fine_of(coarse, fine, A) + cifar.fine_of(coarse, fine, B); idx = {f: i for i, f in enumerate(fs)}
    mtr = tr[np.isin(fine[tr], fs)]; mva = va[np.isin(fine[va], fs)]
    pred = probe(feats[mtr], torch.tensor([idx[f] for f in fine[mtr]]), feats[mva], 10).numpy()
    true = np.array([idx[f] for f in fine[mva]])
    side = lambda i: i // 5
    cross = {fs[i]: float(np.mean(side(pred[true == i]) != side(i))) for i in range(10)}  # coarse error per fine class
    # the most confused cross-side fine pair: share of f's images called g (either direction, max)
    conf = np.zeros((10, 10))
    for i in range(10):
        for j in range(10): conf[i, j] = np.mean(pred[true == i] == j)
    best = max(((i, j) for i in range(10) for j in range(10) if side(i) != side(j)), key=lambda p: conf[p])
    rows.append({"pair": [A, B], "names": [d["coarse_names"][A], d["coarse_names"][B]],
                 "fine_acc": float(np.mean(pred == true)), "coarse_err_mean": float(np.mean(list(cross.values()))),
                 "coarse_err_worst": max(cross.values()), "worst_fine": d["fine_names"][max(cross, key=cross.get)],
                 "top_confusion": [d["fine_names"][fs[best[0]]], d["fine_names"][fs[best[1]]], float(conf[best])]})
json.dump(rows, open("runs_ss/cifar/pair_selection/fine_confusion.json", "w"), indent=1)
