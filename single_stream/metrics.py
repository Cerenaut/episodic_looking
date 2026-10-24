"""
Checkpoint schedule, held-out evaluation and the summary metrics of single_stream_paper plan.md section 4:
final and anytime accuracy, few-shot exposure efficiency E(n) / E(q), retention curve R(a), effective few-shot
exposures N_x and retention time tau.
"""
import numpy as np
import torch


def checkpoint_schedule(epoch_len, epochs, n_log, grid_per_epoch, step=1):
    """
    Exposure counts at which to evaluate. Log-spaced points over the first epoch (to resolve few-shot learning)
    plus a uniform grid of `grid_per_epoch` points per epoch (ending at each epoch end). Rounded up to whole
    steps of `step` exposures. Returns (sorted exposure counts, set of those on the uniform grid).
    """
    def to_step(t):
        return int(min(np.ceil(t / step) * step, epoch_len * epochs))

    grid = {to_step(epoch_len * e + epoch_len * (i + 1) / grid_per_epoch)
            for e in range(epochs) for i in range(grid_per_epoch)}
    log = set()
    if n_log > 0:
        log = {to_step(t) for t in np.geomspace(step, epoch_len, n_log)}
    return sorted(grid | log), grid


@torch.no_grad()
def evaluate(model, data, batch_size=2000):
    """Predicted labels on `data` (int array), with the model in prediction mode."""
    preds = []
    for i in range(0, len(data), batch_size):
        preds.append(model.predict(data.x[i:i + batch_size]).argmax(1).numpy())
    return np.concatenate(preds)


def per_group_accuracy(correct, groups, group_ids):
    return np.array([correct[groups == g].mean() for g in group_ids])


def group_confusion(preds, groups, group_ids, num_classes):
    """counts[i, l] = number of evaluation samples of group group_ids[i] predicted as label l."""
    counts = np.zeros((len(group_ids), num_classes), dtype=np.int64)
    for i, g in enumerate(group_ids):
        counts[i] = np.bincount(preds[groups == g], minlength=num_classes)
    return counts


def group_labels(groups, labels, group_ids):
    """The target label of each group (every group must have a single label)."""
    out = []
    for g in group_ids:
        values = np.unique(labels[groups == g])
        if len(values) != 1:
            raise ValueError(f"group {g} has labels {values}")
        out.append(int(values[0]))
    return np.array(out)


def balanced_group_score(confusion, group_label):
    """
    Per group: mean of recall (its samples predicted as its label) and specificity (samples of groups with a
    different label NOT predicted as this group's label). Insensitive to a bias toward recently seen labels: a
    model predicting one label for everything scores 0.5 on every group. Chance = 0.5.
    """
    confusion = np.asarray(confusion, dtype=np.float64)
    group_label = np.asarray(group_label)
    totals = confusion.sum(1)
    score = np.zeros(len(group_label))
    for i, l in enumerate(group_label):
        recall = confusion[i, l] / totals[i]
        other = group_label != l
        specificity = 1.0 - confusion[other, l].sum() / totals[other].sum()
        score[i] = 0.5 * (recall + specificity)
    return score


def log2_bin(values):
    """Bin edges 0, 1, 2-3, 4-7, ...: bin index 0 for value 0, else floor(log2(v)) + 1."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.int64)
    pos = values > 0
    out[pos] = np.floor(np.log2(values[pos])).astype(np.int64) + 1
    return out


def binned_curve(x, acc):
    """Mean accuracy per log2 bin of x. Returns list of (bin lower edge, mean accuracy, count)."""
    bins = log2_bin(x)
    curve = []
    for b in np.unique(bins):
        sel = bins == b
        lower = 0 if b == 0 else 2 ** (b - 1)
        curve.append((int(lower), float(np.mean(acc[sel])), int(sel.sum())))
    return curve


def summarize(records, group_label=None):
    """
    Summary of one run from its checkpoint records (dicts with t, grid, acc, group_acc, n, q, age; age < 0 means
    the group has not been presented yet). With `group_label` and records carrying a `confusion` matrix, the
    exposure-efficiency and retention curves are also computed on the balanced group score (keys *_balanced),
    which is not inflated by recency bias.
    """
    grid_acc = [r["acc"] for r in records if r["grid"]]
    n = np.concatenate([r["n"] for r in records])
    q = np.concatenate([r["q"] for r in records])
    age = np.concatenate([r["age"] for r in records])
    gacc = np.concatenate([r["group_acc"] for r in records])
    seen = age >= 0
    out = {
        "final_acc": records[-1]["acc"],
        "anytime_acc": float(np.mean(grid_acc)) if grid_acc else None,
        "exposure_efficiency": binned_curve(n, gacc),
        "visit_efficiency": binned_curve(q, gacc),
        "retention": binned_curve(age[seen], gacc[seen]),
    }
    if group_label is not None and all("confusion" in r for r in records):
        bal = np.concatenate([balanced_group_score(r["confusion"], group_label) for r in records])
        out["exposure_efficiency_balanced"] = binned_curve(n, bal)
        out["visit_efficiency_balanced"] = binned_curve(q, bal)
        out["retention_balanced"] = binned_curve(age[seen], bal[seen])
        out["final_group_balanced"] = balanced_group_score(records[-1]["confusion"], group_label).tolist()
    return out


def effective_fewshot_exposures(curve, iid_acc, fraction=0.9):
    """Smallest bin edge n at which E(n) reaches `fraction` of the iid final accuracy (None if never)."""
    for lower, acc, _ in curve:
        if acc >= fraction * iid_acc:
            return lower
    return None


def retention_time(curve, iid_acc, chance):
    """First age bin edge at which R(a) falls below the midpoint between iid accuracy and chance (None if never)."""
    threshold = 0.5 * (iid_acc + chance)
    for lower, acc, _ in curve:
        if acc < threshold:
            return lower
    return None
