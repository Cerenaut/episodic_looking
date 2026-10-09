"""
C9 check (Notes/experiments/plan.md): for a given seed, do the STM, LTM-only and head scripts train on exactly the same
few-shot images? Images are compared by content (sha1 of the raw uint8 CIFAR bytes), never by index.

  reference   the head script's own split_and_subsample(), called on raw images in place of encodings.
  dataset     Cifar100Dataset with subset_seed (the constructor both the STM environment and LTM-only use).
  stm         cifar_main_stm_training.main() run in this process for a short few-shot run, with CifarEnv
              instrumented to record every image it presents in training mode and the target it returns.
  ltm         cifar_main_ltm_fine_tuning.main() run in this process up to its first training step, recording the
              training dataset handed to the DataLoader.

--legacy runs only the stm part against the code in --code-root (e.g. a worktree of the commit that made
runs_fewshot_ref/), without a seed, and reports what the environment trained on: the question there is whether
the earlier runs trained on exactly N images per coarse class, with correct targets.

Run from anywhere; the scripts' relative data paths are resolved from --code-root.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np


def raw_hash(row_uint8) -> str:
    return hashlib.sha1(np.ascontiguousarray(row_uint8, dtype=np.uint8).tobytes()).hexdigest()


def float_hash(image) -> str:
    """Dataset images are float32 [3,32,32] = raw / 255, in the raw CHW order."""
    return raw_hash(np.rint(np.asarray(image, dtype=np.float64).reshape(-1) * 255.0))


def load_train_file(data_path):
    with open(os.path.join(data_path, "train"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"], np.asarray(d[b"coarse_labels"]), np.asarray(d[b"fine_labels"])


def truth_by_hash(data_path):
    X, lc, lf = load_train_file(data_path)
    table = {}
    for i in range(len(X)):
        table.setdefault(raw_hash(X[i]), set()).add((int(lc[i]), int(lf[i])))
    return table


def reference_subsets(data_path, coarse_classes, max_instances, seed, val_holdout, split_seed):
    """The head script's split_and_subsample on raw images: {group name: set of hashes} of the training sets."""
    import cifar_main_head_baselines as heads
    from environment.cifar.cifar_dataset import Cifar100Dataset
    X, lc, lf = load_train_file(data_path)
    enc = {}
    for name, group in heads.GROUPS.items():
        fine = Cifar100Dataset.get_fine_classes(group)
        sel = np.isin(lc, coarse_classes) & np.isin(lf, list(fine))  # file order, as build_dataset
        enc[f"train_{name}"] = (X[sel], lc[sel], lf[sel])
    args = SimpleNamespace(val_holdout=val_holdout, split_seed=split_seed, coarse_classes=coarse_classes,
                           max_instances=max_instances, pretrain_all_instances=True)
    train, _ = heads.split_and_subsample(enc, args, np.random.default_rng(seed))
    return {name: [raw_hash(x) for x in arrays[0]] for name, arrays in train.items()}


def dataset_subset(data_path, coarse_classes, fine_group, max_instances, seed, val_holdout, split_seed):
    from environment.cifar.cifar_dataset import Cifar100Dataset
    others = [g for g in [1, 2, 3, 4, 5] if g not in fine_group]
    ds = Cifar100Dataset(data_path, Cifar100Dataset.LABEL_TYPE_COARSE, training=True,
                         exclude_classes_coarse=Cifar100Dataset.get_coarse_classes_excluded(coarse_classes),
                         exclude_classes_fine=Cifar100Dataset.get_fine_classes(others), max_instances=max_instances,
                         as_tensor=False, val_holdout=val_holdout, split_seed=split_seed, subset_seed=seed)
    hashes = [float_hash(x) for x in ds.images]
    for shm in (ds.shared_memory_images, ds.shared_memory_labels_coarse, ds.shared_memory_labels_fine):
        shm.close()
        shm.unlink()
    return hashes


def run_stm(argv, record):
    """Run the STM script's main() with CifarEnv.set_random_image instrumented to record training presentations."""
    from environment.cifar.cifar_env import CifarEnv
    from util.instrumentation import Instrumentation
    original = CifarEnv.set_random_image

    def recording_set_random_image(self):
        original(self)
        if self.mode == Instrumentation.MODE_TRAINING:
            image, label = self.dataset[self.image_index]
            record["shown"].append((float_hash(image), int(label)))
            key = id(self)
            if key not in record["env_datasets"]:
                record["env_datasets"][key] = sorted(float_hash(x) for x in self.dataset.images)
    CifarEnv.set_random_image = recording_set_random_image
    import cifar_main_stm_training
    sys.argv = ["cifar_main_stm_training.py"] + argv
    try:
        cifar_main_stm_training.main()
    finally:
        CifarEnv.set_random_image = original


class StopAfterDatasets(Exception):
    pass


def run_ltm(argv):
    """Run the LTM-only script's main() until its datasets are built; returns the datasets given to DataLoaders."""
    import torch.utils.data
    import cifar_main_ltm_fine_tuning as ltm
    captured = []
    original_loader = ltm.DataLoader

    def capturing_loader(dataset, *a, **k):
        captured.append((dataset, k.get("shuffle")))
        return original_loader(dataset, *a, **k)

    def stop(*a, **k):
        raise StopAfterDatasets()
    ltm.DataLoader = capturing_loader
    original_writer = ltm.SummaryWriter
    ltm.SummaryWriter = stop  # created right after the model is loaded, before any training
    sys.argv = ["cifar_main_ltm_fine_tuning.py"] + argv
    try:
        ltm.main()
    except StopAfterDatasets:
        pass
    finally:
        ltm.DataLoader = original_loader
        ltm.SummaryWriter = original_writer
    return [d for d, shuffle in captured if shuffle]  # the training loaders shuffle; evaluation ones do not


def summarise_shown(record, truth, n_expected_per_coarse):
    shown = record["shown"]
    distinct = sorted({h for h, _ in shown})
    wrong_targets = sum(1 for h, label in shown if label not in {c for c, _ in truth[h]})
    per_coarse = Counter(next(iter(truth[h]))[0] for h in distinct)
    fine = Counter(next(iter(truth[h]))[1] for h in distinct)
    env_sets = list(record["env_datasets"].values())
    return {
        "presentations": len(shown),
        "distinct_images_shown": len(distinct),
        "distinct_per_coarse_class": dict(per_coarse),
        "fine_labels_of_shown": dict(fine),
        "presentations_with_wrong_target": wrong_targets,
        "env_dataset_sizes": sorted({len(s) for s in env_sets}),
        "all_envs_same_dataset": all(s == env_sets[0] for s in env_sets),
        "exactly_N_per_coarse": bool(per_coarse) and all(v == n_expected_per_coarse for v in per_coarse.values()),
    }, distinct, env_sets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--out", required=True, help="JSON report path")
    p.add_argument("--run-root", required=True, help="--run-root for the short STM runs")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--n", type=int, nargs="+", default=[1, 4, 16, 64, 400])
    p.add_argument("--stm-n", type=int, nargs="+", default=[4, 16])
    p.add_argument("--fine-class", type=int, default=3)
    p.add_argument("--val-holdout", type=int, nargs="+", default=[100, 0])
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--stm-checkpoint", default="../cifar_100_pretrain/variants/stm_rl_fixed_e13_pt12_seed1.pth")
    p.add_argument("--ltm-checkpoint", default="../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth")
    args = p.parse_args()

    out = os.path.abspath(args.out)
    run_root = os.path.abspath(args.run_root)
    os.chdir(args.code_root)
    sys.path.insert(0, args.code_root)
    data_path = "../cifar-100-python"
    cc = [0, 1]
    truth = truth_by_hash(data_path)
    report = {"code_root": args.code_root, "checks": []}

    def stm_argv(n, seed, holdout, tag):
        argv = ["--experiment-type", "few-shot", "--fine-classes", str(args.fine_class), "--coarse-classes", "0", "1",
                "--max-instances", str(n), "--epochs", "1", "--training-steps", str(max(200, 8 * n)),
                "--evaluate-steps", "8", "--eval-bias", "mean", "--ltm-checkpoint", args.ltm_checkpoint,
                "--stm-checkpoint", args.stm_checkpoint, "--run-root", os.path.join(run_root, tag)]
        if seed is not None:
            argv += ["--seed", str(seed)]
        if holdout:
            argv += ["--val-holdout", str(holdout)]
        return argv

    if args.legacy:
        for n in args.stm_n:
            record = {"shown": [], "env_datasets": {}}
            run_stm(stm_argv(n, None, 0, f"legacy_n{n}"), record)
            summary, _, _ = summarise_shown(record, truth, n)
            summary.update({"n": n, "part": "legacy stm, unseeded"})
            report["checks"].append(summary)
            print(json.dumps(summary))
    else:
        for holdout in args.val_holdout:
            for seed in args.seeds:
                for n in args.n:
                    ref = reference_subsets(data_path, cc, n, seed, holdout, 0)
                    for name, group in (("3", [3]), ("4", [4]), ("5", [5])):
                        got = dataset_subset(data_path, cc, group, n, seed, holdout, 0)
                        ok = sorted(got) == sorted(ref[name]) and len(got) == 2 * n
                        report["checks"].append({"part": "dataset vs head reference", "holdout": holdout,
                                                 "seed": seed, "n": n, "group": name, "size": len(got), "same": ok})
                    print(f"dataset holdout={holdout} seed={seed} n={n}: "
                          f"{all(c['same'] for c in report['checks'][-3:])}")
                    if n in args.stm_n:
                        record = {"shown": [], "env_datasets": {}}
                        run_stm(stm_argv(n, seed, holdout, f"h{holdout}_s{seed}_n{n}"), record)
                        summary, distinct, env_sets = summarise_shown(record, truth, n)
                        ref_set = sorted(ref[str(args.fine_class)])
                        summary.update({"part": "stm environment vs head reference", "holdout": holdout,
                                        "seed": seed, "n": n,
                                        "env_dataset_equals_head_subset": all(s == ref_set for s in env_sets),
                                        "shown_subset_of_head_subset": set(distinct) <= set(ref_set),
                                        "all_shown": sorted(distinct) == ref_set})
                        report["checks"].append(summary)
                        print(json.dumps(summary))

                        ltm_argv = ["--experiment-type", "few-shot", "--fine-classes", str(args.fine_class),
                                    "--coarse-classes", "0", "1", "--max-instances", str(n), "--seed", str(seed),
                                    "--run-root", os.path.join(run_root, f"ltm_h{holdout}_s{seed}_n{n}")]
                        if holdout:
                            ltm_argv += ["--val-holdout", str(holdout)]
                        datasets = run_ltm(ltm_argv)
                        # the shuffled loaders are dataset_training_3, _4, _5 in that order
                        got3 = sorted(float_hash(x) for x in datasets[args.fine_class - 3].images)
                        ltm_ok = got3 == ref_set
                        report["checks"].append({"part": "ltm-only training set vs head reference", "holdout": holdout,
                                                 "seed": seed, "n": n, "size": len(got3), "same": ltm_ok})
                        print(f"ltm holdout={holdout} seed={seed} n={n}: {ltm_ok}")
        report["all_pass"] = all(
            c.get("same", True) and c.get("env_dataset_equals_head_subset", True) and c.get("all_shown", True)
            and c.get("presentations_with_wrong_target", 0) == 0 and c.get("exactly_N_per_coarse", True)
            for c in report["checks"])
        print("ALL PASS" if report["all_pass"] else "FAILURES")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
