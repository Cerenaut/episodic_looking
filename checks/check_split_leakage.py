"""
C10 check (Notes/experiments/plan.md): with the validation split on, the held-out images never appear in training,
from inside each script. For every pair A-D:

  stm   cifar_main_stm_training.main() for short pre-training, continual (fine 3, 4, 5) and few-shot runs, with
        CifarEnv instrumented (check_fewshot_subsets.run_stm): every image in every training-mode environment
        dataset, and every image presented, is compared with the held-out images.
  ltm   the training datasets cifar_main_ltm_fine_tuning.main() hands to its shuffled DataLoaders (continual and
        few-shot).
  heads the training arrays of cifar_main_head_baselines.split_and_subsample() on raw images.

Images are compared by content. CIFAR-100 contains a few exact duplicates within a fine label, so an image can be
held out while its twin trains; those overlaps are reported separately and are not leaks of the split.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_fewshot_subsets import float_hash, load_train_file, raw_hash, run_ltm, run_stm  # noqa: E402

PAIRS = {"A": [0, 1], "B": [2, 3], "C": [5, 6], "D": [15, 16]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--out", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--pairs", nargs="+", default=list(PAIRS))
    p.add_argument("--val-holdout", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stm-checkpoint", default="../cifar_100_pretrain/variants/stm_rl_fixed_e13_pt12_seed1.pth")
    p.add_argument("--ltm-checkpoint", default="../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth")
    args = p.parse_args()
    out = os.path.abspath(args.out)
    run_root = os.path.abspath(args.run_root)
    os.chdir(args.code_root)
    sys.path.insert(0, args.code_root)
    from environment.cifar.cifar_dataset import Cifar100Dataset
    import cifar_main_head_baselines as heads

    data_path = "../cifar-100-python"
    X, lc, lf = load_train_file(data_path)
    hashes = np.array([raw_hash(x) for x in X])
    heldout_mask = Cifar100Dataset.get_validation_mask(lf, args.val_holdout, 0)
    heldout = set(hashes[heldout_mask])
    trainable = set(hashes[~heldout_mask])
    twins = heldout & trainable  # content both held out and in the training part: CIFAR duplicates
    report = {"val_holdout": args.val_holdout, "duplicate_twins_across_split_whole_file": len(twins), "checks": []}

    def verdict(part, pair, train_hashes, shown=None):
        train_hashes = set(train_hashes)
        leak = train_hashes & heldout
        rec = {"part": part, "pair": pair, "training_images": len(train_hashes),
               "overlap_with_heldout": len(leak), "of_which_duplicate_twins": len(leak & twins),
               "leaks": len(leak - twins)}
        if shown is not None:
            shown = set(shown)
            rec["presented"] = len(shown)
            rec["presented_leaks"] = len((shown & heldout) - twins)
        report["checks"].append(rec)
        print(json.dumps(rec))

    for letter in args.pairs:
        cc = PAIRS[letter]
        in_pair = np.isin(lc, cc)
        # heads: split_and_subsample on raw images, with and without a few-shot subset
        enc = {}
        for name, group in heads.GROUPS.items():
            sel = in_pair & np.isin(lf, list(Cifar100Dataset.get_fine_classes(group)))
            enc[f"train_{name}"] = (X[sel], lc[sel], lf[sel])
        for max_instances in (None, 16):
            a = SimpleNamespace(val_holdout=args.val_holdout, split_seed=0, coarse_classes=cc,
                                max_instances=max_instances, pretrain_all_instances=True)
            train, _ = heads.split_and_subsample(enc, a, np.random.default_rng(args.seed))
            verdict(f"heads, max_instances={max_instances}", letter,
                    [raw_hash(x) for arrays in train.values() for x in arrays[0]])

        base = ["--coarse-classes", *map(str, cc), "--seed", str(args.seed), "--val-holdout", str(args.val_holdout),
                "--ltm-checkpoint", args.ltm_checkpoint]
        stm_runs = {
            "pretrain": ["--experiment-type", "pretrain", "--fine-classes", "1", "2", "--epochs", "1",
                         "--training-steps", "40", "--evaluate-steps", "8",
                         "--stm-checkpoint", os.path.join(run_root, f"pair{letter}_pretrain_ckpt.pth")],
            "continual": ["--experiment-type", "continual", "--fine-classes", "3", "4", "5", "--epochs", "1",
                          "--training-steps", "40", "--evaluate-steps", "8", "--stm-checkpoint", args.stm_checkpoint],
            "few-shot N=16": ["--experiment-type", "few-shot", "--fine-classes", "3", "--max-instances", "16",
                              "--epochs", "1", "--training-steps", "40", "--evaluate-steps", "8",
                              "--stm-checkpoint", args.stm_checkpoint],
        }
        for tag, extra in stm_runs.items():
            record = {"shown": [], "env_datasets": {}}
            run_stm(base + ["--eval-bias", "mean", "--run-root", os.path.join(run_root, f"pair{letter}_{tag[:9]}")]
                    + extra, record)
            env_images = {h for s in record["env_datasets"].values() for h in s}
            verdict(f"stm {tag}", letter, env_images, shown=[h for h, _ in record["shown"]])

        for tag, extra in {"continual": ["--experiment-type", "continual", "--fine-classes", "3", "4", "5"],
                           "few-shot N=16": ["--experiment-type", "few-shot", "--fine-classes", "3",
                                             "--max-instances", "16"]}.items():
            ltm_args = [a for a in base if a != args.ltm_checkpoint and a != "--ltm-checkpoint"]
            datasets = run_ltm(ltm_args + ["--run-root", os.path.join(run_root, f"pair{letter}_ltm")] + extra)
            verdict(f"ltm {tag}", letter, {float_hash(x) for d in datasets for x in d.images})

    report["all_pass"] = all(c["leaks"] == 0 and c.get("presented_leaks", 0) == 0 for c in report["checks"])
    print("ALL PASS" if report["all_pass"] else "LEAKS FOUND")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
