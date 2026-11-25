"""
Lifelong single stream over a sequence of tasks (Dave, 5 Nov 2026): the clustered stream of task 1 runs for
--epochs epochs on each task in --tasks (single_stream/tasks.py: dihedral image transforms, label permutations), and
switches instantly to the next. The model is never told about switches, clusters or groups. Options are those of
run_stream (the same models), plus --tasks.

Evaluation: at task-relative checkpoints (log-spaced from the switch plus a uniform grid), the model is evaluated on
every task's evaluation set (the current one, earlier ones for retention, later ones zero-shot); at the start of each
task, before its first sample (zero-shot). Few-shot triggers (as run_stream) count a group's exposures within the
current task, and score it on the current task (Generalisation, Memorisation, Protection).

--lifelong (Dave 16 Nov 2026, the plasticity experiment; --tasks plast:N, tasks.py): only the current task is evaluated,
at its start and end; its end record adds acc_seen (accuracy on the classes presented in the task) and the LTM's
plasticity diagnostics (plasticity.py); few-shot triggers only on the tasks in --fewshot-tasks. --snapshot-before saves
the whole model (and the random states) just before the listed tasks, for runs of the reset models from that LTM.

Outputs in --out: args.json, checkpoints.jsonl, fewshot.jsonl, summary.json, log.txt.

  python -m single_stream.run_tasks --tasks id,rot90,hflip,rot180 --cluster-size 128 --epochs 1 ... --out <dir>
"""
import json
import os
import sys
import time

import numpy as np
import torch

from single_stream import datasets, metrics, plasticity, stm_probe, tasks
from single_stream.models import CLSModel
from single_stream.run_stream import Tee, buffer_diagnostics, build_model, build_parser, rng_neutral, stm_diagnostics
from single_stream.stream import ClusteredStream


def advance_hook_ids():
    """
    After unpickling modules: move torch's hook-handle counter past every hook id they carry. The counter is
    process-wide and starts at 0 in a new process, while the loaded modules keep their saved ids, so the next hook
    registered (plasticity.diagnostics') could take a saved hook's id, overwrite it and delete it on removal: on
    resume the LTM's continual-backprop activation recorder was lost at the first task end, and CBP then replaced
    units on stale activations (Dave 26 Nov 2026, plast12). Runs that do not resume are unaffected.
    """
    import gc
    from torch.utils.hooks import RemovableHandle
    ids = [k for m in gc.get_objects() if issubclass(type(m), torch.nn.Module)
           for name, d in vars(m).items() if name.endswith("hooks") and isinstance(d, dict)
           for k in d if isinstance(k, int)]
    RemovableHandle.next_id = max([RemovableHandle.next_id] + [k + 1 for k in ids])

RECOVERY_POINTS = (0, 16, 64, 256, 1024, 4096, 16384)  # exposures since the switch, for the summary


def parse_args(argv=None):
    p = build_parser(__doc__)
    p.add_argument("--tasks", default="id,rot90,hflip,rot180", help="comma-separated task names (tasks.py)")
    p.add_argument("--feature-cache", default="", help="directory for the preprocessor's tokens per task (shared)")
    p.add_argument("--stm-probe-at", default="",
                   help="diagnostic: exposures (counted within each task after the first, 1 = its first sample) at which "
                        "to run the factorial STM probe (stm_probe.py) on the live sample; to stm_probe.jsonl")
    p.add_argument("--stm-probe-temperature", type=float, default=4.0)
    p.add_argument("--stm-probe-ceiling", type=int, default=1024, help="repeats of the probe's ceiling condition")
    p.add_argument("--stm-probe-kind", default="factorial", choices=("factorial", "oracle", "reward"),
                   help="factorial: the STM conditions; oracle: the gradient-optimal bias at several ranges; reward: a "
                        "fresh STM per reward type (stm_probe.py)")
    p.add_argument("--stm-probe-log-p-floor", type=float, default=1e-8, help="reward probe: log-accuracy floor on p(y)")
    p.add_argument("--stm-probe-margin-clip", type=float, default=20.0, help="reward probe: margin clip")
    p.add_argument("--stm-probe-rewards", default="accuracy,log-accuracy,margin",
                   help="reward probe: comma-separated reward types (each from the same random state)")
    p.add_argument("--stm-probe-log-weight", type=float, default=1.0, help="reward probe: accuracy-log weight")
    p.add_argument("--lifelong", action="store_true",
                   help="evaluate the current task only, at its start and end, with acc_seen and plasticity diagnostics")
    p.add_argument("--fewshot-tasks", default="",
                   help="task indices (0 = first) on which few-shot triggers run; empty = all (--lifelong: 1 and last)")
    p.add_argument("--snapshot-before", default="",
                   help="task indices before which the model and random states are saved (snapshot_task<k>.pt)")
    p.add_argument("--resume-from", default="",
                   help="a --snapshot-before file: continue that run from its task, with its model (LTM) and random "
                        "states. --model buffered or head: the snapshot's model (LTM and head) as it is; --model cls: a fresh STM (this run's "
                        "--stm-* flags) on the snapshot's LTM (Dave 21 Nov 2026: the RL STM on the fixed and unfixed LTM)")
    p.add_argument("--stop-after-task", type=int, default=-1, help="last task index to run (-1: all)")
    p.add_argument("--plasticity-probe", type=int, default=1000,
                   help="--lifelong: held-out samples of the task used for the plasticity diagnostics")
    p.add_argument("--full-first-task", action="store_true",
                   help="--max-exposures applies to the tasks after the first only (the first runs in full)")
    return p.parse_args(argv)


def evaluate_all(model, evals):
    """Main predictions on every task's evaluation set: {name: predicted labels}."""
    with rng_neutral():
        return {name: metrics.evaluate_variants(model, d)["main"] for name, d in evals.items()}


def main(argv=None):
    args = parse_args(argv)
    names = tasks.expand(args.tasks)
    for name in names:
        if not tasks.valid(name):
            sys.exit(f"unknown task {name}")
    if os.path.exists(os.path.join(args.out, "summary.json")):
        sys.exit(f"{args.out} already has results; refusing to overwrite")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    log = Tee(os.path.join(args.out, "log.txt"))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_raw, eval_raw = datasets.load(args.dataset, args.data_root, args.val_holdout, args.split_seed)
    pre = None
    if args.preproc:
        from single_stream.preprocessor import load_checkpoint
        pre = load_checkpoint(args.preproc)
        pre.tag = os.path.basename(os.path.dirname(os.path.abspath(args.preproc)))
    cache = args.feature_cache or None
    if args.lifelong:  # built per task (100 tasks' evaluation sets would not fit)
        evals = {}
    else:
        evals = {name: tasks.featurized(name, eval_raw, pre, cache) for name in names}
    if args.fewshot_tasks:
        fewshot_tasks = {int(v) for v in args.fewshot_tasks.split(",") if v.strip()}
    else:
        fewshot_tasks = {1, len(names) - 1} if args.lifelong else set(range(len(names)))
    snapshot_before = {int(v) for v in args.snapshot_before.split(",") if v.strip()}
    token_shape = (49, pre.channels) if pre is not None else None
    train = tasks.featurized(names[0], train_raw, pre, cache)
    start_task = 0
    if args.resume_from:
        snap = torch.load(args.resume_from, weights_only=False)
        advance_hook_ids()
        if args.model == "cls":
            model = build_model(args, train, token_shape, ltm=snap["model"])
        elif args.model in ("buffered", "head"):
            model = snap["model"]
        else:
            sys.exit("--resume-from: --model buffered, head or cls")
        rng.bit_generator.state = snap["np_rng"]
        torch.set_rng_state(snap["torch_rng"])
        start_task = snap["task"]
        if names[start_task] != snap["task_name"]:
            sys.exit(f"--resume-from: task {start_task} is {names[start_task]} here, {snap['task_name']} in the snapshot")
    else:
        model = build_model(args, train, token_shape)
    log(f"tasks {names if len(names) <= 8 else names[:3] + ['...'] + names[-2:]} ({len(names)})  train {len(train)}  eval {len(eval_raw)}  c={args.cluster_size}  epochs/task {args.epochs}  "
        f"memory {model.memory_values() / 1e6:.2f}M values")

    thresholds = np.array([int(v) for v in args.fewshot_thresholds.split(",") if v.strip()], dtype=np.int64)
    ckpt_file = open(os.path.join(args.out, "checkpoints.jsonl"), "w")
    fewshot_file = open(os.path.join(args.out, "fewshot.jsonl"), "w")
    trace_file = None
    probe_at = {int(v) for v in args.stm_probe_at.split(",") if v.strip()}
    probe_file = open(os.path.join(args.out, "stm_probe.jsonl"), "w") if probe_at else None
    if args.stm_trace and isinstance(model, CLSModel):
        model.trace = []
        trace_file = open(os.path.join(args.out, "stm_trace.jsonl"), "w")
    t = snap["t"] if args.resume_from else 0
    train_time = eval_time = 0.0
    summary = {"tasks": names, "per_task": []}

    for ti, name in enumerate(names):
        if ti < start_task:
            continue
        if 0 <= args.stop_after_task < ti:
            break
        if ti > 0:
            train = tasks.featurized(name, train_raw, pre, cache)
        if args.lifelong:
            evals = {name: tasks.featurized(name, eval_raw, pre, cache)}
        cur = evals[name]
        if ti in snapshot_before:
            torch.save({"model": model, "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "t": t, "task": ti, "task_name": name}, os.path.join(args.out, f"snapshot_task{ti}.pt"))
            log(f"snapshot before task {ti} ({name})")
        stream = ClusteredStream(train.groups, args.cluster_size, rng)
        group_ids = stream.group_ids
        group_index = {g: i for i, g in enumerate(group_ids)}
        group_label = metrics.group_labels(train.groups, train.y.numpy(), group_ids)
        epoch_len = len(train)
        total = epoch_len * args.epochs
        if args.max_exposures and not (ti == 0 and args.full_first_task):
            total = min(total, args.max_exposures)
        checkpoints, grid = metrics.checkpoint_schedule(epoch_len, args.epochs, args.n_log_checkpoints,
                                                        args.grid_per_epoch)
        checkpoints = [c for c in checkpoints if c <= total]
        if args.lifelong:
            checkpoints = []
        if not checkpoints or checkpoints[-1] != total:
            checkpoints.append(total)
        n = np.zeros(len(group_ids), dtype=np.int64)
        q = np.zeros(len(group_ids), dtype=np.int64)
        last_seen = np.full(len(group_ids), -1, dtype=np.int64)
        seen = [[] for _ in group_ids]
        online_correct = online_count = 0
        tt = 0  # exposures in this task

        def checkpoint(epoch):
            nonlocal online_correct, online_count, eval_time
            t0 = time.perf_counter()
            preds = evaluate_all(model, evals)
            eval_time += time.perf_counter() - t0
            p = preds[name]
            correct = p == cur.y.numpy()
            rec = {"t": t, "task": ti, "task_name": name, "t_task": tt, "epoch": epoch, "grid": tt in grid,
                   "acc": float(correct.mean()),
                   "task_acc": {k: float((v == evals[k].y.numpy()).mean()) for k, v in preds.items()},
                   "group_acc": metrics.per_group_accuracy(correct, cur.groups, group_ids).tolist(),
                   "confusion": metrics.group_confusion(p, cur.groups, group_ids, cur.num_classes).tolist(),
                   "n": n.tolist(), "q": q.tolist(),
                   "age": np.where(last_seen >= 0, tt - last_seen, -1).tolist(),
                   "online_acc": online_correct / max(online_count, 1) if online_count else None,
                   "train_time": train_time, "eval_time": eval_time}
            if hasattr(model, "buffer"):
                rec.update(buffer_diagnostics(model.buffer, cur.num_classes))
            if isinstance(model, CLSModel):
                rec["stm_online"] = model.online
                with rng_neutral():
                    rec.update(stm_diagnostics(model, cur, args.stm_diag_images))
            ckpt_file.write(json.dumps(rec) + "\n")
            ckpt_file.flush()
            online_correct = online_count = 0
            return rec

        start = checkpoint(-1)  # zero-shot on the new task, before its first sample
        log(f"task {ti} {name}: start  acc {start['acc']:.4f}  tasks {json.dumps({k: round(v, 3) for k, v in start['task_acc'].items()})}")
        next_ckpt = 0
        task_records = [start]
        for epoch in range(args.epochs):
            if tt >= total:
                break
            ep = stream.epoch()
            cluster_start = np.zeros(epoch_len, dtype=bool)
            cluster_start[np.concatenate([[0], np.cumsum(ep.cluster_lengths)[:-1]])] = True
            for i in range(epoch_len):
                if tt >= total:
                    break
                k = int(ep.order[i])
                idx = ep.order[i:i + 1]
                if probe_file is not None and ti > 0 and tt + 1 in probe_at and isinstance(model, CLSModel) \
                        and model.online:
                    gi = group_index[train.groups[k]]
                    if args.stm_probe_kind == "oracle":
                        recs = stm_probe.oracle(model, train.x[idx], train.y[idx])
                    elif args.stm_probe_kind == "reward":
                        recs = stm_probe.rewards(model, train.x[idx], train.y[idx], repeats=args.stm_probe_ceiling,
                                                 log_p_floor=args.stm_probe_log_p_floor,
                                                 margin_clip=args.stm_probe_margin_clip,
                                                 types=tuple(args.stm_probe_rewards.split(",")),
                                                 log_weight=args.stm_probe_log_weight)
                    else:
                        recs = stm_probe.probe(model, train.x[idx], train.y[idx], args.stm_probe_temperature,
                                               args.stm_probe_ceiling)
                    for rec in recs:
                        rec.update(task=ti, t_task=tt + 1, group=int(group_ids[gi]), n=int(n[gi]) + 1,
                                   exposures_since_reset=model.exposures - model.last_reset)
                        probe_file.write(json.dumps(rec) + "\n")
                    probe_file.flush()
                t0 = time.perf_counter()
                pred = model.observe(train.x[idx], train.y[idx])
                train_time += time.perf_counter() - t0
                online_correct += int((pred == train.y[idx]).sum())
                online_count += 1
                gi = group_index[train.groups[k]]
                n_before = int(n[gi])
                n[gi] += 1
                q[gi] += int(cluster_start[i])
                last_seen[gi] = tt + 1
                seen[gi].append(k)
                t += 1
                tt += 1
                if trace_file is not None and model.trace:
                    rec = model.trace.pop()
                    with rng_neutral():  # p(y) of the live sample as evaluated, after the LTM's step (and any sleep)
                        p_eval = float(torch.softmax(model.predict(train.x[idx]), 1)[0, train.y[k]])
                    rec.update(t=t, task=ti, t_task=tt, group=int(group_ids[gi]), n=int(n[gi]),
                               slept_resets=model.resets, p_eval=p_eval)
                    trace_file.write(json.dumps(rec) + "\n")
                crossed = [int(v) for v in thresholds if n_before < v <= n[gi]] if ti in fewshot_tasks else []
                if crossed:
                    t0 = time.perf_counter()
                    with rng_neutral():
                        p = metrics.evaluate_variants(model, cur)["main"]
                        mem_idx = torch.tensor(seen[gi][-args.mem_images:])
                        mem_pred = model.predict(train.x[mem_idx]).argmax(1)
                    conf = metrics.group_confusion(p, cur.groups, group_ids, cur.num_classes)
                    hits = np.array([conf[g, group_label[g]] for g in range(len(group_ids))], dtype=float)
                    tot = conf.sum(1).astype(float)
                    others = np.arange(len(group_ids)) != gi
                    for v in crossed:
                        fewshot_file.write(json.dumps({
                            "t": t, "task": ti, "task_name": name, "t_task": tt, "group": int(group_ids[gi]),
                            "threshold": v, "n": int(n[gi]), "q": int(q[gi]),
                            "recall": float(hits[gi] / tot[gi]),
                            "memorisation": float((mem_pred == train.y[mem_idx]).float().mean()),
                            "protection": float(hits[others].sum() / tot[others].sum()),
                            # other classes' held-out samples labelled as this class (a bias towards the current class
                            # raises recall without generalising; 15 Nov check of the heads' one-shot recall)
                            "to_current": float(conf[others, group_label[gi]].sum() / tot[others].sum()),
                            "acc": float((p == cur.y.numpy()).mean())}) + "\n")
                    fewshot_file.flush()
                    eval_time += time.perf_counter() - t0
                if next_ckpt < len(checkpoints) and tt >= checkpoints[next_ckpt]:
                    rec = checkpoint(epoch)
                    task_records.append(rec)
                    if rec["grid"]:
                        log(f"task {ti} {name}  t {t:>8d} (task {tt:>7d})  acc {rec['acc']:.4f}  "
                            f"online {rec['online_acc'] if rec['online_acc'] is None else round(rec['online_acc'], 4)}  "
                            f"train {train_time:.0f}s  eval {eval_time:.0f}s")
                    while next_ckpt < len(checkpoints) and tt >= checkpoints[next_ckpt]:
                        next_ckpt += 1
        end = task_records[-1]
        if args.lifelong:
            presented = np.isin(cur.groups, [g for g, c in zip(group_ids, n) if c > 0])
            p = evaluate_all(model, evals)[name]
            end["acc_seen"] = float((p[presented] == cur.y.numpy()[presented]).mean())
            end["classes_seen"] = int((n > 0).sum())
            with rng_neutral():
                end.update(plasticity.diagnostics(plasticity.ltm_net(model), cur.x[:args.plasticity_probe]))
            cbp = getattr(getattr(model, "ltm", model), "cbp", None)
            if cbp is not None:
                end["plast_cbp_replaced"] = cbp.replaced  # cumulative units replaced by continual backprop
            ckpt_file.write(json.dumps({"t": t, "task": ti, "task_name": name, "end": True,
                                        **{k: end[k] for k in ("acc", "acc_seen", "classes_seen")},
                                        **{k: v for k, v in end.items() if k.startswith("plast_")}}) + "\n")
            ckpt_file.flush()
        grid_recs = [r for r in task_records if r["grid"]]
        recovery = {}
        for x in RECOVERY_POINTS:  # accuracy on the task at (or just after) x exposures since the switch
            r = next((r for r in task_records if r["t_task"] >= x), end)
            recovery[str(x)] = r["acc"]
        summary["per_task"].append({
            "task": name, "zero_shot": start["acc"], "final": end["acc"],
            "anytime": float(np.mean([r["acc"] for r in grid_recs])) if grid_recs else None,
            "worst_group_final": float(min(end["group_acc"])),
            "worst_group_anytime": float(np.mean([min(r["group_acc"]) for r in grid_recs])) if grid_recs else None,
            "recovery": recovery, "task_acc_end": end["task_acc"], "exposures": tt,
            **({k: end[k] for k in end if k in ("acc_seen", "classes_seen") or k.startswith("plast_")}
               if args.lifelong else {})})
        if args.lifelong:
            log(f"task {ti} {name}: end  acc {end['acc']:.4f}  seen {end['acc_seen']:.4f} ({end['classes_seen']} classes)  "
                + "  ".join(f"{k[6:]} {v:.3f}" for k, v in end.items() if k.startswith("plast_") and k != "plast_cbp_replaced")
                + f"  train {train_time:.0f}s  eval {eval_time:.0f}s")
            continue
        log(f"task {ti} {name}: end  acc {end['acc']:.4f}  tasks {json.dumps({k: round(v, 3) for k, v in end['task_acc'].items()})}")

    ckpt_file.close()
    fewshot_file.close()
    if trace_file is not None:
        trace_file.close()
    if probe_file is not None:
        probe_file.close()
    if hasattr(model, "updates"):
        summary["replay_updates"] = model.updates
    if isinstance(model, CLSModel):
        summary.update({"stm_episodes": model.stm.episodes + model.stm_episodes_before,
                        "stm_live_episodes": model.live_episodes, "stm_replay_episodes": model.replay_episodes,
                        "stm_resets": model.resets})
    summary.update({"exposures": t, "memory_values": model.memory_values(), "train_time": train_time,
                    "eval_time": eval_time, "ms_per_step": 1000 * train_time / max(t, 1)})
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("final " + " ".join(f"{p['task']} {p['final']:.4f}" for p in summary["per_task"]) +
        f"  {summary['ms_per_step']:.2f} ms/step")


if __name__ == "__main__":
    main()
