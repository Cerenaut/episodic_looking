"""
Factorial STM probe (diagnostic, Dave 8 Nov 2026): why does the STM not memorise a new-task sample before the first
sleep? At a chosen exposure, before the model sees the live sample, copy the whole CLS model and run its adaptive
loop on that sample under each condition, all from the same LTM weights:

- STM: as it is (carrying earlier clusters and tasks), or fresh (actor and critic at their initial weights);
- replay: interleaved replay as configured, or none;
- LTM: as it is, or with its output logits divided by a temperature (an artificially equivocal LTM);
- plus a ceiling: fresh STM, no replay, LTM as it is, `ceiling_repeats` repeats.

Each condition records the CLSModel trace (p(y) before, per block, at the end; repeats; stop reason). The live model
is not touched, and the probe runs under a forked RNG, so the training trajectory is unchanged.

Oracle bias test (Dave 9 Nov 2026), `oracle`: on the same sample and LTM weights, optimise the bias itself by gradient
(Adam on the raw bias, cross-entropy on y, no RL) within the LTM's bias range (gain exp(s tanh(a))) and with wider
ranges (s times `scales`). If the oracle flips the sample at s, the range suffices and the STM lacks a signal or
speed; if not even the oracle can, no STM with this bias can.

Reward probe (Dave 9 Nov 2026), `rewards`: a fresh STM, no replay, up to `repeats` repeats, once per reward type
(the run's p(y) "accuracy", log p(y) clipped at a low floor, the logit margin, the p(y) + log p(y) composite), each from the same random state, so
the conditions differ only in the reward. Tests whether a reward with a slope at tiny p(y) lets the STM find the bias
the oracle finds.
"""
import copy
import dataclasses
import itertools
import math

import torch
import torch.nn.functional as F


class SoftLogits:
    """Wraps the LTM network the STM biases: the same features, logits divided by `temperature`."""

    def __init__(self, net, temperature):
        self.net = net
        self.temperature = temperature

    def run(self, x, bias):
        out = dict(self.net.run(x, bias))
        out["logits"] = out["logits"] / self.temperature
        return out

    def __getattr__(self, name):
        return getattr(self.net, name)


def conditions(temperature, ceiling_repeats):
    temps = (1.0,) if temperature == 1.0 else (1.0, temperature)  # temperature 1: no LTM factor
    out = [{"stm": stm, "replay": replay, "temperature": t, "repeats": None}
           for stm, replay, t in itertools.product(("as_is", "fresh"), (True, False), temps)]
    out.append({"stm": "fresh", "replay": False, "temperature": 1.0, "repeats": ceiling_repeats})
    return out


def oracle(model, x, y, scales=(1, 2, 4, 8), steps=500, lr=0.1, record=(0, 10, 25, 50, 100, 200, 500)):
    """Gradient-optimal bias on (x, y) for copies of the LTM with bias range s * scale; one record per scale: p(y)
    at the `record` steps, the final top class, and the fraction of bias elements near the bound (|tanh(a)| > .95)."""
    records = []
    for k in scales:
        net = copy.deepcopy(model.stm.ltm)
        net.eval()
        net.bias_scale = net.bias_scale * k
        for q in net.parameters():
            q.requires_grad_(False)
        a = torch.zeros(len(y), net.bias_size, requires_grad=True)
        opt = torch.optim.Adam([a], lr=lr)
        rec = {"oracle_scale": k, "bias_scale": float(net.bias_scale), "p": []}
        for t in range(steps + 1):
            logits = net.run(x, a)["logits"]
            if t in record:
                with torch.no_grad():
                    rec["p"].append([t, float(F.softmax(logits, 1).gather(1, y[:, None]).mean())])
            if t == steps:
                break
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            rec.update(p_end=rec["p"][-1][1], top=int(logits.argmax(1)[0]),
                       saturated=float((torch.tanh(a).abs() > 0.95).float().mean()))
        records.append(rec)
    return records


def rewards(model, x, y, types=("accuracy", "log-accuracy", "margin"), repeats=1024, log_p_floor=1e-8,
            margin_clip=20.0, log_weight=1.0):
    """One trace record per reward type: fresh STM, no replay, cap `repeats`, the live STM's settings otherwise."""
    records = []
    with torch.random.fork_rng(devices=[]):
        state = torch.get_rng_state()
        for r in types:
            torch.set_rng_state(state)
            m = copy.deepcopy(model)
            m.stm = m.stm_factory()
            m.stm.model.config = dataclasses.replace(m.stm.model.config, reward_type=r,
                                                     log_p_floor=math.log(log_p_floor), margin_clip=margin_clip,
                                                     log_weight=log_weight)
            m.interleave = 0
            m.repeats = repeats
            m.trace = []
            m._live(x, y)
            records.append({**m.trace[0], "reward": r, "log_p_floor": log_p_floor, "margin_clip": margin_clip,
                            "log_weight": log_weight, "repeats_cap": repeats})
    return records


def probe(model, x, y, temperature=4.0, ceiling_repeats=1024):
    """Runs every condition on the live sample (x, y) from copies of `model`; returns one record per condition."""
    records = []
    with torch.random.fork_rng(devices=[]):
        for cond in conditions(temperature, ceiling_repeats):
            m = copy.deepcopy(model)
            if cond["stm"] == "fresh":
                m.stm = m.stm_factory()
            if not cond["replay"]:
                m.interleave = 0
            if cond["temperature"] != 1.0:
                m.stm.ltm = SoftLogits(m.stm.ltm, cond["temperature"])
            if cond["repeats"] is not None:
                m.repeats = cond["repeats"]
            m.trace = []
            m._live(x, y)
            records.append({**m.trace[0], **cond, "repeats_cap": cond["repeats"], "repeats": m.trace[0]["repeats"]})
    return records
