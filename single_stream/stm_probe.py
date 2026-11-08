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
"""
import copy
import itertools

import torch


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
