"""
Loss-of-plasticity diagnostics of the LTM (Dave 16 Nov 2026; Dohare et al. 2024 track these alongside accuracy):
  plast_dormant_b<i>  fraction of block i's feed-forward hidden units that are dormant: mean |GELU output| over the
                      probe samples and tokens, divided by the layer's mean, at most 0.1 (Sokar et al. 2023's
                      tau-dormant; GELU units are never exactly dead)
  plast_eff_rank      effective rank of the LTM's features (the normalised CLS token) on the probe samples:
                      exp of the entropy of the normalised singular values (Roy and Vetterli 2007), at most d_model
  plast_weight_norm   L2 norm of all the LTM's parameters
"""
import torch

DORMANT_TAU = 0.1


def ltm_net(model):
    """The LTM network of any stream model: its own net, or its LTM's (heads, CLS)."""
    return model.ltm.net if hasattr(model, "ltm") else model.net


def effective_rank(features):
    s = torch.linalg.svdvals(features.double())
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def diagnostics(net, x):
    acts = []
    hooks = [block.mlp[1].register_forward_hook(lambda m, i, o: acts.append(o)) for block in net.blocks]
    training = net.training
    net.eval()
    try:
        features = net.run(x)["features"]
    finally:
        net.train(training)
        for h in hooks:
            h.remove()
    out = {}
    for i, a in enumerate(acts):
        score = a.abs().mean((0, 1))
        out[f"plast_dormant_b{i}"] = float((score / score.mean() <= DORMANT_TAU).float().mean())
    out["plast_eff_rank"] = effective_rank(features)
    out["plast_weight_norm"] = float(torch.sqrt(sum((p.double() ** 2).sum() for p in net.parameters())))
    return out
