"""
Biasable transformer: a non-causal, pre-LayerNorm transformer over a set of tokens with a CLS readout and a
classifier head, with hooks for a top-down bias from another module (the STM). Task-agnostic: it sees tokens only.

Bias (one block, chosen by `bias_block`; any block, so it generalises to deeper stacks). The bias input `a` has
shape (B, bias_size); a = 0 is neutral (exactly the unbiased forward pass). Both modes are multiplicative and
bounded by `bias_scale`, so a bias can reweight what the block computes but never remove the bottom-up input:

- "where" (default): an additive per-key term on the attention logits, softmax(QK^T / sqrt(d) + s tanh(a_j)),
  i.e. attention to token j reweighted by a factor in [e^-s, e^s]. bias_size = number of input tokens (the CLS key is
  never biased). Same mechanism as padding masks and ALiBi; applied to every head and every query.
- "what": a per-feature gain exp(s tanh(a)) in [e^-s, e^s] on the block's LayerNorm output, before the Q/K/V
  projections (not on the residual stream, where the next LayerNorm would normalise it away). bias_size = d_model.
- "q", "k", "v" and their combinations ("qk", "qv", "kv", "qkv"; Dave 2 Nov 2026): per-channel gains exp(s tanh(a))
  on the projected queries, keys and/or values (after the Q/K/V projections, before the split into heads), one
  d_model block of the bias per part in q, k, v order. bias_size = d_model x number of parts.

Hooks (`run` returns all of them in one pass):
- "pre_bias": the residual stream entering the bias block (B, 1 + N, d_model); with bias_block 0 it does not depend
  on the bias.
- "features": the CLS output after the final LayerNorm, the input of the classifier head (B, d_model).
- "logits": (B, num_classes).
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self, d_model, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
                                 nn.Linear(mlp_ratio * d_model, d_model))

    def forward(self, x, gain=None, key_bias=None, qkv_gain=None):
        """x (B, L, d); gain (B, d) multiplies the LayerNorm output; key_bias (B, L) is added to the attention logits
        of every query for each key; qkv_gain: {"q"|"k"|"v": (B, d)} multiplies the projected queries / keys /
        values."""
        B, L, d = x.shape
        h = self.ln1(x)
        if gain is not None:
            h = h * gain[:, None, :]
        q, k, v = self.qkv(h).split(d, dim=2)
        if qkv_gain:
            q = q * qkv_gain["q"][:, None, :] if "q" in qkv_gain else q
            k = k * qkv_gain["k"][:, None, :] if "k" in qkv_gain else k
            v = v * qkv_gain["v"][:, None, :] if "v" in qkv_gain else v
        q, k, v = (t.reshape(B, L, self.heads, d // self.heads).transpose(1, 2) for t in (q, k, v))
        mask = None if key_bias is None else key_bias[:, None, None, :].to(q.dtype)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.ln2(x))


class BiasableTransformer(nn.Module):

    BIAS_WHERE = "where"
    BIAS_WHAT = "what"
    BIAS_QKV = ("q", "k", "v", "qk", "qv", "kv", "qkv")

    def __init__(self, num_tokens, token_size, num_classes, d_model=64, heads=4, blocks=1, mlp_ratio=2,
                 bias_block=0, bias_mode=BIAS_WHERE, bias_scale=math.log(4.0)):
        super().__init__()
        if bias_mode not in (self.BIAS_WHERE, self.BIAS_WHAT) + self.BIAS_QKV:
            raise ValueError(f"unknown bias mode {bias_mode}")
        if not 0 <= bias_block < blocks:
            raise ValueError(f"bias_block {bias_block} outside 0..{blocks - 1}")
        self.num_tokens = num_tokens
        self.token_size = token_size
        self.d_model = d_model
        self.bias_block = bias_block
        self.bias_mode = bias_mode
        self.bias_scale = bias_scale
        self.embed = nn.Linear(token_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(Block(d_model, heads, mlp_ratio) for _ in range(blocks))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    @property
    def bias_size(self):
        if self.bias_mode == self.BIAS_WHERE:
            return self.num_tokens
        if self.bias_mode in self.BIAS_QKV:
            return self.d_model * len(self.bias_mode)
        return self.d_model

    def tokens(self, x):
        """Flattened (B, N * token_size) or (B, N, token_size) input to (B, N, token_size)."""
        return x.reshape(len(x), self.num_tokens, self.token_size)

    def run(self, x, bias=None):
        x = self.embed(self.tokens(x))
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos
        pre_bias = None
        for i, block in enumerate(self.blocks):
            gain = key_bias = qkv_gain = None
            if i == self.bias_block:
                pre_bias = x
                if bias is not None:
                    bounded = self.bias_scale * torch.tanh(bias)
                    if self.bias_mode == self.BIAS_WHERE:
                        key_bias = F.pad(bounded, (1, 0))   # CLS key unbiased
                    elif self.bias_mode in self.BIAS_QKV:
                        parts = torch.exp(bounded).split(self.d_model, dim=1)
                        qkv_gain = dict(zip(self.bias_mode, parts))
                    else:
                        gain = torch.exp(bounded)
            x = block(x, gain=gain, key_bias=key_bias, qkv_gain=qkv_gain)
        features = self.norm(x[:, 0])
        return {"pre_bias": pre_bias, "features": features, "logits": self.head(features)}

    def forward(self, x, bias=None):
        return self.run(x, bias)["logits"]
