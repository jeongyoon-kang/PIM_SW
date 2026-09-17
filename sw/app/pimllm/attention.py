# SPDX-License-Identifier: MIT
"""Attention, with Q·Kᵀ and S·V on the card and the softmax between them on the host.

REGISTERED, NOT MONKEYPATCHED.  `transformers` has an `AttentionInterface` for
exactly this — "eager", "sdpa" and "flash_attention_2" are swapped the same way —
so `config._attn_implementation = "pim"` is the supported path and not a workaround.

WHAT RUNS WHERE, AND WHY THE SPLIT IS THERE AND NOT SOMEWHERE ELSE:

    q · Kᵀ        card    a matvec: reduce over D, one output per KEY TOKEN
    mask, softmax HOST    the card has no exp and no max.  fp32, as torch does it
    s · V         card    a matvec: reduce over the SEQUENCE, one output per dim

The softmax in the middle is what forces a GPR round trip per head per layer, and no
rearrangement removes it — it is not a staging decision, it is the one operation in
attention this hardware cannot do.

WHY red_off MATTERS HERE.  Eight KV heads of 128 share one 1024-element DRAM row and
`red_off = kv_head * D` picks which.  Without it the K cache would be 1024/D times
bigger than the data in it.  It is also the thing that would be wrong silently: a
mis-scaled offset reads a neighbouring head's keys and returns a plausible vector of
scores.

GQA IS FREE.  `repeat_kv` materialises H_q/H_kv copies of K and V; we never call it.
Several query heads simply name the same `red_off`.
"""
from __future__ import annotations

import math

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from . import ops


def _scores(rt: ops.Runtime, layer, q: torch.Tensor, kv_head: int,
            n_kv: int) -> torch.Tensor:
    """q · K[0..n_kv) for one head, as BF16 on the host."""
    d = layer.head_dim
    ngroup = (n_kv + rt.per_group - 1) // rt.per_group
    y = rt.matvec(layer.k, q, out_count=ngroup, red_off=kv_head * d, red_len=d)
    return y[:n_kv]


def _weighted_v(rt: ops.Runtime, layer, s: torch.Tensor, kv_head: int,
                n_kv: int) -> torch.Tensor:
    """s · V[0..n_kv) for one head.

    ONLY THIS HEAD'S OUTPUTS ARE COMPUTED.  V's output axis is H_kv*D wide, and
    asking for all of it would do H_kv times the work and throw away all but one
    head's slice.  out_first/out_count name the supergroups covering
    [kv_head*D, (kv_head+1)*D), which needs D to be a multiple of nch*nbank — 64 and
    128 both are on a 2-channel image."""
    d = layer.head_dim
    per = rt.per_group
    if d % per:
        raise NotImplementedError(
            f"head_dim {d} is not a multiple of {per} outputs per supergroup, so a "
            f"head's slice of V does not land on supergroup boundaries"
        )
    y = rt.matvec(layer.v, s, out_first=kv_head * d // per, out_count=d // per,
                  red_len=n_kv)
    return y[:d]


def pim_attention_forward(module, query: torch.Tensor, key, value,
                          attention_mask, scaling: float, dropout: float = 0.0,
                          **kwargs):
    """`key` and `value` are the PimLayer handle `PimLayer.update` returned.

    Returns `[B, S_q, H_q, D]`, which is what `LlamaAttention.forward` reshapes and
    feeds to o_proj — see `eager_attention_forward`, which transposes back the same
    way just before returning.
    """
    layer = key
    if not hasattr(layer, "k"):
        raise TypeError(
            "the 'pim' attention function was given ordinary tensors, which means "
            "the model is not using a PimCache.  Pass past_key_values=PimCache(...) "
            "to generate(), or the KV would be on the host while the weights are not."
        )
    rt = layer.rt
    b, h_q, s_q, d = query.shape
    if b != 1:
        raise NotImplementedError("batch 1 only; see pimllm/cache.py")

    n_kv = layer.cumulative                      # includes the tokens just appended
    group = h_q // layer.n_kv_heads              # GQA: query heads per KV head
    out = torch.empty(b, s_q, h_q, d, dtype=query.dtype)

    # The absolute position of query i, so the causal mask can be applied without
    # consulting `attention_mask` — which is built for a padded [S_q, S_kv] and would
    # have to be sliced to the same thing.
    first_q = n_kv - s_q

    for i in range(s_q):
        valid = first_q + i + 1                  # keys 0 .. first_q+i are visible
        for h in range(h_q):
            kv_head = h // group
            q = query[0, h, i].contiguous()

            sc = _scores(rt, layer, q, kv_head, n_kv).float() * scaling
            if valid < n_kv:                     # prefill; decode has valid == n_kv
                sc[valid:] = float("-inf")
            if attention_mask is not None:
                # Whatever else the model wants masked — padding, a sliding window.
                # Sliced to the keys that exist, since our cache reports its real
                # length rather than the reserved one.
                sc = sc + attention_mask[0, 0, i, :n_kv].float()
            sc = torch.softmax(sc, dim=-1).to(query.dtype).contiguous()

            out[0, i, h] = _weighted_v(rt, layer, sc, kv_head, n_kv)

    # attn_weights is None: returning them would mean keeping an [S_q, S_kv] per head
    # that nothing asked for.  output_attentions=True is unsupported and says so
    # through the None rather than by inventing a tensor.
    return out, None


def register(name: str = "pim") -> None:
    """Make `config._attn_implementation = name` legal.

    `AttentionInterface.get_interface` validates strictly — an unregistered name is a
    KeyError, not a silent fallback — so this has to run before the model is built."""
    ALL_ATTENTION_FUNCTIONS.register(name, pim_attention_forward)
