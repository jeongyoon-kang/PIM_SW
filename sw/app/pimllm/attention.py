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

from . import ops, profile


def pim_attention_forward(module, query: torch.Tensor, key, value,
                          attention_mask, scaling: float, dropout: float = 0.0,
                          **kwargs):
    """`key` and `value` are the PimLayer handle `PimLayer.update` returned.

    Returns `[B, S_q, H_q, D]`, which is what `LlamaAttention.forward` reshapes and
    feeds to o_proj.

    THE HEADS SHARE TWO DOORBELLS, not sixty-four.  Every head's Q.Kᵀ is an
    independent program of about a dozen instructions, and a doorbell costs an IMEM
    transfer, two MMIO writes and a poll loop whatever is behind it.  So all of them
    go into one program, the softmax happens on the host once they are all back, and
    all the S.V go into a second.  Not one MAC changes.

    THE SOFTMAX IS WHY IT IS TWO AND NOT ONE.  S.V needs the scores, and the card has
    no exp and no max — that host round trip is the one thing in attention no
    rearrangement removes.
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
    per = rt.per_group
    if d % per:
        raise NotImplementedError(
            f"head_dim {d} is not a multiple of {per} outputs per supergroup, so a "
            f"head's slice of V does not land on supergroup boundaries"
        )
    ngroup = (n_kv + per - 1) // per
    wide = rt.outputs(ngroup)                    # padding outputs included
    first_q = n_kv - s_q                         # absolute position of query 0

    # ---- 1. every position's, every head's q . K ---------------------------
    #
    # ONE BATCH FOR THE WHOLE LAYER.  The launches split where IMEM says so and
    # nowhere else; whether that is one doorbell or forty is the runtime's business
    # and not this function's, which is the whole point of writing it this way.
    raw = torch.empty(s_q * h_q, wide, dtype=query.dtype)
    with profile.span("attn q.K"), rt.batch() as bat:
        for i in range(s_q):
            for h in range(h_q):
                bat.add(layer.k, query[0, h, i].contiguous(), out_count=ngroup,
                        red_off=(h // group) * d, red_len=d,
                        out=raw[i * h_q + h])

    # ---- 2. mask and softmax, on the host, ALL OF IT AT ONCE ---------------
    #
    # One softmax over [S_q*H_q, S_kv] instead of one per head per position.  At a
    # 512-token prompt that is 1 call where the loop made 16384, and they were small
    # enough that the per-call overhead was most of them.
    with profile.span("attn softmax (host)"):
        sc = raw[:, :n_kv].float().reshape(s_q, h_q, n_kv) * scaling
        # CAUSAL MASK, built once.  Position i may see keys 0 .. first_q+i.
        pos = torch.arange(n_kv)
        allow = pos[None, :] <= (first_q + torch.arange(s_q))[:, None]
        sc = sc.masked_fill(~allow[:, None, :], float("-inf"))
        if attention_mask is not None:
            sc = sc + attention_mask[0, 0, :s_q, :n_kv].float()[:, None, :]
        probs = torch.softmax(sc, dim=-1).to(query.dtype).reshape(s_q * h_q, n_kv)
        probs = probs.contiguous()

    # ---- 3. every position's, every head's s . V ---------------------------
    #
    # ONLY THIS HEAD'S OUTPUTS.  V's output axis is H_kv*D wide; asking for all of
    # it would do H_kv times the work and discard all but one head's slice.
    got = torch.empty(s_q * h_q, rt.outputs(d // per), dtype=query.dtype)
    with profile.span("attn s.V"), rt.batch() as bat:
        for i in range(s_q):
            for h in range(h_q):
                bat.add(layer.v, probs[i * h_q + h],
                        out_first=(h // group) * d // per, out_count=d // per,
                        red_len=n_kv, out=got[i * h_q + h])

    # attn_weights is None: returning them would mean keeping an [S_q, S_kv] per head
    # that nothing asked for.
    return got[:, :d].reshape(1, s_q, h_q, d), None


def register(name: str = "pim") -> None:
    """Make `config._attn_implementation = name` legal.

    `AttentionInterface.get_interface` validates strictly — an unregistered name is a
    KeyError, not a silent fallback — so this has to run before the model is built."""
    ALL_ATTENTION_FUNCTIONS.register(name, pim_attention_forward)
