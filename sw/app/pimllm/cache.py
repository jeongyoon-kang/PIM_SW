# SPDX-License-Identifier: MIT
"""The KV cache, resident on the card.

`transformers` 5.x split the cache: a `Cache` is a list of `CacheLayerMixin`
objects and the per-layer class is what you subclass.  The abstract surface is five
methods, which is what makes this tractable.

WHAT `update` RETURNS IS NOT A TENSOR.  It returns this layer, and that is legal only
because the thing it returns goes straight to OUR attention function and nowhere
else — `LlamaAttention.forward` calls `past_key_values.update(...)` and passes the
result to `attention_interface(...)` with nothing in between.  `repeat_kv` lives
INSIDE `eager_attention_forward`, not in the caller, so our function simply does not
call it; the GQA expansion that materialises H_q/H_kv copies of K and V never
happens.  That is not an optimisation we added, it is work we decline to do.

WHY K AND V HAVE DIFFERENT LAYOUTS.  It follows from which axis each kernel reduces
over and is not a choice — see include/pimrt/pim_tensor.h:

    K   OUT_MAJOR  [S_max, H_kv*D]   a token is an OUTPUT      one contiguous write
    V   RED_MAJOR  [H_kv*D, S_max]   a token is a REDUCTION    a write per output

The second line is the cost of attention on this machine.  It is not a defect of
this file: the bank axis is `row_bytes` apart in the address map, so any write that
spans banks without filling whole rows is a scatter.

NO HOST COPY.  `lazy_initialization` deliberately does not allocate the
`torch.zeros(B, H_kv, S_max, D)` that `StaticLayer` does — a host mirror of the
thing we just put on the card is exactly what this exists to avoid.  The base
class's `offload`, `prefetch` and `reorder_cache` all touch `self.keys` and are
overridden to refuse rather than to silently do nothing.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from . import ops


class PimLayer(CacheLayerMixin):
    """One layer's K and V, on the card.

    Batch size 1 only.  The board has one command stream and every op here is a
    matvec; a batch would be a matmul, which this hardware does not have.
    """

    is_compileable = False
    is_sliding = False

    def __init__(self, rt: ops.Runtime, s_max: int, layer_idx: int = -1):
        super().__init__()
        self.rt = rt
        self.s_max = s_max
        self.layer_idx = layer_idx
        self.k: ops.Tensor | None = None
        self.v: ops.Tensor | None = None
        self.cumulative = 0
        self.n_kv_heads = 0
        self.head_dim = 0

    # ---------------------------------------------------------------- setup ---
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        b, h_kv, _, d = key_states.shape
        if b != 1:
            raise NotImplementedError(
                f"batch {b}: every kernel here is a matvec, so a batch would be a "
                f"matmul and this hardware has none.  Generate one sequence at a time."
            )
        self.dtype, self.device = key_states.dtype, key_states.device
        self.n_kv_heads, self.head_dim = h_kv, d
        width = h_kv * d

        # NO PIM_ALLOC_F_ZERO on either.  K needs none — its reduction length is D,
        # always beat-aligned, and a garbage OUTPUT stays in its own bank and is
        # discarded.  V needs the zero invariant but gets it from append's lazy
        # per-chunk clear, which costs the same in total and only for the chunks a
        # generation actually reaches.
        self.k = ops.Tensor(ops.OUT_MAJOR, self.s_max, width)
        self.v = ops.Tensor(ops.RED_MAJOR, width, self.s_max)
        self.is_initialized = True

    # --------------------------------------------------------------- update ---
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               *args, **kwargs):
        """Append, and hand back a handle rather than tensors.  See the module note."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # [B, H_kv, S, D] -> [S, H_kv*D].  The transpose looks expensive and is not
        # what it appears: the tensor coming in is a VIEW of what k_proj produced,
        # which was [B, S, H_kv, D] already — this puts the memory back the way it
        # was laid out.  .contiguous() is required and is deliberately explicit, per
        # the rule in ops.py.
        s = key_states.shape[2]
        k = key_states.transpose(1, 2).reshape(s, -1).contiguous()
        v = value_states.transpose(1, 2).reshape(s, -1).contiguous()

        if self.cumulative + s > self.s_max:
            raise RuntimeError(
                f"layer {self.layer_idx}: {self.cumulative + s} tokens, past the "
                f"S_max {self.s_max} this cache reserved.  pim_tensor does not grow; "
                f"open the model with a larger s_max (and check it still fits)."
            )
        self.k.append(self.cumulative, k)
        self.v.append(self.cumulative, v)
        self.cumulative += s
        return self, self

    # ------------------------------------------------------- what HF asks -----
    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # The whole cache is valid, from 0.  Unlike StaticLayer this does not report
        # max_cache_len: nothing here computes over the unwritten tail, so there is
        # no padding for a mask to have to cover.
        return self.cumulative, 0

    def get_seq_length(self) -> int:
        return self.cumulative

    def get_max_length(self) -> int:
        return self.s_max

    # ------------------------------------------------------------- rollback ---
    def reset(self) -> None:
        """Between generations.  truncate(0) is free — it retracts the lazy-clear
        bookkeeping so the next append re-clears the chunk it touches, rather than
        moving half a gigabyte now."""
        if self.is_initialized:
            self.k.truncate(0)
            self.v.truncate(0)
        self.cumulative = 0

    def crop(self, max_length: int) -> None:
        """THE CALL THAT CLOSES THE ZERO INVARIANT.  Everything above the new
        frontier holds the previous generation's values, and the final beat of the
        next launch reads there.  Skipping it is NaN with no error."""
        if self.is_initialized and max_length < self.cumulative:
            self.k.truncate(max_length)
            self.v.truncate(max_length)
            self.cumulative = max_length

    # --------------------------------------- what this cache cannot do --------
    # All three touch self.keys/self.values in the base class, which are None here on
    # purpose.  Refusing beats inheriting a method that would quietly do nothing.
    def offload(self):
        raise NotImplementedError("a PIM cache does not offload; it is already off the host")

    def prefetch(self):
        raise NotImplementedError("a PIM cache does not offload, so there is nothing to prefetch")

    def reorder_cache(self, beam_idx) -> None:
        raise NotImplementedError(
            "beam search would reorder the cache, which means permuting both tensors "
            "on the card.  Use greedy or sampling."
        )

    def free(self) -> None:
        for t in (self.k, self.v):
            if t is not None:
                t.free()
        self.k = self.v = None
        self.is_initialized = False

    def __repr__(self):
        return (f"PimLayer(layer={self.layer_idx}, {self.cumulative}/{self.s_max} "
                f"tokens, {self.n_kv_heads}x{self.head_dim})")


class PimCache(Cache):
    """A `Cache` whose layers are all `PimLayer`.

    Built eagerly with one layer per model layer rather than `layer_class_to_replicate`,
    because the layers need the runtime and the S_max and a zero-argument constructor
    cannot have them.
    """

    def __init__(self, rt: ops.Runtime, n_layers: int, s_max: int):
        super().__init__(layers=[PimLayer(rt, s_max, i) for i in range(n_layers)])
        self.s_max = s_max

    def reset(self) -> None:
        for lay in self.layers:
            lay.reset()

    def crop(self, max_length: int) -> None:
        for lay in self.layers:
            lay.crop(max_length)

    def free(self) -> None:
        for lay in self.layers:
            lay.free()

    @property
    def bytes(self) -> int:
        return sum(l.k.bytes + l.v.bytes for l in self.layers if l.is_initialized)
