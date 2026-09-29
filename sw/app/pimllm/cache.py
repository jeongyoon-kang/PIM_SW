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

    K   OUT_PACKED [tokens, H_kv*D]  a token is an OUTPUT      one contiguous write,
                                     and several tokens are one contiguous write too
    V   RED_MAJOR  [H_kv*D, tokens]  a token is a REDUCTION    a write per output

The second line is the cost of attention on this machine.  It is not a defect of
this file: the bank axis is `row_bytes` apart in the address map, so any write that
spans banks without filling whole rows is a scatter.

THE CACHE DOES NOT KNOW HOW LONG THE SEQUENCE WILL BE.  K and V are growable
pim_tensors: before each update the layer grows them to hold every token so far, and
a grow allocates card pages only when a step boundary is crossed — K every
nch*nbank*pack tokens, V every 1024.  Every grow is recorded in PimCache.events, and
running out of card memory raises PimOutOfMemory at the token that needed the page.

NO HOST COPY.  `lazy_initialization` deliberately does not allocate the
`torch.zeros(B, H_kv, S_max, D)` that `StaticLayer` does — a host mirror of the
thing we just put on the card is exactly what this exists to avoid.  The base
class's `offload`, `prefetch` and `reorder_cache` all touch `self.keys` and are
overridden to refuse rather than to silently do nothing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from . import ops


class PimOutOfMemory(RuntimeError):
    """A K or V page could not be allocated: the card memory is full."""


@dataclass(frozen=True)
class GrowEvent:
    """One page allocation.  `tokens` is how many tokens the cache had to hold when
    it happened; `room` is how many it holds after."""
    tokens: int
    layer: int
    which: str        # "K" or "V"
    units: int
    bytes: int
    room: int
    t: float          # time.monotonic()


class PimLayer(CacheLayerMixin):
    """One layer's K and V, on the card.

    Batch size 1 only.  The board has one command stream and every op here is a
    matvec; a batch would be a matmul, which this hardware does not have.
    """

    is_compileable = False
    is_sliding = False

    def __init__(self, rt: ops.Runtime, layer_idx: int = -1,
                 events: list | None = None, unit_bytes: int = 0):
        super().__init__()
        self.rt = rt
        self.layer_idx = layer_idx
        self.events = events
        self.unit_bytes = unit_bytes
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

        # Both start with no room; update() grows them.  K needs no clearing — its
        # reduction length is D, always beat-aligned, and a garbage OUTPUT stays in
        # its own bank and is discarded.  V gets its zero invariant from append's
        # per-chunk clear on first entry, so a fresh page is cleared when the
        # sequence reaches it.
        # K is OUT_PACKED: when H_kv*D is half a row or less, neighbouring tokens
        # share a bank row, so K fills its rows and a run of tokens is one DMA.
        self.k = ops.Tensor.growable(ops.OUT_PACKED, width)
        self.v = ops.Tensor.growable(ops.RED_MAJOR, width)
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

        need = self.cumulative + s
        self._grow("K", self.k, need)
        self._grow("V", self.v, need)
        self.k.append(self.cumulative, k)
        self.v.append(self.cumulative, v)
        self.cumulative += s
        return self, self

    def _grow(self, which: str, t: ops.Tensor, need: int) -> None:
        try:
            units = t.grow(need)
        except RuntimeError as e:
            raise PimOutOfMemory(
                f"layer {self.layer_idx}: no card memory for a {which} page at token "
                f"{need} ({which} holds {t.room}): {e}") from e
        if units and self.events is not None:
            self.events.append(GrowEvent(need, self.layer_idx, which, units,
                                         units * self.unit_bytes, t.room,
                                         time.monotonic()))

    # ------------------------------------------------------- what HF asks -----
    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # The whole cache is valid, from 0.  Unlike StaticLayer this does not report
        # max_cache_len: nothing here computes over the unwritten tail, so there is
        # no padding for a mask to have to cover.
        return self.cumulative, 0

    def get_seq_length(self) -> int:
        return self.cumulative

    def get_max_length(self) -> int:
        # -1 is transformers' "no maximum", as for DynamicLayer.
        return -1

    # ------------------------------------------------------------- rollback ---
    def reset(self) -> None:
        """Between generations.  truncate(0) is free — it retracts the lazy-clear
        bookkeeping so the next append re-clears the chunk it touches, rather than
        moving half a gigabyte now.  The pages stay allocated for the next one."""
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
        room = self.k.room if self.k is not None else 0
        return (f"PimLayer(layer={self.layer_idx}, {self.cumulative} tokens, room "
                f"{room}, {self.n_kv_heads}x{self.head_dim})")


class PimCache(Cache):
    """A `Cache` whose layers are all `PimLayer`.

    Built eagerly with one layer per model layer rather than `layer_class_to_replicate`,
    because the layers need the runtime and a zero-argument constructor cannot have it.

    `events` is every page allocation, in order, across all layers.
    """

    def __init__(self, rt: ops.Runtime, n_layers: int):
        self.events: list[GrowEvent] = []
        unit_bytes = ops.geometry()["unit_bytes"]
        super().__init__(layers=[PimLayer(rt, i, self.events, unit_bytes)
                                 for i in range(n_layers)])

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
