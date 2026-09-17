# SPDX-License-Identifier: MIT
"""Does this model fit, and where would every tensor go.

THE QUESTION HAS TO BE ANSWERED AT STARTUP, not discovered at token 3000.  Weights
and the KV cache come out of ONE pool — 8 GiB on the ch2 image — and the KV half is
reserved for a fixed S_max because pim_tensor does not grow.  So `S_max` is not a
tuning knob you can leave for later: it is what closes the budget, and a model that
does not fit should fail while you can still change something.

PURE ARITHMETIC ON A Geometry.  No device, no libpim call, no board.  You can ask
"would Llama-3.2-3B fit on a 2-channel image at 8K" from a laptop, and that is the
point — the answer decides which model you go and fetch.

PADDING IS COUNTED, NOT ESTIMATED.  Every tensor is padded to whole supergroups on
the output axis and whole beats on the reduction axis, and for a KV cache at D = 64
that is a real 2x on K.  The numbers below are what pim_tensor_bytes() will return,
computed the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .device import Geometry

BF16 = 2


def _roundup(v: int, to: int) -> int:
    return (v + to - 1) // to * to


def tensor_bytes(g: Geometry, nout: int, nred: int) -> int:
    """What pim_tensor_bytes() gives for this shape.  Kept in step with
    runtime/pim_tensor.c by being the same three lines, not by being careful."""
    if not nout or not nred:
        return 0
    noutpad = _roundup(nout, g.outputs_per_group)
    nredpad = _roundup(nred, 16)
    ngroups = noutpad // g.outputs_per_group
    nchunks = (nredpad + g.elems_per_row - 1) // g.elems_per_row
    return ngroups * nchunks * g.unit_bytes


@dataclass(frozen=True)
class ModelShape:
    """The six numbers that decide the layout.  Read off a HF config; see
    `from_hf_config`."""

    name: str
    layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tie_embeddings: bool = True
    # What to hand transformers when it is time to actually load this.  The short
    # names below exist so --dry-run needs no network; without this the same name
    # then fails at AutoTokenizer, which is a confusing place to learn it.
    hf_id: str | None = None

    @property
    def kv_width(self) -> int:
        """H_kv * D — one token's worth of K, and the output axis of V.

        THE NUMBER TO WATCH.  A DRAM row is 1024 BF16, so kv_width == 1024 fills one
        exactly and wastes nothing, while 512 wastes half of K.  It is also the cap
        on head packing: COL + OPSIZE <= 64 beats means kv_width <= 1024."""
        return self.n_kv_heads * self.head_dim

    @classmethod
    def from_hf_config(cls, cfg, name: str | None = None) -> "ModelShape":
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cls(
            name=name or getattr(cfg, "_name_or_path", "model"),
            layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            n_heads=cfg.num_attention_heads,
            n_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim=head_dim,
            intermediate=cfg.intermediate_size,
            vocab=cfg.vocab_size,
            tie_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
        )


@dataclass
class Entry:
    what: str
    nout: int
    nred: int
    count: int
    bytes_each: int

    @property
    def total(self) -> int:
        return self.count * self.bytes_each


@dataclass
class Budget:
    geometry: Geometry
    shape: ModelShape
    s_max: int
    weights: list[Entry] = field(default_factory=list)
    kv: list[Entry] = field(default_factory=list)

    @property
    def weight_bytes(self) -> int:
        return sum(e.total for e in self.weights)

    @property
    def kv_bytes(self) -> int:
        return sum(e.total for e in self.kv)

    @property
    def total(self) -> int:
        return self.weight_bytes + self.kv_bytes

    @property
    def capacity(self) -> int:
        return self.geometry.dram.bytes

    @property
    def fits(self) -> bool:
        return self.total <= self.capacity

    def max_s_max(self) -> int:
        """The largest S_max that still fits, rounded down to a whole DRAM row of
        tokens — V's reduction axis is the sequence, so a partial row buys nothing
        but does cost a whole unit."""
        per_row = self.geometry.elems_per_row
        free = self.capacity - self.weight_bytes
        if free <= 0:
            return 0
        lo, hi = 0, 1 << 20
        while lo < hi:
            mid = (lo + hi + per_row) // 2 // per_row * per_row
            if mid <= lo:
                break
            if plan(self.geometry, self.shape, mid).kv_bytes <= free:
                lo = mid
            else:
                hi = mid - per_row
        return lo

    def report(self) -> str:
        g, s = self.geometry, self.shape
        out = [
            f"{s.name}  —  {s.layers} layers, hidden {s.hidden}, "
            f"{s.n_heads} heads / {s.n_kv_heads} KV of {s.head_dim}, "
            f"ffn {s.intermediate}, vocab {s.vocab}",
            f"on {g.nch} ch x {g.nbank} bank:  H_kv*D = {s.kv_width} of "
            f"{g.elems_per_row} per row"
            + ("  (fills a row exactly)" if s.kv_width == g.elems_per_row
               else f"  ({g.elems_per_row / s.kv_width:.3g}x waste on K)"),
            "",
            f"  {'':<22} {'shape':>16} {'x':>5} {'each':>10} {'total':>10}",
        ]
        for group, title in ((self.weights, "weights"), (self.kv, f"KV at S_max {self.s_max}")):
            out.append(f"  {title}")
            for e in group:
                out.append(
                    f"    {e.what:<20} {f'[{e.nout} x {e.nred}]':>16} {e.count:>5} "
                    f"{_mib(e.bytes_each):>10} {_mib(e.total):>10}"
                )
        out += [
            "",
            f"  {'weights':<22} {_mib(self.weight_bytes):>44}",
            f"  {'KV':<22} {_mib(self.kv_bytes):>44}",
            f"  {'total':<22} {_mib(self.total):>44}   of {_mib(self.capacity)}",
        ]
        if self.fits:
            out.append(
                f"  FITS, {_mib(self.capacity - self.total)} spare "
                f"(S_max could be {self.max_s_max()})"
            )
        else:
            out.append(
                f"  DOES NOT FIT by {_mib(self.total - self.capacity)}.  "
                + (f"S_max {self.max_s_max()} would fit."
                   if self.weight_bytes <= self.capacity
                   else "The weights alone are over capacity; this model needs a "
                        "bigger image, quantisation, or layer streaming.")
            )
        return "\n".join(out)


def _mib(n: int) -> str:
    return f"{n / 2**20:,.1f} MiB" if n < 2**30 else f"{n / 2**30:,.2f} GiB"


def plan(g: Geometry, s: ModelShape, s_max: int) -> Budget:
    """Every tensor the model needs, sized the way pim_tensor will size it.

    THE LAYOUTS ARE NOT A CHOICE, they follow from which axis each kernel reduces
    over — see pim_tensor.h.  Stated here so the table can be read against it:

        every linear   OUT_MAJOR  [out_features x in_features]
        K cache        OUT_MAJOR  [S_max x H_kv*D]   token is an OUTPUT
        V cache        RED_MAJOR  [H_kv*D x S_max]   token is a REDUCTION STEP
    """
    b = Budget(geometry=g, shape=s, s_max=s_max)
    q_out = s.n_heads * s.head_dim
    kv_out = s.kv_width

    per_layer = [
        ("q_proj", q_out, s.hidden),
        ("k_proj", kv_out, s.hidden),
        ("v_proj", kv_out, s.hidden),
        ("o_proj", s.hidden, q_out),
        ("gate_proj", s.intermediate, s.hidden),
        ("up_proj", s.intermediate, s.hidden),
        ("down_proj", s.hidden, s.intermediate),
    ]
    for what, nout, nred in per_layer:
        b.weights.append(Entry(what, nout, nred, s.layers, tensor_bytes(g, nout, nred)))

    # lm_head goes to PIM: it is one matmul per token over the whole vocabulary, and
    # on the host that is hundreds of megabytes read per token.  When the embedding
    # is tied, the HOST still needs its own copy for the lookup — a lookup is not a
    # matmul and does not belong here — so the two coexist rather than share.
    b.weights.append(Entry("lm_head", s.vocab, s.hidden, 1, tensor_bytes(g, s.vocab, s.hidden)))

    b.kv.append(Entry("K (OUT_MAJOR)", s_max, kv_out, s.layers, tensor_bytes(g, s_max, kv_out)))
    b.kv.append(Entry("V (RED_MAJOR)", kv_out, s_max, s.layers, tensor_bytes(g, kv_out, s_max)))
    return b


# Shapes for the two targets, so `generate.py --dry-run` works with no model
# downloaded and no network.  Checked against the published configs; the real ones
# come from `ModelShape.from_hf_config` once transformers has loaded one.
KNOWN = {
    "llama-3.2-1b": ModelShape("Llama-3.2-1B", 16, 2048, 32, 8, 64, 8192, 128256,
                               hf_id="meta-llama/Llama-3.2-1B"),
    "llama-3.2-3b": ModelShape("Llama-3.2-3B", 28, 3072, 24, 8, 128, 8192, 128256,
                               hf_id="meta-llama/Llama-3.2-3B"),
}
