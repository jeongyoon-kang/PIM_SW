# SPDX-License-Identifier: MIT
"""Does this model fit, and where would every tensor go.

Weights and the KV cache come out of ONE pool — 8 GiB on the ch2 image.  The weights
are placed once at load; the KV cache grows page by page as tokens arrive, so what
this answers for the KV is how far a sequence can go: the most tokens whose K and V
pages still fit next to the weights, capped by the model's own context length.

PURE ARITHMETIC ON A Geometry.  No device, no libpim call, no board.  You can ask
"would Llama-3.2-3B fit on a 2-channel image at 8K" from a laptop, and that is the
point — the answer decides which model you go and fetch.

PADDING IS COUNTED, NOT ESTIMATED.  Every tensor is padded to whole supergroups on
the output axis and whole beats on the reduction axis, and the KV grows in whole
units.  The numbers below are what pim_tensor_bytes() and pim_tensor_grow() will
use, computed the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .device import Geometry

BF16 = 2


def _roundup(v: int, to: int) -> int:
    return (v + to - 1) // to * to


def tensor_bytes(g: Geometry, nout: int, nred: int, packed: bool = False) -> int:
    """What pim_tensor_bytes() gives for this shape.  Kept in step with
    runtime/pim_tensor.c by being the same lines.  `packed` is OUT_PACKED: each
    output takes the smallest power-of-two slot that holds nred, and a row holds
    elems_per_row / slot of them."""
    if not nout or not nred:
        return 0
    nredpad = _roundup(nred, 16)
    pack, slot = 1, g.elems_per_row
    while packed and slot // 2 >= nredpad and slot // 2 >= 16:
        slot //= 2
        pack *= 2
    rows = _roundup(nout, g.outputs_per_group * pack) // (g.outputs_per_group * pack)
    nchunks = (nredpad + g.elems_per_row - 1) // g.elems_per_row
    return rows * nchunks * g.unit_bytes


def _k_step(g: Geometry, kv_width: int) -> tuple[int, int]:
    """(tokens, units) one K grow adds per layer.  K is OUT_PACKED with the token on
    the output axis: a step is one row of every bank, nch*nbank*pack tokens, and
    takes one unit per reduction chunk."""
    nredpad = _roundup(kv_width, 16)
    pack, slot = 1, g.elems_per_row
    while slot // 2 >= nredpad and slot // 2 >= 16:
        slot //= 2
        pack *= 2
    return g.outputs_per_group * pack, (nredpad + g.elems_per_row - 1) // g.elems_per_row


def _v_step(g: Geometry, kv_width: int) -> tuple[int, int]:
    """(tokens, units) one V grow adds per layer.  V is RED_MAJOR with the token on
    the reduction axis: a step is one chunk, 1024 tokens, and takes one unit per
    output group."""
    return g.elems_per_row, _roundup(kv_width, g.outputs_per_group) // g.outputs_per_group


def _k_fill(g: Geometry, kv_width: int) -> str:
    """How K sits in its rows.  An OUT_PACKED K gives a token the smallest
    power-of-two slot that holds kv_width and puts row / slot tokens in a row."""
    slot = g.elems_per_row
    while slot // 2 >= kv_width and slot // 2 >= 16:
        slot //= 2
    used = min(kv_width, slot) / slot
    return f"  ({g.elems_per_row // slot} token(s) per row, {used:.0%} of each slot used)"


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
    # max_position_embeddings: the context the model was trained for.  0 if unknown.
    max_position: int = 0
    # What to hand transformers when it is time to actually load this.  The short
    # names below exist so --dry-run needs no network; without this the same name
    # then fails at AutoTokenizer, which is a confusing place to learn it.
    hf_id: str | None = None

    @property
    def kv_width(self) -> int:
        """H_kv * D — one token's worth of K, and the output axis of V.

        A DRAM row is 1024 BF16.  K gives each token the smallest power-of-two
        slot that holds kv_width, so 1024 is one token per row and 512 is two.  It
        is also the cap on head packing: COL + OPSIZE <= 64 beats means
        kv_width <= 1024."""
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
            max_position=int(getattr(cfg, "max_position_embeddings", 0) or 0),
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
    weights: list[Entry] = field(default_factory=list)

    @property
    def weight_bytes(self) -> int:
        return sum(e.total for e in self.weights)

    @property
    def capacity(self) -> int:
        return self.geometry.dram.bytes

    @property
    def fits(self) -> bool:
        """The weights and the first K and V page of every layer."""
        return self.weight_bytes + self.kv_bytes(1) <= self.capacity

    def kv_bytes(self, n: int) -> int:
        """Card memory the KV holds after `n` tokens: whole K and V steps, every layer."""
        if n <= 0:
            return 0
        g, w = self.geometry, self.shape.kv_width
        (kt, ku), (vt, vu) = _k_step(g, w), _v_step(g, w)
        units = -(-n // kt) * ku + -(-n // vt) * vu
        return self.shape.layers * units * g.unit_bytes

    def memory_max_tokens(self) -> int:
        """The most tokens whose K and V pages fit next to the weights."""
        free = self.capacity - self.weight_bytes
        if self.kv_bytes(1) > free:
            return 0
        lo, hi = 1, 1
        while self.kv_bytes(hi) <= free:
            lo, hi = hi, hi * 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.kv_bytes(mid) <= free:
                lo = mid
            else:
                hi = mid
        return lo

    def max_tokens(self) -> tuple[int, str]:
        """(tokens, what limits it): card memory, or the model's trained context."""
        mem = self.memory_max_tokens()
        ctx = self.shape.max_position
        if ctx and ctx < mem:
            return ctx, "model context"
        return mem, "card memory"

    def report(self) -> str:
        g, s = self.geometry, self.shape
        (kt, ku), (vt, vu) = _k_step(g, s.kv_width), _v_step(g, s.kv_width)
        out = [
            f"{s.name}  —  {s.layers} layers, hidden {s.hidden}, "
            f"{s.n_heads} heads / {s.n_kv_heads} KV of {s.head_dim}, "
            f"ffn {s.intermediate}, vocab {s.vocab}",
            f"on {g.nch} ch x {g.nbank} bank:  H_kv*D = {s.kv_width} of "
            f"{g.elems_per_row} per row"
            + _k_fill(g, s.kv_width),
            "",
            f"  {'':<22} {'shape':>16} {'x':>5} {'each':>10} {'total':>10}",
            "  weights",
        ]
        for e in self.weights:
            out.append(
                f"    {e.what:<20} {f'[{e.nout} x {e.nred}]':>16} {e.count:>5} "
                f"{_mib(e.bytes_each):>10} {_mib(e.total):>10}"
            )
        out += [
            "",
            "  KV, grown as tokens arrive (each layer):",
            f"    K  OUT_PACKED  +{_mib(ku * g.unit_bytes)} every {kt} tokens",
            f"    V  RED_MAJOR   +{_mib(vu * g.unit_bytes)} every {vt} tokens",
            f"    {_mib(self.kv_bytes(vt) // vt)} per token over all "
            f"{s.layers} layers",
            "",
            f"  {'weights':<22} {_mib(self.weight_bytes):>44}   of {_mib(self.capacity)}",
        ]
        if not self.fits:
            out.append("  DOES NOT FIT: the weights alone leave no room for the KV.  "
                       "This model needs a bigger image, quantisation, or layer "
                       "streaming.")
            return "\n".join(out)
        n, why = self.max_tokens()
        mem = self.memory_max_tokens()
        line = f"  max tokens  {n:,}  (limited by {why}"
        if why == "model context":
            line += f"; card memory would hold {mem:,}"
        out.append(line + ")")
        return "\n".join(out)


def _mib(n: int) -> str:
    if n < 2**20:
        return f"{n / 2**10:,.0f} KiB"
    return f"{n / 2**20:,.1f} MiB" if n < 2**30 else f"{n / 2**30:,.2f} GiB"


def plan(g: Geometry, s: ModelShape) -> Budget:
    """Every weight the model needs, sized the way pim_tensor will size it.  The KV
    is not planned at a length; Budget.kv_bytes(n) and max_tokens() answer for it.

    THE LAYOUTS ARE NOT A CHOICE, they follow from which axis each kernel reduces
    over — see pim_tensor.h:

        every linear   OUT_MAJOR  [out_features x in_features]
        K cache        OUT_PACKED [tokens x H_kv*D]   token is an OUTPUT
        V cache        RED_MAJOR  [H_kv*D x tokens]   token is a REDUCTION STEP
    """
    b = Budget(geometry=g, shape=s)
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
    return b


# Shapes for the two targets, so `generate.py --dry-run` works with no model
# downloaded and no network.  Checked against the published configs; the real ones
# come from `ModelShape.from_hf_config` once transformers has loaded one.
# The Instruct variants have IDENTICAL shapes to their base models — same layers,
# same hidden, same 8 KV heads — so the budget, the layout and every kernel are
# unchanged.  What differs is entirely on the host: they carry a chat template, and
# without it a prompt is continued rather than answered.
KNOWN = {
    "llama-3.2-1b": ModelShape("Llama-3.2-1B", 16, 2048, 32, 8, 64, 8192, 128256,
                               max_position=131072, hf_id="meta-llama/Llama-3.2-1B"),
    "llama-3.2-3b": ModelShape("Llama-3.2-3B", 28, 3072, 24, 8, 128, 8192, 128256,
                               max_position=131072, hf_id="meta-llama/Llama-3.2-3B"),
    "llama-3.2-1b-instruct": ModelShape("Llama-3.2-1B-Instruct", 16, 2048, 32, 8, 64,
                                        8192, 128256,
                                        max_position=131072,
                                        hf_id="meta-llama/Llama-3.2-1B-Instruct"),
    "llama-3.2-3b-instruct": ModelShape("Llama-3.2-3B-Instruct", 28, 3072, 24, 8, 128,
                                        8192, 128256,
                                        max_position=131072,
                                        hf_id="meta-llama/Llama-3.2-3B-Instruct"),
}
