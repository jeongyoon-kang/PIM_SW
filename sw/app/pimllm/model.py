# SPDX-License-Identifier: MIT
"""Loading a model onto the card.

THE HOST MUST NEVER HOLD THE WHOLE MODEL.  `from_pretrained` puts every weight in
CPU memory, and for 3B that is 6 GB we are about to make a second copy of.  So the
swap below happens module by module and drops each `nn.Linear` as soon as its weight
is on the card — `PimLinear` keeps no host copy, so the original is garbage the
moment the last reference goes.

WHAT STAYS ON THE HOST, and why each one:

    embed_tokens   a lookup, not a matmul.  There is nothing for a MAC to do.
    RMSNorm, RoPE  elementwise over [1, hidden].  Microseconds.
    SiLU, residual the same.
    softmax        the card has no exp and no max.  See attention.py.
    sampling       argmax over the vocabulary, after lm_head has done the work.

lm_head DOES go to the card: it is one matmul per token over the whole vocabulary,
hundreds of megabytes read, and on the host that is the slowest thing in the loop.
When the embedding is tied, the host keeps its own copy for the LOOKUP — a lookup is
not a matmul — so the two coexist rather than share.
"""
from __future__ import annotations

import gc
import sys
import time

import torch
import torch.nn as nn
from transformers import TextStreamer

from . import attention, ops
from .budget import ModelShape, plan
from .cache import PimCache
from .device import Geometry


class TimedStreamer(TextStreamer):
    """Prints each token as it arrives, with the time it took and the ISRs it cost.

    The ISR count per token is the one number that separates "the card is slow" from
    "the host is slow": it does not change with host load, so a token whose time
    moved while its ISRs did not was spent somewhere other than the board.
    """

    def __init__(self, tokenizer, rt, **kw):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True, **kw)
        self.rt = rt
        self.t0 = self.tlast = time.monotonic()
        self.n = 0
        self.isr0 = rt.stats()["nisr"]
        self.text: list[str] = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        now = time.monotonic()
        isr = self.rt.stats()["nisr"]
        if not stream_end:
            # ONE STREAM, ONE LINE PER TOKEN.  The obvious shape — text on stdout,
            # timing on stderr — interleaves wrongly the moment the output is piped,
            # because stdout goes block-buffered there and stderr does not.  A token
            # that appears four tokens late is worse than no streaming at all.
            self.n += 1
            sys.stdout.write(f"{text!r:>18}   [{self.n:>3}  {now - self.tlast:5.2f}s"
                             f"  {isr - self.isr0:>7} ISR]\n")
            sys.stdout.flush()
            self.text.append(text)
        else:
            total = now - self.t0
            print(f"\n{''.join(self.text)}\n", flush=True)
            print(f"  {self.n} tokens in {total:.2f} s  "
                  f"({self.n / total if total else 0:.2f} tok/s)", flush=True)
        self.tlast, self.isr0 = now, isr


class PimLinear(nn.Module):
    """An `nn.Linear` whose weight lives on the card.

    NO HOST COPY OF THE WEIGHT.  That is the whole point, and it is why this is a
    fresh module rather than a subclass that keeps `self.weight` around.
    """

    def __init__(self, rt: ops.Runtime, weight: torch.Tensor,
                 bias: torch.Tensor | None, name: str = ""):
        super().__init__()
        self.rt = rt
        self.name = name
        self.out_features, self.in_features = weight.shape
        self.w = ops.Tensor.from_weight(weight.contiguous())
        # A bias is [out_features] and elementwise; there is no MAC for it and no
        # reason to send it.  Llama's projections have none, but Qwen's q/k/v do.
        self.register_buffer("bias", bias.clone() if bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [..., in_features].  Every row is a separate matvec: this hardware
        # broadcasts one vector against a resident matrix, so a batch of rows is a
        # loop and not a wider operation.
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        out = torch.empty(flat.shape[0], self.out_features, dtype=x.dtype)
        for i in range(flat.shape[0]):
            y = self.rt.matvec(self.w, flat[i].contiguous())
            out[i] = y[:self.out_features]
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape, self.out_features)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"on card ({self.w.bytes >> 20} MiB)")


def _swap_linears(model: nn.Module, rt: ops.Runtime, verbose: bool = True):
    """Replace every nn.Linear with a PimLinear, one at a time.

    `lm_head` included — see the module note.  Walked with named_modules() and
    replaced through the PARENT, because assigning to the module you are iterating
    over is how half a model ends up swapped.
    """
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    total = 0
    for name, mod in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        attr = name.rsplit(".", 1)[-1]
        new = PimLinear(rt, mod.weight.data, getattr(mod, "bias", None), name)
        setattr(parent, attr, new)
        total += new.w.bytes
        # Drop the host weight NOW rather than at the end of the loop, so peak host
        # memory is the model minus what has already moved, not the model plus it.
        del mod
        gc.collect()
        if verbose:
            print(f"    {name:<44} [{new.out_features} x {new.in_features}] "
                  f"{new.w.bytes >> 20:>5} MiB")
    return len(targets), total


class PimModel:
    """A loaded model with its weights on the card and a PIM KV cache.

    One object owns the 8 GiB, because weights and KV come out of one pool and a
    budget that is checked in two places is a budget that closes in neither.
    """

    def __init__(self, hf_id: str, s_max: int = 8192, geometry: Geometry | None = None,
                 verbose: bool = True):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.hf_id, self.s_max, self.verbose = hf_id, s_max, verbose
        cfg = AutoConfig.from_pretrained(hf_id)
        self.shape = ModelShape.from_hf_config(cfg, hf_id)

        # ---- the budget, BEFORE anything is downloaded or allocated ---------
        if geometry is None:
            from .device import geometry as read_geometry
            geometry = read_geometry()
        self.geometry = geometry
        self.budget = plan(geometry, self.shape, s_max)
        if verbose:
            print(self.budget.report())
        if not self.budget.fits:
            raise MemoryError(
                f"{self.shape.name} at S_max {s_max} needs "
                f"{self.budget.total / 2**30:.2f} GiB of the "
                f"{self.budget.capacity / 2**30:.2f} GiB on this image.  "
                f"S_max {self.budget.max_s_max()} would fit."
            )

        # ---- the runtime.  Sized from the shape, not guessed -----------------
        #
        # max_red is the longest reduction any op will ask for: an FFN's input, or
        # the sequence when S·V reduces over it.  max_out_groups is the widest
        # output, which is lm_head's vocabulary.
        per = geometry.outputs_per_group
        self.rt = ops.Runtime(
            max_red=max(self.shape.hidden, self.shape.intermediate, s_max),
            max_out_groups=(self.shape.vocab + per - 1) // per,
        )

        attention.register("pim")
        if verbose:
            print(f"\n  loading {hf_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=torch.bfloat16, attn_implementation="pim")
        self.model.eval()

        if verbose:
            print("  moving the weights onto the card")
        n, total = _swap_linears(self.model, self.rt, verbose)
        gc.collect()
        if verbose:
            print(f"  {n} linear layers, {total / 2**30:.2f} GiB on the card")

        self.cache = PimCache(self.rt, self.shape.layers, s_max)

    # ------------------------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 32, stream: bool = True,
                 **kw) -> str:
        """Generate, printing each token as it lands.

        STREAMING IS NOT A NICETY HERE.  A token takes seconds, so a run that
        printed nothing until the end would be indistinguishable from a hang for
        minutes at a time — and the per-token rate is the number you actually want
        to see, because it is what says whether the card or the host is the limit.
        """
        ids = self.tokenizer(prompt, return_tensors="pt")
        self.cache.reset()
        streamer = TimedStreamer(self.tokenizer, self.rt) if stream else None
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=False,
                past_key_values=self.cache, use_cache=True, streamer=streamer, **kw)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def free(self) -> None:
        self.cache.free()
        for m in self.model.modules():
            if isinstance(m, PimLinear):
                m.w.free()
