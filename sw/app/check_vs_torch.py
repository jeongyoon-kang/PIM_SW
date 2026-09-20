#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Is the card's answer the same answer, or only a plausible one?

    ./check_vs_torch.py [--model llama-3.2-1b] [--prompt "..."]

WHY THE GENERATED TEXT CANNOT ANSWER THIS.  Greedy decoding takes an argmax over
128256 logits.  Two of them within a rounding of each other swap places on the
smallest arithmetic difference, and from there the sequences diverge completely and
stay fluent.  So "the card said something else" is evidence of nothing, in either
direction — a real bug and a last-bit difference look identical after four tokens.

WHAT DOES ANSWER IT is the logit vector from ONE forward pass, compared before any
argmax.  Block floating point is not IEEE: a beat takes the maximum exponent over
its sixteen products and the smaller ones are truncated during alignment, so a
disagreement of a few percent is the hardware working correctly and is not
recoverable by being careful.  What a BUG looks like is different in kind:
uncorrelated, or right for one head and wrong for another, or fine until the
sequence crosses a beat boundary.

So this reports THREE things, and the third is the one that matters:

    max relative error   scale of the disagreement
    correlation          whether it is noise or a different computation
    top-k agreement      whether the disagreement reaches the decision
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from pimllm import budget as B


def torch_logits(hf_id: str, prompt: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16)
    model.eval()
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids)
    del model
    return out.logits[0, -1].float(), ids


def pim_logits(hf_id: str, prompt: str, s_max: int):
    from pimllm.model import PimModel
    m = PimModel(hf_id, s_max=s_max, verbose=False)
    ids = m.tokenizer(prompt, return_tensors="pt")
    m.cache.reset()
    with torch.no_grad():
        out = m.model(**ids, past_key_values=m.cache, use_cache=True)
    lg = out.logits[0, -1].float()
    st = m.rt.stats()
    m.free()
    return lg, st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--s-max", type=int, default=512)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    key = args.model.lower().split("/")[-1]
    hf_id = B.KNOWN[key].hf_id if key in B.KNOWN else args.model

    print(f"one forward pass, {hf_id!r}\n  prompt: {args.prompt!r}")
    ref, ids = torch_logits(hf_id, args.prompt)
    print(f"  {ids['input_ids'].shape[-1]} tokens in, {ref.numel()} logits out")
    got, st = pim_logits(hf_id, args.prompt, args.s_max)

    scale = ref.abs().max()
    err = (got - ref).abs().max() / scale
    corr = torch.corrcoef(torch.stack([got, ref]))[0, 1]

    print(f"\n  max relative error   {err:.4f}   (of a {scale:.1f} logit range)")
    print(f"  correlation          {corr:.6f}")

    kt = ref.topk(args.topk).indices.tolist()
    kg = got.topk(args.topk).indices.tolist()
    print(f"\n  top-{args.topk}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    for i, (a, b) in enumerate(zip(kt, kg)):
        mark = "  " if a == b else "<-"
        print(f"    {i}  torch {tok.decode([a])!r:>14} {ref[a]:8.3f}   "
              f"pim {tok.decode([b])!r:>14} {got[b]:8.3f} {mark}")

    print(f"\n  {st['nop']} ops, {st['nisr']} ISAs, "
          f"{st['launch_us'] / 1e6:.2f} s on the doorbell")

    # A correlation below this is not precision, it is a different computation.
    ok = corr > 0.999 and kt[0] == kg[0]
    print("\n" + ("agrees: same argmax, correlation %.6f — the difference is block "
                  "float precision" % corr if ok else
                  "DISAGREES: argmax differs or correlation is too low.  That is not "
                  "rounding."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
