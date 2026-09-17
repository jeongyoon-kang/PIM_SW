#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Text generation, and the PIM placement the model would get.

    ./generate.py --dry-run --model llama-3.2-3b        # no board, no download
    ./generate.py --model meta-llama/Llama-3.2-1B --prompt "The capital of France is"

WHAT RUNS WHERE, TODAY.  The generation runs on torch.  The PIM path is not wired
in yet — the C layers are built and board-verified (pim_tensor, pim_matvec_logical)
but the per-op binding is not, and until it is this script is honest about it rather
than quietly producing the same text a slower way.  What it DOES do with the card is
the part that has to come first anyway: read the real geometry, and lay out every
weight and both KV caches against it, so `--dry-run` answers "does this model fit and
at what context length" before anyone downloads 6 GB.

RUN IT FROM ANYWHERE; it finds libpim next to itself.  With no pim.ko loaded it
still reports, using whichever geometry you name with --ch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pimllm import budget as B
from pimllm.device import Geometry, PimUnavailable, Region, geometry, meminfo


def synthetic_geometry(nch: int) -> Geometry:
    """A geometry for a board that is not here.

    The numbers are the image's, not invented: a bank window is 256 MiB because ISR
    ROW is 17 bits of 2048 B rows, and that is what platform/ch*.conf records.  This
    is what makes --dry-run worth anything on a laptop."""
    nbank, row = 16, 2048
    unit = nch * nbank * row
    window = 256 << 20
    dram_bytes = nch * nbank * window
    hugepage = 2 << 20
    return Geometry(
        nch=nch,
        nbank=nbank,
        row_bytes=row,
        unit_bytes=unit,
        dram=Region(0, dram_bytes, hugepage, unit, dram_bytes // hugepage,
                    unit.bit_length() - 1, hugepage // unit),
        gpr=Region(0, 4 << 20, 4096, 4096, 1024, 12, 1),
    )


def resolve_shape(name: str):
    """A known shape, or a HF config.  Tried in that order so --dry-run never needs
    the network for the two targets."""
    key = name.lower().split("/")[-1]
    if key in B.KNOWN:
        return B.KNOWN[key], "built-in shape"
    try:
        from transformers import AutoConfig
    except ImportError:
        return None, "transformers is not installed and the name is not a built-in shape"
    try:
        cfg = AutoConfig.from_pretrained(name)
    except Exception as e:                                  # network, auth, typo
        return None, f"could not read a config for {name!r}: {e}"
    return B.ModelShape.from_hf_config(cfg, name), "transformers config"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama-3.2-1b",
                    help="a built-in shape (llama-3.2-1b / -3b) or a HF model id")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--s-max", type=int, default=8192,
                    help="context to reserve KV for.  pim_tensor does not grow, so "
                         "this is what closes the budget")
    ap.add_argument("--ch", type=int, default=None,
                    help="report against an N-channel image instead of the one "
                         "that is loaded")
    ap.add_argument("--dry-run", action="store_true",
                    help="placement only; do not load the model or generate")
    ap.add_argument("--backend", choices=("pim", "torch"), default="pim",
                    help="pim runs the linears and both attention matmuls on the "
                         "card; torch is the reference")
    args = ap.parse_args()

    # ---- 1. the geometry ---------------------------------------------------
    if args.ch:
        g, src = synthetic_geometry(args.ch), f"synthetic ch{args.ch}"
    else:
        try:
            g, src = geometry(), "the loaded pim.ko"
        except PimUnavailable as e:
            print(f"no device: {e}\n", file=sys.stderr)
            g, src = synthetic_geometry(2), "synthetic ch2 (no device)"

    print(f"=== geometry ({src}) ===")
    print(g.describe())
    if src == "the loaded pim.ko":
        alloc, pooled, held = meminfo()
        print(f"  in use: {alloc / 2**20:,.1f} MiB allocated, "
              f"{pooled / 2**20:,.1f} MiB pooled, {held} hugepages held")

    # ---- 2. the placement --------------------------------------------------
    shape, how = resolve_shape(args.model)
    if shape is None:
        print(f"\nERROR: {how}", file=sys.stderr)
        return 2
    print(f"\n=== placement (shape from {how}) ===")
    plan = B.plan(g, shape, args.s_max)
    print(plan.report())

    if args.dry_run:
        return 0 if plan.fits else 1
    if not plan.fits:
        print("\nRefusing to run: the model does not fit and would fail partway "
              "through loading.  Lower --s-max, or use --dry-run to explore.",
              file=sys.stderr)
        return 1

    # ---- 3. generation -----------------------------------------------------
    print("\n=== generation ===")
    hf_id = shape.hf_id or args.model
    try:
        import torch
    except ImportError as e:
        print(f"\nERROR: {e}.  `pip install torch transformers` in this env.",
              file=sys.stderr)
        return 2

    if args.backend == "pim":
        from pimllm.model import PimModel
        print("  BACKEND: pim — every linear and both attention matmuls run on the "
              "card.\n           RMSNorm, RoPE, SiLU, softmax and the embedding "
              "lookup stay on the host;\n           see pimllm/model.py for why "
              "each one does.")
        m = PimModel(hf_id, s_max=args.s_max, geometry=g, verbose=True)
        t0 = time.monotonic()
        text = m.generate(args.prompt, max_new_tokens=args.max_new_tokens)
        dt = time.monotonic() - t0
        st = m.rt.stats()
        print(f"\n{text}")
        print(f"\n  {args.max_new_tokens} tokens in {dt:.2f} s  "
              f"({args.max_new_tokens / dt:.2f} tok/s)")
        print(f"  {st['nop']} ops, {st['nlaunch']} launches, {st['nisr']} ISRs, "
              f"{st['nwrvec']} vector loads, {st['launch_us'] / 1e6:.2f} s on the "
              f"doorbell")
        m.free()
        return 0

    # The reference.  Same model, same prompt, nothing on the card.
    print("  BACKEND: torch (CPU) — the reference path.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt")
    t0 = time.monotonic()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                             do_sample=False)
    dt = time.monotonic() - t0
    new = out.shape[-1] - ids["input_ids"].shape[-1]

    print(f"\n{tok.decode(out[0], skip_special_tokens=True)}")
    print(f"\n  {new} tokens in {dt:.2f} s  ({new / dt:.2f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
