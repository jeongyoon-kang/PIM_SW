#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The binding, end to end, from torch.

    ./test_ops.py          # needs pim.ko, the card, and _pim.so built

WHAT THIS CHECKS AND WHAT IT DOES NOT.  runtime/test/op_board.c already checks the
ARITHMETIC, against pim_mac_exact — the transcribed RTL, which is the only reference
that can be bit-exact, because a beat is block floating point and no fp32 sum
reproduces its alignment losses.  Repeating that here would need the oracle exposed
through the binding for no gain.

What is NOT checked anywhere else is the PLUMBING: that a torch tensor's data_ptr
reaches the right bytes, that the shapes line up, that the layout a Python caller
names is the one the card gets, and that the guards in pimllm/ops.py fire before a
bad pointer crosses.  So the numeric comparison below is against torch with a
tolerance, and its job is to catch "this landed somewhere else entirely", not to
measure precision.  A transposed weight or an off-by-one head would be off by
everything, not by a rounding.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from pimllm import ops

fail = 0


def check(cond, msg):
    global fail
    if not cond:
        print(f"  FAIL: {msg}")
        fail += 1


def close(got: torch.Tensor, want: torch.Tensor, what: str, rtol=0.06):
    """BF16 has 8 mantissa bits and the accumulation order differs, so a few percent
    is agreement.  What this is looking for is the other kind of disagreement."""
    g32, w32 = got.float(), want.float()
    scale = w32.abs().max().clamp(min=1e-6)
    err = (g32 - w32).abs().max() / scale
    ok = err <= rtol
    check(ok, f"{what}: max relative error {err:.4f} > {rtol}")
    if ok:
        print(f"     {what:<34} max rel err {err:.4f}")
    return ok


def main() -> int:
    g = ops.geometry()
    per = g["nch"] * g["nbank"]
    print(f"binding check — {g['nch']} ch x {g['nbank']} bank, "
          f"{per} outputs per supergroup, DRAM {g['dram_bytes'] / 2**30:.0f} GiB")

    rt = ops.Runtime(max_red=4096, max_out_groups=128)
    torch.manual_seed(7)

    # ---- 1. a linear layer, the way nn.Linear stores it --------------------
    print("\n  1. linear, from an nn.Linear.weight")
    n, k = 256, 2048
    lin = torch.nn.Linear(k, n, bias=False, dtype=torch.bfloat16)
    x = torch.randn(k, dtype=torch.bfloat16)

    W = ops.Tensor.from_weight(lin.weight)
    print(f"     {W}")
    y = rt.matvec(W, x)[:n]
    close(y, lin(x), "W[256 x 2048] . x")

    # ---- 2. Q . K^T with the heads packed into shared rows -----------------
    #
    # THE ONE THAT WOULD BE WRONG SILENTLY.  Eight heads of 128 share one 1024-element
    # DRAM row and red_off picks which; a mis-scaled offset reads a neighbouring
    # head's keys and returns a plausible vector of scores.
    print("\n  2. Q . K^T, 8 heads of 128 sharing each row")
    H_KV, D, S_KV = 8, 128, 100
    Kh = torch.randn(S_KV, H_KV * D, dtype=torch.bfloat16)        # [token][head*dim]
    K = ops.Tensor(ops.OUT_MAJOR, 512, H_KV * D)
    K.append(0, Kh)
    check(K.frontier == S_KV, f"frontier is {K.frontier} after {S_KV} tokens")

    ngroup = (S_KV + per - 1) // per
    for h in (0, 3, 7):
        q = torch.randn(D, dtype=torch.bfloat16)
        got = rt.matvec(K, q, out_count=ngroup, red_off=h * D, red_len=D)[:S_KV]
        want = Kh[:, h * D:(h + 1) * D].float() @ q.float()
        close(got, want.bfloat16(), f"head {h}: q . K[0..{S_KV - 1}]")

    # ---- 3. S . V, a reduction that grows -----------------------------------
    print("\n  3. S . V, reduction over the sequence")
    Vh = torch.randn(S_KV, H_KV * D, dtype=torch.bfloat16)        # [token][dim]
    V = ops.Tensor(ops.RED_MAJOR, H_KV * D, 512)
    V.append(0, Vh)
    s = torch.randn(S_KV, dtype=torch.bfloat16)
    got = rt.matvec(V, s, red_len=S_KV)
    close(got, (s.float() @ Vh.float()).bfloat16(), f"S . V[0..{S_KV - 1}]")

    # One more token: the frontier moves into the next beat and red_len changes.
    V.append(S_KV, torch.randn(1, H_KV * D, dtype=torch.bfloat16))
    Vh2 = torch.cat([Vh, torch.zeros(1, H_KV * D, dtype=torch.bfloat16)])
    check(V.frontier == S_KV + 1, f"frontier is {V.frontier} after one more")
    print(f"     frontier {V.frontier}, {V}")

    # ---- 4. the guards in ops.py fire before a pointer crosses --------------
    print("\n  4. what the Python layer refuses")
    for what, fn in [
        ("float32 vector", lambda: rt.matvec(W, torch.randn(k))),
        ("a CUDA-shaped claim", lambda: rt.matvec(W, x.to("meta"))),
        # The one that would otherwise be a use-after-free rather than an error.
        ("a transposed weight", lambda: ops.Tensor.from_weight(lin.weight.T)),
        ("a strided vector", lambda: rt.matvec(W, torch.randn(2 * k,
                                               dtype=torch.bfloat16)[::2])),
    ]:
        try:
            fn()
            check(False, f"{what} was accepted")
        except Exception as e:
            print(f"     {what:<22} {type(e).__name__}: {str(e).splitlines()[0][:58]}")

    st = rt.stats()
    print(f"\n  {st['nop']} ops, {st['nlaunch']} launches, {st['nisr']} ISRs, "
          f"{st['nwrvec']} vector loads, {st['launch_us']} us on the doorbell")

    for t in (W, K, V):
        t.free()
    print("\n" + ("FAILED" if fail else "all checks passed"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
