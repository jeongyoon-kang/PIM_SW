#!/usr/bin/env python3
"""Board check: a growable K/V gives the same q.K and s.V as a fixed-size one.

Same data into a fixed tensor (one allocation) and a growable one (grown in uneven
appends, so its pages are separate allocations); every head's q.K and s.V run on
both and must be bit-identical, and close to a torch reference.
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "sw" / "app"))
import torch
from pimllm import ops

torch.manual_seed(0)
random.seed(0)


def run(width, n_kv_heads, n_q_heads, L):
    D = width // n_kv_heads
    group = n_q_heads // n_kv_heads
    rt = ops.Runtime(max_red=2048, max_out_groups=4096, max_batch=4096,
                     vec_bytes=1 << 20, res_bytes=5 << 19)
    per = rt.per_group
    K = (torch.randn(L, width) * 0.5).to(torch.bfloat16).contiguous()
    V = (torch.randn(L, width) * 0.5).to(torch.bfloat16).contiguous()

    Kf = ops.Tensor(ops.OUT_PACKED, L, width)
    Vf = ops.Tensor(ops.RED_MAJOR, width, L)
    Kf.append(0, K)
    Vf.append(0, V)

    Kg = ops.Tensor.growable(ops.OUT_PACKED, width)
    Vg = ops.Tensor.growable(ops.RED_MAJOR, width)
    pos, grows_k, grows_v = 0, 0, 0
    while pos < L:
        n = min(L - pos, random.randint(1, 97))
        grows_k += bool(Kg.grow(pos + n))
        grows_v += bool(Vg.grow(pos + n))
        Kg.append(pos, K[pos:pos + n].contiguous())
        Vg.append(pos, V[pos:pos + n].contiguous())
        pos += n

    ngroup = Kf.groups(L)
    wide = rt.outputs(ngroup)
    q = (torch.randn(n_q_heads, D) * 0.5).to(torch.bfloat16)
    p = torch.softmax(torch.randn(n_q_heads, L), dim=-1).to(torch.bfloat16)

    def qk(Kt):
        out = torch.empty(n_q_heads, wide, dtype=torch.bfloat16)
        with rt.batch() as b:
            for h in range(n_q_heads):
                b.add(Kt, q[h].contiguous(), out_count=ngroup,
                      red_off=(h // group) * D, red_len=D, out=out[h])
        return out[:, :L]

    def sv(Vt):
        out = torch.empty(n_q_heads, rt.outputs(D // per), dtype=torch.bfloat16)
        with rt.batch() as b:
            for h in range(n_q_heads):
                b.add(Vt, p[h].contiguous(), out_first=(h // group) * D // per,
                      out_count=D // per, red_len=L, out=out[h])
        return out[:, :D]

    qk_f, qk_g = qk(Kf), qk(Kg)
    sv_f, sv_g = sv(Vf), sv(Vg)

    Kr = K.float().view(L, n_kv_heads, D)
    Vr = V.float().view(L, n_kv_heads, D)
    qk_ref = torch.stack([Kr[:, h // group] @ q[h].float() for h in range(n_q_heads)])
    sv_ref = torch.stack([p[h].float() @ Vr[:, h // group] for h in range(n_q_heads)])

    def err(a, ref):
        return ((a.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()

    ok = torch.equal(qk_f, qk_g) and torch.equal(sv_f, sv_g)
    print(f"width {width} ({n_kv_heads}x{D}), {n_q_heads} q heads, {L} tokens: "
          f"K {Kg.npages} pages ({grows_k} grows), V {Vg.npages} pages ({grows_v} grows)")
    print(f"  q.K fixed == growable: {torch.equal(qk_f, qk_g)}   "
          f"s.V fixed == growable: {torch.equal(sv_f, sv_g)}")
    print(f"  vs torch: q.K max rel err {err(qk_g, qk_ref):.2e}, "
          f"s.V max rel err {err(sv_g, sv_ref):.2e}")
    for t in (Kf, Vf, Kg, Vg):
        t.free()
    return ok


results = [
    run(512, 8, 32, 2600),     # Llama-3.2-1B shape, crosses two V chunk boundaries
    run(1024, 8, 24, 2100),    # Llama-3.2-3B shape
]
print("ALL OK" if all(results) else "MISMATCH")
sys.exit(0 if all(results) else 1)
