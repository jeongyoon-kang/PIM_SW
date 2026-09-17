# `sw/app` — the Python side

```bash
./generate.py --dry-run --model llama-3.2-3b     # placement only; no board, no download
./generate.py --model llama-3.2-1b --prompt "..."  # ...and generate
```

## Build

```bash
make -C ..                    # libpim + libpimrt
make PYTHON=$(which python)   # _pim.so, the per-op binding
make check                    # imports it and prints the geometry
./test_ops.py                 # the binding end to end, from torch
```

## It runs

Llama-3.2-1B, every linear and both attention matmuls on the card:

```
The capital of France is Paris. It is the most populous city in France and the world's 4

  16 tokens in 6.64 s  (2.41 tok/s)
  23872 ops, 23936 launches, 2268800 ISRs, 773376 vector loads, 2.32 s on the doorbell
```

**The text differs from torch's, and that is not evidence of anything.** Greedy
decoding argmaxes over 128256 logits; a near-tie swaps on the smallest arithmetic
difference and the sequences diverge and stay fluent. `./check_vs_torch.py` compares
one forward pass *before* the argmax, which is the question that can be answered:

```
  max relative error   0.0090   (of a 20.9 logit range)
  correlation          0.999944
  top-5
    0  torch  ' Paris'  20.875   pim  ' Paris'  20.875
    1  torch      ' a'  19.375   pim      ' a'  19.375
    2  torch    ' the'  19.125   pim    ' the'  19.125
    3  torch    ' one'  18.375   pim   ' also'  18.375 <-
    4  torch   ' also'  18.250   pim    ' one'  18.250 <-
```

Ranks 3 and 4 swapped at *identical logit values* — that is the near-tie, visible.
Block floating point takes the maximum exponent over a beat's sixteen products and
truncates the rest during alignment, so a fraction of a percent is the hardware
working, not a defect.

## What runs where

| on the card | on the host |
|---|---|
| every `nn.Linear`, `lm_head` included | `embed_tokens` — a lookup, not a matmul |
| Q·Kᵀ | mask + softmax — the card has no `exp` and no `max` |
| S·V | RMSNorm, RoPE, SiLU, residuals — elementwise over `[1, hidden]` |
| K and V caches, resident | sampling |

The softmax in the middle of attention is what forces a GPR round trip per head per
layer. No rearrangement removes it: it is the one operation in attention this
hardware cannot do.

**GQA is free.** `repeat_kv` materialises H_q/H_kv copies of K and V; the PIM
attention function never calls it. Several query heads simply name the same
`red_off`.

## Why the budget module has no device in it

`pimllm/budget.py` is arithmetic on a `Geometry` and nothing else. So
`--ch 1 --dry-run` answers "would this fit on a one-channel image" from a laptop,
which is the question you want answered *before* fetching 6 GB of weights.
`pimllm/device.py` is the only file that touches libpim.

## Why ctypes

Three calls that run once at startup. ctypes costs about a microsecond each and
needs no build step. It is the wrong trade for the ~500 per-token calls the PIM path
will make, and those get a compiled shim — `device.py` is deliberately not where
they go.

## What the report tells you

```
on 2 ch x 16 bank:  H_kv*D = 1024 of 1024 per row  (fills a row exactly)
```

The number to watch. A DRAM row holds 1024 BF16; `H_kv * D` is one token of K. At
1024 it fills a row and wastes nothing, at 512 it wastes half — which is why
Llama-3.2-1B's K cache costs the same per token as 3B's despite being half the
data. It is also the packing ceiling, since `COL + OPSIZE <= 64` beats.

Measured on the ch2 image, `S_max 8192`:

| | weights | KV | total | of 8 GiB |
|---|---|---|---|---|
| Llama-3.2-1B | 2.30 GiB | 384 MiB | **2.68 GiB** | fits, S_max up to 123904 |
| Llama-3.2-3B | 5.98 GiB | 896 MiB | **6.86 GiB** | fits, S_max up to 18432 |

On a one-channel image the capacity halves to 4 GiB and 3B's weights alone are over
it. That is what `--ch 1` is for.

## Layout

```
generate.py          the entry point
pimllm/device.py     ctypes over libpim: geometry, meminfo.  The only file that
                     needs pim.ko
pimllm/budget.py     shapes, padding, and where every tensor goes.  No device
```

## The binding

`_pim.cpp` is the only C++ in the tree and does nothing but marshal. It takes raw
pointers because BF16 has no numpy dtype, so no torch tensor can cross through the
buffer protocol — `data_ptr()` is the honest way, and it keeps torch out of the C++
entirely.

`pimllm/ops.py` is where a bad pointer is made hard to pass, because dtype, shape and
contiguity are visible there and not below. It refuses a float32 vector, a non-CPU
tensor and a strided view, and it deliberately does **not** call `.contiguous()` for
you: that would hide a copy of a several-hundred-megabyte weight, and a temporary's
`data_ptr()` is dangling by the time the binding dereferences it. That last one was a
real use-after-free, found by `test_ops.py` on its first run.

```python
from pimllm import ops
rt = ops.Runtime(max_red=4096, max_out_groups=128)
W  = ops.Tensor.from_weight(linear.weight)     # [out, in] IS OUT_MAJOR
y  = rt.matvec(W, x)[:out_features]            # padding outputs come back too
```

One op, not three: `pim_matvec_logical` showed a linear layer, a Q·Kᵀ and an S·V are
the same call with different arguments, so the model's vocabulary stays in Python.

| kernel | `m` | `red_off` | `red_len` |
|---|---|---|---|
| linear | W, OUT_MAJOR | 0 | `in_features` |
| Q·Kᵀ | K, OUT_MAJOR | `head * D` | `D` |
| S·V | V, RED_MAJOR | 0 | `S_kv`, grows |

## Next

`PimKVCache(CacheLayerMixin)` and `ALL_ATTENTION_FUNCTIONS.register("pim", ...)`.
`transformers` 5.x wants a **layer** subclass with five methods, not a `Cache`
subclass.
