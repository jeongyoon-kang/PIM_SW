# `sw/app` — the Python side

```bash
./generate.py --dry-run --model llama-3.2-3b     # placement only; no board, no download
./generate.py --model llama-3.2-1b --prompt "..."  # ...and generate
```

## What it does today, exactly

**The placement is real.** It reads the geometry from the loaded `pim.ko` and sizes
every weight and both KV caches the way `pim_tensor_bytes()` will size them —
padding counted, not estimated. That answers *does this model fit, and at what
context length*, which has to be settled before anything is downloaded.

**The generation runs on torch**, and says so on every run. The C layers below it
(`pim_tensor`, `pim_matvec_logical`) are built and board-verified, but the per-op
binding is not written yet, so nothing dispatches to the card. A fallback that
quietly produced the same text would be the worst kind of scaffolding: it looks
finished.

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

## Next

The per-op binding (`pim_op_linear` / `attn_qk` / `attn_sv` behind pybind11), then
`PimKVCache(CacheLayerMixin)` and `ALL_ATTENTION_FUNCTIONS.register("pim", ...)`.
`transformers` 5.x wants a layer subclass with five methods, not a `Cache`
subclass — see the note in `pimllm/budget.py` on which axis each cache grows along.
