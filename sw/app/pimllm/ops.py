# SPDX-License-Identifier: MIT
"""torch tensors in, torch tensors out.

_pim takes raw pointers, because BF16 has no numpy dtype and so no torch tensor can
expose one through the buffer protocol.  That is the honest way across, and it means
the C++ never has to know what torch is — but it also means a caller can hand it a
pointer to the wrong thing.  THIS is the layer where that is made hard, because here
the dtype, the shape and the contiguity are all visible and down there none of them
are.

Every function below checks three things before it lets a pointer through:

    dtype is bfloat16     a float32 tensor has twice the stride and would be read as
                          interleaved garbage, silently
    contiguous            data_ptr() is the start of a STRIDED view otherwise, and a
                          transpose is the common way to get one
    on the CPU            a CUDA pointer is not a host pointer, and passing one is a
                          segfault rather than a wrong number

None of the three is checkable from C, which is why they are here and not there.

AND NOTHING HERE CALLS .contiguous() FOR YOU.  It would hide a copy of a tensor that
may be hundreds of megabytes, and — the reason it is a rule rather than a taste — a
temporary's data_ptr() is dangling by the time the binding dereferences it.  That is
a use-after-free whose symptom is an intermittent EFAULT from a pwrite, pointing at
nothing.  Callers pass something contiguous; this says so when they do not.
"""
from __future__ import annotations

import time

import torch

import _pim

OUT_MAJOR = _pim.OUT_MAJOR
RED_MAJOR = _pim.RED_MAJOR
OUT_PACKED = _pim.OUT_PACKED
ACC_SINGLE = _pim.ACC_SINGLE
ACC_DUAL = _pim.ACC_DUAL
ALLOC_ZERO = _pim.ALLOC_ZERO

# Open a torch profiler range around every launch when something is profiling.
# OFF BY DEFAULT because record_function is about a microsecond and attention makes
# hundreds of calls a token; pimllm.profile turns it on for the region it watches.
# What it buys is that a chrome trace shows the card's ranges beside RMSNorm and
# softmax, on one timeline, instead of a gap where the PIM work was.
TRACE = False

# Set by pimllm.profile to a callable(name, t0, t1).  It is what turns a launch into
# a slice on a timeline, and it is a hook rather than an import so that ops.py does
# not depend on the profiler it feeds.
SINK = None


class _range:
    __slots__ = ("name", "r", "t0")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter() if SINK is not None else 0.0
        self.r = torch.profiler.record_function(self.name) if TRACE else None
        if self.r is not None:
            self.r.__enter__()
        return self

    def __exit__(self, *exc):
        if self.r is not None:
            self.r.__exit__(*exc)
        if SINK is not None:
            SINK(self.name, self.t0, time.perf_counter())
        return False


def _ptr(t: torch.Tensor, what: str) -> tuple[int, int]:
    if t.dtype is not torch.bfloat16:
        raise TypeError(f"{what} is {t.dtype}; the device is BF16 and nothing "
                        f"converts on the way in")
    if t.device.type != "cpu":
        raise ValueError(f"{what} is on {t.device}; data_ptr() must be a host address")
    if not t.is_contiguous():
        raise ValueError(f"{what} is not contiguous — a view or a transpose.  "
                         f"Call .contiguous() first; data_ptr() on a strided tensor "
                         f"is the start of the underlying storage, not of the view")
    return t.data_ptr(), t.numel()


class Tensor:
    """A matrix resident on the card.

    The layout is not a preference, it follows from which axis the kernel reduces
    over — see include/pimrt/pim_tensor.h.  In model terms:

        linear weight   OUT_MAJOR  [out_features, in_features]  (nn.Linear order)
        K cache         OUT_PACKED [tokens, H_kv*D]  a token is an OUTPUT; tokens
                                                     share a bank row when H_kv*D
                                                     is under half a row
        V cache         RED_MAJOR  [H_kv*D, tokens]  a token is a REDUCTION STEP

    A weight is allocated at its size.  A KV tensor is `growable`: it starts with no
    room for tokens and `grow` adds pages as they arrive.
    """

    def __init__(self, layout: int, nout: int, nred: int, zero: bool = False):
        self._t = _pim.Tensor(layout, nout, nred, ALLOC_ZERO if zero else 0)

    @classmethod
    def growable(cls, layout: int, fixed: int) -> "Tensor":
        """A tensor with no room yet along its growth axis — `out` for OUT_MAJOR and
        OUT_PACKED, `red` for RED_MAJOR.  `fixed` is the other axis, H_kv*D for a KV
        cache.  Nothing is allocated until the first grow."""
        t = cls.__new__(cls)
        t._t = _pim.Tensor.growable(layout, fixed)
        return t

    def grow(self, n: int) -> int:
        """Room for at least `n` along the growth axis.  Allocates the pages it
        needs and returns how many units that was; 0 when there was room."""
        return self._t.grow(n)

    @classmethod
    def from_weight(cls, w: torch.Tensor) -> "Tensor":
        """An nn.Linear.weight, uploaded.  Its [out, in] order IS OUT_MAJOR, so the
        tensor goes across with no permutation on the host."""
        if w.ndim != 2:
            raise ValueError(f"a weight is 2-D; got {tuple(w.shape)}")
        t = cls(OUT_MAJOR, w.shape[0], w.shape[1])
        t.upload(w)
        return t

    def upload(self, src: torch.Tensor) -> None:
        # NOT src.contiguous().  Twice wrong: the copy would be silent — a weight is
        # hundreds of megabytes and the caller should decide when to pay for that —
        # and, worse, a temporary's data_ptr() is dangling by the time the call
        # below runs.  That is a use-after-free, and it showed up as an intermittent
        # "Bad address" from the pwrite rather than as anything pointing at here.
        ptr, n = _ptr(src, "upload source")
        self._t.upload(ptr, n)

    def append(self, first: int, src: torch.Tensor) -> None:
        """Extend along the growth axis — `out` for OUT_MAJOR, `red` for RED_MAJOR.

        `src` is [count, the other axis].  One new token of K is [1, H_kv*D] and of
        V is also [1, H_kv*D]; the difference in what that costs is in the layout,
        not in this call."""
        other = self.nout if self.layout == RED_MAJOR else self.nred
        if src.numel() % other:
            raise ValueError(f"append source has {src.numel()} elements, not a "
                             f"multiple of {other}")
        ptr, n = _ptr(src, "append source")
        self._t.append(first, n // other, ptr, n)

    def truncate(self, pos: int) -> None:
        """Move the frontier BACK and restore the zero invariant.

        Call it on a generation restart, a cache crop, or a rejected speculative
        token.  Skipping it leaves the previous generation's values where the next
        launch's final beat reads, and the result is NaN with no error."""
        self._t.truncate(pos)

    def free(self) -> None:
        self._t.free()

    layout = property(lambda self: self._t.layout)
    nout = property(lambda self: self._t.nout)
    nred = property(lambda self: self._t.nred)
    ngroups = property(lambda self: self._t.ngroups)
    frontier = property(lambda self: self._t.frontier)
    pack = property(lambda self: self._t.pack)
    room = property(lambda self: self._t.room)
    nunits = property(lambda self: self._t.nunits)
    npages = property(lambda self: self._t.npages)

    def groups(self, n: int) -> int:
        """MAC groups that cover outputs [0, n).  With pack > 1 this is not
        ceil(n / per_group): a MAC covers every pack-th output of a row."""
        return self._t.groups(n)
    bytes = property(lambda self: self._t.bytes)

    def __repr__(self):
        return repr(self._t)


class Runtime:
    """The compute side.  One board, one command stream — do not share across
    threads, the same way pim_ctx is not shared."""

    def __init__(self, max_red: int, max_out_groups: int,
                 dual_latch: bool = False, max_batch: int = 1,
                 vec_bytes: int = 0, res_bytes: int = 0):
        # The DRAM timing registers are NOT set here.  hwdef/test/emu_timing owns
        # them; opening an engine reads them and refuses on T_CCD < 2.
        #
        # vec_bytes/res_bytes are the GPR a stacked program may spend on its
        # pieces.  Left at 0 they are derived from max_red, which reserves the FFN's
        # 16 KiB for an attention query of 128 B — see include/pimrt/pim_op.h.
        #
        # dual_latch makes every matvec and batch use both accumulator latches of a
        # bank (ISR[35]): output groups run in pairs, one per latch, and a pair
        # shares its vector loads.  The image must decode ISR[35];
        # runtime/test/tlatch_test checks that.
        self._r = _pim.Runtime(max_red, max_out_groups, dual_latch, max_batch,
                               vec_bytes, res_bytes)
        self.dual_latch = dual_latch
        self.mode = ACC_DUAL if dual_latch else ACC_SINGLE
        self.max_batch = max_batch
        # The IsaTrace recording this runtime's launches, when there is one.  The
        # model's layers read it to label their launches.
        self.tracer = None
        g = _pim.geometry()
        self.nch, self.nbank = g["nch"], g["nbank"]
        self.per_group = self.nch * self.nbank
        self.row_elems = g["row_bytes"] // 2          # BF16 in one DRAM row

    def matvec(self, m: Tensor, v: torch.Tensor, *, out_first: int = 0,
               out_count: int | None = None, red_off: int = 0,
               red_len: int | None = None, mode: int | None = None,
               out: torch.Tensor | None = None) -> torch.Tensor:
        """y = m[.][red_off : red_off+red_len] @ v

        THE RETURNED TENSOR INCLUDES THE PADDING OUTPUTS of the last supergroup,
        because the hardware writes them and pretending otherwise would mean a copy
        on every call.  Slice to your real output count; for a linear layer that is
        `[:weight.shape[0]]`.

        `mode` is ACC_SINGLE or ACC_DUAL; left out, it is the runtime's."""
        if out_count is None:
            out_count = m.ngroups - out_first
        if red_len is None:
            red_len = v.numel()
        if mode is None:
            mode = self.mode
        want = self._r.outputs(out_count)
        if out is None:
            out = torch.empty(want, dtype=torch.bfloat16)
        vp, vn = _ptr(v, "vector")
        yp, yn = _ptr(out, "output")
        with _range(f"pim::matvec[{out_count}g x {red_len}]"):
            self._r.matvec(m._t, out_first, out_count, red_off, red_len, vp, vn,
                           yp, yn, mode)
        return out

    # ---- stacking ----------------------------------------------------------
    # One doorbell for several matvecs.  A launch costs an IMEM transfer, two MMIO
    # writes and a poll loop whatever is behind it, and attention pays that per
    # HEAD — 32 times a layer for programs of eleven instructions.
    #
    # EVERY `out` MUST STAY ALIVE UNTIL submit(), because that is when the card's
    # answer is written into it.  batch() keeps the references so a caller cannot
    # lose one to the garbage collector mid-program.
    def batch(self, mode: int | None = None) -> "Batch":
        return Batch(self, self.mode if mode is None else mode)

    def outputs(self, out_count: int) -> int:
        """How many BF16 a matvec over `out_count` supergroups writes — the padding
        outputs of the last one included, because the hardware writes them."""
        return self._r.outputs(out_count)

    def stats(self) -> dict:
        return self._r.stats()

    def stats_reset(self) -> None:
        self._r.stats_reset()

    # ---- ISA trace ----------------------------------------------------------
    def trace_open(self, path, header: str = "") -> None:
        """Write every launched program to `path` from here on, after `header`.
        The format is pim_rt_trace's; see include/pimrt/pim_op.h."""
        self._r.trace_open(str(path), header)

    def trace_close(self) -> None:
        self._r.trace_close()

    def trace_label(self, label: str) -> None:
        """The label on the header of every launch until the next call."""
        self._r.trace_label(label)


def geometry() -> dict:
    return _pim.geometry()


class Batch:
    """Several matvecs behind one doorbell.

        with rt.batch() as b:
            for h in range(32):
                ys.append(b.add(K, q[h], red_off=h*D, red_len=D, out_count=n))
        # every ys[h] is filled once the block exits

    The results are not there until the block exits.  Reading one inside is not an
    error anywhere — it is whatever the buffer held — so the shape of the API is the
    warning.
    """

    def __init__(self, rt: Runtime, mode: int):
        self.rt, self.mode = rt, mode
        self._keep: list[torch.Tensor] = []

    def __enter__(self):
        self.rt._r.begin(self.mode)
        return self

    def add(self, m: Tensor, v: torch.Tensor, *, out_first: int = 0,
            out_count: int | None = None, red_off: int = 0,
            red_len: int | None = None,
            out: torch.Tensor | None = None) -> torch.Tensor:
        if out_count is None:
            out_count = m.ngroups - out_first
        if red_len is None:
            red_len = v.numel()
        if out is None:
            out = torch.empty(self.rt._r.outputs(out_count), dtype=torch.bfloat16)
        vp, vn = _ptr(v, "vector")
        yp, yn = _ptr(out, "output")
        with _range(f"pim::add[{out_count}g x {red_len}]"):
            self.rt._r.add(m._t, out_first, out_count, red_off, red_len,
                           vp, vn, yp, yn)
        self._keep.append(out)          # alive until submit writes it
        return out

    def add_heads(self, m: Tensor, v: torch.Tensor, *, nhead: int, head_len: int,
                  out: torch.Tensor, out_first: int = 0,
                  out_count: int | None = None, red_off: int = 0) -> torch.Tensor:
        """Several heads behind one vector load: Q.K^T with Q laid out like a row of K.

        `v` is nhead segments of head_len; segment h pairs with m's reduction range
        [red_off + h*head_len, red_off + (h+1)*head_len).  `out` is [nhead, at least
        outputs(out_count)] and row h gets head h's outputs.  Its rows may be any
        distance apart — a strided view of a bigger tensor — but each row must be
        contiguous.  See pim_op_add_heads."""
        if out_count is None:
            out_count = m.ngroups - out_first
        vp, vn = _ptr(v, "vector")
        want = self.rt._r.outputs(out_count)
        if out.dtype is not torch.bfloat16 or out.device.type != "cpu":
            raise TypeError(f"output is {out.dtype} on {out.device}; it must be "
                            f"bfloat16 on the CPU")
        if (out.dim() != 2 or out.shape[0] != nhead or out.shape[1] < want
                or out.stride(1) != 1):
            raise ValueError(f"output must be [{nhead}, >= {want}] with contiguous "
                             f"rows; got {tuple(out.shape)}, strides {out.stride()}")
        span = (nhead - 1) * out.stride(0) + want
        with _range(f"pim::add_heads[{out_count}g x {nhead}h]"):
            self.rt._r.add_heads(m._t, out_first, out_count, red_off, nhead, head_len,
                                 vp, vn, out.data_ptr(), span, out.stride(0))
        self._keep.append(out)          # alive until submit writes it
        return out

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            with _range(f"pim::submit[{len(self._keep)} pieces]"):
                self.rt._r.submit()
        self._keep.clear()
        return False
