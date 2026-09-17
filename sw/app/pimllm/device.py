# SPDX-License-Identifier: MIT
"""The card, as Python sees it.

WHY ctypes AND NOT pybind11.  The binding this file needs is three calls that run
once at startup — open the device, read the geometry, ask how much memory is left.
ctypes costs about a microsecond per call and needs no build step, which is the
right trade for that.  It is the WRONG trade for the per-op calls that will come
later (roughly 500 per token), and those get a compiled shim; this module is
deliberately not where they go.

WHAT IS AND IS NOT AVAILABLE.  libpim is loadable without a board — it only fails
when you try to open the device.  So `geometry()` needs pim.ko loaded; everything in
budget.py is pure arithmetic on a Geometry and runs anywhere.  That split is on
purpose: the question "does this model fit" should be answerable from a laptop.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path

# sw/app/pimllm/device.py -> sw/
SW_ROOT = Path(__file__).resolve().parents[2]

PIM_MEM_DRAM = 0
PIM_MEM_GPR = 1

PIM_ALLOC_F_ZERO = 0x1
PIM_ALLOC_F_CONTIG = 0x2


class PimUnavailable(RuntimeError):
    """The card, the module or the library is not there.  Carries the reason libpim
    gave, which is a sentence and not an errno — see pim.h's error contract."""


@dataclass(frozen=True)
class Region:
    base: int
    bytes: int
    hugepage_bytes: int
    gran: int
    nr_hugepages: int
    gran_shift: int
    grans_per_hugepage: int


@dataclass(frozen=True)
class Geometry:
    """What pim_geometry says, flattened.

    `unit_bytes` is the one number the rest of the stack is built on: nch * nbank *
    row_bytes, the operand of a single all-bank all-channel MAC, and therefore both
    libpim's DRAM granule and the size of one addressable row set.  It tracks the
    channel count because it is not a policy number — it is what one MAC consumes.
    """

    nch: int
    nbank: int
    row_bytes: int
    unit_bytes: int
    dram: Region
    gpr: Region

    @property
    def elems_per_row(self) -> int:
        """BF16 in one DRAM row.  1024 here, and the ceiling on how much of one
        output's reduction axis fits without spilling into another unit."""
        return self.row_bytes // 2

    @property
    def outputs_per_group(self) -> int:
        """Outputs one all-bank MAC covers.  A partial supergroup is not something
        the ISA can express, so every shape pads up to a multiple of this."""
        return self.nch * self.nbank

    def describe(self) -> str:
        return (
            f"{self.nch} ch x {self.nbank} bank, row {self.row_bytes} B "
            f"({self.elems_per_row} BF16), unit {self.unit_bytes >> 10} KiB, "
            f"{self.outputs_per_group} outputs per supergroup\n"
            f"  DRAM {self.dram.bytes / 2**30:.2f} GiB in {self.dram.nr_hugepages} "
            f"hugepages of {self.dram.hugepage_bytes >> 20} MiB, "
            f"granule {self.dram.gran >> 10} KiB\n"
            f"  GPR  {self.gpr.bytes >> 20} MiB in {self.gpr.nr_hugepages} pages of "
            f"{self.gpr.gran >> 10} KiB"
        )


# ---------------------------------------------------------------- the structs ---
# Mirrors of pim_geometry.h.  KEPT IN THE SAME ORDER AS THE HEADER and checked
# against pim_geom()'s answer at load, because a silently shifted field here reads
# as a plausible geometry rather than as an error.
class _CRegion(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("hugepage_bytes", ctypes.c_uint64),
        ("gran", ctypes.c_uint64),
        ("nr_hugepages", ctypes.c_uint64),
        ("gran_shift", ctypes.c_uint32),
        ("grans_per_hugepage", ctypes.c_uint32),
    ]


class _CGeometry(ctypes.Structure):
    _fields_ = [
        ("nch", ctypes.c_uint32),
        ("nbank", ctypes.c_uint32),
        ("row_bytes", ctypes.c_uint32),
        ("map", ctypes.c_int),
        ("unit_bytes", ctypes.c_uint64),
        ("mem", _CRegion * 2),
    ]


_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib

    candidates = [
        os.environ.get("PIMLLM_LIBPIM"),
        SW_ROOT / "lib" / "libpim.so",
        ctypes.util.find_library("pim"),
    ]
    tried = []
    for c in candidates:
        if not c:
            continue
        try:
            _lib = ctypes.CDLL(str(c))
            break
        except OSError as e:
            tried.append(f"{c}: {e}")
    else:
        raise PimUnavailable(
            "libpim.so not found. Build it with `make -C %s` or set PIMLLM_LIBPIM.\n  %s"
            % (SW_ROOT, "\n  ".join(tried))
        )

    _lib.pim_geom.restype = ctypes.POINTER(_CGeometry)
    _lib.pim_last_error.restype = ctypes.c_char_p
    _lib.pim_meminfo.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    return _lib


def last_error() -> str:
    try:
        s = _load().pim_last_error()
    except PimUnavailable:
        return "libpim is not loaded"
    return s.decode() if s else "(no error recorded)"


def geometry() -> Geometry:
    """Ask the driver.  Raises PimUnavailable with libpim's own sentence if the
    module is not loaded — which is the common case and deserves a real message,
    not a None."""
    lib = _load()
    p = lib.pim_geom()
    if not p:
        raise PimUnavailable(
            f"cannot open the PIM device: {last_error()}\n"
            f"  is pim.ko loaded?  `sudo make -C {SW_ROOT}/drv load CH=2`"
        )
    g = p.contents

    def region(i):
        r = g.mem[i]
        return Region(
            base=r.base,
            bytes=r.bytes,
            hugepage_bytes=r.hugepage_bytes,
            gran=r.gran,
            nr_hugepages=r.nr_hugepages,
            gran_shift=r.gran_shift,
            grans_per_hugepage=r.grans_per_hugepage,
        )

    geo = Geometry(
        nch=g.nch,
        nbank=g.nbank,
        row_bytes=g.row_bytes,
        unit_bytes=g.unit_bytes,
        dram=region(PIM_MEM_DRAM),
        gpr=region(PIM_MEM_GPR),
    )

    # The struct mirror above is the thing most likely to rot, and it rots silently:
    # a shifted field gives numbers that look like a geometry.  This is the identity
    # the C side checks at open (pim_geom_check), so if the layout has drifted the
    # two will disagree here rather than three layers later.
    if geo.unit_bytes != geo.nch * geo.nbank * geo.row_bytes:
        raise PimUnavailable(
            f"the ctypes mirror of pim_geometry has drifted from the header: "
            f"unit_bytes {geo.unit_bytes} != {geo.nch} * {geo.nbank} * "
            f"{geo.row_bytes}. Re-check _CGeometry against include/pim/pim_geometry.h"
        )
    return geo


def meminfo(where: int = PIM_MEM_DRAM) -> tuple[int, int, int]:
    """(allocated, pooled, hugepages_held) for one region, in bytes/count."""
    lib = _load()
    a, p, h = (ctypes.c_uint64() for _ in range(3))
    lib.pim_meminfo(where, ctypes.byref(a), ctypes.byref(p), ctypes.byref(h))
    return a.value, p.value, h.value
