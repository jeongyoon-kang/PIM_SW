# SPDX-License-Identifier: MIT
"""Where a token's time goes.

TORCH ALREADY HAS HALF OF THIS, and the half it has is the outer half.
`torch.profiler.profile` records every aten op with its wall time and its Python
stack, and `torch.profiler.record_function(name)` opens a named range that appears
in the same table and the same Chrome trace as the aten ops do.  That is exactly
what we want for the host side — it says how much of a token was RMSNorm, RoPE,
softmax and the generate() loop, without us counting anything ourselves.

THE HALF IT CANNOT HAVE is inside the launch.  A PIM op crosses into C and does
seven different things there — pad a vector, push it over PCIe, build a program,
lower it, ring the doorbell, read the answer back, unpack it — and to torch that is
one opaque call that took four milliseconds.  Those seven have nothing in common:
one is fixed by not rebuilding the program, one by batching transfers, one by the
DRAM timing, and one not at all.  Telling them apart is why pim_rt_stat carries
them, and it is not something a host profiler could ever recover.

    torch.profiler     which MODULE the time was in     (host)
    pim_rt_stat        which PHASE the PIM time was in  (device + driver + codegen)
    the difference     Python itself

SO THIS IS A THIN THING.  It snapshots the C counters around a range, subtracts
nested ranges so the numbers are self time, and prints both columns next to each
other.  Turning it on also makes every PIM op open a `record_function`, so a torch
trace taken at the same time has the card's ranges in it.

    from pimllm import profile
    with profile.Profile(rt) as p:
        model.generate(...)
    print(p.report())

`Profile(rt, torch_trace="trace.json")` additionally runs `torch.profiler` for the
same region and writes a Chrome trace; open it at chrome://tracing or ui.perfetto.dev.
"""
from __future__ import annotations

import time

from . import ops

# The disjoint phases, grouped by WHAT WOULD FIX THEM — which is the only grouping
# that earns its keep.  Code generation is recomputation and is fixed by not doing
# it twice; transfer is bytes over PCIe and is fixed by moving fewer or moving them
# together; execution is the card and is fixed by the DRAM timing or by more banks,
# or not at all.  A breakdown that mixed them would say a number and imply nothing.
#
# `run` contains the doorbell wait.  launch_us is that SUBSET, so it is printed
# inside run rather than beside it — adding the two would count the poll twice.
GROUPS = [
    ("code generation", [
        ("gen",    "pass 1   build the program"),
        ("low",    "pass 2   resolve, check, relocate"),
    ]),
    ("data transfer", [
        ("pad",    "host     zero the tail, stage"),
        ("vup",    "H2C      vector -> GPR"),
        ("back",   "C2H      results -> host"),
        ("unpack", "host     words -> the caller"),
    ]),
    ("execution", [
        ("run",    "card     IMEM, doorbell, poll"),
    ]),
]
PHASES = [ph for _, phs in GROUPS for ph in phs]
_KEYS = [k + "_us" for k, _ in PHASES]
_COUNTS = ("nop", "nlaunch", "nisr", "nwrvec", "polls")


def _snap(rt) -> dict:
    return rt.stats()


def _delta(a: dict, b: dict) -> dict:
    return {k: b[k] - a[k] for k in b}


class _Span:
    """One named range.  Self time, not total: whatever nested spans charged is
    subtracted, so `attention` and the `linear` inside it do not both claim the
    same microsecond and the column still sums to the whole."""

    __slots__ = ("name", "wall", "phase", "child_wall", "child_phase", "n")

    def __init__(self, name):
        self.name = name
        self.wall = 0.0
        self.child_wall = 0.0
        self.phase = {k: 0 for k in _KEYS + ["launch_us"] + list(_COUNTS)}
        self.child_phase = dict(self.phase)
        self.n = 0


class Profile:
    """Counter snapshots around a region, with optional torch tracing.

    ONE PROFILE PER RUNTIME, and it must not be nested with another on the same
    runtime — the C counters are cumulative and a second reader would see the
    first's spans as its own.  Spans inside it may nest freely."""

    def __init__(self, rt, trace: str | None = None,
                 torch_trace: str | None = None, trace_ops: bool = True):
        self.rt = rt
        self.trace = trace
        self.torch_trace = torch_trace
        self.trace_ops = trace_ops
        self.spans: dict[str, _Span] = {}
        self._stack: list[tuple[_Span, float, dict]] = []
        self._prof = None
        self.wall = 0.0
        self._t0 = 0.0
        self._s0: dict = {}
        # (depth-ordered) slices for the timeline: span, then op, then phase.
        self.events: list[tuple[str, float, float, dict | None]] = []
        self._prev: dict = {}

    # ---- the region --------------------------------------------------------
    def __enter__(self):
        global _ACTIVE
        _ACTIVE = self
        if self.trace_ops:
            ops.TRACE = True
        if self.torch_trace:
            import torch.profiler as tp
            self._prof = tp.profile(activities=[tp.ProfilerActivity.CPU],
                                    record_shapes=False, with_stack=False)
            self._prof.__enter__()
        self._s0 = self._prev = _snap(self.rt)
        self._t0 = time.perf_counter()
        self._p0 = time.perf_counter()
        if self.trace:
            ops.SINK = self._sink
        return self

    def __exit__(self, *exc):
        global _ACTIVE
        self.wall = time.perf_counter() - self._t0
        self.total = _delta(self._s0, _snap(self.rt))
        if self._prof is not None:
            self._prof.__exit__(*exc)
            self._prof.export_chrome_trace(self.torch_trace)
        ops.SINK = None
        ops.TRACE = False
        if self.trace:
            self.write_trace(self.trace)
        _ACTIVE = None
        return False

    # ---- named ranges ------------------------------------------------------
    def span(self, name: str):
        return _SpanCtx(self, name)

    def _sink(self, name, t0, t1):
        """One PIM call finished.  The counters are cumulative and nothing else
        moves them, so the difference since the previous call IS this call's — no
        second clock and no per-call state in C."""
        s = _snap(self.rt)
        d = _delta(self._prev, s)
        self._prev = s
        self.events.append((name, t0, t1, d))

    def _enter(self, name):
        sp = self.spans.get(name)
        if sp is None:
            sp = self.spans[name] = _Span(name)
        self._stack.append((sp, time.perf_counter(), _snap(self.rt)))
        return sp

    def _exit(self):
        sp, t0, s0 = self._stack.pop()
        t1 = time.perf_counter()
        wall = t1 - t0
        if self.trace:
            self.events.append((sp.name, t0, t1, None))   # None = not a PIM call
        d = _delta(s0, _snap(self.rt))
        sp.wall += wall - sp.child_wall
        for k, v in d.items():
            sp.phase[k] += v - sp.child_phase.get(k, 0)
        sp.n += 1
        sp.child_wall = 0.0
        sp.child_phase = {k: 0 for k in sp.child_phase}
        if self._stack:                       # charge the parent for our total
            up = self._stack[-1][0]
            up.child_wall += wall
            for k, v in d.items():
                up.child_phase[k] = up.child_phase.get(k, 0) + v

    # ---- the numbers -------------------------------------------------------
    def report(self, detail: bool = True) -> str:
        """The whole region, then every span with its own breakdown under it.

        THE SPAN NUMBERS ARE SELF TIME, so they sum to the wall clock and a span
        that contains another does not claim its time twice.  `detail=False` prints
        the three groups only."""
        t, w = self.total, self.wall
        L = [f"=== profile: {w:.2f} s wall, {t['nop']} ops, "
             f"{t['nlaunch']} launches, {t['nisr']} ISA ===", ""]
        L.append("  where the wall clock went")
        L += _breakdown(t, w, w, indent=2, detail=detail, residual=True)

        for sp in sorted(self.spans.values(), key=lambda s: -s.wall):
            d = sp.phase
            L.append("")
            L.append(f"  {sp.name}   {sp.wall:.2f} s, {100 * sp.wall / w:.1f}% of "
                     f"wall   {sp.n} calls, {d['nlaunch']} launches, "
                     f"{d['nisr']} ISA")
            L += _breakdown(d, sp.wall, w, indent=4, detail=detail, residual=True)

        for f in (self.trace, self.torch_trace):
            if f:
                L += ["", f"  chrome trace: {f}  (ui.perfetto.dev)"]
        return "\n".join(L)


    # ---- the timeline -----------------------------------------------------
    def write_trace(self, path: str) -> None:
        """A Chrome trace of the run, three levels deep.

            span        linear gate_proj / attn q.K / ...
              op        one pim::matvec or pim::submit, REAL timestamps
                phase   gen, low, vup, run, back — real DURATIONS

        WHY WE WRITE THIS AND NOT torch.  torch's exporter records what torch did;
        the seven phases happen inside one opaque C call and there is no way to hand
        them to it after the fact with timestamps it would believe.  The format is
        a list of dicts, so writing it ourselves costs less than fighting that.

        WHAT IS MEASURED AND WHAT IS PLACED.  Every span and every op has a real
        start and a real end.  A PHASE HAS A REAL DURATION BUT NOT A REAL START:
        the counters are sums, so the phases are laid end to end from the op's
        start.  Their ORDER is the true one — a launch really does pad, push the
        vector, build, lower, ring, read back, unpack, in that order — so the
        picture is honest about sequence and about length, and only the boundaries
        between them are drawn rather than observed.  For a batched submit the
        first three are the sum over its pieces.
        """
        import json

        # Short, because a slice is a few pixels wide and the long form in the
        # table is there to be read once, not thirty thousand times.
        ORDER = ["pad", "vup", "gen", "low", "run", "back", "unpack"]
        NAME = {"pad": "stage the vector", "vup": "H2C vector",
                "gen": "pass 1 build", "low": "pass 2 lower", "run": "launch",
                "back": "C2H results", "unpack": "unpack"}
        ev = []

        def slice_(name, t0, dur, args=None):
            e = {"name": name, "ph": "X", "pid": 1, "tid": 1,
                 "ts": (t0 - self._p0) * 1e6, "dur": dur * 1e6}
            if args:
                e["args"] = args
            ev.append(e)

        # Outer first, so a viewer that respects file order still nests correctly.
        for name, t0, t1, d in sorted(self.events, key=lambda e: (e[1], -(e[2] - e[1]))):
            if d is None:                                   # a span
                slice_(name, t0, t1 - t0)
                continue
            slice_(name, t0, t1 - t0,
                   {"launches": d["nlaunch"], "ISA": d["nisr"],
                    "vector loads": d["nwrvec"]})
            # The phases, scaled to fit if the counters and the wall disagree —
            # they can by a microsecond or two, and a child sticking out of its
            # parent makes the whole track unreadable.
            tot = sum(d[k + "_us"] for k in ORDER) / 1e6
            room = t1 - t0
            k = room / tot if tot > room and tot else 1.0
            cur = t0
            for ph in ORDER:
                dur = d[ph + "_us"] / 1e6 * k
                if dur <= 0:
                    continue
                slice_(NAME[ph], cur, dur)
                if ph == "run":
                    pol = d["launch_us"] / 1e6 * k
                    slice_("doorbell -> done", cur, min(pol, dur))
                    if dur > pol:
                        slice_("IMEM + MMIO", cur + pol, dur - pol)
                cur += dur
            if room > tot * k:
                slice_("python + binding", cur, room - tot * k)

        ev.append({"name": "process_name", "ph": "M", "pid": 1, "tid": 1,
                   "args": {"name": "PIM"}})
        ev.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": 1,
                   "args": {"name": "span / op / phase"}})
        with open(path, "w") as f:
            json.dump({"traceEvents": ev, "displayTimeUnit": "ms"}, f)


def _row(indent, label, us, base, extra=""):
    """One line.  `base` is what the percentage is of — the span for a span's rows,
    so a percentage answers "of THIS", not "of the run"."""
    pct = 100.0 * us / 1e6 / base if base else 0.0
    return f"{' ' * indent}{label:<{40 - indent}} {us / 1e6:7.2f} s {pct:6.1f}%{extra}"


def _breakdown(d, base, wall, indent, detail, residual):
    """The three groups, optionally with their phases under them, and what was left
    over for Python.  Used for the whole region and for one span, unchanged — the
    counter dict has the same shape either way, which is the point of charging span
    self time in the same units."""
    out = [" " * indent + "-" * (72 - indent)]
    inside = 0
    for name, phs in GROUPS:
        tot = sum(d[k + "_us"] for k, _ in phs)
        inside += tot
        nl = d["nlaunch"]
        out.append(_row(indent, name, tot, base,
                        f"   {tot / nl:8.1f} us/launch" if nl else ""))
        if not detail or not tot:
            continue                          # an empty group has nothing under it
        for k, what in phs:
            out.append(_row(indent + 2, what, d[k + "_us"], base))
            if k == "run":
                # The poll is the only part of a launch that is the card computing.
                # The rest is one IMEM write and two MMIO stores, and separating
                # them is what says whether a program is too long or too slow.
                pol = d["launch_us"]
                out.append(_row(indent + 4, "doorbell -> done", pol, base))
                out.append(_row(indent + 4, "IMEM + MMIO", d["run_us"] - pol, base))
    if residual:
        # Everything the C layer did not account for: the Python loop, torch's own
        # elementwise work, and the binding.  It is a SUBTRACTION and not a
        # measurement, so it also absorbs any clock skew — which at these sizes is
        # microseconds against seconds.
        out.append(_row(indent, "Python + torch + host math",
                        int(base * 1e6) - inside, base))
    out.append(" " * indent + "-" * (72 - indent))
    out.append(f"{' ' * indent}{'total':<{40 - indent}} {base:7.2f} s")
    return out


class _SpanCtx:
    __slots__ = ("p", "name")

    def __init__(self, p, name):
        self.p, self.name = p, name

    def __enter__(self):
        self.p._enter(self.name)
        return self

    def __exit__(self, *exc):
        self.p._exit()
        return False


# The one live Profile, so instrumented call sites do not have to be handed one.
# None when nothing is profiling, and `span()` below is then free.
_ACTIVE: Profile | None = None


def span(name: str):
    """A range, whether or not anything is profiling.  Costs a global read when
    nothing is."""
    if _ACTIVE is None:
        return _NULL
    return _ACTIVE.span(name)


class _Null:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


_NULL = _Null()
