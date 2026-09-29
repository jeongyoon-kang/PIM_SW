#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measures what one ISR costs on the board and in the AiM simulator, and ranks the
ISRs by how far the two disagree.

    scripts/aim_isr_cost.py                        board, simulator, ranking
    scripts/aim_isr_cost.py --out DIR --no-board   simulator and ranking again
    scripts/aim_isr_cost.py --nccds 2 --out DIR --no-board

Each entry is a pair of short programs that differ by one ISR at the end, in a
stated context (which mode the controller is in, whether the row is open).  The
cost of that ISR is the difference of the two RUN_CYC values on the board and of
the two memory_system_cycles values in the simulator.  The simulator gets the
board's timing registers exactly as scripts/aim_compare.py maps them.

  --nccds N       the simulator's nCCDS instead of the board's T_CCD
"""
from __future__ import annotations

import argparse
import csv
import datetime
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aim_compare as ac  # noqa: E402

# name -> program (emu_aimcmp --prog syntax)
PROGS = {
    "eos":        "eos",
    "w64":        "wrvec 64; eos",
    "w64_w64":    "wrvec 64; wrvec 64; eos",
    "w4":         "wrvec 4; eos",
    "w4_w4":      "wrvec 4; wrvec 4; eos",
    "w1":         "wrvec 1; eos",
    "w1_w1":      "wrvec 1; wrvec 1; eos",
    "m":          "wrvec 64; mac 64 100; eos",
    "m_hit":      "wrvec 64; mac 64 100; mac 64 100; eos",
    "m_miss":     "wrvec 64; mac 64 100; mac 64 101; eos",
    "m4":         "wrvec 4; mac 4 100; eos",
    "m4_hit":     "wrvec 4; mac 4 100; mac 4 100 4; eos",
    "r":          "wrvec 64; mac 64 100; rdmac; eos",
    "r_r":        "wrvec 64; mac 64 100; rdmac; rdmac 2002; eos",
    "r_w":        "wrvec 64; mac 64 100; rdmac; wrvec 64; eos",
    "r_w_m":      "wrvec 64; mac 64 100; rdmac; wrvec 64; mac 64 100; eos",
    "q_w":        "wrvec 4; mac 4 100; rdmac; wrvec 4; eos",
    "q_w_m":      "wrvec 4; mac 4 100; rdmac; wrvec 4; mac 4 100 4; eos",
}

# (the ISR added, its context, program without it, program with it)
ENTRIES = [
    ("EOS",        "a program of one EOS: start-up and end of a run", None, "eos"),
    ("WRVEC 64",   "first ISR: switches BANK -> REGISTER", "eos", "w64"),
    ("WRVEC 64",   "after a WRVEC: already REGISTER", "w64", "w64_w64"),
    ("WRVEC 4",    "after a WRVEC: already REGISTER", "w4", "w4_w4"),
    ("WRVEC 1",    "after a WRVEC: already REGISTER", "w1", "w1_w1"),
    ("WRVEC 64",   "after RD_MAC: already REGISTER", "r", "r_w"),
    ("MAC 64",     "after WRVEC: switches to BANK, opens the row", "w64", "m"),
    ("MAC 64",     "after a MAC on the same row", "m", "m_hit"),
    ("MAC 64",     "after a MAC on another row: closes and opens", "m", "m_miss"),
    ("MAC 4",      "after a MAC on the same row", "m4", "m4_hit"),
    ("MAC 64",     "after RD_MAC, WRVEC: switches to BANK, row open", "r_w", "r_w_m"),
    ("MAC 4",      "after RD_MAC, WRVEC 4: switches to BANK, row open", "q_w", "q_w_m"),
    ("RD_MAC x2",  "after a MAC: switches to REGISTER", "m", "r"),
    ("RD_MAC x2",  "after RD_MAC: already REGISTER", "r", "r_r"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path,
                    help="result directory (default scripts/logs/aimisr-<time>)")
    ap.add_argument("--reps", type=int, default=3, help="board runs per program")
    ap.add_argument("--no-board", action="store_true",
                    help="use the board results already in --out")
    ap.add_argument("--nccds", type=int, help="the simulator's nCCDS")
    ap.add_argument("--sim", type=Path, default=ac.SIM_DEFAULT,
                    help=f"AiM simulator checkout (default {ac.SIM_DEFAULT})")
    a = ap.parse_args()

    out = a.out or ac.EMU_TOP / "scripts" / "logs" / \
        datetime.datetime.now().strftime("aimisr-%Y%m%d-%H%M%S")
    binary = a.sim / "build" / "ramulator2"
    if not binary.exists():
        sys.exit(f"{binary} not found; build the simulator first")

    # ---- the board ----
    if not a.no_board:
        out.mkdir(parents=True, exist_ok=True)
        (out / "board.csv").unlink(missing_ok=True)
        for name, spec in PROGS.items():
            r = subprocess.run([str(ac.PROBE), "--prog", spec, "--name", name,
                                "--out", str(out), "--reps", str(a.reps)],
                               capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"emu_aimcmp --prog '{spec}' failed:\n{r.stdout}{r.stderr}")
        # The programs without an RD_MAC leave the accumulators full; this empties them.
        subprocess.run([str(ac.PROBE), "--prog", "rdmac; eos", "--reps", "1"],
                       capture_output=True)
    with open(out / "board.csv") as f:
        board = {row["workload"]: row for row in csv.DictReader(f)}
    missing = [n for n in PROGS if n not in board]
    if missing:
        sys.exit(f"{out}/board.csv has no result for {', '.join(missing)}")

    # ---- the simulator ----
    timing = ac.read_timing(out)
    params = ac.sim_params(timing)
    if a.nccds is not None:
        params["nCCDS"] = a.nccds
    yaml = out / "aim_isr.yaml"
    ac.write_yaml(a.sim, params, yaml)

    sim = {}
    for name in PROGS:
        sim[name] = ac.run_sim(binary, yaml, out / f"{name}.trace")
    brd = {name: int(board[name]["run_cyc_min"]) for name in PROGS}
    for name in PROGS:
        if int(board[name]["act_fill"]) or int(board[name]["pre_drain"]):
            print(f"warning: {name} raised a timing violation on the board")

    # ---- one ISR each, ranked by the disagreement ----
    rows = []
    for isr, ctx, base, ext in ENTRIES:
        b = brd[ext] - (brd[base] if base else 0)
        s = sim[ext] - (sim[base] if base else 0)
        rows.append((isr, ctx, b, s, s - b))
    rows.sort(key=lambda r: -abs(r[4]))

    print(f"timing: " + " ".join(f"{k}={v}" for k, v in timing.items()))
    print(f"simulator: nCCDS {params['nCCDS']}, nCCDL {params['nCCDL']}\n")
    print(f"  {'ISR':<10} {'board':>7} {'sim':>7} {'sim-board':>9}  context")
    for isr, ctx, b, s, d in rows:
        print(f"  {isr:<10} {b:>7} {s:>7} {d:>+9}  {ctx}")
    print(f"\nresults in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
