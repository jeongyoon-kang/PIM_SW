#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compares the board's cycle counts with the AiM simulator's, on the same programs
with the same timing values.

    scripts/aim_compare.py                         board, then simulator, then the table
    scripts/aim_compare.py --reps 10
    scripts/aim_compare.py --out DIR --no-board    simulator and table from an earlier run

1. hwdef/test/emu_aimcmp runs every workload on the board and writes into DIR the
   timing registers it ran with (timing.txt), RUN_CYC per workload (board.csv) and
   each workload as an AiM trace (<workload>.trace).
2. The board's timing registers become the simulator's timing (MAP) in DIR/aim.yaml,
   on top of the GDDR6_AiM_timing preset of the simulator's test/example.yaml.
3. ramulator2 runs every trace with aim.yaml.  Its memory_system_cycles and the
   board's RUN_CYC are both in board clock cycles, so they are compared directly.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import re
import subprocess
import sys
from pathlib import Path

EMU_TOP = Path(__file__).resolve().parent.parent
PROBE = EMU_TOP / "hwdef" / "test" / "emu_aimcmp"
SIM_DEFAULT = EMU_TOP.parent / "ref" / "aim_simulator"

# Board timing register -> the simulator parameters that take its value.
MAP = [
    ("rcd",  ["nRCDRD", "nRCDRDMAC", "nRCDEWMUL", "nRCDRDAF", "nRCDRDCP",
              "nRCDWR", "nRCDWRCP"]),
    # The simulator spaces WR_GB beats by max(nBL, nCCDS) and MAC16 by
    # max(nCCDS, nCCDL), so this matches the board exactly while T_GB <= T_CCD.
    ("ccd",  ["nCCDL"]),
    ("gb",   ["nCCDS"]),
    ("rtp",  ["nRTP"]),
    ("ras",  ["nRAS"]),
    ("rpab", ["nRP"]),          # one precharge time in the simulator; the workloads close all banks
    ("mod",  ["nMODCH"]),
    ("rrd",  ["nRRDS", "nRRDL"]),
    ("faw",  ["nFAW"]),
]
# The simulator applies WRITE -> PRECHARGE as nCWL + nBL + nWR; these are the
# preset's nCWL and nBL, which stay as they are.
PRESET_NCWL, PRESET_NBL = 6, 2


def sim_params(t: dict[str, int]) -> dict[str, int]:
    """The simulator timing for board timing `t`."""
    p = {name: t[reg] for reg, names in MAP for name in names}
    p["nWR"] = max(0, t["wr"] - PRESET_NCWL - PRESET_NBL)
    # ACT -> ACT on one bank: the board allows it after T_RAS + T_RP_AB (open, close,
    # open), and the simulator gets the same bound.
    p["nRC"] = t["ras"] + t["rpab"]
    return p


def read_timing(out: Path) -> dict[str, int]:
    """The board registers emu_aimcmp recorded.  A run from an image without T_GB
    has no gb line; its WRVEC beats were spaced by the GPR read rate, and T_CCD
    stands in for it as it did before T_GB existed."""
    timing = {}
    for line in (out / "timing.txt").read_text().split():
        k, v = line.split("=")
        timing[k] = int(v)
    timing.setdefault("gb", timing["ccd"])
    return timing


def write_yaml(sim: Path, params: dict[str, int], out: Path) -> None:
    """test/example.yaml with `params` added under the timing preset."""
    lines = (sim / "test" / "example.yaml").read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "preset: GDDR6_AiM_timing" in line:
            indent = line[: len(line) - len(line.lstrip())]
            add = [f"{indent}{k}: {v}\n" for k, v in params.items()]
            out.write_text("".join(lines[: i + 1] + add + lines[i + 1:]))
            return
    sys.exit(f"{sim}/test/example.yaml has no 'preset: GDDR6_AiM_timing' line")


def run_sim(binary: Path, yaml: Path, trace: Path) -> int:
    r = subprocess.run([str(binary), "-f", str(yaml), "-t", str(trace)],
                       capture_output=True, text=True)
    m = re.search(r"memory_system_cycles:\s*(\d+)", r.stdout)
    if r.returncode or not m:
        sys.exit(f"ramulator2 failed on {trace.name} (exit {r.returncode}):\n"
                 f"{r.stderr[-2000:]}")
    return int(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path,
                    help="result directory (default scripts/logs/aimcmp-<time>)")
    ap.add_argument("--reps", type=int, default=5, help="board runs per workload")
    ap.add_argument("--only", help="one workload (see emu_aimcmp --list)")
    ap.add_argument("--no-board", action="store_true",
                    help="use timing.txt, board.csv and the traces already in --out")
    ap.add_argument("--sim", type=Path, default=SIM_DEFAULT,
                    help=f"AiM simulator checkout (default {SIM_DEFAULT})")
    a = ap.parse_args()

    out = a.out or EMU_TOP / "scripts" / "logs" / \
        datetime.datetime.now().strftime("aimcmp-%Y%m%d-%H%M%S")
    binary = a.sim / "build" / "ramulator2"
    if not binary.exists():
        sys.exit(f"{binary} not found; build the simulator first")

    # ---- 1. the board ----
    if a.no_board:
        if not (out / "board.csv").exists():
            sys.exit(f"{out}/board.csv not found; run without --no-board first")
    else:
        cmd = [str(PROBE), "--out", str(out), "--reps", str(a.reps)]
        if a.only:
            cmd += ["--only", a.only]
        r = subprocess.run(cmd)
        if r.returncode == 2 or not (out / "board.csv").exists():
            sys.exit("emu_aimcmp did not produce board.csv")
        print()

    timing = read_timing(out)
    with open(out / "board.csv") as f:
        board = list(csv.DictReader(f))

    # ---- 2. the simulator's timing ----
    params = sim_params(timing)
    yaml = out / "aim.yaml"
    write_yaml(a.sim, params, yaml)
    print("simulator timing from the board registers:")
    for reg, names in MAP:
        print(f"  T_{reg.upper():<5} {timing[reg]:>5}  -> {', '.join(names)}")
    print(f"  T_WR    {timing['wr']:>5}  -> nWR {params['nWR']} "
          f"(nCWL {PRESET_NCWL} + nBL {PRESET_NBL} + nWR = {timing['wr']})")
    print(f"  T_RAS + T_RP_AB   -> nRC {params['nRC']}")
    print()

    # ---- 3. both, side by side ----
    print(f"  {'workload':<11} {'ISRs':>5} {'board':>10} {'simulator':>10} "
          f"{'board-sim':>10} {'board/sim':>9}  violations")
    for row in board:
        name = row["workload"]
        sim = run_sim(binary, yaml, out / f"{name}.trace")
        cyc = int(row["run_cyc_min"])
        spread = int(row["run_cyc_max"]) - cyc
        viol = int(row["act_fill"]) + int(row["pre_drain"])
        print(f"  {name:<11} {row['isrs']:>5} {cyc:>10}{'+' if spread else ' '}"
              f"{sim:>10} {cyc - sim:>10} {cyc / sim:>9.3f}  "
              f"{viol if viol else 'none'}")
    print(f"\nresults in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
