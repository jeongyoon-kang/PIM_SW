#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Linear layers: the board application against the CENT variant in the AiM simulator.

    docs/log/cent_linear_2026-09-29/compare_linear.py --single DIR1 --dual DIR2

DIR1 and DIR2 are `generate.py --isa-trace` outputs of the same run with one latch
per bank and with `--dual-latch`.  For one decode step and one layer this

1. has CENT's trace generator write the seven linear GEMVs of a Llama-3.2-1B layer
   on two channels in our order (`--GEMV reuse-bank`, one or two latches);
2. applies the board's rules to that trace: no activation-unit commands, and one
   RD_MAC per channel;
3. cuts the trace into one file per op and compares each with the ISRs the board
   ran for that op:
     - the command list: op, OPSIZE and channel mask, in order (EOS left out)
     - the row pattern: each MAC's row, renumbered by first appearance
     - the chunk order: which input chunk each vector load carries
     - the repeating unit: its commands, how often it repeats, and how far the
       rows move from one repeat to the next
4. runs each op in the AiM simulator with the board's timing registers, once from
   the CENT trace and once from the board's own ISRs with the board's row numbers,
   and sets both beside the board's RUN_CYC.

Results go to --out: results.csv, the simulator traces, aim.yaml, and extracts of
the board dumps (every launch header of the step, and the ISRs of the layer).
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMU_TOP = HERE.parents[2]
sys.path.insert(0, str(EMU_TOP / "scripts"))
import aim_compare as ac  # noqa: E402  (sim_params, write_yaml, read_timing)

CENT_DEFAULT = EMU_TOP.parent / "ref" / "CENT" / "cent_simulation"
CENT_PYTHON = Path("/home/kjy/miniconda3/envs/cent/bin/python3")
SIM_DEFAULT = EMU_TOP.parent / "ref" / "aim_simulator"

BANKS = 32                                  # 2 channels x 16 banks
NODES = [                                   # nn.Linear shapes of a Llama-3.2-1B layer
    ("q_proj", 2048, 2048), ("k_proj", 512, 2048), ("v_proj", 512, 2048),
    ("o_proj", 2048, 2048), ("gate_proj", 8192, 2048), ("up_proj", 8192, 2048),
    ("down_proj", 2048, 8192),
]
# CENT's generator fixes head_dim at 128, so these arguments give Llama-3.2-1B's
# linear shapes (q 2048x2048, k/v 512x2048, FFN 8192) but not its attention.
CENT_SHAPE = ["--Llama-GQA", "--n_heads", "16", "--n_kv_heads", "4", "--ffn_dim", "8192"]
LETTER = {"WRVEC": "V", "MAC": "M", "RD_MAC": "R"}


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def macs(n: int, k: int) -> int:
    return cdiv(n, BANKS) * cdiv(k, 1024)


# ---- the CENT side -----------------------------------------------------------

def cent_trace(cent: Path, latches: int, path: Path) -> None:
    """CENT's generator, our order, `latches` MAC registers per bank."""
    cmd = [str(CENT_PYTHON), "function_sim.py", *CENT_SHAPE, "--only-FC", "--only-trace",
           "--num-channels", "2", "--FC-devices", "1", "--model-parallel",
           "--seqlen", "128", "--op-trace", "--GEMV", "reuse-bank",
           "--reuse-size", str(latches), "--trace-file", str(path)]
    r = subprocess.run(cmd, cwd=cent, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"CENT generator failed:\n{r.stderr[-2000:]}")


def board_rules(lines: list[str]) -> list[str]:
    """No activation unit on the board, and each RD_MAC lands one channel's word."""
    out = []
    for line in lines:
        f = line.split()
        if not f or f[1] in ("AF", "RD_AF", "EOC"):
            continue
        if f[1] == "RD_MAC":
            out += ["AiM RD_MAC 0 0x1", "AiM RD_MAC 0 0x2"]
        else:
            out.append(line.strip())
    return out


def split_ops(lines: list[str]) -> dict[str, list[str]]:
    """A whole-layer trace cut into the seven GEMVs by each one's MAC count."""
    out, i, n = collections.defaultdict(list), 0, 0
    for line in lines:
        f = line.split()
        name, nout, nred = NODES[i]
        if f[1] in ("WR_GB", "MAC_ABK") and n == macs(nout, nred) and i < len(NODES) - 1:
            i, n = i + 1, 0
        if f[1] == "MAC_ABK":
            n += 1
        out[NODES[i][0]].append(line)
    return out


def dual_both_latches(nout: int, nred: int) -> list[str]:
    """Two output groups per vector load, both latches used, in the order the
    runtime issues it.  CENT keeps one of its two latches for the activation result
    in the layer that carries it, so for gate_proj this stands in for its output."""
    groups, chunks, out = cdiv(nout, BANKS), cdiv(nred, 1024), []
    for g0 in range(0, groups, 2):
        pair = [g for g in (g0, g0 + 1) if g < groups]
        for c in range(chunks):
            out.append("AiM WR_GB 64 0 0x3")
            out += [f"AiM MAC_ABK 64 0x3 {100 + g * chunks + c}" for g in pair]
        for _ in pair:
            out += ["AiM RD_MAC 0 0x1", "AiM RD_MAC 0 0x2"]
    return out


def parse_sim(lines: list[str]) -> list[tuple]:
    """AiM lines as (op, OPSIZE, channel mask, row).  CENT's WR_GB and RD_MAC carry
    no address the board's could be compared with, so their row is None."""
    s = []
    for line in lines:
        f = line.split()
        if f[1] == "WR_GB":
            s.append(("WRVEC", int(f[2]), "0x3", None))
        elif f[1] == "MAC_ABK":
            s.append(("MAC", int(f[2]), "0x3", int(f[4])))
        elif f[1] == "RD_MAC":
            s.append(("RD_MAC", 0, f[3], None))
    return s


# ---- the board side ----------------------------------------------------------

def read_dump(isa: Path, step: int, layer: int, out: Path, tag: str):
    """RUN_CYC per (layer, op) over every launch of the step, and the ISRs of
    `layer` as (op, OPSIZE, channel mask, ROW, COL, T).  Writes both extracts."""
    cyc = collections.defaultdict(int)
    isrs = collections.defaultdict(list)
    names = {n for n, _, _ in NODES}
    cur = None
    with open(isa / f"step{step:05d}.isa") as f, \
         open(out / f"launches_{tag}.tsv", "w") as hl:
        hl.write("launch\tlayer\tnode\tpart\tisrs\trun_cyc\n")
        for line in f:
            if line.startswith("# launch"):
                t = line.split()
                lay = t[t.index("layer") + 1] if "layer" in t else "-"
                node = t[t.index("layer") + 2] if "layer" in t else t[t.index("pos") + 2]
                run = int(t[t.index("run_cyc") + 1])
                hl.write(f"{t[2]}\t{lay}\t{node}\t{t[t.index('part') + 1]}\t"
                         f"{t[t.index('isrs') + 1]}\t{run}\n")
                cur = None
                if node in names and lay != "-":
                    cyc[(int(lay), node)] += run
                    if int(lay) == layer:
                        cur = node
            elif cur and line[:1] == " " and not line.lstrip().startswith("#"):
                t = line.split()
                if t[1] != "EOS":
                    isrs[cur].append((t[1], int(t[2]), t[5], int(t[3]), int(t[4]), int(t[8])))
    with open(out / f"layer{layer}_isrs_{tag}.tsv", "w") as f:
        f.write("node\top\topsize\tch\trow\tcol\tt\n")
        for node, _, _ in NODES:
            for op, size, ch, row, col, tbit in isrs[node]:
                f.write(f"{node}\t{op}\t{size}\t{ch}\t{row}\t{col}\t{tbit}\n")
    return cyc, isrs


def board_as_aim(isrs: list[tuple]) -> list[str]:
    """The board's ISRs as an AiM trace, keeping the board's row numbers."""
    out = []
    for op, size, ch, row, _, _ in isrs:
        if op == "WRVEC":
            out.append(f"AiM WR_GB {size} 0 {ch}")
        elif op == "MAC":
            out.append(f"AiM MAC_ABK {size} {ch} {row}")
        elif op == "RD_MAC":
            out.append(f"AiM RD_MAC 0 {ch}")
    return out


# ---- the pattern -------------------------------------------------------------

def renumber(xs: list) -> list[int]:
    """Each value replaced by the order in which it first appears."""
    seen: dict = {}
    return [seen.setdefault(x, len(seen)) for x in xs]


def board_chunks(isrs: list[tuple]) -> list[int]:
    """The input chunk of each vector load: its GPR word, renumbered.  The op's
    input starts at chunk 0 and its chunks sit 64 words apart."""
    return renumber([row for op, _, _, row, _, _ in isrs if op == "WRVEC"])


def cent_chunks(seq: list[tuple], chunks: int) -> list[int]:
    """CENT's WR_GB carries no address.  The chunk is the one the next MAC reduces:
    in reuse-bank order a MAC's row is base + group*chunks + chunk."""
    base = min(r for op, _, _, r in seq if op == "MAC")
    out = []
    for i, s in enumerate(seq):
        if s[0] == "WRVEC":
            nxt = next(t[3] for t in seq[i + 1:] if t[0] == "MAC")
            out.append((nxt - base) % chunks)
    return out


def unit_of(cmds: list[tuple], rows: list) -> tuple[str, int, list[int], int, bool]:
    """The shortest block the command list repeats, as letters; how often it
    repeats; its MACs' rows relative to its first MAC; the row step from one repeat
    to the next; and whether every repeat is the first one moved by that step."""
    n = len(cmds)
    p = next(p for p in range(1, n + 1)
             if n % p == 0 and all(cmds[i] == cmds[i % p] for i in range(n)))
    reps = n // p
    unit_rows = [[r for (c, r) in zip(cmds[k * p:(k + 1) * p], rows[k * p:(k + 1) * p])
                  if c[0] == "MAC"] for k in range(reps)]
    first = unit_rows[0]
    step = unit_rows[1][0] - first[0] if reps > 1 else 0
    regular = all(u == [r + k * step for r in first] for k, u in enumerate(unit_rows))
    letters = "".join(LETTER[c[0]] for c in cmds[:p])
    return letters, reps, [r - first[0] for r in first], step, regular


def run_sim(binary: Path, yaml: Path, lines: list[str], path: Path) -> int:
    path.write_text("\n".join(lines + ["AiM EOC"]) + "\n")
    r = subprocess.run([str(binary), "-f", str(yaml), "-t", str(path)],
                       capture_output=True, text=True)
    m = re.search(r"memory_system_cycles:\s*(\d+)", r.stdout)
    if r.returncode or not m:
        sys.exit(f"ramulator2 failed on {path.name}:\n{r.stderr[-2000:]}")
    return int(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--single", type=Path, required=True, help="--isa-trace dir, one latch")
    ap.add_argument("--dual", type=Path, required=True, help="--isa-trace dir, --dual-latch")
    ap.add_argument("--step", type=int, default=1, help="forward step (1 = first decode)")
    ap.add_argument("--layer", type=int, default=0, help="layer whose ISRs are checked")
    ap.add_argument("--timing", type=Path,
                    default=EMU_TOP / "scripts" / "logs" / "aimcmp-1hot",
                    help="a dir with emu_aimcmp's timing.txt (the board registers)")
    ap.add_argument("--out", type=Path, default=HERE / "run")
    ap.add_argument("--cent", type=Path, default=CENT_DEFAULT)
    ap.add_argument("--sim", type=Path, default=SIM_DEFAULT)
    a = ap.parse_args()

    a.out = a.out.resolve()             # the CENT generator runs in its own directory
    for d in ("", "sim", "sim_board_rows"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    timing = ac.read_timing(a.timing)
    yaml = a.out / "aim.yaml"
    ac.write_yaml(a.sim, ac.sim_params(timing), yaml)
    (a.out / "timing.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in timing.items()) + "\n")
    binary = a.sim / "build" / "ramulator2"

    rows = []
    for tag, latches, isa in (("L1", 1, a.single), ("L2", 2, a.dual)):
        raw = a.out / f"cent_reuse_bank_{tag}.trace"
        cent_trace(a.cent, latches, raw)
        ops = split_ops(board_rules(raw.read_text().splitlines()))
        cyc, board = read_dump(isa, a.step, a.layer, a.out, tag)
        for name, nout, nred in NODES:
            lines, source = ops[name], "CENT reuse-bank"
            if latches == 2 and name == "gate_proj":
                lines, source = dual_both_latches(nout, nred), "own (both latches)"
            cent = parse_sim(lines)
            bisr = board[name]
            c_cmds = [s[:3] for s in cent]
            b_cmds = [s[:3] for s in bisr]
            c_rows = [s[3] for s in cent]
            b_rows = [s[3] if s[0] == "MAC" else None for s in bisr]
            c_unit = unit_of(c_cmds, c_rows)
            b_unit = unit_of(b_cmds, b_rows)
            sim = run_sim(binary, yaml, lines, a.out / "sim" / f"{name}_{tag}.trace")
            sim_b = run_sim(binary, yaml, board_as_aim(bisr),
                            a.out / "sim_board_rows" / f"{name}_{tag}.trace")
            per_layer = [cyc[(l, name)] for l in sorted({l for l, _ in cyc})]
            rows.append(dict(
                latch=latches, op=name, board=cyc[(a.layer, name)], sim=sim,
                diff_pct=round(100 * (cyc[(a.layer, name)] - sim) / sim, 3),
                cmds_same=c_cmds == b_cmds,
                rows_same=renumber([r for r in c_rows if r is not None])
                          == renumber([r for r in b_rows if r is not None]),
                chunks_same=cent_chunks(cent, cdiv(nred, 1024)) == board_chunks(bisr),
                unit_same=c_unit == b_unit,
                unit=b_unit[0], repeats=b_unit[1],
                unit_rows="+" + ",".join(map(str, b_unit[2])), row_step=b_unit[3],
                regular=b_unit[4] and c_unit[4],
                board_cols=",".join(map(str, sorted({s[4] for s in bisr if s[0] == "MAC"}))),
                board_t_unit="".join(str(s[5]) for s in bisr[:len(b_unit[0])]),
                sim_board_rows=sim_b, isrs=len(lines), source=source,
                layers=len(per_layer), board_min=min(per_layer), board_max=max(per_layer)))

    with open(a.out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for latches in (1, 2):
        print(f"latch {latches}: step {a.step}, layer {a.layer}")
        b = s = 0
        for r in (r for r in rows if r["latch"] == latches):
            b, s = b + r["board"], s + r["sim"]
            same = all(r[k] for k in ("cmds_same", "rows_same", "chunks_same", "unit_same",
                                      "regular"))
            print(f"  {r['op']:10s} board {r['board']:>10,}  sim {r['sim']:>10,}  "
                  f"{r['diff_pct']:+.2f}%  pattern {'same' if same else 'DIFFERENT'}: "
                  f"{r['unit']} x{r['repeats']} rows {r['unit_rows']} step {r['row_step']}  "
                  f"sim from board rows {'equal' if r['sim_board_rows'] == r['sim'] else r['sim_board_rows']}"
                  f"  ({r['source']}; {r['layers']} layers {r['board_min']:,}..{r['board_max']:,})")
        print(f"  {'sum':10s} board {b:>10,}  sim {s:>10,}  {100 * (b - s) / s:+.2f}%")
    print(f"\nresults in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
