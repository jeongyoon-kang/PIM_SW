# SPDX-License-Identifier: MIT
"""Every ISA program the card runs during generation, written to a directory.

One file per forward step of the model:

    step00000.isa   the prompt, positions 0 .. P-1; it produces generated token 1
    step00001.isa   generated token 1 fed back at position P; it produces token 2
    step0000k.isa   generated token k at position P-1+k; it produces token k+1

In a file, every launch is a header line followed by its program, one line per ISR:

    # launch 812  step 1  pos 7  layer 3  q_proj  part 1  isrs 194  run_cyc 51234  us 60
      #     op      OPSIZE  ROW      COL  CH    PU     GBMC   T  word (bit 255 first)
      0     WRVEC   64      0        0    0x3   0      0      0  0000...

    pos      the sequence position the launch computes.  Attention over the prompt
             covers every prompt position at once and shows the range, e.g. 0-5.
    layer    the decoder layer.  lm_head has none.
    node     q_proj k_proj v_proj o_proj gate_proj up_proj down_proj lm_head, and
             attn_qk / attn_sv for the two attention matmuls.
    part     which launch of this node's op it is; a large op takes several.
    run_cyc  the board's RUN_CYC for the launch: PL cycles from the doorbell to done.

`grep '^# launch' step*.isa` lists every launch.  steps.tsv has one row per step with
its positions, launches, ISRs and RUN_CYC summed.
"""
from __future__ import annotations

from pathlib import Path


class IsaTrace:
    """Records the launches of one ops.Runtime under `directory`.

    The model calls begin_step() at the start of every forward pass and label()
    before every op; end_step() closes the current file."""

    def __init__(self, rt, directory):
        self.rt = rt
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step = -1
        self.pos_first = self.pos_last = 0
        self._at: dict | None = None          # rt.stats() when the open step began
        self._steps = open(self.dir / "steps.tsv", "w")
        self._steps.write("step\tpos_first\tpos_last\tfile\tlaunches\tisrs\trun_cyc\n")
        rt.tracer = self

    def file(self, step: int) -> str:
        return f"step{step:05d}.isa"

    def begin_step(self, pos_first: int, pos_last: int) -> None:
        """A forward pass over positions pos_first .. pos_last starts."""
        self.end_step()
        self.step += 1
        self.pos_first, self.pos_last = pos_first, pos_last
        self.rt.trace_open(self.dir / self.file(self.step),
                           f"# step {self.step}  positions {pos_first}-{pos_last}\n")
        self._at = self.rt.stats()

    def label(self, node: str, layer: int | None, pos_first: int,
              pos_last: int | None = None) -> None:
        """Label the launches of the next op."""
        pos = (str(pos_first) if pos_last is None or pos_last == pos_first
               else f"{pos_first}-{pos_last}")
        where = f"layer {layer}  " if layer is not None else ""
        self.rt.trace_label(f"step {self.step}  pos {pos}  {where}{node}")

    def end_step(self) -> None:
        """Close the open step's file and add its row to steps.tsv."""
        if self._at is None:
            return
        self.rt.trace_close()
        now = self.rt.stats()
        self._steps.write(
            f"{self.step}\t{self.pos_first}\t{self.pos_last}\t{self.file(self.step)}\t"
            f"{now['nlaunch'] - self._at['nlaunch']}\t{now['nisr'] - self._at['nisr']}\t"
            f"{now['run_cyc'] - self._at['run_cyc']}\n")
        self._steps.flush()
        self._at = None

    def close(self) -> None:
        self.end_step()
        self._steps.close()
        if self.rt.tracer is self:
            self.rt.tracer = None
