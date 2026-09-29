#!/usr/bin/env python3
"""KV-cache growth experiment, one model per process.

    kv_experiment.py --model HFID --mode alloc --out DIR
    kv_experiment.py --model HFID --mode gen   --out DIR --max-time SECONDS

alloc  Load the model (weights on the card), then grow every layer's K and V one
       token at a time through PimLayer._grow — the cache's own allocation path —
       until the card refuses a page.  Records every grow and where it stopped.
       No K/V data is written; only allocation is exercised.

gen    Real generation with the PIM backend, forced to keep going (no EOS stop)
       until max_tokens or --max-time.  Records, per generated token, the time,
       the ISAs it took and the grows it caused.

Output: DIR/<tag>.<mode>.jsonl (one record per line, flushed as it goes) and
DIR/<tag>.<mode>.summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "sw" / "app"))

import torch
from transformers.generation.streamers import BaseStreamer

from pimllm.cache import PimOutOfMemory
from pimllm.device import meminfo
from pimllm.model import PimModel

PROMPT = ("Write a very long, detailed technical book about computer architecture. "
          "Cover every topic from transistors to datacenters, chapter by chapter, "
          "and never stop writing.")


def jdump(f, rec):
    f.write(json.dumps(rec) + "\n")
    f.flush()


def ev_dict(e, **extra):
    d = dict(tokens=e.tokens, layer=e.layer, which=e.which, units=e.units,
             bytes=e.bytes, room=e.room, t=e.t)
    d.update(extra)
    return d


def run_alloc(m, out, tag):
    cache = m.cache
    s = m.shape
    kv = torch.zeros(1, s.n_kv_heads, 1, s.head_dim, dtype=torch.bfloat16)
    for lay in cache.layers:
        lay.lazy_initialization(kv, kv)
    f = open(out / f"{tag}.alloc.jsonl", "w")
    t0 = time.monotonic()
    seen = 0
    stop = None
    tok = 0
    # Past the estimate on purpose: the point is to find where the card says no.
    limit = m.budget.memory_max_tokens() + 4096
    while tok < limit:
        tok += 1
        try:
            for lay in cache.layers:
                lay._grow("K", lay.k, tok)
                lay._grow("V", lay.v, tok)
        except PimOutOfMemory as e:
            stop = dict(token=tok, error=str(e))
            break
        for e in cache.events[seen:]:
            jdump(f, ev_dict(e, dt=e.t - t0))
        seen = len(cache.events)
    f.close()
    alloc, pooled, held = meminfo()
    return dict(stop=stop, last_token_ok=tok - 1 if stop else tok,
                events=len(cache.events), seconds=time.monotonic() - t0,
                dram_allocated=alloc, dram_pooled=pooled, hugepages_held=held)


class Recorder(BaseStreamer):
    """One record per generated token: its index in the sequence, when it landed,
    the ISAs since the previous one, and the grows the cache made for it."""

    def __init__(self, m, f, n_prompt, t0):
        self.m, self.f, self.n_prompt, self.t0 = m, f, n_prompt, t0
        self.first = True
        self.n = 0
        self.last_t = t0
        self.last_isr = m.rt.stats()["nisr"]
        self.seen = 0

    def put(self, value):
        now = time.monotonic()
        isr = self.m.rt.stats()["nisr"]
        evs = self.m.cache.events[self.seen:]
        self.seen = len(self.m.cache.events)
        if self.first:
            # The prompt.  Its grows happened during prefill.
            self.first = False
            jdump(self.f, dict(kind="prefill", tokens=self.n_prompt, t=now - self.t0,
                               isr=isr - self.last_isr,
                               grows=[ev_dict(e, dt=e.t - self.t0) for e in evs]))
        else:
            self.n += 1
            jdump(self.f, dict(kind="token", i=self.n, seq=self.n_prompt + self.n,
                               t=now - self.t0, dt=now - self.last_t,
                               isr=isr - self.last_isr,
                               grows=[ev_dict(e, dt=e.t - self.t0) for e in evs]))
        self.last_t, self.last_isr = now, isr

    def end(self):
        pass


def run_gen(m, out, tag, max_time, chat):
    tok = m.tokenizer
    if chat and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
    else:
        ids = tok(PROMPT, return_tensors="pt")
    n_prompt = ids["input_ids"].shape[1]
    n_new = m.max_tokens - n_prompt
    f = open(out / f"{tag}.gen.jsonl", "w")
    m.cache.reset()
    t0 = time.monotonic()
    rec = Recorder(m, f, n_prompt, t0)
    stopped = None
    try:
        with torch.no_grad():
            m.model.generate(**ids, max_new_tokens=n_new, min_new_tokens=n_new,
                             do_sample=False, past_key_values=m.cache, use_cache=True,
                             streamer=rec, pad_token_id=tok.eos_token_id,
                             max_time=max_time)
    except PimOutOfMemory as e:
        stopped = str(e)
    f.close()
    st = m.rt.stats()
    alloc, pooled, held = meminfo()
    return dict(n_prompt=n_prompt, generated=rec.n, seq_end=n_prompt + rec.n,
                seconds=time.monotonic() - t0, max_time=max_time, oom=stopped,
                grows=len(m.cache.events), nisr=st["nisr"], nlaunch=st["nlaunch"],
                dram_allocated=alloc, dram_pooled=pooled, hugepages_held=held)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=("alloc", "gen"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-time", type=float, default=3600)
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.model.split("/")[-1]

    t_load = time.monotonic()
    m = PimModel(a.model, verbose=False)
    b = m.budget
    info = dict(model=a.model, mode=a.mode, layers=m.shape.layers,
                kv_width=m.shape.kv_width, weights_bytes=b.weight_bytes,
                capacity=b.capacity, memory_max_tokens=b.memory_max_tokens(),
                model_max_position=m.shape.max_position, max_tokens=m.max_tokens,
                max_tokens_limit=m.max_tokens_limit, kv_bytes_1024=b.kv_bytes(1024),
                load_seconds=time.monotonic() - t_load,
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps(info), flush=True)
    if a.mode == "alloc":
        res = run_alloc(m, out, tag)
    else:
        res = run_gen(m, out, tag, a.max_time, a.chat)
    info.update(res, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / f"{tag}.{a.mode}.summary.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    m.free()


if __name__ == "__main__":
    main()
