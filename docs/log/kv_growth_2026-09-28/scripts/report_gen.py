#!/usr/bin/env python3
"""Build the KV growth report (HTML) from the experiment output in ../data/."""
from __future__ import annotations

import collections
import html
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent      # the experiment folder
D = HERE / "data"
OUT = HERE / "kv_report.html"

MODELS = [
    ("Llama-3.2-1B-Instruct", "llama1b", "Llama-3.2-1B-Instruct"),
    ("Llama-3.2-3B-Instruct", "llama3b", "Llama-3.2-3B-Instruct"),
    ("Qwen3-0.6B", "qwen06b", "Qwen3-0.6B"),
]
MiB = 1 << 20
GiB = 1 << 30
UNIT = 64 * 1024


def esc(s):
    return html.escape(str(s))


def n(v):
    return f"{v:,}"


def size(b):
    if b >= GiB:
        return f"{b / GiB:.2f} GiB"
    if b >= MiB:
        v = b / MiB
        return f"{int(v)} MiB" if v == int(v) else f"{v:.2f} MiB"
    return f"{b // 1024} KiB"


def dur(s):
    if s >= 3600:
        return f"{int(s // 3600)}시간 {int(s % 3600 // 60)}분"
    if s >= 60:
        return f"{int(s // 60)}분 {int(s % 60)}초"
    return f"{s:.1f}초"


# --------------------------------------------------------------------- data ---
def load_alloc(tag):
    s = json.loads((D / f"{tag}.alloc.summary.json").read_text())
    evs = [json.loads(l) for l in open(D / f"{tag}.alloc.jsonl")]
    g = collections.OrderedDict()
    for e in evs:
        a = g.setdefault((e["tokens"], e["which"]), dict(layers=0, units=0, bytes=0))
        a["layers"] += 1
        a["units"] += e["units"]
        a["bytes"] += e["bytes"]
    out = dict(s=s, groups=g)
    for w in ("K", "V"):
        toks = [t for (t, ww) in g if ww == w]
        diffs = collections.Counter(b - a for a, b in zip(toks, toks[1:]))
        out[w] = dict(tokens=toks, count=len(toks),
                      interval=diffs.most_common(1)[0][0] if diffs else None,
                      regular=len(diffs) <= 1,
                      bytes=g[(toks[1], w)]["bytes"] if len(toks) > 1 else g[(toks[0], w)]["bytes"],
                      layers=g[(toks[0], w)]["layers"])
    stop = s.get("stop")
    if stop:
        m = re.search(r"layer (\d+): no card memory for a (\w) page at token (\d+)", stop["error"])
        out["stop"] = dict(token=stop["token"], layer=int(m.group(1)), which=m.group(2))
    return out


def load_gen(tag):
    p = D / f"{tag}.gen.jsonl"
    if not p.exists():
        return None
    recs = []
    for l in open(p):
        try:
            recs.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    sp = D / f"{tag}.gen.summary.json"
    summ = json.loads(sp.read_text()) if sp.exists() else None
    pre = next((r for r in recs if r["kind"] == "prefill"), None)
    toks = [r for r in recs if r["kind"] == "token"]
    if not pre or not toks:
        return dict(summary=summ, prefill=pre, tokens=toks, events=[], done=summ is not None)
    events = []  # (seq position that needed it, which, layers, units, bytes, t)
    for r in toks:
        g = collections.OrderedDict()
        for e in r["grows"]:
            a = g.setdefault((e["tokens"], e["which"]), dict(layers=0, units=0, bytes=0, t=e["dt"]))
            a["layers"] += 1
            a["units"] += e["units"]
            a["bytes"] += e["bytes"]
        for (tk, w), a in g.items():
            events.append(dict(tokens=tk, which=w, i=r["i"], **a))
    return dict(summary=summ, prefill=pre, tokens=toks, events=events,
                done=summ is not None)


# ------------------------------------------------------------------- charts ---
def nice_ticks(lo, hi, count=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / count
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= count:
            break
    first = math.ceil(lo / step) * step
    ticks = []
    v = first
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


CHART_ID = [0]


def chart(title, sub, series, xmax, ymax, xlab, ylab, xfmt, yfmt, vlines=(), hlines=(),
          ticks_bottom=(), step=True, key=None, xticks=None, yticks=None, height=240):
    """series: list of dict(name, cls, xs, ys).  Returns a <figure>."""
    CHART_ID[0] += 1
    cid = f"c{CHART_ID[0]}"
    W, H = 720, height
    L, R, T, B = 58, 18, 14, 34
    pw, ph = W - L - R, H - T - B

    def X(x):
        return L + pw * (x / xmax if xmax else 0)

    def Y(y):
        return T + ph * (1 - (y / ymax if ymax else 0))

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" '
             f'data-l="{L}" data-pw="{pw}" data-xmax="{xmax}" id="{cid}-svg">']
    yt = yticks or nice_ticks(0, ymax, 4)
    for v in yt:
        if v > ymax * 1.0001:
            continue
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{esc(yfmt(v))}</text>')
    xt = xticks or nice_ticks(0, xmax, 6)
    for v in xt:
        if v > xmax * 1.0001:
            continue
        parts.append(f'<text class="tick" x="{X(v):.1f}" y="{H - B + 18}" text-anchor="middle">{esc(xfmt(v))}</text>')
    parts.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{T + ph}" y2="{T + ph}"/>')
    for hl in hlines:
        parts.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{Y(hl["y"]):.1f}" y2="{Y(hl["y"]):.1f}"/>')
        parts.append(f'<text class="reflabel" x="{W - R - 4}" y="{Y(hl["y"]) - 6:.1f}" text-anchor="end">{esc(hl["label"])}</text>')
    for i, vl in enumerate(vlines):
        x = X(vl["x"])
        parts.append(f'<line class="vmark {vl.get("cls", "")}" x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{T + ph}"/>')
        if vl.get("label"):
            anchor = "end" if x > L + pw * 0.8 else "start"
            dx = -5 if anchor == "end" else 5
            parts.append(f'<text class="vlabel" x="{x + dx:.1f}" y="{T + 12 + 13 * vl.get("row", 0):.1f}" '
                         f'text-anchor="{anchor}">{esc(vl["label"])}</text>')
    for tb in ticks_bottom:
        x = X(tb)
        parts.append(f'<line class="ktick" x1="{x:.1f}" x2="{x:.1f}" y1="{T + ph - 7}" y2="{T + ph}"/>')
    for s in series:
        xs, ys = s["xs"], s["ys"]
        if not xs:
            continue
        d = [f"M{X(xs[0]):.1f},{Y(ys[0]):.1f}"]
        for k in range(1, len(xs)):
            if step:
                d.append(f"H{X(xs[k]):.1f}V{Y(ys[k]):.1f}")
            else:
                d.append(f"L{X(xs[k]):.1f},{Y(ys[k]):.1f}")
        if step and s.get("extend") is not None:
            d.append(f"H{X(s['extend']):.1f}")
        parts.append(f'<path class="line {s["cls"]}" d="{"".join(d)}"/>')
    parts.append(f'<text class="axlabel" x="{L}" y="{T - 2}" text-anchor="start">{esc(ylab)}</text>')
    parts.append(f'<text class="axlabel" x="{W - R}" y="{H - 2}" text-anchor="end">{esc(xlab)}</text>')
    parts.append(f'<line class="cross" x1="0" x2="0" y1="{T}" y2="{T + ph}" visibility="hidden"/>')
    parts.append("</svg>")
    data = dict(step=step, xmax=xmax, series=[dict(name=s["name"], cls=s["cls"], xs=s["xs"],
                                                   ys=s["ys"], fmt=s.get("fmt", "")) for s in series])
    legend = ""
    if len(series) > 1 or key:
        items = "".join(f'<span class="key"><i class="sw {s["cls"]}"></i>{esc(s["name"])}</span>'
                        for s in series)
        legend = f'<div class="legend">{items}{key or ""}</div>'
    return (f'<figure class="chart" id="{cid}"><figcaption><strong>{esc(title)}</strong>'
            f'<span>{esc(sub)}</span></figcaption>{legend}<div class="plot">{"".join(parts)}'
            f'<div class="tip" hidden></div></div>'
            f'<script type="application/json" class="cdata">{json.dumps(data)}</script></figure>')


def fmt_tok(v):
    v = int(round(v))
    return f"{v // 1000}k" if v >= 1000 and v % 1000 == 0 else f"{v:,}"


# ------------------------------------------------------------------ sections ---
def staircase(a, upto):
    """Cumulative K and V MiB (all layers) for tokens 0..upto, from the alloc events."""
    out = {}
    for w in ("K", "V"):
        xs, ys, tot = [0], [0.0], 0
        for (tk, ww), g in a["groups"].items():
            if ww != w or tk > upto:
                continue
            tot += g["bytes"]
            xs.append(tk)           # the page is there from the token that needed it
            ys.append(tot / MiB)
        out[w] = (xs, ys)
    return out


def fullrange(a):
    xs, ys, tot = [0], [0.0], 0
    evs = sorted(a["groups"].items(), key=lambda kv: kv[0][0])
    for (tk, w), g in evs:
        tot += g["bytes"]
        xs.append(tk)
        ys.append(tot / GiB)
    return xs, ys


def model_section(tag, anchor, label, a, gen):
    s = a["s"]
    K, V = a["K"], a["V"]
    layers = s["layers"]
    free = s["capacity"] - s["weights_bytes"]
    stop = a.get("stop")
    kpl = K["bytes"] // K["layers"]
    vpl = V["bytes"] // V["layers"]

    # ---- the event timeline (it is a sequence, so it is numbered by token)
    ev = []
    ev.append((1, "K + V", f"첫 할당. 레이어마다 K {size(kpl)}, V {size(vpl)} (전체 {size(K['bytes'] + V['bytes'])})"))
    ev.append((K["tokens"][1], "K", f"이후 {K['interval']}토큰마다 ({', '.join(n(t) for t in K['tokens'][1:4])}, …). "
               f"레이어마다 {size(kpl)}, 전체 {size(K['bytes'])}. 총 {n(K['count'])}회"))
    ev.append((V["tokens"][1], "V", f"이후 {V['interval']}토큰마다 ({', '.join(n(t) for t in V['tokens'][1:4])}, …). "
               f"레이어마다 {size(vpl)}, 전체 {size(V['bytes'])}. 총 {n(V['count'])}회"))
    if s["model_max_position"] and s["model_max_position"] < s["memory_max_tokens"]:
        ev.append((s["model_max_position"], "한계", "모델 최대 문맥(max_position_embeddings). 표시되는 최대 토큰"))
    if stop:
        ev.append((stop["token"] - 1, "마지막", "모든 레이어의 K, V 할당 성공"))
        ev.append((stop["token"], "실패", f"{stop['layer']}번 레이어의 {stop['which']} 페이지 할당 실패. "
                   f"그 앞 레이어들은 이 토큰의 페이지를 이미 받은 상태. 직후 카드 DRAM "
                   f"{n(s['dram_allocated'] // MiB)} MiB / {n(s['capacity'] // MiB)} MiB 사용 중"))
    wcls = {"K + V": "w-kv", "K": "w-k", "V": "w-v", "한계": "w-lim", "마지막": "w-last", "실패": "w-fail"}
    rows = "".join(f'<li><span class="at">{n(t)}</span><span class="what {wcls[w]}">{esc(w)}</span>'
                   f'<span class="desc">{esc(d)}</span></li>' for t, w, d in ev)
    timeline = f'<ol class="timeline" aria-label="{esc(label)} 할당 타임라인">{rows}</ol>'

    # ---- charts from the allocation run
    upto = 2200
    st = staircase(a, upto)
    kmax = st["K"][1][-1]
    vmax = st["V"][1][-1]
    ymax = max(kmax, vmax) * 1.15
    c1 = chart(f"처음 {n(upto)}토큰의 누적 할당", "전 레이어 합계, 할당 전용 실행", [
        dict(name="K", cls="k", xs=st["K"][0], ys=st["K"][1], extend=upto, fmt="mib"),
        dict(name="V", cls="v", xs=st["V"][0], ys=st["V"][1], extend=upto, fmt="mib"),
    ], upto, ymax, "토큰", "MiB", fmt_tok, lambda v: f"{v:g}")
    fx, fy = fullrange(a)
    last = (stop["token"] - 1) if stop else fx[-1]
    vl = []
    if s["model_max_position"] and s["model_max_position"] < last:
        vl.append(dict(x=s["model_max_position"], label=f"모델 문맥 {n(s['model_max_position'])}", cls="ctx"))
    vl.append(dict(x=last, label=f"메모리 한계 {n(last)}", cls="lim", row=1 if vl else 0))
    c2 = chart("전체 구간: 토큰 수에 따른 KV 사용량", "할당 전용 실행, 카드가 거부할 때까지", [
        dict(name="KV 합계", cls="t", xs=fx, ys=fy, extend=last, fmt="gib"),
    ], last * 1.04, free / GiB * 1.12, "토큰", "GiB", fmt_tok, lambda v: f"{v:g}",
        vlines=vl, hlines=[dict(y=free / GiB, label=f"가중치 뒤 남은 DRAM {size(free)}")])

    # ---- the real generation
    gsec = ""
    if gen and gen["tokens"]:
        toks = gen["tokens"]
        pre = gen["prefill"]
        summ = gen["summary"]
        xs = [r["seq"] for r in toks[1:]]
        dts = [r["dt"] for r in toks[1:]]
        isr = [r["isr"] for r in toks[1:]]
        # A grow for "the cache holds N tokens" happens while output token N+1 is
        # computed, so its cost shows in that token's time: the record it was logged in.
        seq_of = {r["i"]: r["seq"] for r in toks}
        vpos = [seq_of[e["i"]] for e in gen["events"] if e["which"] == "V" and e["i"] > 1]
        kpos = [seq_of[e["i"]] for e in gen["events"] if e["which"] == "K" and e["i"] > 1]
        xmax = max(xs[-1], pre["tokens"] + 10) if xs else pre["tokens"] + 10
        # spike at a V grow: the token's time against the median of its 10 neighbours
        idx = {r["seq"]: k for k, r in enumerate(toks)}
        # One entry per token.  A token where V grew usually has a K grow too (both
        # steps divide 1024), so it counts as V; the rest are K-only.
        spikes = []
        vset = set(vpos)
        for p in sorted(vset | set(kpos)):
            k = idx.get(p)
            if k is None or k < 6 or k + 6 > len(toks):
                continue
            nb = sorted(toks[j]["dt"] for j in range(k - 5, k + 6) if j != k)
            spikes.append((p, "V" if p in vset else "K", toks[k]["dt"], nb[len(nb) // 2]))
        vsp = [x for x in spikes if x[1] == "V"]
        ksp = [x for x in spikes if x[1] == "K"]
        c3 = chart("실제 생성: 토큰당 시간", "생성 토큰마다 한 점. 세로선은 V grow", [
            dict(name="토큰당 시간", cls="t", xs=xs, ys=dts, fmt="sec"),
        ], xmax, max(dts) * 1.12 if dts else 1, "시퀀스 위치(토큰)", "초", fmt_tok,
            lambda v: f"{v:g}", step=False,
            vlines=[dict(x=p, cls="vgrow") for p in vpos],
            ticks_bottom=kpos,
            key='<span class="key"><i class="sw vg"></i>V grow</span>'
                '<span class="key"><i class="sw kt"></i>K grow (아래 눈금)</span>')
        c4 = chart("실제 생성: 토큰당 ISA", "문맥이 길어질수록 q·K가 늘어남", [
            dict(name="토큰당 ISA", cls="t", xs=xs, ys=isr, fmt="int"),
        ], xmax, max(isr) * 1.12 if isr else 1, "시퀀스 위치(토큰)", "ISA", fmt_tok,
            lambda v: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}", step=False)
        # the event list of the real run
        erows = []
        for e in gen["events"][:14]:
            where = "prefill" if e["i"] == 1 else f"출력 {n(seq_of[e['i']])}번째 토큰 계산 중"
            erows.append(f'<tr><td class="num">{n(e["tokens"])}</td><td>{esc(e["which"])}</td>'
                         f'<td>{esc(where)}</td><td class="num">{e["layers"]}</td>'
                         f'<td class="num">{n(e["units"])}</td><td class="num">{esc(size(e["bytes"]))}</td>'
                         f'<td class="num">{e["t"]:.1f}</td></tr>')
        more = len(gen["events"]) - 14
        state = ("완료" if gen["done"] else "진행 중")
        span = summ["seconds"] if summ else toks[-1]["t"]
        first_dt = toks[1]["dt"] if len(toks) > 1 else 0.0
        last_dt = sum(dts[-20:]) / min(20, len(dts)) if dts else 0.0
        parts = []
        for name, sp in (("V(와 K)가 grow한", vsp), ("K만 grow한", ksp)):
            if sp:
                ratio = sum(x[2] for x in sp) / sum(x[3] for x in sp)
                parts.append(f"{name} 토큰 {n(len(sp))}개는 {ratio:.2f}배")
        vtxt = ("grow가 일어난 토큰의 생성 시간을 앞뒤 10토큰의 중앙값과 비교하면 " + ", ".join(parts) +
                "였습니다. V grow 토큰에는 새 V 페이지를 0으로 채우는 쓰기가 함께 들어갑니다.") if parts else ""
        gsec = f"""
      <h3>실제 생성 ({state})</h3>
      <p>프롬프트 {n(pre['tokens'])}토큰으로 시작해 {n(xs[-1] if xs else pre['tokens'])}번째 토큰까지
      {n(len(toks))}개를 생성했습니다({dur(span)}). 토큰당 시간은 처음 {first_dt:.2f}초에서
      마지막 20토큰 평균 {last_dt:.2f}초로 늘었습니다. {esc(vtxt)}</p>
      {c3}{c4}
      <div class="tablewrap"><table class="data">
        <caption>실제 생성 중 할당 (처음 {min(14, len(gen['events']))}건{f', 이후 {n(more)}건 생략' if more > 0 else ''})</caption>
        <thead><tr><th>캐시 토큰 수</th><th>K/V</th><th>시점</th><th>레이어</th><th>unit</th><th>크기</th><th>시작 후 초</th></tr></thead>
        <tbody>{''.join(erows)}</tbody></table></div>"""
    else:
        gsec = "<h3>실제 생성</h3><p class=\"muted\">아직 이 모델의 생성 실행이 시작되지 않았습니다.</p>"

    return f"""
    <section class="model" id="{anchor}">
      <h2>{esc(label)}</h2>
      <p class="meta">{layers}개 레이어 · KV 너비 {s['kv_width']} ({'토큰 2개가 한 행을 나눠 씀' if s['kv_width'] <= 512 else '토큰 하나가 한 행 전체'}) ·
      가중치 {size(s['weights_bytes'])} · 추정 최대 토큰 {n(s['max_tokens'])} ({'모델 문맥 한계' if s['max_tokens_limit'] == 'model context' else '카드 메모리 한계'})</p>
      <h3>할당 타임라인</h3>
      {timeline}
      {c1}{c2}
      {gsec}
    </section>"""


def summary_tables(data):
    r1, r2 = [], []
    for tag, anchor, label in MODELS:
        a, g = data[tag]
        s, K, V, stop = a["s"], a["K"], a["V"], a.get("stop")
        kpl = K["bytes"] // K["layers"]
        vpl = V["bytes"] // V["layers"]
        lim = "모델 문맥" if s["max_tokens_limit"] == "model context" else "카드 메모리"
        r1.append(f"""<tr><th scope="row"><a href="#{anchor}">{esc(label)}</a></th>
          <td class="num">{s['layers']}</td><td class="num">{s['kv_width']}</td>
          <td class="num">{K['interval']}토큰</td><td class="num">{size(kpl)} / {size(K['bytes'])}</td>
          <td class="num">{V['interval']}토큰</td><td class="num">{size(vpl)} / {size(V['bytes'])}</td>
          <td class="num">{n(K['count'])} / {n(V['count'])}</td></tr>""")
        stopc = f"{n(stop['token'])} ({stop['layer']}번 레이어 {stop['which']})" if stop else "-"
        match = "일치" if stop and stop["token"] - 1 == s["memory_max_tokens"] else "불일치"
        if g and g["tokens"]:
            toks = g["tokens"]
            reach = f"{n(toks[-1]['seq'])}"
            dts = [r["dt"] for r in toks[1:]]
            speed = (f"{toks[1]['dt']:.2f} → {sum(dts[-20:]) / min(20, len(dts)):.2f}"
                     if len(toks) > 1 else "-")
            span = dur(g["summary"]["seconds"] if g["summary"] else toks[-1]["t"])
            state = "" if g["done"] else " (진행 중)"
        else:
            reach, speed, span, state = "-", "-", "-", " (대기)"
        r2.append(f"""<tr><th scope="row"><a href="#{anchor}">{esc(label)}</a></th>
          <td class="num">{n(s['memory_max_tokens'])}</td><td class="num">{n(s['model_max_position'])}</td>
          <td class="num"><strong>{n(s['max_tokens'])}</strong> <span class="muted">{lim}</span></td>
          <td class="num">{n(stop['token'] - 1) if stop else '-'}</td><td class="num">{esc(stopc)}</td>
          <td>{match}</td>
          <td class="num">{reach}{state}</td><td class="num">{span}</td><td class="num">{speed}</td></tr>""")
    t1 = f"""<div class="tablewrap"><table class="data">
      <caption>Grow 규칙 (레이어당 / 전 레이어 합계)</caption>
      <thead><tr><th>모델</th><th>레이어</th><th>KV 너비</th><th>K 간격</th><th>K 1회 할당</th>
      <th>V 간격</th><th>V 1회 할당</th><th>K / V 횟수<br><span class="muted">한계까지</span></th></tr></thead>
      <tbody>{''.join(r1)}</tbody></table></div>"""
    t2 = f"""<div class="tablewrap"><table class="data">
      <caption>한계: 추정과 실측</caption>
      <thead><tr><th>모델</th><th>메모리 추정</th><th>모델 문맥</th><th>표시 최대 토큰</th>
      <th>실측 마지막 성공</th><th>첫 실패 토큰</th><th>추정 대비</th>
      <th>실제 생성 도달</th><th>생성 시간</th><th>토큰당 초<br><span class="muted">처음 → 마지막</span></th></tr></thead>
      <tbody>{''.join(r2)}</tbody></table></div>"""
    return t1, t2


CSS = r"""
:root{
  --page:#f5f6f4; --surface:#fcfcfb; --ink:#141617; --ink2:#4d5356; --muted:#80868a;
  --hair:#dfe2de; --axis:#c3c6c1; --k:#2a78d6; --v:#eb6834; --t:#2a78d6; --ref:#6b7074;
  --fail:#c23b3b; --chip:#eceeeb; --link:#1f5fae;
  --sans:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --page:#101211; --surface:#181a19; --ink:#eff1ee; --ink2:#c0c4bf; --muted:#8b908c;
    --hair:#2a2d2b; --axis:#3a3d3b; --k:#3987e5; --v:#d95926; --t:#3987e5; --ref:#9aa09b;
    --fail:#e66767; --chip:#232624; --link:#7fb0ef;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#101211; --surface:#181a19; --ink:#eff1ee; --ink2:#c0c4bf; --muted:#8b908c;
  --hair:#2a2d2b; --axis:#3a3d3b; --k:#3987e5; --v:#d95926; --t:#3987e5; --ref:#9aa09b;
  --fail:#e66767; --chip:#232624; --link:#7fb0ef;
}
body{background:var(--page);color:var(--ink);font:15px/1.65 var(--sans);}
.doc{max-width:1000px;margin:0 auto;padding-inline:20px;padding-block:28px 64px;display:grid;gap:34px}
header{display:grid;gap:10px}
.eyebrow{font:500 12px/1.4 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0}
h1{font:600 30px/1.25 var(--sans);margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font:600 22px/1.3 var(--sans);margin:0;text-wrap:balance}
h3{font:600 16px/1.4 var(--sans);margin:18px 0 6px}
p{margin:0;max-width:68ch}
.lede{font-size:16px;color:var(--ink2)}
.muted{color:var(--muted)}
a{color:var(--link)}
code,.mono{font-family:var(--mono);font-size:.92em}
.facts{display:flex;flex-wrap:wrap;gap:6px 18px;margin:4px 0 0;padding:0}
.facts div{display:flex;gap:6px;align-items:baseline}
.facts dt{font:500 12px var(--mono);color:var(--muted);letter-spacing:.04em}
.facts dd{margin:0;font-size:14px}
section{display:grid;gap:10px}
section.model{border-top:1px solid var(--hair);padding-top:26px}
.meta{color:var(--ink2);font-size:14px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--hair);border-radius:6px}
table.data{border-collapse:collapse;width:100%;font-size:13.5px}
table.data caption{text-align:left;font-weight:600;padding:10px 12px 4px;caption-side:top}
table.data th,table.data td{padding:7px 12px;border-bottom:1px solid var(--hair);text-align:left;vertical-align:top;white-space:nowrap}
table.data thead th{font-weight:500;color:var(--ink2);font-size:12.5px;border-bottom:1px solid var(--axis)}
table.data tbody tr:last-child th,table.data tbody tr:last-child td{border-bottom:0}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:12.5px}
.timeline{list-style:none;margin:0;padding:0;display:grid;gap:0;background:var(--surface);border:1px solid var(--hair);border-radius:6px}
.timeline li{display:grid;grid-template-columns:92px 64px 1fr;gap:12px;padding:8px 12px;border-bottom:1px solid var(--hair);align-items:baseline}
.timeline li:last-child{border-bottom:0}
.timeline .at{font:500 13px var(--mono);font-variant-numeric:tabular-nums;text-align:right}
.timeline .what{font:600 11.5px var(--mono);letter-spacing:.04em;padding:1px 8px;border-radius:99px;background:var(--chip);justify-self:start}
.timeline .what.w-k::before,.timeline .what.w-v::before,.timeline .what.w-kv::before{content:"";display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}
.timeline .what.w-k::before{background:var(--k)}
.timeline .what.w-v::before{background:var(--v)}
.timeline .what.w-kv::before{background:linear-gradient(90deg,var(--k) 50%,var(--v) 50%)}
.timeline .what.w-fail{color:var(--fail)}
.timeline .desc{color:var(--ink2);font-size:14px}
figure.chart{margin:8px 0 0;background:var(--surface);border:1px solid var(--hair);border-radius:6px;padding:12px 14px 8px;display:grid;gap:6px}
figure.chart figcaption{display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline}
figure.chart figcaption strong{font-weight:600;font-size:14px}
figure.chart figcaption span{color:var(--muted);font-size:12.5px}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12.5px;color:var(--ink2)}
.key{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:14px;height:2px;border-radius:1px;background:var(--ink2)}
.sw.k{background:var(--k)} .sw.v{background:var(--v)} .sw.t{background:var(--t)}
.sw.vg{width:2px;height:12px;background:var(--v);opacity:.55}
.sw.kt{width:2px;height:7px;background:var(--k);opacity:.6}
.plot{position:relative}
.plot svg{display:block;width:100%;height:auto;overflow:visible;touch-action:pan-y}
svg .grid{stroke:var(--hair);stroke-width:1}
svg .axis{stroke:var(--axis);stroke-width:1}
svg .tick{fill:var(--muted);font:11px var(--mono)}
svg .axlabel{fill:var(--muted);font:11px var(--sans)}
svg .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .line.k{stroke:var(--k)} svg .line.v{stroke:var(--v)} svg .line.t{stroke:var(--t)}
svg .ref{stroke:var(--ref);stroke-width:1;stroke-dasharray:4 3}
svg .reflabel{fill:var(--ink2);font:11.5px var(--sans)}
svg .vmark{stroke:var(--ref);stroke-width:1}
svg .vmark.lim{stroke:var(--fail)}
svg .vmark.vgrow{stroke:var(--v);opacity:.45}
svg .vlabel{fill:var(--ink2);font:11.5px var(--sans)}
svg .ktick{stroke:var(--k);stroke-width:1;opacity:.55}
svg .cross{stroke:var(--ink2);stroke-width:1;opacity:.6}
.tip{position:absolute;top:6px;pointer-events:none;background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:6px 9px;font-size:12.5px;box-shadow:0 2px 8px rgba(0,0,0,.08);min-width:120px}
.tip .v{font:600 13px var(--mono);font-variant-numeric:tabular-nums}
.tip .row{display:flex;align-items:center;gap:7px;white-space:nowrap}
.tip .x{color:var(--muted);font:11.5px var(--mono);margin-bottom:2px}
ul.plain{margin:0;padding-left:18px;display:grid;gap:4px;max-width:78ch}
ul.plain li{color:var(--ink)}
.note{background:var(--surface);border:1px solid var(--hair);border-radius:6px;padding:12px 14px}
@media (max-width:560px){
  h1{font-size:25px}
  .timeline li{grid-template-columns:76px 1fr;}
  .timeline .desc{grid-column:1 / -1}
}
"""

JS = r"""
(function(){
  function fmt(v, f){
    if(f==='mib') return (Math.round(v*100)/100).toLocaleString()+' MiB';
    if(f==='gib') return (Math.round(v*1000)/1000).toLocaleString()+' GiB';
    if(f==='sec') return v.toFixed(2)+' 초';
    if(f==='int') return Math.round(v).toLocaleString();
    return String(v);
  }
  function idx(xs, x, step){
    var lo=0, hi=xs.length-1;
    if(x<=xs[0]) return 0;
    if(x>=xs[hi]) return hi;
    while(hi-lo>1){ var m=(lo+hi)>>1; if(xs[m]<=x) lo=m; else hi=m; }
    if(step) return lo;
    return (x-xs[lo] < xs[hi]-x) ? lo : hi;
  }
  document.querySelectorAll('figure.chart').forEach(function(fig){
    var svg=fig.querySelector('svg'), tip=fig.querySelector('.tip');
    var data=JSON.parse(fig.querySelector('.cdata').textContent);
    var cross=svg.querySelector('.cross');
    var L=+svg.dataset.l, PW=+svg.dataset.pw, XM=+svg.dataset.xmax;
    function show(clientX){
      var r=svg.getBoundingClientRect(); var sx=r.width/720;
      var px=(clientX-r.left)/sx; var x=(px-L)/PW*XM;
      if(px<L||px>L+PW){ hide(); return; }
      tip.textContent='';
      var head=document.createElement('div'); head.className='x';
      var xv=Math.max(0,Math.round(x));
      var s0=data.series[0]; var k0=idx(s0.xs,x,data.step);
      var xs=data.step? xv : s0.xs[k0];
      head.textContent=Math.round(xs).toLocaleString()+'번째 토큰';
      tip.appendChild(head);
      data.series.forEach(function(s){
        var k=idx(s.xs, data.step? x : xs, data.step);
        var row=document.createElement('div'); row.className='row';
        var sw=document.createElement('i'); sw.className='sw '+s.cls;
        var val=document.createElement('span'); val.className='v'; val.textContent=fmt(s.ys[k], s.fmt);
        var nm=document.createElement('span'); nm.textContent=s.name;
        row.appendChild(sw); row.appendChild(val); row.appendChild(nm); tip.appendChild(row);
      });
      var cx=L+PW*((data.step? x : xs)/XM);
      cross.setAttribute('x1',cx); cross.setAttribute('x2',cx); cross.setAttribute('visibility','visible');
      tip.hidden=false;
      var left=cx*sx+12; if(left+tip.offsetWidth>r.width) left=cx*sx-tip.offsetWidth-12;
      tip.style.left=Math.max(0,left)+'px';
    }
    function hide(){ tip.hidden=true; cross.setAttribute('visibility','hidden'); }
    svg.addEventListener('pointermove', function(e){ show(e.clientX); });
    svg.addEventListener('pointerleave', hide);
    svg.setAttribute('tabindex','0');
    svg.addEventListener('focus', function(){ var r=svg.getBoundingClientRect(); show(r.left+r.width*0.5); });
    svg.addEventListener('blur', hide);
  });
})();
"""


def main():
    data = {tag: (load_alloc(tag), load_gen(tag)) for tag, _, _ in MODELS}
    t1, t2 = summary_tables(data)
    secs = "".join(model_section(tag, anc, lab, *data[tag]) for tag, anc, lab in MODELS)
    any_gen = any(g and g["tokens"] for _, g in data.values())
    all_done = all(g and g["done"] for _, g in data.values())
    status = ("세 모델의 생성 실행이 모두 끝났습니다." if all_done else
              "생성 실행 일부가 아직 진행 중이며, 표와 그림은 그 시점까지의 기록입니다.")
    page = f"""<title>KV 페이지 할당 실측</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">
<style>{CSS}</style>
<main class="doc">
  <header>
    <p class="eyebrow">PIM emulator · ch2 이미지 · 2026-09-28 밤 측정</p>
    <h1>KV 페이지 할당 실측</h1>
    <p class="lede">KV 캐시를 토큰이 들어오는 만큼 늘리도록 바꾼 뒤, 모델마다 몇 번째 토큰에서
    K와 V 페이지가 할당되는지, 얼마나 자주 할당되는지, 어디서 카드 메모리가 바닥나는지를 보드에서 측정했습니다.
    {status}</p>
    <dl class="facts">
      <div><dt>보드</dt><dd>2채널 × 16뱅크, DRAM 8 GiB</dd></div>
      <div><dt>할당 단위</dt><dd>64 KiB (행 2 KiB × 32뱅크)</dd></div>
      <div><dt>주소 매핑</dt><dd>RoChBaCo</dd></div>
      <div><dt>보드 타이밍</dt><dd>emu_timing ×7 설정 상태</dd></div>
    </dl>
  </header>

  <section id="summary">
    <h2>요약</h2>
    <p>세 모델 모두 grow 간격이 처음부터 한계까지 한 번도 어긋나지 않았습니다. K는 토큰이 뱅크 행 하나를 채울 때마다
    (1B 64토큰, 3B와 Qwen 32토큰), V는 1024토큰마다 모든 레이어에서 동시에 할당됩니다.
    할당 전용 실행이 멈춘 토큰은 세 모델 모두 메모리 추정값 바로 다음 토큰이었습니다.</p>
    {t1}
    {t2}
    <p class="muted">"K 1회 할당"은 레이어 하나의 페이지 / 한 번의 grow에서 전 레이어가 받는 합계입니다.
    K와 V는 1024토큰당 같은 양을 쓰지만, K는 작은 페이지를 자주, V는 큰 페이지를 드물게 받습니다.</p>
  </section>
  {secs}
  <section id="method">
    <h2>측정 방법</h2>
    <ul class="plain">
      <li><strong>할당 전용 실행.</strong> 모델 가중치를 카드에 올린 상태에서, KV 캐시가 실제로 쓰는 경로
      (<code>PimLayer._grow</code> → <code>pim_tensor_grow</code> → <code>pim_alloc</code> → pim.ko)로
      토큰을 하나씩 늘려 카드가 페이지를 거부할 때까지 grow했습니다. K/V 데이터는 쓰지 않았습니다.
      모델 문맥 한계를 넘어서도 계속해, 메모리 추정이 맞는지 확인했습니다.</li>
      <li><strong>실제 생성.</strong> PIM 백엔드로 긴 글을 요청하고 EOS에서 멈추지 않게 한 뒤, 생성 토큰마다 시간,
      그 토큰에 든 ISA 수, 그 토큰에서 일어난 grow를 기록했습니다. 모델마다 시간 제한(3.5~4시간) 안에서 간 데까지입니다.
      문맥이 길어질수록 q·K 비용이 선형으로 늘어서 최대 토큰까지는 가지 못합니다.</li>
      <li><strong>보드 상태.</strong> pim.ko는 2채널 RoChBaCo 파라미터로 로드된 상태였고, DRAM 타이밍은 다른 작업이
      걸어 둔 emu_timing ×7 설정 그대로 두었습니다. 토큰당 시간은 이 설정 기준입니다.</li>
    </ul>
  </section>

  <section id="changes">
    <h2>구현 변경</h2>
    <ul class="plain">
      <li><code>pim_tensor</code>: growable 텐서 추가. 시작할 때 room이 0이고, <code>pim_tensor_grow</code>가 필요한 unit을
      <code>pim_alloc</code>으로 받아 텐서의 unit 테이블에 붙입니다. 이미 쓴 unit은 움직이지 않습니다.</li>
      <li>V(<code>RED_MAJOR</code>)의 unit 순서를 chunk 우선으로 변경. 토큰이 늘면 새 chunk의 unit이 뒤에 붙습니다.</li>
      <li>MAC 명령은 unit 주소를 테이블에서 받습니다. 여러 할당에 흩어진 unit도 같은 명령 수로 처리합니다.</li>
      <li>런타임의 호스트 스테이징 버퍼(<code>vbuf</code>)는 더 긴 벡터가 오면 늘어납니다. 모델의 <code>s_max</code>와
      <code>--s-max</code> 옵션은 없앴습니다.</li>
      <li><code>budget</code>: 최대 토큰 = min(카드 메모리로 가능한 토큰, 모델 문맥)을 시작할 때 보여 줍니다.
      생성 중 페이지를 못 받으면 그 토큰에서 멈추고 알립니다.</li>
      <li>검증: 오프라인 테스트 5종, 보드 테스트(gemv, tlatch, tensor_board, op_board), growable과 고정 크기의 q·K, s·V
      결과 비트 일치, Llama-3.2-1B와 Qwen3-0.6B의 PIM 대 torch logit 비교(argmax 동일, 상관 0.9999, 0.9996).
      변경은 아직 커밋하지 않았습니다.</li>
    </ul>
  </section>
</main>
<script>{JS}</script>
"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page) / 1024:.0f} KiB), gen data: {any_gen}, all done: {all_done}")


if __name__ == "__main__":
    main()
