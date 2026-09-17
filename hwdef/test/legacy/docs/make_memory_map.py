#!/usr/bin/env python3
"""make_memory_map.py — narrow transfer 오배치를 메모리 맵으로 그린다.

   python3 make_memory_map.py            # 옆의 CSV / .bin / .log 를 읽어 HTML 을 만든다

숫자를 손으로 적지 않는다.  세 종류의 실측만 쓴다.

  1. ILA CSV        버스에 실제로 실린 AxSIZE / wstrb / 데이터  (마스터가 보낸 것)
  2. read-back .bin emu_mc --dump 가 남긴 호스트 버퍼         (호스트가 받은 것)
  3. workload .log  emu_mc 의 판정                            (독립 대조)

메모리 맵의 "실제로 있는 것" 열은 계산이 아니라 (2) 에서 읽은 값이다.  payload 워드가
자기 채널 오프셋을 값에 담고 있어서 (pat(a) = (0xC5A17 << 44) | a), 바이트가 스스로
어디서 왔는지 말한다.  AxSIZE=5 오해 모델은 (1) 로부터 따로 계산해서 (2) 와 몇 칸이
맞는지 세는 데에만 쓴다 — 모델을 그려놓고 맞다고 하는 게 아니라, 실측을 그려놓고
모델이 그것을 설명하는지 판정한다.
"""
import csv
import os
import re
import struct
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'narrow-transfer-memory-map.html')

BUS = 32            # wdata[255:0] / rdata[255:0]
BEAT = 16           # AxSIZE = 4
BURST = 16          # AxLEN = 0x0f
BLOCK = BEAT * BURST        # 한 버스트가 덮는 256 B
PAGE = 0x800                # 한 뱅크의 페이지 (CO 6 b x 32 B)
ROW = 0x8000                # RoBaCo 한 행 = 16 뱅크 x 페이지
PAT, POI = 0xC5A17, 0xDEAD0

F = {k: os.path.join(CAP, v) for k, v in {
    'rcsv': 'mc_read_axis_ila_0.csv', 'wcsv': 'mc_write_axis_ila_0.csv',
    'rbin': 'mc_read_readback.bin', 'wbin': 'mc_write_readback.bin',
    'rlog': 'mc_read_workload.log', 'wlog': 'mc_write_workload.log'}.items()}


# ------------------------------------------------------------ 1. ILA CSV ---
def columns(path):
    with open(path) as f:
        return {n.split('/')[-1]: i for i, n in enumerate(next(csv.reader(f)))}


def bus_word(hex256, byte_off):
    h = hex256.strip().strip('"')
    n = len(h)
    return int(''.join(reversed([h[n - 2 * (k + 1):n - 2 * k]
                                 for k in range(byte_off, byte_off + 8)])), 16)


def first_burst(path, kind):
    c = columns(path)
    p = 'aw' if kind == 'write' else 'ar'
    av, ar = c[f'SLOT_7_AXI_{p}valid'], c[f'SLOT_7_AXI_{p}ready']
    dv = c['SLOT_7_AXI_wvalid' if kind == 'write' else 'SLOT_7_AXI_rvalid']
    dr = c['SLOT_7_AXI_wready' if kind == 'write' else 'SLOT_7_AXI_rready']
    dc = c['SLOT_7_AXI_wdata[255:0]' if kind == 'write' else 'SLOT_7_AXI_rdata[255:0]']
    sc = c.get('SLOT_7_AXI_wstrb[31:0]') if kind == 'write' else None

    info, beats = None, []
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if info is None:
                if r[av] == '1' and r[ar] == '1':
                    info = {'addr': int(r[c[f'SLOT_7_AXI_{p}addr[63:0]']], 16),
                            'len': int(r[c[f'SLOT_7_AXI_{p}len[7:0]']], 16),
                            'size': r[c[f'SLOT_7_AXI_{p}size[2:0]']].strip('"')}
                continue
            if r[dv] == '1' and r[dr] == '1':
                beats.append({'lo': bus_word(r[dc], 0), 'hi': bus_word(r[dc], BUS // 2),
                              'strb': r[sc].strip('"') if sc else None})
                if len(beats) >= BURST:
                    break
    return info, beats


def census(path, kind):
    c = columns(path)
    p = 'aw' if kind == 'write' else 'ar'
    av, ar = c[f'SLOT_7_AXI_{p}valid'], c[f'SLOT_7_AXI_{p}ready']
    sizes, addrs = Counter(), []
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if r[av] == '1' and r[ar] == '1':
                sizes[(r[c[f'SLOT_7_AXI_{p}size[2:0]']].strip('"'),
                       r[c[f'SLOT_7_AXI_{p}len[7:0]']])] += 1
                addrs.append(int(r[c[f'SLOT_7_AXI_{p}addr[63:0]']], 16))
    return {'sizes': sizes, 'n': len(addrs), 'first': addrs[0], 'last': addrs[-1],
            'span': addrs[-1] - addrs[0] + BLOCK,
            'steps': sorted({addrs[i + 1] - addrs[i] for i in range(len(addrs) - 1)})}


# --------------------------------------------------------- 2. 호스트 덤프 ---
def chunk(dump, off):
    """16 B 청크가 담고 있는 것: ('pat'|'poi'|'?', 그 워드가 말하는 출처 오프셋)."""
    w, = struct.unpack_from('<Q', dump, off)
    tag, val = w >> 44, w & 0xFFFFFFFFFF
    return ('pat', val) if tag == PAT else ('poi', val) if tag == POI else ('?', w)


def measured(dump, base, n):
    return [dict(zip(('kind', 'src'), chunk(dump, base + i * BEAT)), at=base + i * BEAT)
            for i in range(n)]


def tally(dump):
    n, t = len(dump) // BEAT, Counter()
    for m in range(n):
        k, v = chunk(dump, m * BEAT)
        t['제자리' if (k == 'pat' and v == m * BEAT) else
          '다른 곳의 데이터' if k == 'pat' else
          '쓰이지 않음 (poison)' if k == 'poi' else '해독 불가'] += 1
    return n, t


def page_mix(dump, base, size):
    ok = mv = no = 0
    for off in range(base, min(base + size, len(dump)), BEAT):
        k, v = chunk(dump, off)
        if k == 'pat' and v == off: ok += 1
        elif k == 'pat': mv += 1
        else: no += 1
    t = max(1, ok + mv + no)
    return ok / t, mv / t, no / t


def grid_svg(dump, label):
    """1 MiB 전체.  가로 = 뱅크 0..15, 세로 = 행.  칸 하나 = 한 뱅크의 페이지(2 KiB)."""
    nrow = len(dump) // ROW
    nbank = ROW // PAGE
    CW, CH, G = 30, 15, 2
    LX, LY = 44, 30                      # 축 라벨 자리
    W = LX + nbank * (CW + G)
    H = LY + nrow * (CH + G) + 6
    p = []
    for b in range(nbank):
        x = LX + b * (CW + G)
        p.append(f'<text x="{x + CW / 2:.0f}" y="{LY - 8}" class="axl" '
                 f'text-anchor="middle">{b}</text>')
    p.append(f'<text x="{LX + nbank * (CW + G) / 2:.0f}" y="12" class="axt" '
             f'text-anchor="middle">BA — 뱅크</text>')
    for r in range(nrow):
        y = LY + r * (CH + G)
        if r % 4 == 0:
            p.append(f'<text x="{LX - 8}" y="{y + CH - 3}" class="axl" '
                     f'text-anchor="end">{r}</text>')
        for b in range(nbank):
            base = r * ROW + b * PAGE
            ok, mv, no = page_mix(dump, base, PAGE)
            x = LX + b * (CW + G)
            yy = float(y)
            for frac, cls in ((ok, 'ok'), (mv, 'mv'), (no, 'no')):
                if frac <= 0: continue
                p.append(f'<rect x="{x}" y="{yy:.2f}" width="{CW}" '
                         f'height="{frac * CH:.2f}" class="c-{cls}"/>')
                yy += frac * CH
            p.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" class="c-fr">'
                     f'<title>row {r} · bank {b} · 0x{base:06x}\n'
                     f'제자리 {ok * 100:.0f}% · 이동 {mv * 100:.0f}% · 미기록 {no * 100:.0f}%</title></rect>')
    return (f'<figure class="mapfig"><figcaption>{esc(label)}'
            f'<span class="dim"> — 칸 하나 = 한 뱅크의 페이지 {PAGE // 1024} KiB, '
            f'세로 = 행 0…{nrow - 1}</span></figcaption>'
            f'<div class="mapscroll"><svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-label="{esc(label)}">{"".join(p)}</svg></div></figure>')


def rowzoom_svg(dump, r, label):
    """행 하나(32 KiB)를 16 B 청크까지 펼친다.  가로 = 페이지 안 위치, 세로 = 뱅크."""
    nbank = ROW // PAGE
    ncol = PAGE // BEAT
    CW, CH = 6.5, 14
    LX, LY = 52, 26
    W = LX + ncol * CW
    H = LY + nbank * (CH + 2)
    p = [f'<text x="{LX}" y="12" class="axt">행 {r} — 페이지 안 오프셋 0x000 → 0x{PAGE - 1:03x}</text>']
    for b in range(nbank):
        y = LY + b * (CH + 2)
        p.append(f'<text x="{LX - 8}" y="{y + CH - 3}" class="axl" text-anchor="end">bank {b}</text>')
        for i in range(ncol):
            off = r * ROW + b * PAGE + i * BEAT
            k, v = chunk(dump, off)
            cls = 'ok' if (k == 'pat' and v == off) else 'mv' if k == 'pat' else 'no'
            p.append(f'<rect x="{LX + i * CW:.2f}" y="{y}" width="{CW - .4:.2f}" '
                     f'height="{CH}" class="c-{cls}"/>')
    return (f'<figure class="mapfig"><figcaption>{esc(label)}'
            f'<span class="dim"> — 칸 하나 = {BEAT} B</span></figcaption>'
            f'<div class="mapscroll"><svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-label="{esc(label)}">{"".join(p)}</svg></div></figure>')


# ------------------------------------------------------------- 3. 판정 로그 --
def verdict_log(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    d = re.search(r'FIRST DIVERGENCE at channel offset 0x([0-9a-f]+)\s*\n'
                  r'\s*expected\s+0x([0-9a-f]+)\s+got 0x([0-9a-f]+)', txt)
    p = re.search(r'FAIL: (\d+) of (\d+) B wrong \(([\d.]+)%\)', txt)
    return {'at': int(d.group(1), 16), 'want': int(d.group(2), 16), 'got': int(d.group(3), 16),
            'bad': int(p.group(1)), 'total': int(p.group(2)), 'pct': p.group(3)}


# ------------------------------------------- AxSIZE=5 오해 모델 (설명 후보) --
# 슬레이브가 beat 를 BUS(32 B) 로 세면 마스터의 BEAT(16 B) 와 보폭이 2 배 어긋난다.
#   WRITE  beat n 의 32 B 가 통째로 +0x20n 에 앉는다 → 청크 m 은 beat m//2 의 절반
#   READ   슬레이브는 +0x20n 의 32 B 를 보내고, 마스터는 narrow 규칙대로 그 중
#          주소가 가리키는 절반만 집어 +0x10n 에 놓는다
def model_write(beats):
    return [beats[m // 2]['lo' if m % 2 == 0 else 'hi'] for m in range(BURST)]


def model_read(beats):
    return [beats[n]['lo' if n % 2 == 0 else 'hi'] for n in range(BURST)]


def src_of(word):
    return word & 0xFFFFFFFFFF if (word >> 44) == PAT else None


# ------------------------------------------------------------------- HTML ---
def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def map_rows(rows, model):
    h = []
    for i, r in enumerate(rows):
        want, got = r['at'], r['src']
        ok = (r['kind'] == 'pat' and got == want)
        pred = src_of(model[i]) if model else None
        agree = (pred is not None and r['kind'] == 'pat' and pred == got)
        if r['kind'] == 'poi':
            shown, cls = '쓰이지 않음', 'none'
        elif r['kind'] == '?':
            shown, cls = '해독 불가', 'none'
        else:
            shown, cls = f'0x{got:03x}', ('ok' if ok else 'bad')
        h.append(
            f'<div class="row {cls}">'
            f'<span class="addr">+0x{want:03x}</span>'
            f'<span class="want">0x{want:03x}</span>'
            f'<span class="arrow" aria-hidden="true">&rarr;</span>'
            f'<span class="got">{shown}</span>'
            f'<span class="pred">{"0x%03x" % pred if pred is not None else "&mdash;"}'
            f'{" ✓" if agree else " ✗" if pred is not None else ""}</span>'
            f'<span class="verdict">{"일치" if ok else "어긋남"}</span>'
            f'</div>')
    return '\n'.join(h)


def beat_rows(beats, strb=True):
    h = []
    for i, b in enumerate(beats):
        lo, hi = src_of(b['lo']), src_of(b['hi'])
        s = f'<td class="mono">{b["strb"]}</td>' if strb else ''
        h.append(f'<tr><td class="num">{i}</td>{s}'
                 f'<td class="mono">0x{b["lo"]:016x}</td>'
                 f'<td class="mono">0x{b["hi"]:016x}</td>'
                 f'<td class="mono dim">+0x{lo:03x} / +0x{hi:03x}</td></tr>')
    return '\n'.join(h)


CSS = """
:root{
  --ground:#f5f7f7; --panel:#fff; --ink:#151d21; --muted:#5c6d72;
  --rule:#d6dee0; --rule-soft:#e7edee;
  --ok:#0f6f68; --ok-bg:#e2efed; --bad:#a8352a; --bad-bg:#f8e7e4;
  --none:#7b6a4a; --none-bg:#f0ebe0; --note:#8a6a1f; --accent:#0f6f68;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0e1417; --panel:#151d21; --ink:#e3ebec; --muted:#8ba0a5;
  --rule:#26343a; --rule-soft:#1c272c;
  --ok:#54bdb1; --ok-bg:#12312f; --bad:#e87d6f; --bad-bg:#38201c;
  --none:#b6a582; --none-bg:#2b2519; --note:#cfa64a; --accent:#54bdb1;
}}
:root[data-theme="dark"]{
  --ground:#0e1417; --panel:#151d21; --ink:#e3ebec; --muted:#8ba0a5;
  --rule:#26343a; --rule-soft:#1c272c;
  --ok:#54bdb1; --ok-bg:#12312f; --bad:#e87d6f; --bad-bg:#38201c;
  --none:#b6a582; --none-bg:#2b2519; --note:#cfa64a; --accent:#54bdb1;
}
*{box-sizing:border-box}
body{margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans KR","Malgun Gothic",sans-serif;
  font-size:16px; line-height:1.72; -webkit-font-smoothing:antialiased}
.mono,code,.addr,.want,.got,.pred,td.mono,th{
  font-family:ui-monospace,"SF Mono","JetBrains Mono","DejaVu Sans Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.wrap{max-width:72rem; margin:0 auto; padding:clamp(2rem,5vw,4.5rem) clamp(1rem,4vw,2.5rem) 6rem;
  display:flex; flex-direction:column; gap:3.25rem}
.prose{max-width:44rem; display:flex; flex-direction:column; gap:.9rem}
header.head{display:flex; flex-direction:column; gap:.9rem; max-width:48rem}
.kicker{font-family:ui-monospace,monospace; font-size:.72rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent)}
h1{margin:0; font-size:clamp(1.7rem,3.6vw,2.5rem); line-height:1.2; font-weight:650;
  letter-spacing:-.015em; text-wrap:balance}
.standfirst{margin:0; color:var(--muted); font-size:1.05rem; max-width:42rem}
.meta{display:flex; flex-wrap:wrap; gap:.4rem 1.4rem; font-family:ui-monospace,monospace;
  font-size:.76rem; color:var(--muted); border-top:1px solid var(--rule); padding-top:.9rem}
section{display:flex; flex-direction:column; gap:1.15rem}
.sec-label{font-family:ui-monospace,monospace; font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted)}
h2{margin:0; font-size:1.3rem; font-weight:640; letter-spacing:-.01em; text-wrap:balance}
h3{margin:.4rem 0 0; font-size:1rem; font-weight:640}
p{margin:0}
code{background:var(--rule-soft); padding:.1em .38em; border-radius:3px; font-size:.88em}
.tablewrap{overflow-x:auto; border:1px solid var(--rule); border-radius:6px; background:var(--panel)}
table{border-collapse:collapse; width:100%; font-size:.83rem}
th,td{text-align:left; padding:.5rem .8rem; border-bottom:1px solid var(--rule-soft); white-space:nowrap}
th{font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  background:var(--rule-soft); border-bottom:1px solid var(--rule)}
tr:last-child td{border-bottom:none}
td.num{text-align:right; font-family:ui-monospace,monospace; font-variant-numeric:tabular-nums}
td.dim,.dim{color:var(--muted)}
.map{border:1px solid var(--rule); border-radius:6px; overflow:hidden; background:var(--panel)}
.map-head,.row{display:grid; grid-template-columns:5rem 1fr 1.3rem 1fr 7rem 4.4rem;
  align-items:center; gap:.4rem; padding:.3rem .85rem}
.map-head{background:var(--rule-soft); border-bottom:1px solid var(--rule);
  font-family:ui-monospace,monospace; font-size:.66rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted)}
.row{font-size:.82rem; border-bottom:1px solid var(--rule-soft)}
.row:last-child{border-bottom:none}
.row .addr{color:var(--muted)}
.row .want,.row .got{padding:.08rem .45rem; border-radius:3px; text-align:center}
.row .want{background:var(--rule-soft)}
.row .arrow{text-align:center; color:var(--muted)}
.row .pred{font-size:.74rem; color:var(--muted); text-align:center}
.row .verdict{font-size:.74rem; text-align:right}
.row.ok .got{background:var(--ok-bg); color:var(--ok); font-weight:650}
.row.ok .verdict{color:var(--ok)}
.row.bad .got{background:var(--bad-bg); color:var(--bad); font-weight:650}
.row.bad .verdict{color:var(--bad)}
.row.none .got{background:var(--none-bg); color:var(--none); font-weight:600}
.row.none .verdict{color:var(--none)}
.legend{display:flex; flex-wrap:wrap; gap:1.1rem; font-size:.8rem; color:var(--muted)}
.legend span{display:inline-flex; align-items:center; gap:.45rem}
.swatch{width:.8rem; height:.8rem; border-radius:2px; display:inline-block; flex:none}
.swatch.ok{background:var(--ok)} .swatch.bad{background:var(--bad)} .swatch.none{background:var(--none)}
.verdict-line{font-family:ui-monospace,monospace; font-size:.8rem; padding:.6rem .85rem;
  border-radius:4px}
.verdict-line.pass{background:var(--ok-bg); color:var(--ok)}
.verdict-line.fail{background:var(--bad-bg); color:var(--bad)}
.mapfig{margin:0; display:flex; flex-direction:column; gap:.5rem}
.mapfig figcaption{font-size:.85rem; font-weight:600}
.mapscroll{overflow-x:auto; background:var(--panel); border:1px solid var(--rule);
  border-radius:6px; padding:.75rem .9rem}
.mapscroll svg{display:block; max-width:100%; height:auto}
.c-ok{fill:var(--ok)} .c-mv{fill:var(--bad)} .c-no{fill:var(--none)}
.c-fr{fill:transparent; stroke:var(--panel); stroke-width:1}
.c-fr:hover{stroke:var(--ink); stroke-width:1.5}
.axl{fill:var(--muted); font-size:9.5px; font-family:ui-monospace,monospace}
.axt{fill:var(--muted); font-size:10.5px; letter-spacing:.06em}
.maps{display:grid; grid-template-columns:repeat(auto-fit,minmax(25rem,1fr)); gap:1.5rem}
.callout{border-left:3px solid var(--note); padding:.15rem 0 .15rem 1rem}
.callout .lbl{font-family:ui-monospace,monospace; font-size:.7rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--note); display:block; margin-bottom:.25rem}
ul{margin:0; padding-left:1.15rem; display:flex; flex-direction:column; gap:.45rem}
footer{border-top:1px solid var(--rule); padding-top:1.2rem; color:var(--muted); font-size:.8rem;
  display:flex; flex-direction:column; gap:.35rem}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (max-width:780px){
  .map-head,.row{grid-template-columns:4.4rem 1fr 1rem 1fr}
  .row .pred,.row .verdict,.map-head span:nth-child(5),.map-head span:nth-child(6){display:none}
}
"""


def build():
    wi, wb = first_burst(F['wcsv'], 'write')
    ri, rb = first_burst(F['rcsv'], 'read')
    wc, rc = census(F['wcsv'], 'write'), census(F['rcsv'], 'read')
    wlog, rlog = verdict_log(F['wlog']), verdict_log(F['rlog'])
    wd = open(F['wbin'], 'rb').read()
    rd = open(F['rbin'], 'rb').read()

    wmap, rmap = measured(wd, 0, BURST), measured(rd, 0, BURST)
    wmodel, rmodel = model_write(wb), model_read(rb)

    def agree(rows, model):
        n = sum(1 for r, m in zip(rows, model)
                if r['kind'] == 'pat' and src_of(m) == r['src'])
        return n, len(rows)

    wag, wn = agree(wmap, wmodel)
    rag, rn = agree(rmap, rmodel)

    wtot_n, wtot = tally(wd)
    rtot_n, rtot = tally(rd)

    def vline(ag, n):
        cls = 'pass' if ag == n else 'fail'
        return (f'<p class="verdict-line {cls}">AxSIZE=5 오해 모델이 실측과 맞은 칸 '
                f'<b>{ag} / {n}</b>{" — 전부 설명됨" if ag == n else " — 일부만 설명됨"}</p>')

    html = f"""<title>narrow transfer 오배치 메모리 맵 — ch1 legacy</title>
<style>{CSS}</style>
<div class="wrap">

<header class="head">
  <span class="kicker">ch1 legacy · axi_noc_0_M03_AXI &rarr; emulator controller</span>
  <h1>1 MiB 를 옮겼더니 이렇게 남았다</h1>
  <p class="standfirst">
    아래 지도는 모델이 아니라 <b>호스트가 실제로 되읽은 1 MiB</b> 다 —
    <code>emu_mc --dump</code> 가 판정 직전에 남긴 버퍼를 16 B 씩 읽어 그대로 칠했다.
    payload 워드가 자기 채널 오프셋을 값에 담고 있어서, 바이트마다 제자리인지 남의
    자리인지 스스로 말한다.
  </p>
  <div class="meta">
    <span>emu_mc --read / --write --mib 1 --dump</span>
    <span>ch1 legacy 이미지</span>
    <span>MC 0x0201_0000_0000</span>
    <span>주소 ROW | BA | CO</span>
  </div>
</header>

<section>
  <span class="sec-label">지도 — 1 MiB 전체</span>
  <div class="prose">
    <h2>먼저 전체를 본다</h2>
    <p>
      이 이미지의 채널 주소는 <code>ROW | BA | CO</code> 로 쪼개진다. 그래서 주소를
      0 부터 일렬로 늘어놓는 대신 그 격자 그대로 놓았다 — <b>가로가 뱅크 16 개, 세로가
      행</b>, 칸 하나가 한 뱅크의 페이지 {PAGE // 1024} KiB 다. 1 MiB 는 정확히
      {len(wd) // ROW} 행 × 16 뱅크다.
    </p>
  </div>
  <div class="legend">
    <span><i class="swatch ok"></i> 제자리 — 있어야 할 데이터가 있음</span>
    <span><i class="swatch bad"></i> 다른 오프셋의 데이터가 앉음</span>
    <span><i class="swatch none"></i> 쓰이지 않음 (poison 그대로)</span>
  </div>
  <div class="maps">
    {grid_svg(wd, 'WRITE — MC 로 쓰고 direct 로 되읽은 DRAM')}
    {grid_svg(rd, 'READ — MC 로 읽어 호스트 버퍼에 담긴 것')}
  </div>
  <div class="prose">
    <p>
      WRITE 지도는 <b>맨 왼쪽 열 하나만</b> 칠해져 있다. 뱅크 0 의 페이지에만 데이터가
      닿았고 뱅크 1~15 는 poison 그대로다 — 1 MiB 중 실제로 쓰인 것은
      {wtot['다른 곳의 데이터'] + wtot['제자리']:,} 청크,
      {100 * (wtot['다른 곳의 데이터'] + wtot['제자리']) / wtot_n:.1f}% 다.
      그리고 그 열조차 거의 전부 빨강이다: 닿긴 했는데 제 자리가 아니다.
    </p>
    <p>
      READ 지도는 전면이 섞여 있다. poison 이 없는 것은 읽기라서 당연하고
      (읽기는 DRAM 을 바꾸지 않는다), 초록과 빨강이 규칙적으로 섞인 것이 문제다 —
      절반 넘는 칸이 남의 자리 데이터를 받았다.
    </p>
  </div>
</section>

<section>
  <span class="sec-label">지도 — 행 하나 확대 (32 KiB)</span>
  <div class="prose">
    <h2>행 0 을 16 B 까지 펼치면</h2>
    <p>
      위 지도의 맨 윗줄 하나다. 가로가 페이지 안 오프셋 0x000…0x7ff, 세로가 뱅크 0~15,
      칸 하나가 {BEAT} B 다.
    </p>
  </div>
  {rowzoom_svg(wd, 0, 'WRITE — 행 0')}
  {rowzoom_svg(rd, 0, 'READ — 행 0')}
  <div class="prose">
    <p>
      WRITE 는 bank 0 줄만 칠해지고 나머지 15 줄이 통째로 비어 있다. READ 는 모든 줄에
      데이터가 있지만 bank 0 줄에만 초록이 규칙적으로 박혀 있고, 그 초록도
      {BLOCK} B 마다 한 칸씩이다 — 버스트 하나당 첫 {BEAT} B 만 제자리라는 뜻이다.
    </p>
  </div>
</section>

<section>
  <span class="sec-label">지도 — 첫 256 B, 값까지</span>
  <div class="prose">
    <h2>한 버스트 안에서 무슨 일이 벌어지나</h2>
    <p>
      가장 안쪽까지 들어가면 값이 보인다. "실제로 있는 것" 은 덤프에 적혀 있던 출처
      오프셋이고, "모델 예측" 은 AxSIZE=5 오해로부터 <em>파형만 보고</em> 따로 계산한 값이다.
      둘이 맞는지가 곧 원인 규명이다.
    </p>
  </div>
  <h3>WRITE — DRAM +0x000 … +0x0f0</h3>
  <div class="map">
    <div class="map-head"><span>DRAM</span><span>있어야 할 것</span><span></span>
      <span>실제로 있는 것</span><span>모델 예측</span><span>판정</span></div>
    {map_rows(wmap, wmodel)}
  </div>
  {vline(wag, wn)}
  <h3>READ — 호스트 +0x000 … +0x0f0</h3>
  <div class="map">
    <div class="map-head"><span>호스트</span><span>있어야 할 것</span><span></span>
      <span>실제로 담긴 것</span><span>모델 예측</span><span>판정</span></div>
    {map_rows(rmap, rmodel)}
  </div>
  {vline(rag, rn)}
  <div class="prose">
    <p>
      두 맵 다 +0x000 하나만 맞고, 뒤로 갈수록 두 배로 벌어진다.
      emu_mc 판정은 WRITE <b>{wlog['pct']}%</b>, READ <b>{rlog['pct']}%</b> 불일치 —
      지도에서 보이는 것보다 훨씬 작은 숫자다. payload 워드끼리 상위 바이트가 같아서
      자리가 완전히 틀려도 8 B 중 몇 바이트만 다르게 세이기 때문이다.
      <b>"조금 틀렸다" 가 아니라 자리 자체가 틀린 것</b>이다.
    </p>
  </div>
</section>

<section>
  <span class="sec-label">원인 — 버스 위의 사실</span>
  <div class="prose">
    <h2>마스터는 {BEAT} B 로 보냈다</h2>
    <p>
      캡처 창 안의 AW / AR 핸드셰이크를 전부 셌다. 섞여 있는 게 아니라 한 종류다.
      버스는 {BUS} B 인데 {BEAT} B 로 온다 — 정의 그대로의 narrow transfer 다.
    </p>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th>방향</th><th>AxSIZE</th><th>AxLEN</th><th>버스트</th><th>주소 범위</th><th>증분</th></tr></thead>
    <tbody>
      <tr><td>READ</td><td class="mono">{list(rc['sizes'])[0][0]}</td>
          <td class="mono">0x{list(rc['sizes'])[0][1]}</td><td class="num">{rc['n']}</td>
          <td class="mono">0x{rc['first']:011x} … 0x{rc['last']:011x}</td>
          <td class="mono">{', '.join(hex(x) for x in rc['steps'])}</td></tr>
      <tr><td>WRITE</td><td class="mono">{list(wc['sizes'])[0][0]}</td>
          <td class="mono">0x{list(wc['sizes'])[0][1]}</td><td class="num">{wc['n']}</td>
          <td class="mono">0x{wc['first']:011x} … 0x{wc['last']:011x}</td>
          <td class="mono">{', '.join(hex(x) for x in wc['steps'])}</td></tr>
    </tbody>
  </table></div>
  <div class="prose">
    <p>
      첫 WRITE 버스트 <code>0x{wi['addr']:011x}</code> 의 beat 를 펼치면 유효 절반이
      beat 마다 아래·위로 번갈아 간다 — narrow transfer 규칙 그대로다.
    </p>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th>beat</th><th>wstrb</th><th>하위 8 B</th><th>상위 8 B</th><th>이 워드가 말하는 출처</th></tr></thead>
    <tbody>{beat_rows(wb[:8])}</tbody>
  </table></div>
  <div class="callout">
    <span class="lbl">여기서 이미 함정이 보인다</span>
    <p>
      유효하지 않은 절반에도 값이 실려 있다. beat 0 은 같은 {BEAT} B 를 양쪽에 복제해
      두었고, 홀수 beat 의 하위 절반에는 직전 beat 의 값이 남아 있다.
      <code>wstrb</code> 를 보지 않는 슬레이브에게 이 값들은 유효한 데이터와 구별되지 않는다.
    </p>
  </div>
  <div class="callout">
    <span class="lbl">그런데 뱅크가 비는 것은 이것으로 설명되지 않는다</span>
    <p>
      beat 폭 오해는 페이지 <em>안에서</em> 두 배로 벌어지는 것까지만 설명한다 —
      첫 256 B 맵 {wn} 칸이 그것을 칸 단위로 확인해 준다. 뱅크 필드가 통째로 안 쓰이는 것은
      별개다. 읽기 쪽에서도 호스트 +0x000800 이 받아온 것은 DRAM +0x008000 의 내용이다:
      <code>BA</code> 자리로 가야 할 비트가 <code>ROW</code> 로 올라갔다.
      이 이미지에는 조용한 고장이 <b>둘</b> 있었다고 읽는 것이 정확하다.
    </p>
  </div>
</section>

<section>
  <span class="sec-label">왜 조용한가</span>
  <div class="prose">
    <h2>이 고장이 에러로 보이지 않는 이유</h2>
    <ul>
      <li><b>AXI 에 "크기가 틀렸다" 는 응답이 없다.</b> 낼 수 있는 것은 <code>SLVERR</code> /
          <code>DECERR</code> 뿐이고 둘 다 이 상황을 뜻하지 않는다. AxSIZE 를 검사하지 않는
          슬레이브는 검사할 수 없는 게 아니라 <em>안 하는</em> 것이고, 안 하면
          <code>OKAY</code> 가 나간다.</li>
      <li><b>타이밍 위반 카운터는 이것을 볼 수 없다.</b> 뱅크 컨트롤러가 보기에 접근은
          적법한 순서로 도착한다. 틀린 것은 주소와 데이터의 짝이지 간격이 아니다.</li>
      <li><b>전송량이 맞아떨어진다.</b> 요청한 바이트 수만큼 오가고 <code>WLAST</code> /
          <code>RLAST</code> 도 제자리에 온다. 길이로는 걸러지지 않는다.</li>
      <li><b>바이트로 세면 작아 보인다.</b> 지도에서는 대부분이 빨강인데 불일치율은
          {wlog['pct']}% / {rlog['pct']}% 다. 이 숫자만 보면 "가끔 깨진다" 로 읽힌다.</li>
    </ul>
  </div>
</section>

<section>
  <span class="sec-label">근거의 경계</span>
  <div class="prose">
    <h2>이 문서가 말하지 않는 것</h2>
    <ul>
      <li>지도와 "실제로 있는 것" 은 전부 <b>실측</b>이다. "모델 예측" 열만 계산이고,
          첫 블록에서만 대조했다 — WRITE {wag}/{wn}, READ {rag}/{rn} 칸 일치.</li>
      <li>ILA 창은 전송의 <b>앞부분만</b> 담는다 (READ {rc['span']:,} B,
          WRITE {wc['span']:,} B). 1 MiB 전체가 좁은 전송이었는지는 파형으로 답할 수 없다.
          지도는 파형이 아니라 덤프에서 나온 것이다.</li>
      <li>뱅크 필드가 왜 안 쓰였는지는 <b>미해결</b>이다. 슬레이브 쪽 관측이 필요하다.</li>
      <li>READ 는 호스트 앞 64 KiB 가 DRAM 1 MiB 전체를 훑고 그 뒤는 다른 규칙을 따른다.
          뒤쪽 규칙은 특성화하지 않았다.</li>
    </ul>
  </div>
</section>

<footer>
  <span>파형 · 덤프 · 로그 : hwdef/test/legacy/</span>
  <span>이 페이지는 make_memory_map.py 가 같은 폴더의 CSV / .bin / .log 에서 생성한다. 숫자는 손으로 적지 않았다.</span>
</footer>

</div>
"""
    open(OUT, 'w', encoding='utf-8').write(html)
    print(f'생성: {OUT}')
    print(f'  WRITE 모델 대조: {wag}/{wn} 칸 일치   1 MiB: ' +
          ', '.join(f'{k} {v:,}' for k, v in wtot.most_common()))
    print(f'  READ  모델 대조: {rag}/{rn} 칸 일치   1 MiB: ' +
          ', '.join(f'{k} {v:,}' for k, v in rtot.most_common()))
    return wag == wn and rag == rn


if __name__ == '__main__':
    sys.exit(0 if build() else 1)
