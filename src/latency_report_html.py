"""Build the HTML version of BUCKET_LATENCY_RESULTS.md from the raw trial records. Run from the repo root:

    python src/latency_report_html.py   # writes report/bucket-latency-report.html
"""
import json, math, html
from collections import Counter, defaultdict

recs = [json.loads(l) for l in open("logs/latency-20260903-075808.jsonl")]
ok = [r for r in recs if "error" not in r]
SIZES = [1024, 65536, 1_048_576, 4_700_000, 36_700_160, 134_217_728]
SN = {1024: "1 KB", 65536: "64 KB", 1_048_576: "1 MB", 4_700_000: "4.7 MB", 36_700_160: "35 MiB", 134_217_728: "128 MiB"}
FOCUS = 4_700_000

def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs: return None
    k = (len(xs) - 1) * p; lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
def base(r): return r["mode"] == "rename" and r["probe"] == "open" and not r["warm"] and r["direction"] == "A>B"
by_size = defaultdict(list)
for r in ok:
    if base(r): by_size[r["size"]].append(r)
bf = by_size[FOCUS]
ta50, tg50, tm50 = (pct([r[k] for r in bf], .5) for k in ("t_api", "t_api_get", "t_mount"))
cc = [r["t_mount"] - r["t_api"] for r in bf]

# ---------- chart 1: decomposition timeline at 4.7 MB ----------
def timeline():
    W, H, L, R = 720, 150, 24, 24
    x = lambda s: L + (W - L - R) * s / 31.0
    y = 70
    parts = []
    parts.append(f'<rect x="{x(0):.1f}" y="{y-9}" width="{x(ta50)-x(0):.1f}" height="18" rx="4" class="c-api"><title>close() → listed by bucket API: {ta50:.2f} s p50</title></rect>')
    parts.append(f'<rect x="{x(ta50)+2:.1f}" y="{y-9}" width="{x(tm50)-x(ta50)-2:.1f}" height="18" rx="4" class="c-mount"><title>listed by API → visible on the other mount: {tm50-ta50:.1f} s p50</title></rect>')
    for s in range(0, 31, 5):
        parts.append(f'<line x1="{x(s):.1f}" x2="{x(s):.1f}" y1="{y+14}" y2="{y+19}" class="axis"/><text x="{x(s):.1f}" y="{y+33}" class="tick" text-anchor="middle">{s} s</text>')
    parts.append(f'<line x1="{x(0):.1f}" x2="{x(31):.1f}" y1="{y+14}" y2="{y+14}" class="axis"/>')
    # annotations
    parts.append(f'<text x="{x(0):.1f}" y="{y-22}" class="lbl">write + close() returns · ≤ 0.0001 s</text>')
    parts.append(f'<text x="{x(ta50):.1f}" y="{y+52}" class="lbl c-api-t" text-anchor="start">↑ readable via bucket API · {ta50:.1f} s (verified bytes {tg50:.1f} s)</text>')
    parts.append(f'<text x="{x(tm50):.1f}" y="{y-22}" class="lbl c-mount-t" text-anchor="end">visible on reader’s mount · {tm50:.1f} s ↓</text>')
    parts.append(f'<text x="{(x(ta50)+x(tm50))/2:.1f}" y="{y+4}" class="inbar" text-anchor="middle">metadata-cache cost {pct(cc,.5):.1f} s</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Timeline of one 4.7 MB sync: API readable at {ta50:.1f} s, mount visible at {tm50:.1f} s">{"".join(parts)}</svg>'

# ---------- chart 2: size dependence, log x, two series, p50 with min–max whiskers ----------
def sizes_chart():
    W, H = 720, 300; L, R, T, B = 52, 20, 18, 46
    xs = [math.log10(s) for s in SIZES]
    x = lambda s: L + (W - L - R) * (math.log10(s) - xs[0]) / (xs[-1] - xs[0])
    y = lambda v: T + (H - T - B) * (1 - v / 32.0)
    p = []
    for v in range(0, 33, 8):
        p.append(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/><text x="{L-8}" y="{y(v)+4:.1f}" class="tick" text-anchor="end">{v} s</text>')
    for s in SIZES:
        p.append(f'<text x="{x(s):.1f}" y="{H-B+20}" class="tick" text-anchor="middle">{SN[s]}</text>')
    p.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-8}" class="tick" text-anchor="middle">payload size (log scale)</text>')
    for key, cls, name in (("t_mount", "mount", "visible on reader’s mount"), ("t_api", "api", "readable via bucket API")):
        pts = []
        for s in SIZES:
            rs = by_size[s]; v = [r[key] for r in rs]
            p50, lo, hi = pct(v, .5), min(v), max(v)
            pts.append((x(s), y(p50)))
            p.append(f'<line x1="{x(s):.1f}" x2="{x(s):.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" class="whisk c-{cls}-s"/>')
        p.append('<polyline points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" class="line c-{cls}-s"/>')
        for s, (a, b) in zip(SIZES, pts):
            rs = by_size[s]; v = [r[key] for r in rs]
            p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5" class="dot c-{cls}"><title>{name} · {SN[s]} · p50 {pct(v,.5):.2f} s (min {min(v):.2f}, max {max(v):.2f}, n={len(v)})</title></circle>')
        # direct label at the right end
        a, b = pts[-1]
        dy = -10 if key == "t_mount" else 18
        p.append(f'<text x="{a-8:.1f}" y="{b+dy:.1f}" class="lbl c-{cls}-t" text-anchor="end">{name}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Latency by payload size: mount flat near 30 s, API 2.3 to 3.6 s">{"".join(p)}</svg>'

# ---------- chart 3: strip plot of secondary variables ----------
VARIANTS = [
    ("baseline", base),
    ("publish = direct", lambda r: r["mode"] == "direct"),
    ("probe = listdir parent", lambda r: r["probe"] == "listdir"),
    ("cache warming on", lambda r: r["warm"]),
    ("direction B→A", lambda r: r["direction"] == "B>A"),
]
focus = [r for r in ok if r["size"] == FOCUS]
def strip_chart():
    W = 720; rowh = 34; L, R, T, B = 210, 20, 14, 40; H = T + rowh * len(VARIANTS) + B
    x = lambda v: L + (W - L - R) * v / 32.0
    p = []
    for v in range(0, 33, 5):
        p.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{T}" y2="{H-B+4}" class="grid"/><text x="{x(v):.1f}" y="{H-B+20}" class="tick" text-anchor="middle">{v} s</text>')
    p.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-6}" class="tick" text-anchor="middle">t_mount, seconds after the writer’s close() returned · 4.7 MB · 10 trials per row</text>')
    for i, (name, pred) in enumerate(VARIANTS):
        rs = [r for r in focus if pred(r)]; cy = T + rowh * i + rowh / 2
        p.append(f'<text x="{L-12}" y="{cy+4:.1f}" class="lbl" text-anchor="end">{html.escape(name)}</text>')
        med = pct([r["t_mount"] for r in rs], .5)
        p.append(f'<line x1="{x(med):.1f}" x2="{x(med):.1f}" y1="{cy-13:.1f}" y2="{cy+13:.1f}" class="median c-mount-s"><title>p50 {med:.1f} s</title></line>')
        for r in sorted(rs, key=lambda r: r["t_mount"]):
            p.append(f'<circle cx="{x(r["t_mount"]):.1f}" cy="{cy:.1f}" r="5" class="dot-open c-mount-s"><title>trial {r["trial"]} · t_mount {r["t_mount"]:.1f} s · t_api {r["t_api"]:.2f} s</title></circle>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Strip plot of mount visibility time per secondary variable">{"".join(p)}</svg>'

# ---------- chart 4: histogram ----------
def histogram():
    ab = [r["t_mount"] for r in ok if r["direction"] == "A>B"]
    c = Counter(int(v) for v in ab)
    W, H = 720, 220; L, R, T, B = 40, 20, 14, 40
    bins = list(range(0, 32)); mx = max(c.values())
    bw = (W - L - R) / len(bins)
    y = lambda n: T + (H - T - B) * (1 - n / mx)
    p = [f'<line x1="{L}" x2="{W-R}" y1="{H-B}" y2="{H-B}" class="axis"/>']
    for b in bins:
        n = c.get(b, 0)
        if n:
            p.append(f'<rect x="{L+b*bw+1:.1f}" y="{y(n):.1f}" width="{bw-2:.1f}" height="{H-B-y(n):.1f}" rx="3" class="c-mount"><title>{b}–{b+1} s: {n} trials</title></rect>')
            if n >= 4 or b < 28:
                p.append(f'<text x="{L+b*bw+bw/2:.1f}" y="{y(n)-5:.1f}" class="tick" text-anchor="middle">{n}</text>')
        if b % 5 == 0:
            p.append(f'<text x="{L+b*bw:.1f}" y="{H-B+18}" class="tick" text-anchor="middle">{b}</text>')
    p.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-6}" class="tick" text-anchor="middle">t_mount, 1 s bins · all {len(ab)} A→B trials, every size and variant</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Histogram of mount visibility time, concentrated at 29 to 31 seconds">{"".join(p)}</svg>'

def spread(xs, d=1):
    xs = [x for x in xs if x is not None]
    return f"{min(xs):.{d}f} / <b>{pct(xs,.5):.{d}f}</b> / {pct(xs,.9):.{d}f} / {max(xs):.{d}f}"

# ---------- follow-up run: gated first access ----------
import os as _os
GATED = "logs/latency-gated-20260903-185609.jsonl"
grecs = [json.loads(l) for l in open(GATED)] if _os.path.exists(GATED) else []
GCELLS = [("gated/stale", "gated · mount untouched 40 s · first access open()"), ("gated/recent", "gated · parent listed 1 s before · first access open()"),
          ("gated/listdir", "gated · first access listdir(parent), then open()"), ("cold", "control · open() every 0.25 s from the write reply")]
def gated_strip():
    if not grecs: return ""
    W = 720; rowh = 34; L, R, T, B = 300, 20, 14, 40; H = T + rowh * len(GCELLS) + B
    x = lambda v: L + (W - L - R) * v / 32.0
    p = []
    for v in range(0, 33, 5):
        p.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{T}" y2="{H-B+4}" class="grid"/><text x="{x(v):.1f}" y="{H-B+20}" class="tick" text-anchor="middle">{v} s</text>')
    p.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-6}" class="tick" text-anchor="middle">t_mount, seconds after the writer’s close() returned · 4.7 MB</text>')
    for i, (cell, name) in enumerate(GCELLS):
        rs = [r for r in grecs if r["cell"] == cell]; cy = T + rowh * i + rowh / 2
        p.append(f'<text x="{L-12}" y="{cy+4:.1f}" class="lbl" text-anchor="end">{html.escape(name)}</text>')
        for r in rs:
            cls = "dot c-api" if r.get("first_access_ok") else "dot-open c-mount-s"
            p.append(f'<circle cx="{x(r["t_mount"]):.1f}" cy="{cy:.1f}" r="5" class="{cls}"><title>trial {r["trial"]} · t_mount {r["t_mount"]:.2f} s · t_api {r["t_api"]:.2f} s · first access {"hit" if r.get("first_access_ok") else "miss"}</title></circle>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Gated first access: all mount times at 2.6 s; control at 13 to 30 s">{"".join(p)}</svg>'
grows = ""
for cell, name in GCELLS:
    rs = [r for r in grecs if r["cell"] == cell]
    if not rs: continue
    nok = sum(1 for r in rs if r.get("first_access_ok")); tm = [r["t_mount"] for r in rs]
    grows += f"<tr><td>{html.escape(name)}</td><td>{len(rs)}</td><td>{nok}/{len(rs)}</td><td>{pct([r['t_api'] for r in rs],.5):.2f}</td><td>{min(tm):.2f} / <b>{pct(tm,.5):.2f}</b> / {max(tm):.2f}</td><td>{pct([r['t_mount']-r['t_api'] for r in rs],.5):.2f}</td></tr>"

# ---------- before / after the hf-mount fix (minimal repro, github.com/AmineDiro/hf-mount-repro) ----------
import re as _re
def _parse_repro(path):
    out = []
    if not _os.path.exists(path): return out
    for line in open(path):
        m = _re.match(r"(cold|gated)\s+on Hub after\s+(n/a|[\d.]+)s?\s+on this mount after\s+([\d.]+)s", line)
        if m: out.append(dict(mode=m.group(1), t_hub=None if m.group(2) == "n/a" else float(m.group(2)), t_mount=float(m.group(3))))
    return out
BEFORE = _parse_repro("logs/repro-before-fix-20260904.txt")
AFTER = _parse_repro("logs/repro-after-fix-20260908.txt")
FIXROWS = [("before fix · cold: open() every 0.25 s from the write", BEFORE, "cold"),
           ("before fix · gated: first open() after the Hub lists it", BEFORE, "gated"),
           ("after fix · cold: open() every 0.25 s from the write", AFTER, "cold"),
           ("after fix · gated: first open() after the Hub lists it", AFTER, "gated")]
def fix_strip():
    if not (BEFORE and AFTER): return ""
    W = 720; rowh = 34; L, R, T, B = 330, 20, 14, 40; H = T + rowh * len(FIXROWS) + B
    x = lambda v: L + (W - L - R) * v / 32.0
    p = []
    for v in range(0, 33, 5):
        p.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{T}" y2="{H-B+4}" class="grid"/><text x="{x(v):.1f}" y="{H-B+20}" class="tick" text-anchor="middle">{v} s</text>')
    p.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-6}" class="tick" text-anchor="middle">time until open() succeeds on the other Job’s mount, seconds after the writer’s close() · 4.7 MB</text>')
    hub = [r["t_hub"] for r in BEFORE + AFTER if r["t_hub"] is not None]
    xh = x(pct(hub, .5))
    p.append(f'<line x1="{xh:.1f}" x2="{xh:.1f}" y1="{T}" y2="{H-B+4}" class="median c-api-s" stroke-dasharray="4 3"><title>upload lands: Hub lists the file at {pct(hub,.5):.2f} s (median of all gated trials)</title></line>')
    p.append(f'<text x="{xh+6:.1f}" y="{T+10}" class="tick c-api-t">upload landed ({pct(hub,.5):.1f} s)</text>')
    for i, (name, recs, mode) in enumerate(FIXROWS):
        rs = [r for r in recs if r["mode"] == mode]; cy = T + rowh * i + rowh / 2
        p.append(f'<text x="{L-12}" y="{cy+4:.1f}" class="lbl" text-anchor="end">{html.escape(name)}</text>')
        cls = "dot c-mount" if mode == "cold" else "dot c-api"
        for r in rs:
            p.append(f'<circle cx="{x(r["t_mount"]):.1f}" cy="{cy:.1f}" r="5" class="{cls}"><title>{name}: {r["t_mount"]:.2f} s</title></circle>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Before the fix cold polling took 27 to 28 s, after the fix 2 to 3 s">{"".join(p)}</svg>'
fixrows = ""
for name, recs, mode in FIXROWS:
    rs = [r["t_mount"] for r in recs if r["mode"] == mode]
    if rs: fixrows += f"<tr><td>{html.escape(name)}</td><td>{len(rs)}</td><td>{min(rs):.2f} / <b>{pct(rs,.5):.2f}</b> / {max(rs):.2f}</td></tr>"

rows1 = "".join(
    f"<tr><td>{SN[s]}</td><td>{len(by_size[s])}</td><td>{sum(r['writer_write_s']+r['writer_fsync_s'] for r in by_size[s])/len(by_size[s]):.3f}</td>"
    f"<td>{spread([r['t_api'] for r in by_size[s]])}</td><td>{spread([r['t_mount'] for r in by_size[s]])}</td>"
    f"<td>{spread([r['t_mount']-r['t_api'] for r in by_size[s]])}</td></tr>" for s in SIZES)
rows2 = "".join(
    f"<tr><td>{SN[s]}</td><td>{pct([r['t_api'] for r in by_size[s]],.5):.2f}</td><td>{pct([r['t_api_get'] for r in by_size[s]],.5):.2f}</td>"
    f"<td>{pct([r['api_download_s'] for r in by_size[s]],.5):.2f}</td><td>{pct([r['t_mount'] for r in by_size[s]],.5):.1f}</td><td>{pct([r['mount_read_s'] for r in by_size[s]],.5):.2f}</td></tr>" for s in SIZES)
LONG = {"baseline": "baseline · rename, open, cold, A→B"}
rows4 = ""
for name, pred in VARIANTS:
    rs = [r for r in focus if pred(r)]; m = pct([r["t_mount"] for r in rs], .5)
    rows4 += (f"<tr><td>{html.escape(LONG.get(name, name))}</td><td>{len(rs)}</td><td>{pct([r['t_api'] for r in rs],.5):.2f}</td><td>{spread([r['t_mount'] for r in rs])}</td>"
              f"<td>{pct([r['t_mount']-r['t_api'] for r in rs],.5):.1f}</td><td>{m-tm50:+.1f}</td></tr>")
rev = sorted(r["t_mount"] for r in focus if r["direction"] == "B>A")

page = f'''<title>Bucket Sync Latency Decomposition</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f4f6f8;--surface:#ffffff;--ink:#14202b;--ink2:#5a6771;--rule:#d9dfe5;--api:#2a78d6;--mount:#eb6834;--api-soft:#e3eefb;--mount-soft:#fdeae1;color-scheme:light}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#15191d;--surface:#1d2328;--ink:#edf0f3;--ink2:#a3adb6;--rule:#2e373f;--api:#3987e5;--mount:#d95926;--api-soft:#1b2a3d;--mount-soft:#3a2419;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#15191d;--surface:#1d2328;--ink:#edf0f3;--ink2:#a3adb6;--rule:#2e373f;--api:#3987e5;--mount:#d95926;--api-soft:#1b2a3d;--mount-soft:#3a2419;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:16px;line-height:1.55;margin:0}}
main{{max-width:76ch;margin:0 auto;padding:40px 20px 72px}}
h1{{font-size:1.9rem;font-weight:600;letter-spacing:-.01em;line-height:1.2;margin:0 0 6px;text-wrap:balance}}
h2{{font-size:1.15rem;font-weight:600;margin:44px 0 10px;text-wrap:balance}}
p{{margin:0 0 14px}} p,li{{max-width:68ch}}
.eyebrow{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin-bottom:14px}}
.lede{{font-size:1.1rem;color:var(--ink);margin:14px 0 22px}}
.verdict{{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:18px 20px 12px;margin:22px 0 8px}}
.verdict .num{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}
figure{{margin:18px 0 8px}} figcaption{{font-size:.85rem;color:var(--ink2);margin-top:6px;max-width:68ch}}
svg{{width:100%;height:auto;display:block;overflow:visible}}
.c-api{{fill:var(--api)}} .c-mount{{fill:var(--mount)}} .c-api-s{{stroke:var(--api)}} .c-mount-s{{stroke:var(--mount)}}
.c-api-t{{fill:var(--api)}} .c-mount-t{{fill:var(--mount)}}
.tick{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;fill:var(--ink2)}}
.lbl{{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:12px;fill:var(--ink)}}
.inbar{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;fill:#fff;font-weight:500}}
.axis{{stroke:var(--ink2);stroke-width:1}} .grid{{stroke:var(--rule);stroke-width:1}}
.whisk{{stroke-width:1.5;opacity:.7}} .line{{fill:none;stroke-width:2}} .dot{{stroke:var(--surface);stroke-width:2}}
.dot-open{{fill:var(--surface);stroke-width:2}} .median{{stroke-width:2.5}}
.dot:hover,.dot-open:hover,rect:hover{{filter:brightness(1.15)}}
.tbl{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface);margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}}
th,td{{padding:8px 12px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{font-weight:500;color:var(--ink2);font-size:.78rem;letter-spacing:.03em}}
tr:last-child td{{border-bottom:0}} td b{{font-weight:600}}
code,.mono{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.88em}}
code{{background:var(--surface);border:1px solid var(--rule);border-radius:4px;padding:0 4px}}
.legend{{display:flex;gap:18px;font-size:.85rem;color:var(--ink2);margin:6px 0 0}} .legend span::before{{content:"";display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.legend .l-api::before{{background:var(--api)}} .legend .l-mount::before{{background:var(--mount)}}
ul{{padding-left:20px}} li{{margin-bottom:6px}}
.facts li{{font-size:.95rem}}
.small{{font-size:.85rem;color:var(--ink2)}}
a{{color:var(--api)}}
</style>
<main>
<div class="eyebrow">HF Jobs · Storage Bucket · 2026-09-03 to 09-08 · 151 trials · two cpu-basic Jobs · fixed upstream</div>
<h1>Where the 30 s goes when two Jobs share a bucket</h1>
<p class="lede">An async GRPO run spent 29 of every 30.8 s weight sync waiting for the reading Job’s mount to notice a 4.7 MB adapter. This experiment splits that wait into upload time and cache time, all measured on the reader’s clock.</p>

<div class="verdict">
<p><b>The upload is done in <span class="num">{ta50:.1f} s</span>. The mount takes <span class="num">{tm50:.1f} s</span> to admit it.</b> The follow-up run shows why: the reader looked before the upload had landed, and that negative answer was cached for ~30 s. When the first lookup waits for the bucket API to list the object, the mount answers in <span class="num">0.1 s</span>. Reported to the hf-mount team; after their negative-cache fix of 2026-09-08 the plain polling loop sees the file in <span class="num">2.3 to 3.2 s</span>, so the 30 s is gone without any change to the trainer or to vLLM.</p>
<figure>{timeline()}
<figcaption>One 4.7 MB sync, medians of 10 trials. Stopwatch starts when the writer’s <code>close()</code> has returned and its HTTP reply reaches the reader. Blue: until the bucket API lists the object. Orange: until the reader’s own <code>hf-mount</code> lets <code>open()</code> succeed.</figcaption></figure>
</div>

<h2>The four questions the spec asked</h2>
<ul>
<li><b>Is <code>t_mount</code> size-independent from 1 KB to 128 MiB?</b> Yes. Its median is 29.4 to 29.9 s at every size. <code>t_api</code> is the one that moves, from 2.3 s to 3.6 s.</li>
<li><b>What is <code>t_mount − t_api</code> at 4.7 MB?</b> min {min(cc):.1f} / p50 {pct(cc,.5):.1f} / p90 {pct(cc,.9):.1f} / max {max(cc):.1f} s (n=10).</li>
<li><b>Does warming, probe style or publish mode change <code>t_mount</code> by more than 2 s?</b> No. Each moved the median by 0.1 s or less. Direction did, but by splitting the distribution, not shifting it.</li>
<li><b>Would an API read be faster?</b> Yes: {tm50:.1f} s to {tg50:.1f} s at 4.7 MB, and 29.4 s to 5.2 s at 128 MiB.</li>
</ul>

<h2>Latency by payload size</h2>
<figure>{sizes_chart()}
<div class="legend"><span class="l-mount">visible on reader’s mount, p50 with min–max</span><span class="l-api">readable via bucket API, p50 with min–max</span></div>
<figcaption>Baseline cells only: rename publish, direct <code>open</code> probe, cold cache, A writes and B reads, 10 trials per size. Five orders of magnitude of payload move the API time by 1.3 s and the mount time by nothing.</figcaption></figure>
<div class="tbl"><table>
<thead><tr><th>size</th><th>n</th><th>writer write+fsync mean</th><th>t_api min / p50 / p90 / max</th><th>t_mount min / p50 / p90 / max</th><th>cache cost min / p50 / p90 / max</th></tr></thead>
<tbody>{rows1}</tbody></table></div>
<div class="tbl"><table>
<thead><tr><th>size</th><th>t_api listed</th><th>t_api bytes fetched</th><th>API download s</th><th>t_mount</th><th>mount read s</th></tr></thead>
<tbody>{rows2}</tbody></table></div>
<p class="small">All values are seconds and medians unless a spread is shown. <code>close()</code> returned in at most 0.0001 s for every one of the 100 writes, so <code>write_s</code> is omitted from the headline: the writer never waits on the upload.</p>

<h2>What did and did not move the number</h2>
<figure>{strip_chart()}
<figcaption>Every row is 10 trials at 4.7 MB with one variable changed from the baseline. The bar is the median. Publish mode, probe style and a warming thread listing the parent every 0.5 s leave the ~30 s ceiling untouched. Reversing direction produced a second mode at 8 to 15 s in half the trials.</figcaption></figure>
<div class="tbl"><table>
<thead><tr><th>cell</th><th>n</th><th>t_api p50</th><th>t_mount min / p50 / p90 / max</th><th>cache cost p50</th><th>Δ p50 vs baseline</th></tr></thead>
<tbody>{rows4}</tbody></table></div>
<p>The B→A values, sorted: <span class="mono">{", ".join(f"{v:.1f}" for v in rev)}</span>. The early mode is not unique to that direction. Of the six A→B trials under 27 s, five immediately followed a B→A trial, in which the reading mount had itself just written into the bucket. The numbers say a mount that recently wrote sometimes sees a new entry sooner. They do not say why, and the spec asks that nothing more be concluded about mechanism.</p>

<h2>Shape of the distribution</h2>
<figure>{histogram()}
<figcaption>84 of 90 A→B trials land in 28 to 31 s. The remaining six scatter between 12 and 24 s with no cluster at 10 or 20 s. This is neither a clean 10 s TTL nor a uniform phase over a 30 s poll: it is a hard ceiling near 30 s from the reader’s first probe, with occasional early visibility. It matches the earlier production data, where 51 adapters landed in 26 to 31 s.</figcaption></figure>

<h2>Follow-up: let the first look come after the upload, and the mount answers at once</h2>
<p>The 30 s was anchored to the reader’s <i>first</i> lookup, so a second run (35 trials, 10 minutes) kept the reader off its mount until <code>paths-info</code> listed the object, then made a single <code>open()</code>.</p>
<figure>{gated_strip()}
<div class="legend"><span class="l-api">first access succeeded</span><span class="l-mount">first access missed, polled until visible</span></div>
<figcaption>Every gated <code>open()</code> succeeded on the first try, 0.08 s after the API listed the object, whether the parent directory had been touched 1 s or 40 s earlier. A <code>listdir</code> of the parent still showed the old snapshot in 9 of 10 trials, but an <code>open()</code> of the exact path right after it succeeded. The control, which is what the trainer does today, reproduced the 13 to 30 s wait.</figcaption></figure>
<div class="tbl"><table>
<thead><tr><th>cell</th><th>n</th><th>first access ok</th><th>t_api p50</th><th>t_mount min / p50 / max</th><th>t_mount − t_api p50</th></tr></thead>
<tbody>{grows}</tbody></table></div>
<p><b>What this changes.</b> A lookup of a name the mount has never seen goes to the Hub. A lookup made <i>before</i> the object exists is cached as “not found” for up to 30 s, and every retry is served from that answer. The trainer’s first <code>load_lora_adapter</code> lands 0.7 s after the write, so it poisons the path on every sync. The fix needs no change to vLLM and no mount polling: after the rename, ask <code>paths-info</code> until the adapter files are listed (about 2.5 s), then call <code>load_lora_adapter</code> once. Expected wait per sync: 2.6 s instead of 30 s.</p>

<h2>After the fix: the mount catches up with the upload</h2>
<p>The finding went to the hf-mount team with a two-script repro (<span class="mono">github.com/AmineDiro/hf-mount-repro</span>). A release with a shorter negative-cache TTL landed on 2026-09-08, and the same repro was run again, unchanged, on Jobs.</p>
<figure>{fix_strip()}
<div class="legend"><span class="l-mount">cold: polled the mount from the start</span><span class="l-api">gated: first open() after the Hub listed the file</span></div>
<figcaption>Each dot is one trial. Before the fix, polling the mount from the moment the write returned cost 27 to 28 s; after it, the same loop sees the file 2.3 to 3.2 s after the write, within one retry of the upload landing. The gated path is unchanged at 2.5 s because it never depended on the negative cache.</figcaption></figure>
<div class="tbl"><table>
<thead><tr><th>cell</th><th>n</th><th>t_mount min / p50 / max</th></tr></thead>
<tbody>{fixrows}</tbody></table></div>
<p><b>What it means for the trainer.</b> Nothing has to change in the sync code. The existing retry loop, which asks vLLM to load the adapter every 2 s, now converges about 3 to 4 s after the write instead of about 30 s. A negative lookup is forgotten within a fraction of a second rather than held for the poll interval, so looking early no longer poisons the path.</p>

<h2>Facts worth writing down even though they are boring</h2>
<ul class="facts">
<li>Both Jobs mount the bucket as <code>hf-mount /lora fuse rw,nosuid,nodev,relatime,idmapped,user_id=0,group_id=0,default_permissions,allow_other</code>.</li>
<li><code>close()</code> and <code>fsync()</code> return before the upload lands. A 128 MiB write returns in 0.08 s and is listed by the API 3.6 s later, so the mount uploads at well over 100 MB/s in the background.</li>
<li>The staged <code>.tmp</code> path never showed up in the API before the final path (0 of 90 rename trials). <code>os.rename</code> of the staged directory took 0.0001 s.</li>
<li>Payloads were complete the instant they were visible: mount <code>stat</code> size matched in 100 of 100 trials, mount read hash-verified in 100 of 100, API download hash-verified in 100 of 100.</li>
<li>No trial timed out. Round trip from reader to writer through the jobs proxy was 0.053 s, which bounds the clock-alignment error of the B→A cells.</li>
<li>A directory created with <code>os.makedirs</code> on the reader’s mount and left empty for 24 s had vanished by the time the first file was written into it (<code>ENOENT</code>). Result files were recovered from the Job log instead.</li>
<li>Runtime: 48 min per Job on <code>cpu-basic</code>, under $0.05 total. The 1.96 GB of probe payloads were deleted from the bucket afterwards.</li>
</ul>

<h2>Method</h2>
<p>Side A runs an HTTP server on an exposed port with the bucket mounted at <code>/lora</code>. Side B, with the same mount, drives 100 trials in shuffled order: it asks A to write <i>N</i> fresh <code>os.urandom</code> bytes under a new key, starts a stopwatch when A’s reply arrives, then polls every 0.25 s in two threads. One thread calls the bucket API (<code>paths-info</code>, then a real <code>GET</code> through the resolve route, hash-checked). The other opens the exact path on B’s mount, then reads and hashes it. Sixty baseline trials cover six sizes; forty more at 4.7 MB change one variable each. For the reverse direction B writes and A watches, reporting durations from the moment the watch request arrived.</p>
<p class="small">Code: <span class="mono">run_latency.sh</span>, <span class="mono">src/latency_writer.py</span>, <span class="mono">src/latency_reader.py</span>, <span class="mono">src/latency_common.py</span>, <span class="mono">src/latency_analyze.py</span>. Raw trials: <span class="mono">logs/latency-20260903-075808.jsonl</span>. Written version of this page: <span class="mono">BUCKET_LATENCY_RESULTS.md</span>.</p>
</main>
'''
out = "report/bucket-latency-report.html"
open(out, "w").write(page)
print(len(page), "bytes ->", out)
