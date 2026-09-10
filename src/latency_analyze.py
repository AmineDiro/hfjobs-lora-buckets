"""Turn the bucket-latency probe's TRIAL records into the tables BUCKET_LATENCY_EXPERIMENT.md asks for.

    python src/latency_analyze.py logs/latency-<run>.log [more files...]   # reader Job log (TRIAL lines) or .jsonl

Prints Markdown. Every number is seconds unless the column says otherwise.
"""

import json
import sys
from collections import Counter, defaultdict

SIZE_NAMES = {1024: "1 KB", 65536: "64 KB", 1_048_576: "1 MB", 4_700_000: "4.7 MB", 36_700_160: "35 MiB", 134_217_728: "128 MiB"}
FOCUS = 4_700_000


def load(paths):
    recs = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line.startswith("TRIAL "):
                line = line[6:]
            if line.startswith("{"):
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    # de-duplicate (a log snapshot may have been concatenated)
    seen, out = set(), []
    for r in recs:
        k = (r.get("run_id"), r.get("trial"))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def f(v, d=1):
    return "–" if v is None else f"{v:.{d}f}"


def spread(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    return f"{f(min(xs))} / **{f(pct(xs, .5))}** / {f(pct(xs, .9))} / {f(max(xs))}"


def is_baseline(r):
    return r["mode"] == "rename" and r["probe"] == "open" and not r["warm"] and r["direction"] == "A>B"


def cache_cost(rs):
    return [r["t_mount"] - r["t_api"] for r in rs if r.get("t_mount") is not None and r.get("t_api") is not None]


def main(paths):
    recs = load(paths)
    ok = [r for r in recs if "error" not in r]
    errs = [r for r in recs if "error" in r]
    print(f"Records: {len(recs)} trials, {len(errs)} errored, run(s) {sorted({r.get('run_id') for r in recs})}\n")

    # 1. decomposition table
    print("## 1. Decomposition (baseline cells: rename, open-probe, cold cache, A writes → B reads)\n")
    print("| size | n | writer `write_s` mean | writer `close_s` mean | `t_api` min / **p50** / p90 / max | `t_mount` min / **p50** / p90 / max | **cache cost** min / **p50** / p90 / max | never seen |")
    print("|---|---|---|---|---|---|---|---|")
    by_size = defaultdict(list)
    for r in ok:
        if is_baseline(r):
            by_size[r["size"]].append(r)
    for size in sorted(by_size):
        rs = by_size[size]
        ws = [r["writer_write_s"] + r.get("writer_fsync_s", 0) for r in rs]
        cs = [r["writer_close_s"] for r in rs]
        never = sum(1 for r in rs if r.get("t_mount") is None)
        print(f"| {SIZE_NAMES.get(size, size)} | {len(rs)} | {f(sum(ws)/len(ws), 3)} | {f(sum(cs)/len(cs), 3)} | "
              f"{spread([r['t_api'] for r in rs])} | {spread([r['t_mount'] for r in rs])} | {spread(cache_cost(rs))} | {never} |")

    # 2. size dependence
    print("\n## 2. Size dependence (baseline cells, p50)\n")
    print("| size | `t_api` p50 | `t_api_get` p50 (bytes fetched) | API download s p50 | `t_mount` p50 | mount read s p50 |")
    print("|---|---|---|---|---|---|")
    for size in sorted(by_size):
        rs = by_size[size]
        print(f"| {SIZE_NAMES.get(size, size)} | {f(pct([r['t_api'] for r in rs], .5), 2)} | {f(pct([r['t_api_get'] for r in rs], .5), 2)} | "
              f"{f(pct([r['api_download_s'] for r in rs], .5), 2)} | {f(pct([r['t_mount'] for r in rs], .5), 1)} | {f(pct([r['mount_read_s'] for r in rs], .5), 2)} |")
    sizes = sorted(by_size)
    if len(sizes) >= 2:
        a0, a1 = pct([r["t_api"] for r in by_size[sizes[0]]], .5), pct([r["t_api"] for r in by_size[sizes[-1]]], .5)
        m0, m1 = pct([r["t_mount"] for r in by_size[sizes[0]]], .5), pct([r["t_mount"] for r in by_size[sizes[-1]]], .5)
        print(f"\n`t_api` p50 goes {f(a0,2)} s → {f(a1,2)} s from {SIZE_NAMES[sizes[0]]} to {SIZE_NAMES[sizes[-1]]}; "
              f"`t_mount` p50 goes {f(m0)} s → {f(m1)} s.")

    # 3. the fix
    base_focus = by_size.get(FOCUS, [])
    if base_focus:
        ta, tg, tm = (pct([r[k] for r in base_focus], .5) for k in ("t_api", "t_api_get", "t_mount"))
        print(f"\n## 3. Does the fix work?\n\nAt 4.7 MB (n={len(base_focus)}): reading through the bucket API instead of the mount would cut the wait "
              f"from **{f(tm)} s** (mount p50) to **{f(ta)} s** (API lists it, p50) / **{f(tg)} s** (bytes fetched and hash-verified, p50).")

    # 4. secondary variables
    print("\n## 4. Secondary variables (all at 4.7 MB, one variable changed from baseline at a time)\n")
    print("| cell | n | `t_api` p50 | `t_mount` min / **p50** / p90 / max | cache cost p50 | Δ `t_mount` p50 vs baseline | never seen |")
    print("|---|---|---|---|---|---|---|")
    focus = [r for r in ok if r["size"] == FOCUS]
    variants = [
        ("baseline: rename, open, cold, A>B", is_baseline),
        ("publish mode = direct", lambda r: r["mode"] == "direct"),
        ("probe = listdir parent", lambda r: r["probe"] == "listdir"),
        ("cache warming on (listdir every 0.5 s)", lambda r: r["warm"]),
        ("direction B writes → A reads", lambda r: r["direction"] == "B>A"),
    ]
    base_p50 = pct([r["t_mount"] for r in base_focus], .5) if base_focus else None
    for name, pred in variants:
        rs = [r for r in focus if pred(r)]
        if not rs:
            continue
        p50 = pct([r["t_mount"] for r in rs], .5)
        delta = None if (p50 is None or base_p50 is None) else p50 - base_p50
        never = sum(1 for r in rs if r.get("t_mount") is None)
        print(f"| {name} | {len(rs)} | {f(pct([r['t_api'] for r in rs], .5), 2)} | {spread([r['t_mount'] for r in rs])} | "
              f"{f(pct(cache_cost(rs), .5))} | {'–' if delta is None else f'{delta:+.1f}'} | {never} |")
    ld = [r["mount_open_after_listdir_s"] for r in focus if r["probe"] == "listdir" and r.get("mount_open_after_listdir_s") is not None]
    if ld:
        print(f"\nAfter `listdir` showed the entry, `open` on the child succeeded after {f(pct(ld,.5),2)} s p50 / {f(max(ld),2)} s max.")
    warm = [r["warm_listings"] for r in focus if r["warm"]]
    if warm:
        print(f"Warming thread issued {f(sum(warm)/len(warm),0)} parent listings per trial on average.")

    # 5. histogram
    print("\n## 5. Shape of the distribution: `t_mount` in 1 s bins, all A→B trials\n")
    ab = [r["t_mount"] for r in ok if r["direction"] == "A>B" and r.get("t_mount") is not None]
    if ab:
        c = Counter(int(x) for x in ab)
        print("```")
        for b in range(min(c), max(c) + 1):
            print(f"{b:3d}–{b+1:<3d}s | {'█' * c[b]}{' ' + str(c[b]) if c[b] else ''}")
        print("```")
        rev = [r["t_mount"] for r in ok if r["direction"] == "B>A" and r.get("t_mount") is not None]
        if rev:
            print(f"\nB→A trials (n={len(rev)}): {', '.join(f'{x:.1f}' for x in sorted(rev))}")
    # t_api histogram too, small
    api = [r["t_api"] for r in ok if r["direction"] == "A>B" and r.get("t_api") is not None]
    if api:
        c = Counter(round(x * 2) / 2 for x in api)
        print("\n`t_api` in 0.5 s bins, all A→B trials:\n```")
        for b in sorted(c):
            print(f"{b:4.1f}s | {'█' * c[b]} {c[b]}")
        print("```")

    # 6. boring but required
    print("\n## 6. The boring facts\n")
    close_s = [r["writer_close_s"] for r in ok if "writer_close_s" in r]
    rename_s = [r["writer_rename_s"] for r in ok if r["mode"] == "rename"]
    print(f"- `close()` returned in {f(max(close_s),4)} s at worst across all {len(close_s)} writes (and `write+fsync` p50 "
          f"{f(pct([r['writer_write_s']+r['writer_fsync_s'] for r in ok],.5),3)} s), while the API first listed the object "
          f"{f(pct([r['t_api'] for r in ok],.5),2)} s later (p50): the mount acknowledges the write before the upload lands.")
    if rename_s:
        print(f"- `os.rename` of the staged directory took {f(pct(rename_s,.5),4)} s p50 / {f(max(rename_s),3)} s max.")
    tmp = [r for r in ok if r["mode"] == "rename" and r.get("t_api_tmp") is not None]
    print(f"- Staged `.tmp` path visible through the API before the final path: {len(tmp)} of {sum(1 for r in ok if r['mode']=='rename')} rename trials"
          + (f" (p50 {f(pct([r['t_api_tmp'] for r in tmp],.5),2)} s)" if tmp else "") + ".")
    mv = Counter(str(r.get("mount_verified")) for r in ok)
    av = Counter(str(r.get("api_verified")) for r in ok)
    print(f"- Hash verification: via mount {dict(mv)}, via API {dict(av)}.")
    size_ok = sum(1 for r in ok if r.get("mount_size_first_seen") == r["size"])
    print(f"- Mount `stat` size equalled the payload size the instant the entry appeared in {size_ok} of {len(ok)} trials; "
          f"API-reported size matched in {sum(1 for r in ok if r.get('api_size_first_seen') == r['size'])}.")
    never = [r for r in ok if r.get("t_mount") is None or r.get("t_api") is None]
    if never:
        print(f"- Trials where something never appeared within the watch timeout ({len(never)}):")
        for r in never:
            print(f"  - trial {r['trial']}: {SIZE_NAMES.get(r['size'], r['size'])} {r['mode']} {r['probe']} warm={r['warm']} {r['direction']} "
                  f"t_api={f(r.get('t_api'))} t_mount={f(r.get('t_mount'))} api_errors={r.get('api_errors')} mount_errors={r.get('mount_errors')}")
    else:
        print("- Every object appeared on both the API and the mount within the watch timeout.")
    if errs:
        print(f"- {len(errs)} trials errored before producing a measurement: " + "; ".join(f"trial {r['trial']}: {r['error'][:80]}" for r in errs))
    rtt = [r.get("rtt_proxy_median_s") for r in ok if r.get("rtt_proxy_median_s")]
    if rtt:
        print(f"- Reader→writer round trip through the jobs proxy: {f(rtt[0],3)} s median (bounds the B→A clock-alignment error).")


if __name__ == "__main__":
    main(sys.argv[1:] or ["logs/latency.log"])
