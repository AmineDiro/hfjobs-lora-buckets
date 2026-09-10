"""Bucket-latency probe, side B: drives the sweep and owns the clock.

For every trial the reader asks the writer (side A) to write a fresh random payload, starts its stopwatch the
instant the writer's HTTP reply arrives, and polls the bucket API and its own mount in parallel until both admit
the object exists. For the reverse-direction cells the reader writes and asks A to watch; A reports durations
measured on its own clock from the moment the watch request arrived, and the reader subtracts its own write time.

One JSON line per trial on stdout, mirrored to `/lora/results/<RUN_ID>/`. Nothing here imports beyond stdlib.
"""

import json
import os
import random
import statistics
import sys
import time
import urllib.parse

import latency_common as lc

WRITER_URL = os.environ["WRITER_URL"].rstrip("/")
RUN_ID = os.environ.get("RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
N = int(os.environ.get("TRIALS_PER_CELL", "10"))
SIZES = [int(x) for x in os.environ.get("SIZES", "1024,65536,1048576,4700000,36700160,134217728").split(",")]
FOCUS = 4_700_000
RESULTS_DIR = os.path.join(lc.MOUNT, "results", RUN_ID)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def writer(method, route, timeout=400, **q):
    url = f"{WRITER_URL}{route}"
    if q:
        url += "?" + urllib.parse.urlencode({k: v for k, v in q.items() if v is not None})
    st, _, body = lc.http(method, url, headers=lc.hub_headers(), timeout=timeout)
    if st != 200:
        raise RuntimeError(f"{method} {route} -> {st}: {body[:200]!r}")
    return json.loads(body)


def build_plan():
    base = dict(mode="rename", probe="open", warm=False, direction="A>B")
    plan = [dict(base, size=s) for s in SIZES for _ in range(N)]
    plan += [dict(base, size=FOCUS, mode="direct") for _ in range(N)]
    plan += [dict(base, size=FOCUS, probe="listdir") for _ in range(N)]
    plan += [dict(base, size=FOCUS, warm=True) for _ in range(N)]
    plan += [dict(base, size=FOCUS, direction="B>A") for _ in range(N)]
    random.Random(7).shuffle(plan)
    return plan


def run_trial(i, cell):
    key = f"{RUN_ID}-t{i:03d}-{random.randrange(16**6):06x}"
    rec = dict(cell, trial=i, key=key, run_id=RUN_ID)
    if cell["direction"] == "A>B":
        parent = "probe"
        t_send = time.monotonic()
        w = writer("POST", "/write", parent=parent, key=key, size=cell["size"], mode=cell["mode"])
        t_reply = time.monotonic()
        watch = lc.Watch(
            parent=parent, key=key, size=cell["size"], sha256=w["sha256"], rel_path=w["rel_path"],
            rel_tmp_path=w.get("rel_tmp_path"), probe=cell["probe"], warm=cell["warm"], t0=t_reply,
        )
        res = watch.wait()
        rec.update({f"writer_{k}": v for k, v in w.items() if k.endswith("_s")})
        rec["writer_sha256"] = w["sha256"]
        rec["request_rtt_s"] = t_reply - t_send
        rec.update(res)
    else:
        parent = "probe-rev"
        rel_path = f"{parent}/{key}/payload.bin"
        rel_tmp = f"{parent}/.{key}.tmp/payload.bin" if cell["mode"] == "rename" else None
        payload, sha = lc.make_payload(cell["size"])
        t_send = time.monotonic()
        writer("POST", "/watch", timeout=30, parent=parent, key=key, size=cell["size"], sha256=sha,
               rel_path=rel_path, rel_tmp_path=rel_tmp, probe=cell["probe"], warm=int(cell["warm"]))
        t_reply = time.monotonic()
        w = lc.write_payload(parent, key, cell["size"], cell["mode"], payload=payload)
        t_done = time.monotonic()
        rec.update({f"writer_{k}": v for k, v in w.items() if k.endswith("_s")})
        rec["writer_sha256"] = w["sha256"]
        rec["request_rtt_s"] = t_reply - t_send
        rec["write_offset_s"] = t_done - t_send  # A's t0 sits inside this window, within one one-way latency
        deadline = time.monotonic() + lc.WATCH_TIMEOUT_S + 30
        while time.monotonic() < deadline:
            res = writer("GET", "/watch", timeout=30, key=key)
            if res.get("done"):
                break
            time.sleep(1.0)
        # Shift A's durations to start at our write completion. Uncertainty is one one-way proxy latency.
        for k in ("t_api", "t_api_tmp", "t_api_get", "t_mount"):
            if res.get(k) is not None:
                res[k] = res[k] - rec["write_offset_s"]
        rec.update(res)
    return rec


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(recs):
    cells = {}
    for r in recs:
        c = (r["size"], r["mode"], r["probe"], r["warm"], r["direction"])
        cells.setdefault(c, []).append(r)
    log("SUMMARY  size mode probe warm dir | n | t_api p50 | t_mount p50 | cache p50 | never")
    for c in sorted(cells):
        rs = cells[c]
        ta = [r["t_api"] for r in rs]
        tm = [r["t_mount"] for r in rs]
        cc = [r["t_mount"] - r["t_api"] for r in rs if r["t_mount"] is not None and r["t_api"] is not None]
        never = sum(1 for r in rs if r["t_mount"] is None)
        f = lambda v: "-" if v is None else f"{v:6.2f}"
        log(f"SUMMARY  {c[0]:>10} {c[1]:6} {c[2]:7} {str(c[3]):5} {c[4]} | {len(rs):2} | {f(pct(ta,.5))} | {f(pct(tm,.5))} | {f(pct(cc,.5))} | {never}")


def main():
    log(f"run_id={RUN_ID} writer={WRITER_URL} bucket={lc.BUCKET} trials_per_cell={N}")
    log(f"reader mount: {lc.mount_line()}")
    for attempt in range(120):
        try:
            writer("GET", "/health", timeout=10)
            break
        except Exception as e:
            if attempt % 10 == 0:
                log(f"waiting for writer: {e}")
            time.sleep(5)
    else:
        log("writer never became healthy")
        sys.exit(1)
    log(f"writer mount: {writer('GET', '/mount', timeout=10)}")
    rtts = []
    for _ in range(5):
        t = time.monotonic()
        writer("GET", "/health", timeout=10)
        rtts.append(time.monotonic() - t)
    log(f"writer RTT through jobs proxy: min={min(rtts):.3f}s median={statistics.median(rtts):.3f}s")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plan = build_plan()
    log(f"{len(plan)} trials planned")
    recs = []
    prev_dir = None
    t_run = time.monotonic()
    for i, cell in enumerate(plan):
        try:
            rec = run_trial(i, cell)
        except Exception as e:
            rec = dict(cell, trial=i, error=f"{type(e).__name__}: {e}")
            log(f"trial {i} failed: {rec['error']}")
        rec["prev_direction"] = prev_dir
        prev_dir = cell["direction"]
        rec["rtt_proxy_median_s"] = statistics.median(rtts)
        recs.append(rec)
        print("TRIAL " + json.dumps(rec), flush=True)
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(os.path.join(RESULTS_DIR, f"trial-{i:03d}.json"), "w") as f:
                json.dump(rec, f)
        except Exception as e:
            log(f"could not mirror trial {i} to bucket: {e}")
        if "error" not in rec:
            ta, tm = rec.get("t_api"), rec.get("t_mount")
            log(f"trial {i:3d}/{len(plan)} {cell['size']:>10}B {cell['mode']:6} {cell['probe']:7} "
                f"warm={int(cell['warm'])} {cell['direction']} | write={rec.get('writer_write_s', 0):.2f}s "
                f"t_api={'-' if ta is None else f'{ta:.2f}'}s t_mount={'-' if tm is None else f'{tm:.2f}'}s "
                f"| {(time.monotonic() - t_run) / 60:.1f} min elapsed")
    # An empty directory created with `os.makedirs` on hf-mount was gone again by the time the first file went into
    # it (ENOENT on open, observed 2026-09-03), so re-create the directory right before every write and never let
    # the mirror break the run: stdout `TRIAL` lines are the primary record.
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "results.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    except Exception as e:
        log(f"could not write results.jsonl to bucket: {e}")
    summarize([r for r in recs if "error" not in r])
    log(f"done in {(time.monotonic() - t_run) / 60:.1f} min; results in {RESULTS_DIR}")
    try:
        writer("POST", "/shutdown", timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    main()
