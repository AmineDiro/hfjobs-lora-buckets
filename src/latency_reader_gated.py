"""Follow-up to the bucket-latency probe: does the reader's mount answer correctly if its FIRST look at a new path
comes after the upload has landed?

The first experiment probed the mount from the instant the write returned and saw ~30 s in every cell. Here the
reader stays off its mount, waits until the bucket API (`paths-info`) lists the object, and only then makes one
`open()`. If that first access succeeds, gating the load on the API is enough; if it still waits ~30 s, the
mount only learns about new paths from its background poll and the adapter has to be fetched through the API.

Cells (all 4.7 MB, rename publish, A writes -> B reads):
  gated/stale   B has not touched the mount for >= STALE_S before the trial
  gated/recent  B listed the parent directory 1 s before requesting the write
  gated/listdir first access after the gate is listdir(parent), then open
  cold          the original behaviour: open() every 0.25 s from the write reply (control)

One `TRIAL {json}` line per trial on stdout. The reader never writes to the mount, so nothing here refreshes it.
"""

import json
import os
import random
import statistics
import sys
import time
import urllib.parse
import hashlib

import latency_common as lc

WRITER_URL = os.environ["WRITER_URL"].rstrip("/")
RUN_ID = os.environ.get("RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
N = int(os.environ.get("TRIALS_PER_CELL", "10"))
N_COLD = int(os.environ.get("COLD_TRIALS", "5"))
SIZE = int(os.environ.get("SIZE", "4700000"))
STALE_S = float(os.environ.get("STALE_S", "40"))
PARENT = "probe"
ROOT = os.path.join(lc.MOUNT, PARENT)

last_mount_access = time.monotonic()


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


def touch_mount(fn):
    """Every access to the mount goes through here so `last_mount_access` stays honest."""
    global last_mount_access
    try:
        return fn()
    finally:
        last_mount_access = time.monotonic()


def try_open(path):
    try:
        touch_mount(lambda: open(path, "rb").close())
        return True
    except OSError:
        return False


def try_listdir(key):
    try:
        return key in touch_mount(lambda: os.listdir(ROOT))
    except OSError:
        return False


def read_hash(path):
    def _r():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    return touch_mount(_r)


def wait_api(rel_path, t0):
    """Poll paths-info every 0.25 s until the final path is listed. Returns (t_api, n_requests, errors)."""
    n = err = 0
    while time.monotonic() - t0 < lc.WATCH_TIMEOUT_S:
        n += 1
        try:
            seen, st = lc.api_paths_info([rel_path])
            if seen is None:
                err += 1
            elif rel_path in seen:
                return time.monotonic() - t0, n, err
        except Exception:
            err += 1
        time.sleep(lc.POLL_S)
    return None, n, err


def run_trial(i, cell):
    global last_mount_access
    key = f"{RUN_ID}-t{i:03d}-{random.randrange(16**6):06x}"
    path = os.path.join(ROOT, key, "payload.bin")
    rec = dict(cell, trial=i, key=key, run_id=RUN_ID, size=SIZE)

    # --- pre-conditioning of the reader's mount ---
    if cell["parent"] == "stale":
        wait = STALE_S - (time.monotonic() - last_mount_access)
        if wait > 0:
            log(f"trial {i}: idling {wait:.0f}s so the mount goes untouched for {STALE_S:.0f}s")
            time.sleep(wait)
    elif cell["parent"] == "recent":
        touch_mount(lambda: os.listdir(ROOT))
        time.sleep(1.0)
    rec["mount_idle_before_write_s"] = time.monotonic() - last_mount_access

    # --- the write, on A ---
    t_send = time.monotonic()
    w = writer("POST", "/write", parent=PARENT, key=key, size=SIZE, mode="rename")
    t0 = time.monotonic()
    rec["request_rtt_s"] = t0 - t_send
    rec["writer_sha256"] = w["sha256"]
    rec.update({f"writer_{k}": v for k, v in w.items() if k.endswith("_s")})

    if cell["gate"]:
        # Stay off the mount until the Hub says the object exists.
        t_api, n_req, err = wait_api(w["rel_path"], t0)
        rec.update(t_api=t_api, api_requests=n_req, api_errors=err)
        if t_api is None:
            rec["error"] = "API never listed the object"
            return rec
        # The first access. This is the measurement.
        t_first = time.monotonic()
        ok = try_listdir(key) if cell["first"] == "listdir" else try_open(path)
        rec["t_first_access"] = t_first - t0
        rec["first_access_ok"] = ok
        if ok and cell["first"] == "listdir":
            rec["open_after_listdir_ok"] = try_open(path)
        if ok:
            rec["t_mount"] = time.monotonic() - t0
        else:
            # Fall back to polling so we still learn how long it takes when the first access misses.
            while time.monotonic() - t0 < lc.WATCH_TIMEOUT_S:
                time.sleep(lc.POLL_S)
                if try_open(path):
                    rec["t_mount"] = time.monotonic() - t0
                    break
            else:
                rec["t_mount"] = None
    else:
        # Control: the original behaviour, both pollers from the reply.
        watch = lc.Watch(parent=PARENT, key=key, size=SIZE, sha256=w["sha256"], rel_path=w["rel_path"],
                         rel_tmp_path=w.get("rel_tmp_path"), probe="open", warm=False, t0=t0)
        res = watch.wait()
        last_mount_access = time.monotonic()
        rec.update(t_api=res["t_api"], t_mount=res["t_mount"], first_access_ok=False, t_first_access=lc.POLL_S)
        rec["mount_verified"] = res["mount_verified"]
        return rec

    if rec.get("t_mount") is not None:
        rec["mount_verified"] = read_hash(path) == w["sha256"]
    return rec


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    log(f"run_id={RUN_ID} writer={WRITER_URL} bucket={lc.BUCKET} size={SIZE} n={N} cold={N_COLD} stale={STALE_S}s")
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

    plan = []
    plan += [dict(cell="gated/stale", gate=True, parent="stale", first="open") for _ in range(N)]
    plan += [dict(cell="gated/recent", gate=True, parent="recent", first="open") for _ in range(N)]
    plan += [dict(cell="gated/listdir", gate=True, parent="none", first="listdir") for _ in range(N)]
    plan += [dict(cell="cold", gate=False, parent="none", first="open") for _ in range(N_COLD)]
    random.Random(11).shuffle(plan)
    log(f"{len(plan)} trials planned")

    recs = []
    t_run = time.monotonic()
    for i, c in enumerate(plan):
        try:
            rec = run_trial(i, c)
        except Exception as e:
            rec = dict(c, trial=i, error=f"{type(e).__name__}: {e}")
        recs.append(rec)
        print("TRIAL " + json.dumps(rec), flush=True)
        if "error" in rec:
            log(f"trial {i} failed: {rec['error']}")
            continue
        f = lambda v: "-" if v is None else f"{v:.2f}"
        log(f"trial {i:2d}/{len(plan)} {c['cell']:13} idle={rec.get('mount_idle_before_write_s', 0):5.1f}s "
            f"t_api={f(rec.get('t_api'))}s first_access@{f(rec.get('t_first_access'))}s ok={rec.get('first_access_ok')} "
            f"t_mount={f(rec.get('t_mount'))}s | {(time.monotonic() - t_run) / 60:.1f} min")

    log("SUMMARY cell | n | first access ok | t_api p50 | t_mount min/p50/max")
    for cell in ("gated/stale", "gated/recent", "gated/listdir", "cold"):
        rs = [r for r in recs if r.get("cell") == cell and "error" not in r]
        if not rs:
            continue
        ok = sum(1 for r in rs if r.get("first_access_ok"))
        tm = [r["t_mount"] for r in rs if r.get("t_mount") is not None]
        f = lambda v: "-" if v is None else f"{v:.2f}"
        log(f"SUMMARY {cell:13} | {len(rs):2} | {ok}/{len(rs)} | {f(pct([r['t_api'] for r in rs], .5))} | "
            f"{f(min(tm) if tm else None)}/{f(pct(tm, .5))}/{f(max(tm) if tm else None)}")
    log(f"done in {(time.monotonic() - t_run) / 60:.1f} min")
    try:
        writer("POST", "/shutdown", timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    main()
