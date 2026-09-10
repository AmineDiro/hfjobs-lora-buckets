"""Preflight, client half: cross-job bucket visibility, and what the jobs proxy will carry.

Runs in a second `cpu-basic` Job with the same bucket mounted at `/lora` and `SERVER_URL` pointing at the
server half's exposed port. It answers, with numbers rather than documentation:

* how many seconds after a write the other job's mount can see a new adapter-shaped directory, and how much
  longer until its bytes are complete (hash-verified, not just present);
* whether the exposed port rejects an unauthenticated request;
* the longest response the jobs proxy will hold open, which caps how long a generation request can take.

Everything it prints is a number the real run's settings are derived from.
"""

import hashlib
import json
import os
import threading
import time

import requests


MOUNT = os.environ.get("MOUNT", "/lora")
ROOT = os.path.join(MOUNT, "preflight")
SERVER_URL = os.environ["SERVER_URL"].rstrip("/")
TOKEN = os.environ["HF_TOKEN"]
BEACONS = int(os.environ.get("BEACONS", "12"))
# Generous: the point is to find where the *proxy* gives up, not to be cut off by our own client.
CLIENT_TIMEOUT = 900


def probe_visibility():
    """Wait for each beacon the server publishes and report first-seen and hash-verified latency."""
    beacons = os.path.join(ROOT, "beacons")
    print(f"[vis] watching {beacons}", flush=True)
    results = []
    for version in range(1, BEACONS + 1):
        meta_path = os.path.join(beacons, f"beacon-v{version}", "meta.json")
        payload_path = os.path.join(beacons, f"beacon-v{version}", "payload.bin")
        deadline = time.time() + 300
        meta = None
        while time.time() < deadline:
            try:
                # Listing the parent is what forces a directory revalidation on a polling mount; a bare stat of
                # the child can be answered from a cached negative lookup.
                os.listdir(beacons)
                with open(meta_path) as f:
                    meta = json.load(f)
                break
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                time.sleep(0.5)
        if meta is None:
            print(f"[vis] v{version} NEVER APPEARED within 300s", flush=True)
            results.append((version, None, None))
            continue
        t_seen = time.time()
        # Present is not the same as complete. Read until the payload hashes to what the writer recorded.
        while time.time() < deadline:
            try:
                with open(payload_path, "rb") as f:
                    data = f.read()
                if hashlib.sha256(data).hexdigest() == meta["sha256"]:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        t_ok = time.time()
        seen = t_seen - meta["published_at"]
        complete = t_ok - meta["published_at"]
        print(f"[vis] v{version}: first seen +{seen:.1f}s, hash-verified +{complete:.1f}s", flush=True)
        results.append((version, seen, complete))

    ok = [r for r in results if r[1] is not None]
    if ok:
        seens = sorted(r[1] for r in ok)
        completes = sorted(r[2] for r in ok)
        print(
            f"[vis] SUMMARY over {len(ok)}/{len(results)} beacons: "
            f"first-seen min {seens[0]:.1f}s / median {seens[len(seens) // 2]:.1f}s / max {seens[-1]:.1f}s; "
            f"hash-verified min {completes[0]:.1f}s / median {completes[len(completes) // 2]:.1f}s / "
            f"max {completes[-1]:.1f}s",
            flush=True,
        )


def probe_proxy():
    """Find out what the jobs proxy requires and what it will carry."""
    print(f"[proxy] target {SERVER_URL}", flush=True)
    auth = {"Authorization": f"Bearer {TOKEN}"}

    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=60)
        print(f"[proxy] no auth  -> {r.status_code} (401/403 means the header is required)", flush=True)
    except Exception as e:
        print(f"[proxy] no auth  -> {type(e).__name__}: {e}", flush=True)

    try:
        r = requests.get(f"{SERVER_URL}/health", headers=auth, timeout=60)
        print(f"[proxy] with auth-> {r.status_code} {r.text[:120]}", flush=True)
    except Exception as e:
        print(f"[proxy] with auth-> {type(e).__name__}: {e}", flush=True)

    # A 1 MB POST stands in for a rollout request body; a 32 MB response for a batch of completions.
    for name, call in [
        ("POST 1 MiB", lambda: requests.post(f"{SERVER_URL}/echo", data=b"x" * (1 << 20), headers=auth, timeout=120)),
        ("GET 32 MiB", lambda: requests.get(f"{SERVER_URL}/big?n={32 << 20}", headers=auth, timeout=300)),
    ]:
        t0 = time.time()
        try:
            r = call()
            print(f"[proxy] {name} -> {r.status_code} in {time.time() - t0:.1f}s ({len(r.content)} bytes)", flush=True)
        except Exception as e:
            print(f"[proxy] {name} -> {type(e).__name__} after {time.time() - t0:.1f}s: {e}", flush=True)

    # The load-bearing one. Escalating hold times until the proxy cuts one off; the last value that returns 200
    # is the ceiling a single generation request has to stay under.
    for seconds in (5, 30, 60, 120, 240, 400, 600):
        t0 = time.time()
        try:
            r = requests.get(f"{SERVER_URL}/slow?s={seconds}", headers=auth, timeout=CLIENT_TIMEOUT)
            print(f"[proxy] hold {seconds:>3}s -> {r.status_code} after {time.time() - t0:.1f}s", flush=True)
            if r.status_code != 200:
                print(f"[proxy] proxy stops holding somewhere below {seconds}s", flush=True)
                break
        except Exception as e:
            print(f"[proxy] hold {seconds:>3}s -> {type(e).__name__} after {time.time() - t0:.1f}s: {e}", flush=True)
            break


if __name__ == "__main__":
    # Concurrently: the beacons are published on a cadence starting when the server half boots, so waiting for
    # the proxy probe to finish first would find every beacon long since visible and measure nothing.
    watcher = threading.Thread(target=probe_visibility)
    watcher.start()
    probe_proxy()
    watcher.join()
    print("[preflight] client done", flush=True)
