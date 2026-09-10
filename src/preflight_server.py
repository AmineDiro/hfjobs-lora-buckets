"""Preflight, server half: what a mounted Storage Bucket supports, and a target for the jobs proxy.

Runs in a `cpu-basic` Job with the shared bucket mounted read-write at `/lora` and port 8000 exposed. Two jobs
answer between them the four questions the disaggregated LoRA setup rests on, and every one of them is a
`hf jobs`/bucket property rather than anything about trl:

1. Does the mount support the calls `save_lora_adapter` makes -- `os.makedirs`, write, `os.rename` of a
   *directory*, `shutil.rmtree`? The atomic-rename publish is what stops vLLM reading a half-written adapter.
2. How long after a write does the *other* job's mount see it? `hf-mount` documents a 10s metadata TTL and a
   30s background poll, so this is the per-sync latency the trainer has to absorb.
3. Does an exposed port need the `Authorization` header, and can another Job reach it at all?
4. How long a response will the jobs proxy hold open? A rollout request occupies the connection for as long as
   generation takes, so a proxy response timeout is a hard cap on `max_completion_length`.

This half runs (1), publishes the beacons (2) reads, and serves the endpoints (3)/(4) probe.
"""

import hashlib
import http.server
import json
import mmap
import os
import shutil
import socketserver
import threading
import time


MOUNT = os.environ.get("MOUNT", "/lora")
ROOT = os.path.join(MOUNT, "preflight")
# An r=32 adapter on a 1.5B model is ~35 MB, r=1 is ~1 MB. Beacons carry the larger size so the visibility
# measurement includes shipping a realistic adapter payload, not just a directory entry.
PAYLOAD_MB = int(os.environ.get("PAYLOAD_MB", "35"))
BEACONS = int(os.environ.get("BEACONS", "12"))
BEACON_INTERVAL = float(os.environ.get("BEACON_INTERVAL", "20"))


def check(name, fn):
    """Run one filesystem capability check and print PASS/FAIL with the exception if it failed."""
    try:
        detail = fn()
        print(f"[fs] PASS  {name}" + (f"  ({detail})" if detail else ""), flush=True)
        return True
    except Exception as e:
        print(f"[fs] FAIL  {name}  {type(e).__name__}: {e}", flush=True)
        return False


def fs_probe():
    """Exercise every filesystem operation the adapter publish path performs, in the order it performs them."""
    print(f"[fs] probing {MOUNT}", flush=True)
    print(f"[fs] mount line: {[l.strip() for l in open('/proc/mounts') if MOUNT in l]}", flush=True)
    base = os.path.join(ROOT, "fs")
    shutil.rmtree(base, ignore_errors=True)

    check("makedirs", lambda: (os.makedirs(f"{base}/a/b", exist_ok=True), "")[1])
    check("write+close", lambda: (open(f"{base}/a/b/x.bin", "wb").write(b"x" * 4096), "4 KiB")[1])
    check("read back", lambda: f"{len(open(f'{base}/a/b/x.bin', 'rb').read())} bytes")
    check("listdir", lambda: str(os.listdir(f"{base}/a/b")))

    # The one that matters: `save_lora_adapter` stages into `.<name>.tmp` and publishes with a directory rename.
    check("rename dir", lambda: (os.rename(f"{base}/a/b", f"{base}/a/published"), "")[1])
    check("read after rename", lambda: f"{len(open(f'{base}/a/published/x.bin', 'rb').read())} bytes")
    # And the one that reclaims a version that fell out of the staleness window.
    check("rmtree", lambda: (shutil.rmtree(f"{base}/a/published"), "")[1])

    # Overwrite in place: hf-mount's default bucket write mode is documented as append-only streaming, so a
    # second write to an existing path may not be supported. The publish path never needs it (every version is a
    # fresh name), but a trainer checkpoint written to the same bucket does.
    check("write again", lambda: (open(f"{base}/over.bin", "wb").write(b"1" * 1024), "")[1])
    check("overwrite", lambda: (open(f"{base}/over.bin", "wb").write(b"2" * 2048), "")[1])
    check("overwrite size", lambda: f"{os.path.getsize(f'{base}/over.bin')} bytes (expect 2048)")

    # safetensors mmaps the file it loads, and vLLM loads the adapter through safetensors. A previous run on
    # this mount took a SIGBUS from two ranks mmapping a 7 GB checkpoint; an adapter is ~1000x smaller, but the
    # call has to work at all.
    def _mmap():
        with open(f"{base}/over.bin", "rb") as f:
            with mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) as m:
                return f"first byte {m[0:1]!r}"

    check("mmap read", _mmap)
    print("[fs] done", flush=True)


def publish_beacons():
    """Publish adapter-shaped directories on a fixed cadence, the way a weight sync publishes `trl-policy-vN`."""
    payload = os.urandom(PAYLOAD_MB * 1024 * 1024)
    digest = hashlib.sha256(payload).hexdigest()
    beacons = os.path.join(ROOT, "beacons")
    shutil.rmtree(beacons, ignore_errors=True)
    os.makedirs(beacons, exist_ok=True)
    for version in range(1, BEACONS + 1):
        dest = os.path.join(beacons, f"beacon-v{version}")
        tmp = os.path.join(beacons, f".beacon-v{version}.tmp")
        os.makedirs(tmp, exist_ok=True)
        t0 = time.time()
        with open(os.path.join(tmp, "payload.bin"), "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        t_written = time.time()
        # `meta.json` is written last and inside the staged directory, so a reader that can see it can already
        # see the payload -- the same ordering guarantee the atomic rename gives the real adapter.
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump({"version": version, "sha256": digest, "bytes": len(payload), "published_at": t_written}, f)
        try:
            os.rename(tmp, dest)
            how = "rename"
        except OSError as e:
            # If the mount cannot rename a directory, fall back to what the trainer would have to do instead.
            print(f"[beacon] rename failed ({e}); falling back to copytree", flush=True)
            shutil.copytree(tmp, dest)
            how = "copytree"
        print(
            f"[beacon] v{version} published via {how}: {PAYLOAD_MB} MiB write took {t_written - t0:.1f}s, "
            f"publish took {time.time() - t_written:.1f}s",
            flush=True,
        )
        time.sleep(BEACON_INTERVAL)
    print("[beacon] done", flush=True)


class Handler(http.server.BaseHTTPRequestHandler):
    """Endpoints the client half probes the jobs proxy with. `/slow` is the one that matters."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, body: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/slow"):
            # Sleep, then answer. A generation request looks exactly like this from the proxy's point of view:
            # one connection held open with nothing on it until the whole completion is ready.
            seconds = float(self.path.partition("s=")[2] or 1)
            time.sleep(seconds)
            self._send(json.dumps({"slept": seconds}).encode())
        elif self.path.startswith("/big"):
            n = int(self.path.partition("n=")[2] or 1024)
            self._send(b"." * n)
        else:
            self._send(json.dumps({"ok": True, "headers": dict(self.headers)}).encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        self._send(json.dumps({"received_bytes": len(body)}).encode())


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    fs_probe()
    threading.Thread(target=publish_beacons, daemon=True).start()
    print("[http] serving on 0.0.0.0:8000", flush=True)
    Server(("0.0.0.0", 8000), Handler).serve_forever()
