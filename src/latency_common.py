"""Shared pieces of the bucket-latency probe: the write, the two-sided watch, and a tiny HTTP helper.

Both Jobs mount the same Storage Bucket at `/lora`. A trial writes `size` random bytes on one side and, on the
other side, times two things from a single monotonic clock: when the object becomes readable through the bucket
HTTP API (`t_api`, the upload flush) and when the reader's own FUSE mount admits the entry exists (`t_mount`).
`t_mount - t_api` is the pure metadata-cache cost, which is the number the experiment is for.

Stdlib only, so both Jobs start on a bare `python:3.12` image with no pip step.
"""

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

MOUNT = os.environ.get("MOUNT", "/lora")
BUCKET = os.environ.get("BUCKET", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
POLL_S = float(os.environ.get("POLL_S", "0.25"))
WARM_S = float(os.environ.get("WARM_S", "0.5"))
WATCH_TIMEOUT_S = float(os.environ.get("WATCH_TIMEOUT_S", "180"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def http(method, url, headers=None, data=None, timeout=60):
    """One HTTP call that never follows redirects. Returns (status, headers, body); raises only on transport errors."""
    req = urllib.request.Request(url, method=method, data=data, headers=headers or {})
    try:
        with _opener.open(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def hub_headers():
    return {"Authorization": f"Bearer {HF_TOKEN}"}


def mount_line():
    return [l.strip() for l in open("/proc/mounts") if f" {MOUNT} " in l]


def make_payload(size):
    """Fresh random bytes every call: Xet deduplicates at chunk level, so reused bytes would measure a dedup hit."""
    payload = os.urandom(size)
    return payload, hashlib.sha256(payload).hexdigest()


def write_payload(parent, key, size, mode, payload=None):
    """Write `size` fresh random bytes to `<MOUNT>/<parent>/<key>/payload.bin`, either directly or via a staged
    directory plus `os.rename` (what `save_lora_adapter` does). Times every step separately."""
    root = os.path.join(MOUNT, parent)
    if payload is None:
        payload, sha = make_payload(size)
    else:
        sha = hashlib.sha256(payload).hexdigest()
    final_dir = os.path.join(root, key)
    stage = os.path.join(root, f".{key}.tmp") if mode == "rename" else final_dir
    t_start = time.monotonic()
    os.makedirs(stage, exist_ok=True)
    t_mkdir = time.monotonic()
    f = open(os.path.join(stage, "payload.bin"), "wb")
    f.write(payload)
    f.flush()
    t_written = time.monotonic()
    os.fsync(f.fileno())
    t_fsync = time.monotonic()
    f.close()
    t_close = time.monotonic()
    rename_s = 0.0
    if mode == "rename":
        os.rename(stage, final_dir)
        rename_s = time.monotonic() - t_close
    return {
        "key": key,
        "parent": parent,
        "size": size,
        "mode": mode,
        "sha256": sha,
        "rel_path": f"{parent}/{key}/payload.bin",
        "rel_tmp_path": f"{parent}/.{key}.tmp/payload.bin" if mode == "rename" else None,
        "mkdir_s": t_mkdir - t_start,
        "write_s": t_written - t_mkdir,  # open -> last write returned
        "fsync_s": t_fsync - t_written,
        "close_s": t_close - t_fsync,
        "rename_s": rename_s,
        "total_s": time.monotonic() - t_start,
    }


def api_paths_info(paths):
    """Existence check for several bucket paths in one round trip. Returns {path: entry} for those that exist."""
    body = urllib.parse.urlencode([("paths", p) for p in paths]).encode()
    st, _, raw = http(
        "POST",
        f"{HUB}/api/buckets/{BUCKET}/paths-info",
        headers={**hub_headers(), "Content-Type": "application/x-www-form-urlencoded"},
        data=body,
        timeout=30,
    )
    if st != 200:
        return None, st
    return {e["path"]: e for e in json.loads(raw)}, st


def api_download(rel_path):
    """GET the object through the bucket resolve route: hub 302 -> presigned CAS URL (fetched without auth).
    Returns (status, bytes_or_None)."""
    url = f"{HUB}/buckets/{BUCKET}/resolve/{urllib.parse.quote(rel_path, safe='')}"
    st, hdrs, body = http("GET", url, headers=hub_headers(), timeout=120)
    if st in (301, 302, 303, 307, 308):
        loc = hdrs.get("Location") or hdrs.get("location")
        st, _, body = http("GET", loc, timeout=300)
    return st, (body if st == 200 else None)


class Watch:
    """Poll the reader's mount and the bucket API for one object. Every time is seconds after `t0` (monotonic on
    this machine). `probe` is how the mount is polled: `open` the exact path, or `listdir` the parent and look
    for the key. `warm=True` adds a thread listing the parent every WARM_S from the start."""

    def __init__(self, parent, key, size, sha256, rel_path, rel_tmp_path=None, probe="open", warm=False, t0=None):
        self.parent, self.key, self.size, self.sha256 = parent, key, size, sha256
        self.rel_path, self.rel_tmp_path, self.probe, self.warm = rel_path, rel_tmp_path, probe, warm
        self.t0 = time.monotonic() if t0 is None else t0
        self.root = os.path.join(MOUNT, parent)
        self.path = os.path.join(self.root, key, "payload.bin")
        self.r = {
            "t_api": None,  # paths-info lists the final path
            "t_api_tmp": None,  # paths-info lists the staged .tmp path (rename mode only)
            "t_api_get": None,  # a real GET of the final path returned 200
            "api_size_first_seen": None,
            "api_download_s": None,
            "api_verified": None,
            "api_errors": 0,
            "t_mount": None,  # mount admits the entry exists (per `probe`)
            "mount_size_first_seen": None,
            "mount_open_after_listdir_s": None,
            "mount_read_s": None,
            "mount_verified": None,
            "mount_errors": 0,
            "warm_listings": 0,
            "probe": probe,
            "warm": warm,
        }
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._mount_loop, daemon=True),
            threading.Thread(target=self._api_loop, daemon=True),
        ]
        if warm:
            self._threads.append(threading.Thread(target=self._warm_loop, daemon=True))
        for t in self._threads:
            t.start()

    def _now(self):
        return time.monotonic() - self.t0

    def _expired(self):
        return self._now() > WATCH_TIMEOUT_S

    def _warm_loop(self):
        while not self._stop.is_set() and self.r["t_mount"] is None and not self._expired():
            try:
                os.listdir(self.root)
                self.r["warm_listings"] += 1
            except OSError:
                pass
            time.sleep(WARM_S)

    def _mount_loop(self):
        r = self.r
        while not self._stop.is_set() and not self._expired():
            try:
                if self.probe == "listdir":
                    if self.key in os.listdir(self.root):
                        r["t_mount"] = self._now()
                        t = time.monotonic()
                        # A parent listing that shows the entry does not prove the child path resolves yet.
                        while not self._expired():
                            try:
                                open(self.path, "rb").close()
                                break
                            except OSError:
                                time.sleep(POLL_S)
                        r["mount_open_after_listdir_s"] = time.monotonic() - t
                else:
                    open(self.path, "rb").close()
                    r["t_mount"] = self._now()
            except OSError:
                pass
            except Exception:
                r["mount_errors"] += 1
            if r["t_mount"] is not None:
                break
            time.sleep(POLL_S)
        if r["t_mount"] is None:
            return
        try:
            r["mount_size_first_seen"] = os.stat(self.path).st_size
            t = time.monotonic()
            h = hashlib.sha256()
            with open(self.path, "rb") as f:
                for chunk in iter(lambda: f.read(8 << 20), b""):
                    h.update(chunk)
            r["mount_read_s"] = time.monotonic() - t
            r["mount_verified"] = h.hexdigest() == self.sha256
        except Exception as e:
            r["mount_errors"] += 1
            r["mount_verified"] = f"error: {type(e).__name__}: {e}"

    def _api_loop(self):
        r = self.r
        paths = [self.rel_path] + ([self.rel_tmp_path] if self.rel_tmp_path else [])
        while not self._stop.is_set() and not self._expired():
            try:
                seen, st = api_paths_info(paths)
                now = self._now()
                if seen is None:
                    r["api_errors"] += 1
                else:
                    if self.rel_tmp_path and self.rel_tmp_path in seen and r["t_api_tmp"] is None:
                        r["t_api_tmp"] = now
                    if self.rel_path in seen:
                        r["t_api"] = now
                        r["api_size_first_seen"] = seen[self.rel_path].get("size")
                        break
            except Exception:
                r["api_errors"] += 1
            time.sleep(POLL_S)
        if r["t_api"] is None:
            return
        # Now prove it is actually readable, not just listed, and time the transfer.
        while not self._stop.is_set() and not self._expired():
            try:
                t = time.monotonic()
                st, body = api_download(self.rel_path)
                if st == 200:
                    r["t_api_get"] = self._now()
                    r["api_download_s"] = time.monotonic() - t
                    r["api_verified"] = hashlib.sha256(body).hexdigest() == self.sha256
                    break
                r["api_errors"] += 1
            except Exception:
                r["api_errors"] += 1
            time.sleep(POLL_S)

    def done(self):
        return not any(t.is_alive() for t in self._threads[:2])

    def wait(self, timeout=None):
        deadline = time.monotonic() + (timeout if timeout is not None else WATCH_TIMEOUT_S + 30)
        for t in self._threads[:2]:
            t.join(max(0.0, deadline - time.monotonic()))
        self._stop.set()
        self.r["done"] = self.done()
        self.r["elapsed_s"] = self._now()
        return dict(self.r)
