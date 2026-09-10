"""Bucket-latency probe, side A: an HTTP server on an exposed port that writes on request and can watch on request.

Runs in a `cpu-basic` Job with the shared bucket mounted read-write at `/lora` and port 8000 exposed. The reader
(side B) drives everything so that every duration is measured on one clock.

    GET  /health                       -> {"ok": true}
    GET  /mount                        -> the /proc/mounts line(s) for /lora
    POST /write?parent=&key=&size=&mode= -> writes `size` random bytes, returns the write timings + sha256
    POST /watch?parent=&key=&size=&sha256=&rel_path=&rel_tmp_path=&probe=&warm=
                                       -> starts polling this side's mount and the API; t0 = request arrival
    GET  /watch?key=                   -> current watch result (times are seconds after t0)
    POST /shutdown                     -> exits so the Job stops billing
"""

import http.server
import json
import os
import socketserver
import threading
import urllib.parse

import latency_common as lc

WATCHES = {}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[http] {fmt % args}", flush=True)

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _q(self):
        return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}

    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        q = self._q()
        if route == "/health":
            self._send({"ok": True})
        elif route == "/mount":
            self._send({"mount": lc.mount_line(), "hostname": os.uname().nodename})
        elif route == "/watch":
            w = WATCHES.get(q.get("key"))
            if w is None:
                self._send({"error": "no such watch"}, 404)
            else:
                r = dict(w.r)
                r["done"] = w.done()
                r["elapsed_s"] = w._now()
                self._send(r)
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        route = urllib.parse.urlparse(self.path).path
        q = self._q()
        n = int(self.headers.get("Content-Length", 0))
        if n:
            self.rfile.read(n)
        if route == "/write":
            try:
                res = lc.write_payload(q.get("parent", "probe"), q["key"], int(q["size"]), q.get("mode", "rename"))
                print(f"[write] {res['key']} {res['size']}B {res['mode']} write={res['write_s']:.3f}s "
                      f"fsync={res['fsync_s']:.3f}s close={res['close_s']:.3f}s rename={res['rename_s']:.3f}s", flush=True)
                self._send(res)
            except Exception as e:
                self._send({"error": f"{type(e).__name__}: {e}"}, 500)
        elif route == "/watch":
            w = lc.Watch(
                parent=q.get("parent", "probe-rev"),
                key=q["key"],
                size=int(q["size"]),
                sha256=q["sha256"],
                rel_path=q["rel_path"],
                rel_tmp_path=q.get("rel_tmp_path") or None,
                probe=q.get("probe", "open"),
                warm=q.get("warm", "0") == "1",
            )
            WATCHES[q["key"]] = w
            self._send({"watching": q["key"]})
        elif route == "/shutdown":
            self._send({"bye": True})
            threading.Thread(target=lambda: (__import__("time").sleep(0.5), os._exit(0)), daemon=True).start()
        else:
            self._send({"error": "not found"}, 404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    for parent in ("probe", "probe-rev", "results"):
        os.makedirs(os.path.join(lc.MOUNT, parent), exist_ok=True)
    print(f"[mount] {lc.mount_line()}", flush=True)
    print(f"[http] serving on 0.0.0.0:8000 bucket={lc.BUCKET} token={'yes' if lc.HF_TOKEN else 'NO'}", flush=True)
    Server(("0.0.0.0", 8000), Handler).serve_forever()
