"""Tests for the shared core: types and the SSRF guard. Offline.

Run: python3 tests/test_core.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.net import FakeFetcher, FetchResult, UrllibFetcher, check_public_host


def test_guard_refuses_internal_targets():
    for u in ("http://127.0.0.1/", "http://localhost/", "http://169.254.169.254/latest/meta-data/",
              "http://10.0.0.5/", "http://192.168.1.1/", "http://[::1]/", "http://0.0.0.0/"):
        assert check_public_host(u) not in (None, "dns"), u
    for u in ("file:///etc/passwd", "ftp://example.com/", "gopher://x/", "javascript:alert(1)"):
        assert check_public_host(u) not in (None, "dns"), u


def test_real_fetcher_reports_blocked_not_dns_for_internal():
    r = UrllibFetcher().fetch("http://127.0.0.1:9/")
    assert r.error == "blocked" and r.status is None


def test_fake_fetcher_unknown_route_is_a_failure_not_a_pass():
    f = FakeFetcher({"http://a/": FetchResult("http://a/", "http://a/", 200, b"x")})
    assert f.fetch("http://a/").status == 200
    assert f.fetch("http://b/").error == "network"
    assert f.calls == ["http://a/", "http://b/"]


def _with_getaddrinfo(exc, fn):
    import socket
    real = socket.getaddrinfo
    def boom(*a, **k): raise exc
    socket.getaddrinfo = boom
    try:
        return fn()
    finally:
        socket.getaddrinfo = real


def test_nxdomain_is_dns_but_resolver_failure_is_not():
    import socket
    assert _with_getaddrinfo(socket.gaierror(socket.EAI_NONAME, "nx"),
                             lambda: check_public_host("http://nope.invalid/")) == "dns"
    assert _with_getaddrinfo(socket.gaierror(socket.EAI_AGAIN, "try again"),
                             lambda: check_public_host("http://example.com/")) == "resolver"
    r = _with_getaddrinfo(socket.gaierror(socket.EAI_AGAIN, "try again"),
                          lambda: UrllibFetcher().fetch("http://example.com/"))
    assert r.error == "network"   # NOT "dns": a down resolver must never read as a dead link


def test_http_error_body_is_kept():
    import http.server, threading
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"responseCode":100}')
        def log_message(self, *a): pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    import engine.net as net
    real = net.check_public_host
    net.check_public_host = lambda u: None      # loopback fixture: bypass the guard for this test only
    try:
        r = UrllibFetcher().fetch(f"http://127.0.0.1:{srv.server_port}/")
    finally:
        net.check_public_host = real; srv.shutdown()
    assert r.status == 404 and r.body == b'{"responseCode":100}'


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
