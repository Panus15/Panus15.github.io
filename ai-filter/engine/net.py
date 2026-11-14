"""One place where the filter touches the network, and the only place.

Two reasons it is isolated:

1. TESTS. Every test in this project must pass with the network denied (CI
   proves it). Checkers therefore take a `fetcher` argument and tests hand them
   a FakeFetcher. Nothing else in the package may import urllib/socket/http.

2. SSRF. The URLs we fetch come from the text of an AI answer, i.e. from
   untrusted input. Without a guard, an answer containing
   http://169.254.169.254/latest/meta-data/ would make THIS tool read a cloud
   metadata service and then quote it in a report. So: http(s) only, and every
   hop of every redirect is re-resolved and refused if it lands on a loopback,
   private, link-local, reserved or multicast address.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol

USER_AGENT = "ai-filter/0.1 (fact-check tool; +https://github.com/panus15/panus15.github.io)"
MAX_BYTES_DEFAULT = 512 * 1024
MAX_REDIRECTS = 5


@dataclass(frozen=True)
class FetchResult:
    url: str                        # the URL asked for
    final_url: str                  # after redirects (== url if none)
    status: Optional[int]           # HTTP status; None when no response was obtained
    body: bytes = b""               # truncated to max_bytes
    # None on success. Otherwise one of:
    #   "dns"      the host does not exist (NXDOMAIN)            -> positive evidence of a bad link
    #   "blocked"  refused by our own SSRF guard / bad scheme    -> says nothing about the link
    #   "timeout" | "ssl" | "network" | "other"                  -> our failure, says nothing about the link
    error: Optional[str] = None
    detail: str = ""
    # Response headers, keys lower-cased (e.g. "last-modified"). Empty for fakes that don't set it.
    headers: Mapping[str, str] = field(default_factory=dict)


class Fetcher(Protocol):
    def fetch(self, url: str, *, headers: Optional[Mapping[str, str]] = None,
              timeout: float = 10.0, max_bytes: int = MAX_BYTES_DEFAULT) -> FetchResult: ...


def _is_nxdomain(e: socket.gaierror) -> bool:
    return e.errno in {getattr(socket, n) for n in ("EAI_NONAME", "EAI_NODATA") if hasattr(socket, n)}


_CANARY_HOSTS = ("example.com", "iana.org")
_resolver_ok_until = 0.0


def _resolver_healthy() -> bool:
    """Can THIS machine resolve a name that certainly exists? Cached for a minute.

    Why it exists: "NXDOMAIN" is only evidence that a link is fake if our resolver is
    trustworthy. Behind an egress proxy or in a sandbox, local DNS often fails for
    EVERY external name, which would make every real link look like an invented
    domain. So a name that fails is only called non-existent when a canary resolves.
    """
    global _resolver_ok_until
    import time
    now = time.monotonic()
    if now < _resolver_ok_until:
        return True
    for h in _CANARY_HOSTS:
        try:
            socket.getaddrinfo(h, 443, proto=socket.IPPROTO_TCP)
            _resolver_ok_until = now + 60.0
            return True
        except socket.gaierror:
            continue
    return False


_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".lan", ".home.arpa", ".intranet", ".corp")


def _bad_ip(ip) -> bool:
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified)


def _proxied(url: str, host: str) -> bool:
    """True if urllib will send this request through an HTTP(S) proxy (env-configured)."""
    scheme = urllib.parse.urlsplit(url).scheme
    try:
        return bool(urllib.request.getproxies().get(scheme)) and not urllib.request.proxy_bypass(host)
    except Exception:
        return False


def check_public_host(url: str) -> Optional[str]:
    """Return None if `url` is safe to fetch, else a reason string.

    Sentinels (not human reasons): "dns" = the name verifiably does not exist;
    "resolver" = we could not resolve it and that says nothing about the link.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return f"scheme {parts.scheme!r} not allowed"
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        return "no host"
    try:                                   # an IP literal needs no DNS at all
        ip = ipaddress.ip_address(host)
        return f"{host} is a non-public address" if _bad_ip(ip) else None
    except ValueError:
        pass
    # Name-based refusals that do not depend on resolution.
    if host == "localhost" or host.endswith(_LOCAL_SUFFIXES) or "." not in host:
        return f"{host} is a local/internal name"
    if _proxied(url, host):
        # An egress proxy resolves and connects on our behalf; local DNS is irrelevant (and often
        # unavailable). We cannot see the address it will use, so the proxy's own policy has to stop
        # internal targets -- documented in the README. The name checks above still apply.
        return None
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        # Only "no such name" from a HEALTHY resolver is evidence the link is bad.
        if _is_nxdomain(e) and _resolver_healthy():
            return "dns"
        return "resolver"
    for info in infos:
        if _bad_ip(ipaddress.ip_address(info[4][0].split("%")[0])):
            return f"{host} resolves to non-public address {info[4][0]}"
    return None


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        problem = check_public_host(urllib.parse.urljoin(req.full_url, newurl))
        if problem:
            raise urllib.error.URLError(f"blocked redirect: {problem}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class UrllibFetcher:
    """The real fetcher. stdlib only, like the rest of this repository."""

    def fetch(self, url, *, headers=None, timeout=10.0, max_bytes=MAX_BYTES_DEFAULT):
        problem = check_public_host(url)
        if problem == "dns":
            return FetchResult(url, url, None, error="dns", detail="host does not resolve")
        if problem == "resolver":
            return FetchResult(url, url, None, error="network", detail="DNS lookup failed (resolver unavailable)")
        if problem:
            return FetchResult(url, url, None, error="blocked", detail=problem)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
        opener = urllib.request.build_opener(_GuardedRedirect)
        try:
            with opener.open(req, timeout=timeout) as resp:
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                return FetchResult(url, resp.geturl(), resp.status, resp.read(max_bytes), headers=hdrs)
        except urllib.error.HTTPError as e:     # 4xx/5xx still carry a real status AND often a
            try:                                 # meaningful body (doi.org answers 404 + JSON for unregistered DOIs)
                body = e.read(max_bytes)
            except Exception:
                body = b""
            return FetchResult(url, e.geturl() or url, e.code, body,
                               headers={k.lower(): v for k, v in (e.headers or {}).items()})
        except urllib.error.URLError as e:
            reason = e.reason
            if isinstance(reason, socket.gaierror):
                kind = "dns" if _is_nxdomain(reason) else "network"
                return FetchResult(url, url, None, error=kind, detail=str(reason))
            if isinstance(reason, socket.timeout) or "timed out" in str(reason):
                return FetchResult(url, url, None, error="timeout", detail=str(reason))
            if "CERTIFICATE" in str(reason).upper() or "SSL" in str(reason).upper():
                return FetchResult(url, url, None, error="ssl", detail=str(reason))
            if "blocked redirect" in str(reason):
                return FetchResult(url, url, None, error="blocked", detail=str(reason))
            return FetchResult(url, url, None, error="network", detail=str(reason))
        except socket.timeout:
            return FetchResult(url, url, None, error="timeout", detail="timed out")
        except Exception as e:                  # never let a checker crash the run
            return FetchResult(url, url, None, error="other", detail=repr(e))


class FakeFetcher:
    """For tests. `routes` maps URL -> FetchResult, or -> callable(url) -> FetchResult.
    Unknown URLs return error='network' so a test that forgets a route fails loudly
    on the assertion rather than silently hitting the internet."""

    def __init__(self, routes: Mapping[str, object]):
        self.routes = dict(routes)
        self.calls: list[str] = []

    def fetch(self, url, *, headers=None, timeout=10.0, max_bytes=MAX_BYTES_DEFAULT):
        self.calls.append(url)
        r = self.routes.get(url)
        if r is None:
            return FetchResult(url, url, None, error="network", detail="FakeFetcher: no route")
        return r(url) if callable(r) else r


class OfflineFetcher:
    """`--offline` mode: never touches the network, and says so. Every lookup
    therefore comes back UNVERIFIED (error='network'), never CONTRADICTED."""

    def fetch(self, url, *, headers=None, timeout=10.0, max_bytes=MAX_BYTES_DEFAULT):
        return FetchResult(url, url, None, error="network", detail="offline mode")
