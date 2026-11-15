"""Code-reference checker: do the packages and functions an AI answer uses actually exist?

scan(text, claims, *, fetcher=None, registries=True, today=None) -> list[Finding], module="codecheck".

A classic AI failure is code that imports a library, or calls a function, that DOES NOT
EXIST.  Attackers register those invented names ("slopsquatting"), so installing what the
model made up can be a supply-chain compromise.  This is checkable on any programming
question without a knowledge base.  The checker looks at CODE claims only and emits NEW
claims of kind API_REF (ids "k1", "k2", ... in document order) whose span is the exact name
inside the ORIGINAL text.

CHECK A -- Python standard-library attributes (offline, deterministic).
  `import json` + `json.parse(...)`, `os.getcwdu()`, `from os import getcwdu`, `import os.foo`.
  Done by introspection in an ISOLATED SUBPROCESS (`python -I -S -B`, cwd = fresh temp dir,
  minimal environment, stdin /dev/null, 5 s timeout, output via a size-limited file, rlimits,
  own process group).  Nothing is ever imported in this process.  Modules whose import has
  side effects (antigravity, this, tkinter, idlelib, turtle, webbrowser, test, ensurepip,
  venv, ... and every `*.__main__`) are on a blocklist and are never imported.  A missing
  attribute is UNVERIFIED, never CONTRADICTED: whether it exists depends on the Python
  version, and the one we run is not necessarily the one the user has.  Present -> no finding.
  Not checked: names rebound in the snippet (assigned, parameters, defs, classes, loop
  variables, shadowing imports), attributes assigned in the snippet, chains deeper than 3.

CHECK B -- package existence (only if `registries` is true; needs the network via `fetcher`).
  PyPI  https://pypi.org/pypi/<name>/json      npm  https://registry.npmjs.org/<name>
  Names come from (1) explicit install commands (`pip install x`, `python -m pip install x`,
  `npm i x`, `yarn add x`, `pnpm add x`, `npx x`), (2) Python third-party imports (import name
  mapped to a PyPI distribution through a built-in table of well-known mismatches such as
  cv2 -> opencv-python), (3) JS `require('x')` / `import ... from 'x'` / `import('x')`.
    200  -> VERIFIED, with the caveat below
    404  -> CONTRADICTED only where a 404 really means "invented": an explicit install of a
            distribution name, or a JS import (npm name == specifier root).  A Python IMPORT
            name that 404s is only UNVERIFIED (import names can differ from distribution
            names and the module may be local).  Also only UNVERIFIED when the command names a
            custom index/registry (the package may be private).
    anything else / any failure (timeout, offline, 429, 5xx, captive portal) -> UNVERIFIED:
            that is our failure, not the answer's.
  Names are validated against the PyPI/npm grammars (length <= 214) BEFORE any URL is built;
  anything odd is skipped, never sent.  Only the package name leaves this process -- never
  the answer text or the code.  Lookups are de-duplicated and capped at 30 per run; beyond
  that the finding is UNVERIFIED ("lookup limit reached").

POLICY.  This proves a NAME exists, not that the API usage is right.  Typosquats, look-alike
and malicious packages exist, and a local module can share a name with a PyPI project, so a
VERIFIED here is weak and its reason says so.  Our failures are UNVERIFIED.  Legacy Python-2 and
removed-stdlib module names (urllib2, imp, distutils, ...) and well-known local-alias roots in
JS (`components/...`, `src/...`) are skipped rather than looked up.  The `today` argument is
accepted for interface symmetry with other checkers and is unused.
"""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

from .models import Claim, ClaimKind, Finding, Status
from .net import UrllibFetcher

MODULE = "codecheck"

MAX_LOOKUPS = 30
MAX_CODE_CHARS = 200_000
MAX_QUERIES = 300
MAX_NAME_LEN = 214
INTROSPECT_TIMEOUT = 5.0
LOOKUP_TIMEOUT = 8.0
LOOKUP_BUDGET_S = 90.0       # total wall-clock for all registry lookups in one scan
RESULT_CAP = 1 << 20
MAX_CHAIN_DEPTH = 3

PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"

# --------------------------------------------------------------------------- tables

_PY_FAMILY = {"python", "py", "python3", "py3"}
_JS_FAMILY = {"javascript", "js", "node", "nodejs", "jsx", "mjs", "cjs", "typescript", "ts", "tsx"}
_SH_FAMILY = {"bash", "sh", "shell", "console", "zsh", "terminal", "shell-session", "shellsession",
              "sh-session", "shsession"}

# Modules whose import has side effects (opens a browser, prints, starts a GUI, touches the
# terminal, runs a test suite, ...).  Never imported, even in the sandboxed child.  A skipped
# module can only cause a missed finding, never a false one.
BLOCKED_MODULES = frozenset({
    "antigravity", "this", "tkinter", "_tkinter", "idlelib", "turtle", "turtledemo", "webbrowser",
    "pydoc_data", "pydoc", "test", "ensurepip", "venv", "__hello__", "__phello__", "__main__",
    "site", "sitecustomize", "usercustomize", "lib2to3", "pdb", "bdb", "pstats", "profile",
    "cProfile", "trace", "timeit", "zipapp", "_pyrepl", "readline", "rlcompleter", "curses",
    "_curses", "_curses_panel", "msilib", "winreg", "_winapi", "msvcrt", "nt",
    "_testcapi", "_testinternalcapi", "_testbuffer", "_testmultiphase", "_xxtestfuzz",
    "_xxsubinterpreters", "_xxinterpchannels", "xxlimited", "xxlimited_35", "xxsubtype",
})

# Formerly-stdlib and Python-2 module names.  They are not in sys.stdlib_module_names of a new
# interpreter, but a PyPI lookup would be meaningless (or misleading), so they are skipped.
_LEGACY_MODULES = frozenset({
    "imp", "distutils", "asyncore", "asynchat", "smtpd", "cgi", "cgitb", "crypt", "nntplib",
    "telnetlib", "pipes", "sndhdr", "spwd", "sunau", "uu", "xdrlib", "aifc", "audioop", "chunk",
    "imghdr", "mailcap", "msilib", "nis", "ossaudiodev", "lib2to3", "binhex", "formatter",
    "parser", "symbol", "macpath", "dummy_threading", "_dummy_thread",
    "urllib2", "urlparse", "ConfigParser", "Queue", "StringIO", "cStringIO", "cPickle", "Tkinter",
    "tkMessageBox", "tkFileDialog", "httplib", "HTMLParser", "xmlrpclib", "SocketServer", "commands",
    "__builtin__", "thread", "md5", "sha", "sets", "exceptions", "UserDict", "copy_reg", "Cookie",
    "cookielib", "BaseHTTPServer", "SimpleHTTPServer", "htmlentitydefs", "anydbm", "dbhash",
    "repr", "itertools_izip", "future_builtins", "urllib3_placeholder",
})

# import name -> PyPI distribution name, for the well-known cases where they differ.
IMPORT_TO_DIST: dict[str, str] = {
    "cv2": "opencv-python", "PIL": "pillow", "sklearn": "scikit-learn", "yaml": "pyyaml",
    "bs4": "beautifulsoup4", "dateutil": "python-dateutil", "skimage": "scikit-image",
    "Crypto": "pycryptodome", "Cryptodome": "pycryptodomex", "OpenSSL": "pyopenssl",
    "jwt": "pyjwt", "dotenv": "python-dotenv", "attr": "attrs", "serial": "pyserial",
    "usb": "pyusb", "magic": "python-magic", "MySQLdb": "mysqlclient", "psycopg2": "psycopg2-binary",
    "git": "gitpython", "docx": "python-docx", "pptx": "python-pptx", "fitz": "pymupdf",
    "wx": "wxpython", "gi": "pygobject", "zmq": "pyzmq", "slugify": "python-slugify",
    "jose": "python-jose", "socks": "pysocks", "Xlib": "python-xlib", "nacl": "pynacl",
    "bson": "pymongo", "grpc": "grpcio", "pkg_resources": "setuptools", "umap": "umap-learn",
    "faiss": "faiss-cpu", "mpl_toolkits": "matplotlib", "dns": "dnspython",
    "telegram": "python-telegram-bot", "discord": "discord.py", "win32api": "pywin32",
    "win32con": "pywin32", "win32com": "pywin32", "win32gui": "pywin32", "pythoncom": "pywin32",
    "pywintypes": "pywin32", "rest_framework": "djangorestframework", "flask_cors": "flask-cors",
    "flask_sqlalchemy": "flask-sqlalchemy", "flask_login": "flask-login",
    "flask_migrate": "flask-migrate", "flask_wtf": "flask-wtf", "flask_restful": "flask-restful",
    "jinja2": "jinja2", "google.protobuf": "protobuf", "google.cloud.storage": "google-cloud-storage",
    "google.cloud.bigquery": "google-cloud-bigquery", "google.generativeai": "google-generativeai",
    "google.oauth2": "google-auth", "googleapiclient": "google-api-python-client",
    "ruamel.yaml": "ruamel.yaml", "cairo": "pycairo", "memcache": "python-memcached",
    "ldap": "python-ldap", "snappy": "python-snappy", "Image": "pillow", "markdown": "markdown",
    "mx": "egenix-mx-base", "pyaudio": "pyaudio", "speech_recognition": "speechrecognition",
    "sqlalchemy": "sqlalchemy", "tensorflow": "tensorflow", "torch": "torch",
    "langchain_openai": "langchain-openai", "langchain_community": "langchain-community",
    "pydantic_settings": "pydantic-settings", "typing_extensions": "typing-extensions",
    "importlib_metadata": "importlib-metadata", "ibm_db": "ibm-db", "Levenshtein": "levenshtein",
    "sentence_transformers": "sentence-transformers", "azure.storage.blob": "azure-storage-blob",
    "azure.identity": "azure-identity", "pymysql": "pymysql", "yaml_include": "pyyaml-include",
    "lxml": "lxml", "tabulate": "tabulate", "regex": "regex", "decouple": "python-decouple",
    "multipart": "python-multipart", "engineio": "python-engineio", "socketio": "python-socketio",
    "ujson": "ujson", "Bio": "biopython", "matplotlib": "matplotlib", "cx_Oracle": "cx-oracle",
}
# Namespace-package roots: the top-level name alone says nothing about a PyPI project.
_NAMESPACE_ROOTS = frozenset({"google", "azure", "backports", "zope", "ruamel", "mpl_toolkits",
                              "jaraco", "sphinxcontrib", "paste", "repoze"})

NODE_BUILTINS = frozenset({
    "assert", "assert/strict", "async_hooks", "buffer", "child_process", "cluster", "console",
    "constants", "crypto", "dgram", "diagnostics_channel", "dns", "dns/promises", "domain",
    "events", "fs", "fs/promises", "http", "http2", "https", "inspector", "inspector/promises",
    "module", "net", "os", "path", "path/posix", "path/win32", "perf_hooks", "process", "punycode",
    "querystring", "readline", "readline/promises", "repl", "stream", "stream/consumers",
    "stream/promises", "stream/web", "string_decoder", "sys", "timers", "timers/promises", "tls",
    "trace_events", "tty", "url", "util", "util/types", "v8", "vm", "wasi", "worker_threads", "zlib",
})
# Roots that in real projects are far more often a tsconfig/webpack alias or a local folder
# than an npm package: never looked up (a 404 would be a false accusation, a 200 meaningless).
_JS_LOCAL_ROOTS = frozenset({
    "src", "app", "lib", "components", "utils", "util", "config", "pages", "hooks", "store", "services",
    "assets", "styles", "helpers", "models", "controllers", "routes", "middleware", "public", "test",
    "tests", "server", "client", "common", "shared", "types", "api", "db", "context", "constants",
    "features", "modules", "actions", "reducers", "layouts", "views", "templates", "data", "core",
    "domain", "infra", "internal", "libs", "pkg", "dist", "build", "mocks", "fixtures", "theme",
})
_NPX_BIN_SKIP = frozenset({"tsc", "ng", "tsserver", "node", "npm", "npx", "yarn", "pnpm"})

# --------------------------------------------------------------------------- name grammars

_PYPI_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_NPM_UNSCOPED = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]*$")
_NPM_SCOPED = re.compile(r"^@[a-z0-9][a-z0-9._~-]*/[a-z0-9][a-z0-9._~-]*$")
_PY_IDENT_TOP = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_FILE_EXT = (".txt", ".whl", ".zip", ".tar.gz", ".tgz", ".tar", ".cfg", ".toml", ".in", ".py", ".egg",
             ".json", ".yml", ".yaml", ".lock", ".md", ".sh", ".js", ".ts")


def _valid_pypi(name: str) -> bool:
    return (0 < len(name) <= MAX_NAME_LEN and bool(_PYPI_NAME.match(name))
            and any(c.isalpha() for c in name) and not name.lower().endswith(_FILE_EXT))


def _valid_npm(name: str) -> bool:
    if not (0 < len(name) <= MAX_NAME_LEN) or not any(c.isalpha() for c in name):
        return False
    if name in ("node_modules", "favicon.ico"):
        return False
    return bool(_NPM_SCOPED.match(name)) if name.startswith("@") else bool(_NPM_UNSCOPED.match(name))


def _pypi_norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# --------------------------------------------------------------------------- text helpers

def _lines(text: str) -> list[tuple[int, int, int]]:
    """[(start, content_end, next_start)] handling \\n, \\r\\n and \\r."""
    out = []
    pos, n = 0, len(text)
    rx = re.compile(r"[^\r\n]*")
    while pos < n:
        end = rx.match(text, pos).end()
        if text.startswith("\r\n", end):
            nxt = end + 2
        elif end < n:
            nxt = end + 1
        else:
            nxt = end
        out.append((pos, end, nxt))
        pos = nxt
    return out


_FENCE_OPEN = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\r\n]*)$")


def _body_region(ct: str) -> tuple[int, int]:
    """[start, end) of the code inside the claim text, i.e. without the fence lines.
    Hand-made claims without fences are their own body."""
    lines = _lines(ct)
    if not lines:
        return 0, 0
    s, e, nxt = lines[0]
    m = _FENCE_OPEN.match(ct[s:e])
    if not m or (m.group(1)[0] == "`" and "`" in m.group(2)):
        return 0, len(ct)
    marker = m.group(1)
    bs, be = nxt, len(ct)
    if len(lines) > 1:
        ls, le, _ = lines[-1]
        if re.match(r"^[ \t]*" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*$", ct[ls:le]):
            be = ls
    return bs, max(bs, be)


def _family(lang: str) -> Optional[str]:
    lang = (lang or "").strip().lower()
    if lang in _PY_FAMILY:
        return "python"
    if lang in _JS_FAMILY:
        return "js"
    if lang in _SH_FAMILY:
        return "shell"
    if lang == "":
        return ""
    return None


# --------------------------------------------------------------------------- references

@dataclass
class _Ref:
    claim: Claim
    start: int                       # absolute offsets into the ORIGINAL text
    end: int
    name: str                        # what the report shows (and Claim.text)
    kind: str                        # "package" | "attribute"
    lang: str                        # "python" | "javascript"
    origin: str                      # "install" | "py_import" | "js_import" | "stdlib"
    registry: Optional[str] = None   # "pypi" | "npm"
    dist: Optional[str] = None       # the name asked of the registry
    mapped: bool = False
    custom_index: bool = False
    cmd: str = ""
    # --- stdlib introspection
    query: Optional[dict] = None
    spans: Optional[list] = None     # per attribute index: (start, end, name) | None
    modspan: Optional[tuple] = None  # where to point if the imported module itself is missing
    anchor: Optional[tuple] = None   # where to point if introspection could not run


# ------------------------------------------------------------------ Python (ast based)

class _PyCtx:
    def __init__(self, claim: Claim, region_lines, code_lines, shifts):
        self.claim, self.region_lines, self.code_lines, self.shifts = claim, region_lines, code_lines, shifts

    def rel(self, lineno: int, byte_col: int) -> Optional[int]:
        """offset inside claim.text for an ast (1-based line, utf-8 byte column) position."""
        i = lineno - 1
        if not (0 <= i < len(self.code_lines)):
            return None
        line = self.code_lines[i]
        try:
            col = len(line.encode("utf-8")[:byte_col].decode("utf-8", "ignore"))
        except UnicodeError:
            return None
        return self.region_lines[i][0] + self.shifts[i] + col


def _py_context(claim: Claim, region_lines) -> _PyCtx:
    ct = claim.text
    raw = [ct[s:e] for s, e in region_lines]
    code = claim.data.get("code")
    code_lines = raw
    if isinstance(code, str):
        cl = code.split("\n")
        if len(cl) <= len(raw) and all(raw[i].endswith(cl[i]) for i in range(len(cl))):
            code_lines = cl
    shifts = [len(raw[i]) - len(code_lines[i]) if i < len(code_lines) else 0 for i in range(len(raw))]
    return _PyCtx(claim, region_lines, code_lines, shifts)


def _bound_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            out.add(n.rest)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
    return out


def _stdlib_top(top: str) -> bool:
    return top in sys.stdlib_module_names


def _checkable(mod: str) -> bool:
    parts = mod.split(".")
    return (bool(mod) and parts[0] in sys.stdlib_module_names and parts[0] not in BLOCKED_MODULES
            and "__main__" not in parts and all(parts))


def _dist_for_import(module: str, first_name: str = "") -> tuple[Optional[str], bool]:
    """(distribution name, mapped?) for a Python import path, or (None, False) = do not look up."""
    parts = module.split(".")
    cands = [".".join(parts[:k]) for k in range(len(parts), 0, -1)]
    if first_name:
        cands.insert(0, module + "." + first_name)
    for c in cands:
        if c in IMPORT_TO_DIST:
            return IMPORT_TO_DIST[c], True
    top = parts[0]
    if top in _NAMESPACE_ROOTS:
        return None, False
    return top, False


def _python_refs(claim: Claim, region_lines) -> list[_Ref]:
    ctx = _py_context(claim, region_lines)
    src = "\n".join(ctx.code_lines)
    if not src.strip() or len(src) > MAX_CODE_CHARS:
        return []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []

    ct = claim.text
    base = claim.start
    refs: list[_Ref] = []

    def span(lineno, c0, end_lineno, c1, expect) -> Optional[tuple[int, int]]:
        if end_lineno != lineno:
            return None
        a, b = ctx.rel(lineno, c0), ctx.rel(end_lineno, c1)
        if a is None or b is None or ct[a:b] != expect:
            return None
        return a, b

    # ---- collect bindings
    bindings: dict[str, set[tuple]] = {}
    top_imports: dict[str, set[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    bindings.setdefault(a.asname, set()).add(("alias", a.name))
                else:
                    top = a.name.split(".")[0]
                    bindings.setdefault(top, set()).add(("top", top))
                    top_imports.setdefault(top, set()).add(a.name)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            for a in n.names:
                if a.name != "*":
                    bindings.setdefault(a.asname or a.name, set()).add(("from", n.module, a.name))
    shadow = _bound_names(tree)
    stored: set[tuple[str, str]] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)):
            v = n.value
            while isinstance(v, ast.Attribute):
                v = v.value
            if isinstance(v, ast.Name):
                # record (root, first attribute) of the stored chain
                chain = []
                c = n
                while isinstance(c, ast.Attribute):
                    chain.append(c.attr)
                    c = c.value
                stored.add((v.id, chain[-1]))

    seen_pkg: set[str] = set()

    def third_party(top: str, module: str, first_name: str, tok_span: Optional[tuple[int, int]]):
        if tok_span is None or top in seen_pkg:
            return
        if _stdlib_top(top) or top in _LEGACY_MODULES or not _PY_IDENT_TOP.match(top):
            return
        dist, mapped = _dist_for_import(module, first_name)
        if not dist or not _valid_pypi(dist):
            return
        seen_pkg.add(top)
        refs.append(_Ref(claim, base + tok_span[0], base + tok_span[1], ct[tok_span[0]:tok_span[1]],
                         "package", "python", "py_import", registry="pypi", dist=dist, mapped=mapped,
                         cmd="import"))

    # ---- imports
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                top = a.name.split(".")[0]
                if not hasattr(a, "end_col_offset") or a.end_lineno != a.lineno:
                    continue
                sp = span(a.lineno, a.col_offset, a.lineno, a.col_offset + len(top.encode("utf-8")), top)
                if _stdlib_top(top):
                    if "." in a.name and _checkable(a.name):
                        full = span(a.lineno, a.col_offset, a.lineno,
                                    a.col_offset + len(a.name.encode("utf-8")), a.name)
                        if full:
                            refs.append(_Ref(claim, base + full[0], base + full[1], a.name, "attribute",
                                             "python", "stdlib",
                                             query={"imports": [a.name], "base": a.name, "attrs": []},
                                             spans=[], modspan=(base + full[0], base + full[1], a.name),
                                             anchor=(base + full[0], base + full[1], a.name)))
                else:
                    third_party(top, a.name, "", sp)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            top = n.module.split(".")[0]
            # the module token: first occurrence after "from" on the statement's first line
            i = n.lineno - 1
            if not (0 <= i < len(ctx.code_lines)):
                continue
            line = ctx.code_lines[i]
            try:
                c0 = len(line.encode("utf-8")[:n.col_offset].decode("utf-8", "ignore"))
            except UnicodeError:
                continue
            m = re.compile(r"from\s+(" + re.escape(n.module) + r")(?=\s)").match(line, c0)
            mod_rel = None
            if m:
                mod_rel = ctx.region_lines[i][0] + ctx.shifts[i] + m.start(1)
                if ct[mod_rel:mod_rel + len(n.module)] != n.module:
                    mod_rel = None
            first_name = next((a.name for a in n.names if a.name != "*"), "")
            if _stdlib_top(top):
                if not _checkable(n.module):
                    continue
                modspan = ((base + mod_rel, base + mod_rel + len(n.module), n.module)
                           if mod_rel is not None else None)
                for a in n.names:
                    if a.name == "*" or a.end_lineno != a.lineno:
                        continue
                    asp = span(a.lineno, a.col_offset, a.lineno,
                               a.col_offset + len(a.name.encode("utf-8")), a.name)
                    if not asp:
                        continue
                    refs.append(_Ref(claim, base + asp[0], base + asp[1], a.name, "attribute", "python",
                                     "stdlib",
                                     query={"imports": [n.module], "base": n.module, "attrs": [a.name]},
                                     spans=[(base + asp[0], base + asp[1], n.module + "." + a.name)],
                                     modspan=modspan,
                                     anchor=(base + asp[0], base + asp[1], n.module + "." + a.name)))
            elif mod_rel is not None:
                third_party(top, n.module, first_name, (mod_rel, mod_rel + len(top)))

    # ---- attribute chains rooted at an imported name
    inner = {id(a.value) for a in ast.walk(tree) if isinstance(a, ast.Attribute)
             and isinstance(a.value, ast.Attribute)}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Attribute) or id(n) in inner or not isinstance(n.ctx, ast.Load):
            continue
        nodes = []
        cur: ast.AST = n
        while isinstance(cur, ast.Attribute):
            nodes.append(cur)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            continue
        nodes.reverse()
        root = cur.id
        tg = bindings.get(root)
        if not tg or len(tg) != 1 or root in shadow or (root, nodes[0].attr) in stored:
            continue
        target = next(iter(tg))
        nodes = nodes[:MAX_CHAIN_DEPTH]
        attrs = [x.attr for x in nodes]
        if target[0] == "top":
            imports = sorted(top_imports[root])
            qbase, qattrs, skip_first = root, attrs, 0
        elif target[0] == "alias":
            imports, qbase, qattrs, skip_first = [target[1]], target[1], attrs, 0
        else:
            imports, qbase, qattrs, skip_first = [target[1]], target[1], [target[2]] + attrs, 1
        if not all(_checkable(m) for m in imports) or not _checkable(qbase):
            continue
        spans: list = [None] * skip_first
        names = []
        for k, x in enumerate(nodes):
            names.append(root + "".join("." + y for y in attrs[:k + 1]))
            sp = None
            if x.end_lineno == x.lineno:
                a = ctx.rel(x.lineno, x.col_offset)
                b = ctx.rel(x.end_lineno, x.end_col_offset)
                if a is not None and b is not None:
                    seg = ct[a:b]
                    if "\n" not in seg and "\r" not in seg and re.sub(r"\s+", "", seg) == names[-1]:
                        sp = (base + a, base + b, names[-1])
            spans.append(sp)
        if spans[skip_first] is None:
            continue
        refs.append(_Ref(claim, spans[skip_first][0], spans[skip_first][1], names[-1], "attribute", "python",
                         "stdlib", query={"imports": imports, "base": qbase, "attrs": qattrs},
                         spans=spans, anchor=spans[skip_first]))
    return refs


# --------------------------------------------------------------------------- JS (regex over comment-masked text)

def _mask_js(s: str) -> str:
    """Blank out // and /* */ comments (keeping offsets and newlines); strings are kept."""
    out = list(s)
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in "'\"`":
            q = c
            i += 1
            while i < n and s[i] != q:
                if s[i] == "\\":
                    i += 1
                elif s[i] in "\r\n" and q != "`":
                    break
                i += 1
            i += 1
        elif c == "/" and i + 1 < n and s[i + 1] == "/":
            j = i
            while j < n and s[j] not in "\r\n":
                out[j] = " "
                j += 1
            i = j
        elif c == "/" and i + 1 < n and s[i + 1] == "*":
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] not in "\r\n":
                    out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


_JS_PATTERNS = (
    re.compile(r"""(?<![\w$.])(?:import|export)\b[^'"`;]{0,400}?\bfrom\s*(['"])([^'"\n]{1,300})\1"""),
    re.compile(r"""(?<![\w$.])import\s*(['"])([^'"\n]{1,300})\1"""),
    re.compile(r"""(?<![\w$.])import\s*\(\s*(['"])([^'"\n]{1,300})\1\s*[,)]"""),
    re.compile(r"""(?<![\w$.])require\s*\(\s*(['"])([^'"\n]{1,300})\1\s*\)"""),
)


def _npm_root(spec: str) -> Optional[str]:
    if not spec or spec[0] in "./~#" or ":" in spec or "\\" in spec or any(c.isspace() for c in spec):
        return None
    parts = spec.split("/")
    if spec.startswith("@"):
        if len(parts) < 2:
            return None
        return parts[0] + "/" + parts[1]
    return parts[0]


def _js_refs(claim: Claim, bs: int, be: int) -> list[_Ref]:
    ct = claim.text
    region = ct[bs:be]
    if len(region) > MAX_CODE_CHARS:
        return []
    masked = _mask_js(region)
    refs: list[_Ref] = []
    seen_pos: set[int] = set()
    seen_names: set[str] = set()
    found = []
    for rx in _JS_PATTERNS:
        for m in rx.finditer(masked):
            if m.start(2) in seen_pos:
                continue
            seen_pos.add(m.start(2))
            found.append((m.start(2), m.group(2)))
    for pos, spec in sorted(found):
        root = _npm_root(spec)
        if not root or root in NODE_BUILTINS or spec in NODE_BUILTINS or root in seen_names:
            continue
        if root.startswith("@"):
            if root[1:].split("/")[0] in _JS_LOCAL_ROOTS:
                continue
        elif root in _JS_LOCAL_ROOTS:
            continue
        if not _valid_npm(root):
            continue
        seen_names.add(root)
        a = bs + pos
        refs.append(_Ref(claim, claim.start + a, claim.start + a + len(root), root, "package", "javascript",
                         "js_import", registry="npm", dist=root, cmd="import"))
    return refs


# --------------------------------------------------------------------------- install commands

_PIP_CMD = re.compile(
    r"(?<![\w./-])(?:(?:python[\d.]*|py)(?:[ \t]+-[\w.]+)*[ \t]+-m[ \t]+pip[\d.]*|pip[\d.]*|uv[ \t]+pip)"
    r"[ \t]+install(?=[ \t]|$)", re.M)
_NPM_CMD = re.compile(
    r"(?<![\w./-])(?P<c>npm[ \t]+(?:install|i|add)|pnpm[ \t]+(?:install|i|add)|yarn[ \t]+(?:global[ \t]+)?add"
    r"|npx|pnpm[ \t]+dlx|yarn[ \t]+dlx)(?=[ \t]|$)", re.M)
_TOKEN = re.compile(r""""[^"\n]*"|'[^'\n]*'|\S+""")

_PIP_VALUE_FLAGS = frozenset({
    "-r", "--requirement", "-c", "--constraint", "-e", "--editable", "-t", "--target", "-i", "--index-url",
    "--extra-index-url", "-f", "--find-links", "--prefix", "--root", "--src", "--proxy", "--cert",
    "--client-cert", "--cache-dir", "--python", "--platform", "--implementation", "--abi",
    "--python-version", "--only-binary", "--no-binary", "--global-option", "--install-option",
    "--config-settings", "-C", "--progress-bar", "--log", "--report", "--exists-action", "--retries",
    "--timeout", "--trusted-host", "--upgrade-strategy", "--use-feature", "--use-deprecated",
    "--keyring-provider", "--root-user-action", "--no-build-isolation-package", "-b", "--build",
})
_PIP_INDEX_FLAGS = frozenset({"-i", "--index-url", "--extra-index-url", "-f", "--find-links"})
_NPM_VALUE_FLAGS = frozenset({
    "--registry", "--prefix", "--tag", "--workspace", "-w", "--cache", "--userconfig", "--globalconfig",
    "--loglevel", "--save-prefix", "--scope", "--omit", "--include", "--before", "--otp", "--location",
    "--cwd", "--filter", "-F", "--dir", "-C", "--modules-folder", "--use-yarnrc", "--network-concurrency",
    "--offline-mirror", "--link-folder", "--global-folder", "--cache-folder", "--audit-level",
    "--install-strategy", "--lockfile-version", "--prefer-offline-folder", "--store-dir",
    "--node-arg", "--shell", "--userconfig",
})
_NPM_NAME = re.compile(r"^(@[a-z0-9][a-z0-9._~-]*/[a-z0-9][a-z0-9._~-]*|[A-Za-z0-9][A-Za-z0-9._~-]*)(?:@(.*))?$")
_NPM_VERSION = re.compile(r"^[\w.^~<>=*|+ -]*$")
_PIP_REQ = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)[ \t]*(\[[^\]]*\])?[ \t]*(.*)$", re.S)
_WORDS = re.compile(r"^(?:(?:sudo|RUN|exec|time|command|then|do|else|nohup|env(?:[ \t]+\w+=\S*)*)(?:[ \t]+-\S+)*[ \t]+)+")


def _prefix_ok(prefix: str, shell: bool) -> bool:
    p = prefix.strip()
    p = re.sub(r"^\([\w.-]+\)[ \t]*", "", p)
    if shell:
        for _ in range(4):                       # prompts: "$", "user@host:~$", "#", ">", "%", "!"
            q = re.sub(r"^(?:\S*[$#%>][ \t]+|[!%$>#])", "", p)
            if q == p:
                break
            p = q
        for sep in ("&&", "||", ";", "|"):
            if sep in p:
                p = p.rsplit(sep, 1)[-1]
        p = p.strip()
        p = _WORDS.sub("", p + " ", count=1).strip() if p else p
        return p == ""
    p = re.sub(r"^(?:(?:#+|//+|/\*+|\*+|!|%|\$|>)[ \t]*)+", "", p)
    p = _WORDS.sub("", p + " ", count=1).strip() if p else p
    return p == ""


def _tokens(line_text: str, line_abs: int):
    """Yield (content, abs_start_of_content, quoted, terminated)."""
    for m in _TOKEN.finditer(line_text):
        raw = m.group(0)
        if raw[0] in "\"'" and len(raw) >= 2 and raw[-1] == raw[0]:
            yield raw[1:-1], line_abs + m.start() + 1, True, False
            continue
        cut = None
        for k, ch in enumerate(raw):
            if ch in ";|&<>":
                cut = k
                break
        if cut is None:
            yield raw, line_abs + m.start(), False, False
        else:
            yield raw[:cut], line_abs + m.start(), False, True
            return


def _install_refs(claim: Claim, bs: int, be: int, shell: bool) -> list[_Ref]:
    ct = claim.text
    region = ct[bs:be]
    if len(region) > MAX_CODE_CHARS or not any(k in region for k in ("pip", "npm", "npx", "yarn", "pnpm")):
        return []
    # join backslash continuations without moving anything (same length)
    joined = re.sub(r"\\[ \t]*(\r\n|\r|\n)", lambda m: " " * (len(m.group(0))), region)
    refs: list[_Ref] = []
    seen: set[tuple] = set()

    def emit(name, a, kind_lang, registry, cmd, custom):
        key = (registry, name)
        if key in seen:
            return
        seen.add(key)
        gs = claim.start + bs + a
        refs.append(_Ref(claim, gs, gs + len(name), name, "package", kind_lang, "install", registry=registry,
                         dist=name, cmd=cmd, custom_index=custom))

    for rx, which in ((_PIP_CMD, "pip"), (_NPM_CMD, "npm")):
        for m in rx.finditer(joined):
            ls = max(joined.rfind("\n", 0, m.start()), joined.rfind("\r", 0, m.start())) + 1
            if not _prefix_ok(joined[ls:m.start()], shell):
                continue
            le = len(joined)
            for ch in "\r\n":
                k = joined.find(ch, m.end())
                if k != -1:
                    le = min(le, k)
            line = joined[m.end():le]
            cmd = " ".join(m.group(0).split())
            if which == "pip":
                _scan_pip(line, m.end(), cmd, emit)
            else:
                _scan_npm(line, m.end(), cmd, m.group("c").split()[0], emit)
    return refs


def _scan_pip(line: str, line_abs: int, cmd: str, emit) -> None:
    toks = []
    for content, pos, quoted, term in _tokens(line, line_abs):
        if content == "" or (not quoted and content.startswith("#")):
            break
        toks.append((content, pos, quoted))
        if term:
            break
    items = []
    custom = False
    pending = False
    for i, (content, pos, quoted) in enumerate(toks):
        if pending:
            pending = False
        elif content.startswith("-") and not quoted:
            flag = content.split("=", 1)[0]
            if flag in _PIP_INDEX_FLAGS:
                custom = True
            if flag in _PIP_VALUE_FLAGS and "=" not in content:
                pending = True
        else:
            m = _PIP_REQ.match(content)
            direct_ref = i + 1 < len(toks) and toks[i + 1][0].startswith("@")      # "name @ url"
            if (m and not direct_ref and (m.group(3) == "" or m.group(3)[0] in "=<>!~;(")
                    and _valid_pypi(m.group(1))):
                items.append((m.group(1), pos + m.start(1)))
    for name, p in items:
        emit(name, p, "python", "pypi", cmd, custom)


def _scan_npm(line: str, line_abs: int, cmd: str, tool: str, emit) -> None:
    items = []
    custom = False
    pending = False
    pkg_value = False
    only_first = tool in ("npx", "dlx") or cmd.endswith("dlx")
    got_package_flag = False
    for content, pos, quoted, term in _tokens(line, line_abs):
        if content == "":
            break
        if not quoted and content.startswith("#"):
            break
        if pending:
            pending = False
            if pkg_value:
                pkg_value = False
                _npm_item(content, pos, items)
        elif content.startswith("-") and not quoted:
            flag = content.split("=", 1)[0]
            if flag in ("--registry",) or flag.endswith(":registry"):
                custom = True
            if only_first and flag in ("-p", "--package"):
                got_package_flag = True
                if "=" in content:
                    _npm_item(content.split("=", 1)[1], pos + len(flag) + 1, items)
                else:
                    pending = pkg_value = True
            elif only_first and flag in ("-c", "--call"):
                break
            elif flag in _NPM_VALUE_FLAGS and "=" not in content:
                pending = True
        else:
            if only_first:
                if not got_package_flag and content not in _NPX_BIN_SKIP:
                    _npm_item(content, pos, items)
                break
            _npm_item(content, pos, items)
        if term:
            break
    for name, p in items:
        emit(name, p, "javascript", "npm", cmd, custom)


def _npm_item(content: str, pos: int, items: list) -> None:
    m = _NPM_NAME.match(content)
    if not m or not _valid_npm(m.group(1)):
        return
    ver = m.group(2)
    if ver is not None and not _NPM_VERSION.match(ver):
        return                                  # npm:alias, github:user/repo, URLs, paths ...
    items.append((m.group(1), pos))


# --------------------------------------------------------------------------- stdlib introspection (subprocess)

_PYTHON = sys.executable

_CHILD_SRC = r'''
import sys, json, importlib, types
try:
    import resource
    for _n, _v in (("RLIMIT_CPU", 6), ("RLIMIT_AS", 1 << 30), ("RLIMIT_FSIZE", 1 << 20), ("RLIMIT_CORE", 0)):
        try:
            resource.setrlimit(getattr(resource, _n), (_v, _v))
        except Exception:
            pass
except Exception:
    pass
spec = json.load(open(sys.argv[1], encoding="utf-8"))
blocked = set(spec["blocked"])
std = sys.stdlib_module_names

def bad(name):
    parts = name.split(".")
    return (not name or parts[0] not in std or parts[0] in blocked or "__main__" in parts
            or any(not p for p in parts))

def owner_of(o):
    if isinstance(o, types.ModuleType):
        return o.__name__
    mod = getattr(o, "__module__", None)
    q = getattr(o, "__qualname__", None) or getattr(o, "__name__", None)
    if isinstance(o, type) and mod and q:
        return mod + "." + q
    return type(o).__name__ + " object"

def resolve(q):
    for m in q["imports"]:
        if bad(m):
            return {"s": "skip"}
        try:
            importlib.import_module(m.split(".")[0])
        except BaseException:
            return {"s": "skip"}
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as e:
            nm = e.name or ""
            if "." in m and nm and (nm == m or m.startswith(nm + ".")):
                return {"s": "nomodule", "mod": nm}
            return {"s": "skip"}
        except BaseException:
            return {"s": "skip"}
    base = q["base"]
    if bad(base):
        return {"s": "skip"}
    obj = sys.modules.get(base)
    if obj is None:
        obj = importlib.import_module(base)
    for i, a in enumerate(q["attrs"]):
        try:
            obj = getattr(obj, a)
            continue
        except AttributeError:
            pass
        except BaseException:
            return {"s": "skip"}
        if isinstance(obj, types.ModuleType):
            sub = obj.__name__ + "." + a
            if bad(sub):
                return {"s": "skip"}
            try:
                importlib.import_module(sub)
                obj = sys.modules[sub]
                continue
            except ModuleNotFoundError:
                pass
            except BaseException:
                return {"s": "skip"}
        return {"s": "missing", "i": i, "owner": owner_of(obj)}
    return {"s": "ok"}

out = []
for q in spec["queries"]:
    try:
        out.append(resolve(q))
    except BaseException:
        out.append({"s": "skip"})
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f)
'''


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run_introspection(queries: list[dict], timeout: Optional[float] = None) -> tuple[Optional[list], str]:
    """Run all queries in ONE isolated child interpreter.  Returns (results, "") or (None, why)."""
    timeout = INTROSPECT_TIMEOUT if timeout is None else timeout
    with tempfile.TemporaryDirectory(prefix="aifilter-cc-", ignore_cleanup_errors=True) as tmp:
        work = os.path.join(tmp, "work")
        os.mkdir(work)
        spec_path = os.path.join(tmp, "spec.json")
        out_path = os.path.join(tmp, "out.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump({"queries": queries, "blocked": sorted(BLOCKED_MODULES)}, f)
        env = {"PYTHONIOENCODING": "utf-8"}
        if os.name == "nt":  # pragma: no cover
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        try:
            proc = subprocess.Popen(
                [_PYTHON, "-I", "-S", "-B", "-X", "utf8", "-c", _CHILD_SRC, spec_path, out_path],
                cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
        except (OSError, ValueError) as e:
            return None, f"could not start Python ({type(e).__name__})"
        try:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.wait()
                return None, f"timed out after {timeout:g} s"
        finally:
            _kill_group(proc)                       # stragglers
            if proc.poll() is None:
                proc.wait()
        if proc.returncode != 0:
            return None, f"the helper process failed (exit status {proc.returncode})"
        try:
            if os.path.getsize(out_path) > RESULT_CAP:
                return None, "the helper returned too much output"
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None, "the helper returned no readable result"
    if (not isinstance(data, list) or len(data) != len(queries)
            or not all(isinstance(d, dict) for d in data)):
        return None, "the helper returned malformed output"
    return data, ""


def _qkey(q: dict) -> tuple:
    return (tuple(q["imports"]), q["base"], tuple(q["attrs"]))


# --------------------------------------------------------------------------- registry lookups

_REGISTRY = {
    "pypi": ("PyPI", "pypi.org"),
    "npm": ("npm", "registry.npmjs.org"),
}


def _lookup_url(registry: str, name: str) -> str:
    if registry == "pypi":
        return f"https://pypi.org/pypi/{_pypi_norm(name)}/json"
    return "https://registry.npmjs.org/" + name.replace("/", "%2F")


def _host_of(url: str) -> str:
    m = re.match(r"^https?://([^/:?#]+)", url or "", re.I)
    return m.group(1).lower() if m else ""


def _do_lookup(fetcher, registry: str, name: str) -> tuple[str, str, str]:
    """-> (state, detail, url); state in exists | missing | unknown."""
    url = _lookup_url(registry, name)
    label, host = _REGISTRY[registry]
    headers = {"Accept": "application/json"}
    if registry == "npm":
        headers["Accept"] = "application/vnd.npm.install-v1+json; q=1.0, application/json; q=0.8"
    try:
        r = fetcher.fetch(url, headers=headers, timeout=LOOKUP_TIMEOUT, max_bytes=64 * 1024)
    except Exception as e:                       # a fetcher must never crash the run
        return "unknown", f"the {label} lookup failed ({type(e).__name__})", url
    if r.error:
        return "unknown", f"could not reach {label} ({r.error})", url
    if r.final_url and _host_of(r.final_url) not in ("", host):
        return "unknown", f"the {label} lookup was redirected elsewhere", url
    if r.status == 200:
        if r.body.lstrip()[:1] == b"<":
            return "unknown", f"{label} returned a web page instead of package data", url
        return "exists", "", url
    if r.status == 404:
        return "missing", "", url
    return "unknown", f"{label} answered HTTP {r.status}", url


# --------------------------------------------------------------------------- findings

def _reason_exists(ref: _Ref) -> str:
    label = _REGISTRY[ref.registry][0]
    if ref.origin == "py_import" and ref.dist != ref.name:
        what = f"The import `{ref.name}` is provided by the {label} package `{ref.dist}`, which exists."
    elif ref.origin == "py_import":
        what = f"A package named `{ref.dist}` exists on {label}."
    else:
        what = f"`{ref.dist}` exists on {label}."
    return (what + " That only shows the name is registered: existing does not mean it is the right or a safe "
            "package (look-alike and malicious packages exist), and the way it is used here is not checked.")


def _reason_missing(ref: _Ref) -> tuple[Status, str]:
    label = _REGISTRY[ref.registry][0]
    if ref.custom_index:
        return Status.UNVERIFIED, (f"`{ref.dist}` was not found on the public {label}, but this command names a custom "
                                   f"index/registry, so the package may live there; not checked.")
    if ref.origin == "install":
        return Status.CONTRADICTED, (f"No such package on {label}: `{ref.dist}` does not exist (the registry says 404). "
                                     f"It looks like a name invented by the model, and an attacker could register it "
                                     f"(slopsquatting). Do not install it.")
    if ref.origin == "js_import":
        return Status.CONTRADICTED, (f"No such package on npm: `{ref.dist}` does not exist (the registry says 404). It is "
                                     f"imported by this code, so it looks like a name invented by the model, and one an "
                                     f"attacker could register (slopsquatting). Unless it is a local file or workspace "
                                     f"package, do not install it.")
    return Status.UNVERIFIED, (f"No PyPI package named exactly `{ref.dist}`; import names can differ from package names, "
                               f"and this may be a local module, so this is not proof it is invented.")


def _api_claim(ref: _Ref, text: str, n: int) -> Claim:
    data = {"language": ref.lang, "name": ref.dist if ref.kind == "package" and ref.dist else ref.name,
            "kind": ref.kind, "origin": ref.origin}
    if ref.registry:
        data["registry"] = ref.registry
    return Claim(f"k{n}", ClaimKind.API_REF, text[ref.start:ref.end], ref.start, ref.end, data)


def scan(text: str, claims: Sequence[Claim], *, fetcher=None, registries: bool = True,
         today=None) -> list[Finding]:
    refs: list[_Ref] = []
    for claim in claims:
        if claim.kind != ClaimKind.CODE or not hasattr(claim.data, "get"):
            continue
        if (not (0 <= claim.start <= claim.end <= len(text)) or text[claim.start:claim.end] != claim.text
                or len(claim.text) > MAX_CODE_CHARS * 2):
            continue
        fam = _family(str(claim.data.get("language") or ""))
        if fam is None:
            continue
        bs, be = _body_region(claim.text)
        region_lines = [(s, e) for s, e, _n in _lines(claim.text) if bs <= s < be]
        if fam in ("python", ""):
            refs += _python_refs(claim, region_lines)
        if fam == "js":
            refs += _js_refs(claim, bs, be)
        refs += _install_refs(claim, bs, be, shell=fam in ("shell", ""))

    # ---- A. stdlib introspection (one subprocess for everything)
    results: dict[tuple, dict] = {}
    fail_reason = ""
    stdlib = [r for r in refs if r.origin == "stdlib"]
    if stdlib:
        uniq: dict[tuple, dict] = {}
        for r in stdlib:
            uniq.setdefault(_qkey(r.query), r.query)
        keys = list(uniq)[:MAX_QUERIES]
        res, fail_reason = _run_introspection([uniq[k] for k in keys])
        if res is not None:
            results = dict(zip(keys, res))

    out: list[tuple[int, int, _Ref, Status, str, tuple]] = []   # (start, end, ref, status, reason, evidence)
    seen: set[tuple] = set()

    def add(ref: _Ref, start: int, end: int, name: str, status: Status, reason: str, evidence: tuple = ()):
        key = (ref.claim.id, ref.kind, name, ref.registry)
        if key in seen or text[start:end] == "":
            return
        seen.add(key)
        r2 = _Ref(ref.claim, start, end, name, ref.kind, ref.lang, ref.origin, ref.registry, ref.dist,
                  ref.mapped, ref.custom_index, ref.cmd)
        out.append((start, end, r2, status, reason, evidence))

    if stdlib and fail_reason:
        by_claim: dict[str, list[_Ref]] = {}
        for r in stdlib:
            by_claim.setdefault(r.claim.id, []).append(r)
        for rs in by_claim.values():
            rs.sort(key=lambda r: r.start)
            a = rs[0]
            s, e, nm = a.anchor
            n_unique = len({_qkey(r.query) for r in rs})
            add(a, s, e, nm, Status.UNVERIFIED,
                f"Could not check Python's standard library for the names used in this code block "
                f"({fail_reason}); {n_unique} reference(s) were not checked. This is a limit of the checker, "
                f"not a problem with the answer.")
    elif stdlib:
        for r in sorted(stdlib, key=lambda r: r.start):
            res = results.get(_qkey(r.query))
            if not res:
                continue
            ev = (f"Checked by importing the module in an isolated Python {PY_VER} subprocess.",)
            if res.get("s") == "missing":
                i = res.get("i", 0)
                sp = r.spans[i] if r.spans and 0 <= i < len(r.spans) else None
                if sp is None:
                    continue
                owner = str(res.get("owner", "the module"))[:120]
                add(r, sp[0], sp[1], sp[2], Status.UNVERIFIED,
                    f"`{sp[2]}` was not found in Python {PY_VER}'s `{owner}`: possibly invented, or from a "
                    f"newer/older Python than the one used for this check.", ev)
            elif res.get("s") == "nomodule" and r.modspan:
                s, e, nm = r.modspan
                add(r, s, e, nm, Status.UNVERIFIED,
                    f"No module `{nm}` was found in Python {PY_VER}'s standard library: possibly invented, or "
                    f"from a newer/older Python than the one used for this check.", ev)

    # ---- B. registries
    if registries:
        fetcher = fetcher if fetcher is not None else UrllibFetcher()
        cache: dict[tuple, tuple[str, str, str]] = {}
        t0 = time.monotonic()
        for r in sorted((r for r in refs if r.registry), key=lambda r: r.start):
            key = (r.registry, _pypi_norm(r.dist) if r.registry == "pypi" else r.dist)
            label = _REGISTRY[r.registry][0]
            if key not in cache:
                if len([1 for v in cache.values() if v[0] != "skipped"]) >= MAX_LOOKUPS:
                    cache[key] = ("skipped", f"lookup limit reached ({MAX_LOOKUPS} registry lookups per run)", "")
                elif time.monotonic() - t0 > LOOKUP_BUDGET_S:
                    cache[key] = ("skipped", "the time budget for registry lookups ran out", "")
                else:
                    cache[key] = _do_lookup(fetcher, r.registry, r.dist)
            state, detail, url = cache[key]
            ev = (url,) if url else ()
            if state == "exists":
                add(r, r.start, r.end, r.name, Status.VERIFIED, _reason_exists(r), ev)
            elif state == "missing":
                st, why = _reason_missing(r)
                add(r, r.start, r.end, r.name, st, why, ev)
            elif state == "skipped":
                add(r, r.start, r.end, r.name, Status.UNVERIFIED, f"Not checked: {detail}.", ev)
            else:
                add(r, r.start, r.end, r.name, Status.UNVERIFIED,
                    f"Could not check whether `{r.dist}` exists on {label}: {detail}. This is a limit of the "
                    f"checker, not a problem with the answer.", ev)

    out.sort(key=lambda t: (t[0], t[1]))
    findings = []
    for n, (s, e, ref, status, reason, ev) in enumerate(out, 1):
        findings.append(Finding(_api_claim(ref, text, n), status, MODULE, reason, ev))
    return findings
