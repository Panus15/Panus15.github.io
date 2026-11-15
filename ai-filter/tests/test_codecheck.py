"""Tests for engine/codecheck.py.  Offline: registries are a FakeFetcher; the stdlib introspection
really spawns a local, isolated Python subprocess (no network needed).

Run: python3 tests/test_codecheck.py
"""
import json, os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import Claim, ClaimKind, Status
from engine.net import FakeFetcher, FetchResult
from engine import codecheck
from engine.extract import extract

V, U, C = Status.VERIFIED, Status.UNVERIFIED, Status.CONTRADICTED
FENCE = "`" * 3


# ------------------------------------------------------------------ helpers

def fence(lang, body, nl="\n"):
    return FENCE + lang + nl + body.replace("\n", nl) + nl + FENCE


def ok(url, body=b"{}"):
    return FetchResult(url, url, 200, body)


def nf(url):
    return FetchResult(url, url, 404, b"Not Found")


def pypi(name):
    return f"https://pypi.org/pypi/{name}/json"


def npm(name):
    return "https://registry.npmjs.org/" + name.replace("/", "%2F")


def run(text, routes=None, registries=True, fetcher=None, claims=None):
    f = fetcher or FakeFetcher(routes or {})
    cl = extract(text) if claims is None else claims
    out = codecheck.scan(text, cl, fetcher=f, registries=registries)
    for x in out:
        assert x.module == "codecheck"
        assert x.claim.kind == ClaimKind.API_REF
        assert text[x.claim.start:x.claim.end] == x.claim.text, (x.claim, text[x.claim.start:x.claim.end])
    assert [x.claim.id for x in out] == [f"k{i}" for i in range(1, len(out) + 1)]
    return out, f


def by_name(out):
    return {x.claim.text: x for x in out}


def block(lang, body):
    return fence(lang, body)


# ------------------------------------------------------------------ pip install

def test_pip_install_404_is_contradicted_and_200_verified():
    t = block("bash", "pip install totally-made-up-lib requests")
    out, f = run(t, {pypi("totally-made-up-lib"): nf(pypi("totally-made-up-lib")),
                     pypi("requests"): ok(pypi("requests"))})
    d = by_name(out)
    assert d["totally-made-up-lib"].status == C
    assert "attacker" in d["totally-made-up-lib"].reason and "PyPI" in d["totally-made-up-lib"].reason
    assert d["requests"].status == V
    assert "right or a safe" in d["requests"].reason
    assert d["requests"].claim.data["kind"] == "package" and d["requests"].claim.data["language"] == "python"


def test_extras_versions_and_quotes_stripped():
    t = block("bash", 'pip install "requests[security]>=2.0" flask==2.0 numpy~=1.2 \'pandas >= 1.0\' Django[argon2]')
    routes = {pypi(n): ok(pypi(n)) for n in ("requests", "flask", "numpy", "pandas", "django")}
    out, f = run(t, routes)
    assert sorted(x.claim.text for x in out) == ["Django", "flask", "numpy", "pandas", "requests"]
    assert all(x.status == V for x in out)
    assert pypi("django") in f.calls            # PEP 503 normalisation


def test_flags_urls_paths_and_requirement_files_skipped():
    t = block("bash", "\n".join([
        "pip install -r requirements.txt",
        "pip install -e .",
        "pip install -e ./pkg[dev]",
        "pip install git+https://github.com/a/b.git",
        "pip install https://example.com/x.whl",
        "pip install ./dist/thing-1.0.whl",
        "pip install ../sibling",
        "pip install --index-url https://my.index/simple somepkg",
        "pip install -t build/lib --upgrade --no-cache-dir realone",
        "pip install pkg @ https://example.com/p.zip",
        "pip install requirements.txt",
    ]))
    out, f = run(t, {})
    names = sorted(x.claim.text for x in out)
    assert names == ["realone", "somepkg"], names
    assert all(f != "https://my.index/simple" for f in f.calls)
    assert all(c.startswith("https://pypi.org/pypi/") for c in f.calls)


def test_custom_index_404_is_only_unverified():
    t = block("bash", "pip install --extra-index-url https://corp.example/simple internal-thing")
    out, _ = run(t, {pypi("internal-thing"): nf(pypi("internal-thing"))})
    assert len(out) == 1 and out[0].status == U and "custom index" in out[0].reason


def test_python_m_pip_pip3_uv_variants_and_prompts():
    t = block("console", "$ python3 -m pip install aaa\n(venv) $ pip3 install bbb\n# uv pip install ccc\n"
                         "sudo -H pip install ddd && echo done\nRUN pip install --no-cache-dir eee\n"
                         "echo 'run pip install fff'\n")
    routes = {pypi(n): ok(pypi(n)) for n in "aaa bbb ccc ddd eee".split()}
    out, f = run(t, routes)
    assert sorted(x.claim.text for x in out) == ["aaa", "bbb", "ccc", "ddd", "eee"]
    assert pypi("fff") not in f.calls


def test_pip_command_stops_at_shell_operators_and_continuations():
    t = block("bash", "pip install one two \\\n   three && pip uninstall notthis\npip install four; rm -rf x\n"
                      "pip install five | tee log\npip install six # seven")
    routes = {pypi(n): ok(pypi(n)) for n in "one two three four five six".split()}
    out, f = run(t, routes)
    assert sorted(x.claim.text for x in out) == sorted("one two three four five six".split())


def test_install_lines_inside_python_comments_and_notebook_bang():
    t = block("python", "# pip install rich\n!pip install tqdm\nprint('x = pip install nope')\n")
    out, f = run(t, {pypi("rich"): ok(pypi("rich")), pypi("tqdm"): ok(pypi("tqdm"))})
    assert sorted(x.claim.text for x in out) == ["rich", "tqdm"]


def test_unlabelled_block_with_pip_line():
    t = block("", "$ pip install ghostly-lib")
    out, _ = run(t, {pypi("ghostly-lib"): nf(pypi("ghostly-lib"))})
    assert len(out) == 1 and out[0].status == C


# ------------------------------------------------------------------ npm

def test_npm_yarn_pnpm_npx_commands():
    t = block("bash", "npm i express\nnpm install --save-dev jest@29.0.0 @types/node\nyarn add react\n"
                      "pnpm add -D vite\nyarn global add nodemon\nnpx create-react-app my-app\n"
                      "npx -y degit user/repo\nnpm install\nnpm i github:user/repo ./local file:../x user/repo")
    names = ["express", "jest", "@types/node", "react", "vite", "nodemon", "create-react-app", "degit"]
    routes = {npm(n): ok(npm(n)) for n in names}
    out, f = run(t, routes)
    assert sorted(x.claim.text for x in out) == sorted(names)
    assert all(x.claim.data["registry"] == "npm" and x.claim.data["language"] == "javascript" for x in out)
    assert "https://registry.npmjs.org/@types%2Fnode" in f.calls


def test_scoped_name_is_url_encoded_and_404_contradicted():
    t = block("bash", "npm install @fake-scope/does-not-exist@^1.2.3")
    u = "https://registry.npmjs.org/@fake-scope%2Fdoes-not-exist"
    out, f = run(t, {u: nf(u)})
    assert f.calls == [u]
    assert out[0].status == C and out[0].claim.text == "@fake-scope/does-not-exist"


def test_npx_package_flag_and_registry_flag():
    t = block("bash", "npx --package cowsay-x cowsay hi\nnpm i --registry https://corp/npm private-lib\n"
                      "npx -c 'echo hi'\nnpx tsc --init")
    routes = {npm("cowsay-x"): nf(npm("cowsay-x")), npm("private-lib"): nf(npm("private-lib"))}
    out, f = run(t, routes)
    d = by_name(out)
    assert set(d) == {"cowsay-x", "private-lib"}, d.keys()
    assert d["cowsay-x"].status == C
    assert d["private-lib"].status == U                       # custom registry
    assert npm("cowsay") not in f.calls and npm("tsc") not in f.calls


def test_npm_version_with_url_or_alias_skipped():
    t = block("bash", "npm i foo@npm:bar foo2@github:u/r baz@latest")
    out, f = run(t, {npm("baz"): ok(npm("baz"))})
    assert [x.claim.text for x in out] == ["baz"]


# ------------------------------------------------------------------ JS imports

def test_js_require_import_dynamic_export_sources():
    t = block("javascript", "\n".join([
        "const a = require('ghost-a');",
        "import b from \"ghost-b/sub/path\";",
        "import { c } from 'ghost-c';",
        "const d = await import('ghost-d');",
        "export * from 'ghost-e';",
        "import 'ghost-f';",
        "import g, * as h from '@ghost/g/x';",
        "import type { T } from 'ghost-t';",
    ]))
    names = ["ghost-a", "ghost-b", "ghost-c", "ghost-d", "ghost-e", "ghost-f", "@ghost/g", "ghost-t"]
    out, f = run(t, {npm(n): nf(npm(n)) for n in names})
    assert sorted(x.claim.text for x in out) == sorted(names)
    assert all(x.status == C for x in out)
    assert all("npm" in x.reason for x in out)


def test_js_builtins_relative_and_prefixed_skipped():
    t = block("js", "\n".join([
        "const fs = require('fs'); const p = require('node:path'); import {x} from 'fs/promises';",
        "import y from './local'; import z from '../up/x'; import w from '/abs'; import v from '~/home';",
        "import q from '@/alias'; import u from '#internal'; import e from 'node:test'; import s from 'bun:sqlite';",
        "import u2 from 'components/Header'; import u3 from 'src/utils'; import c from 'https://cdn/x.js';",
        "import k from 'real-one';",
    ]))
    out, f = run(t, {npm("real-one"): ok(npm("real-one"))})
    assert [x.claim.text for x in out] == ["real-one"]
    assert f.calls == [npm("real-one")]


def test_js_comments_are_not_scanned_but_strings_with_slashes_are_fine():
    t = block("js", "// const a = require('in-line-comment');\n/* import x from 'in-block' */\n"
                    "const u = 'http://a/b'; const r = require('after-url');")
    out, f = run(t, {npm("after-url"): ok(npm("after-url"))})
    assert [x.claim.text for x in out] == ["after-url"]


def test_typescript_family_languages():
    for lang in ("ts", "typescript", "tsx", "node", "mjs", "jsx"):
        t = block(lang, "import a from 'pkg-x';")
        out, _ = run(t, {npm("pkg-x"): ok(npm("pkg-x"))})
        assert len(out) == 1 and out[0].status == V, lang


def test_js_import_custom_registry_n_a_but_install_comment_in_js():
    t = block("js", "// npm install left-pad-ng\nconst x = 1;")
    out, _ = run(t, {npm("left-pad-ng"): nf(npm("left-pad-ng"))})
    assert len(out) == 1 and out[0].status == C


# ------------------------------------------------------------------ python imports -> PyPI

def test_mapping_table_has_well_known_entries():
    m = codecheck.IMPORT_TO_DIST
    assert len(m) >= 40
    for k, v in {"cv2": "opencv-python", "PIL": "pillow", "sklearn": "scikit-learn", "yaml": "pyyaml",
                 "bs4": "beautifulsoup4", "dateutil": "python-dateutil"}.items():
        assert m[k] == v


def test_mapped_import_uses_distribution_name_and_404_is_unverified():
    t = block("python", "import cv2\nfrom PIL import Image\nimport sklearn.linear_model\nimport yaml, bs4, dateutil.parser\n")
    names = ["opencv-python", "pillow", "scikit-learn", "pyyaml", "beautifulsoup4", "python-dateutil"]
    routes = {pypi(n): ok(pypi(n)) for n in names}
    out, f = run(t, routes)
    assert sorted(x.claim.text for x in out) == sorted(["cv2", "PIL", "sklearn", "yaml", "bs4", "dateutil"])
    assert sorted(c.split("/")[-2] for c in f.calls) == sorted(names)
    assert all(x.status == V for x in out)
    assert by_name(out)["cv2"].claim.data["name"] == "opencv-python"
    out, _ = run(t, {pypi("opencv-python"): nf(pypi("opencv-python"))})
    assert by_name(out)["cv2"].status == U


def test_404_on_explicit_install_contradicted_vs_unmapped_import_unverified():
    t = block("bash", "pip install fancylib") + "\n\n" + block("python", "import fancylib\n")
    out, _ = run(t, {pypi("fancylib"): nf(pypi("fancylib"))})
    assert [x.status for x in out] == [C, U]
    assert "no PyPI package named exactly" in out[1].reason.lower().replace("No PyPI", "no PyPI") or "PyPI" in out[1].reason
    assert "import names can differ" in out[1].reason


def test_relative_stdlib_legacy_and_namespace_imports_skipped():
    t = block("python", "from . import sibling\nfrom .x import y\nfrom ..pkg import z\nimport os, sys, json\n"
                        "import urllib2, imp, ConfigParser\nfrom google.cloud import something\nimport __future__\n")
    out, f = run(t, {})
    assert out == [] or all(x.claim.data.get("origin") == "stdlib" for x in out)
    assert f.calls == []


def test_third_party_from_import_and_import_inside_function():
    t = block("python", "def f():\n    import numpy as np\n    from requests.adapters import HTTPAdapter\n")
    out, f = run(t, {pypi("numpy"): ok(pypi("numpy")), pypi("requests"): ok(pypi("requests"))})
    assert sorted(x.claim.text for x in out) == ["numpy", "requests"]


def test_dedupe_lookups_and_findings():
    t = block("python", "import foo\nimport foo.bar\nfrom foo import baz\n") + "\n\n" + block("bash", "pip install foo")
    out, f = run(t, {pypi("foo"): ok(pypi("foo"))})
    assert f.calls == [pypi("foo")]
    assert len(out) == 2                      # once per block (import) + the install command


# ------------------------------------------------------------------ stdlib attributes (real subprocess)

def test_json_parse_and_os_getcwdu_flagged_real_attributes_silent():
    t = block("python", "import json, os\nx = json.loads('1')\ny = json.parse('1')\nz = os.getcwdu()\nw = os.path.join('a')\n"
                        "from os import getcwdu, path, environ\nprint(os.environ.get('A'))\n")
    out, f = run(t, registries=False)
    d = by_name(out)
    assert set(d) == {"json.parse", "os.getcwdu"}, d.keys()      # repeated name reported once per block
    assert all(x.status == U for x in out)
    assert "not found in Python" in d["json.parse"].reason and "newer/older" in d["json.parse"].reason
    assert "CONTRADICTED" not in " ".join(x.status.name for x in out)
    assert f.calls == []


def test_from_import_missing_name_points_at_the_name():
    t = block("python", "from os import getcwdu\nfrom collections import OrderedDict, namedtuple\n")
    out, _ = run(t, registries=False)
    assert [x.claim.text for x in out] == ["getcwdu"]
    assert out[0].claim.data["kind"] == "attribute" and out[0].claim.data["name"] == "os.getcwdu"


def test_missing_submodule_and_aliases_and_class_attributes():
    t = block("python", "import os.foo\nimport os as o\nimport json.nothing\nfrom datetime import datetime\n"
                        "o.getcwdu()\ndatetime.nowx()\ndatetime.now()\no.path.joinx('a')\no.path.join('a')\n")
    out, _ = run(t, registries=False)
    names = sorted(x.claim.text for x in out)
    assert names == ["datetime.nowx", "json.nothing", "o.getcwdu", "o.path.joinx", "os.foo"], names


def test_depth_limit_and_unrooted_chains_ignored():
    t = block("python", "import os\nos.path.join('a').foo.bar.baz\nx = os.getcwd().nothing\nos.a.b.c.d\n")
    out, _ = run(t, registries=False)
    assert [x.claim.text for x in out] == ["os.a"]


def test_shadowed_or_assigned_names_not_checked():
    t = block("python", "import json\njson = {}\njson.parse\n\n") + "\n\n" + block("python", "import os\ndef f(os):\n    return os.nothing\n") \
        + "\n\n" + block("python", "import sys\nsys.myflag = 1\nprint(sys.myflag)\n") \
        + "\n\n" + block("python", "try:\n    import simplejson as json\nexcept ImportError:\n    import json\njson.nothing\n")
    out, _ = run(t, registries=False)
    assert out == [], [x.claim.text for x in out]


def test_syntax_error_unknown_language_and_non_code_claims_produce_nothing():
    t = (block("python", "import json\njson.parse(\n") + "\n\n" + block("python", "print 'hi'\nimport json\n")
         + "\n\n" + block("rust", "use json::parse; pip install nothing") + "\n\n" + block("json", '{"a": "import foo"}'))
    out, f = run(t, registries=True)
    assert out == [] and f.calls == []


def test_unlabelled_block_parsing_as_python_is_checked():
    t = block("", "import json\njson.parse('1')\n")
    out, _ = run(t, registries=False)
    assert [x.claim.text for x in out] == ["json.parse"]


def test_blocklisted_modules_never_imported_sentinel():
    d = tempfile.mkdtemp(prefix="cc-sentinel-")
    log = os.path.join(d, "imports.log")
    hook = (f"import sys\n_L = open({log!r}, 'a')\n"
            "def _h(ev, args):\n    if ev == 'import':\n        _L.write(args[0] + '\\n'); _L.flush()\n"
            "sys.addaudithook(_h)\n")
    real = codecheck._CHILD_SRC
    codecheck._CHILD_SRC = hook + real
    try:
        # (1) the parent filters: these never even reach the child
        t = block("python", "import this\nimport antigravity\nimport tkinter\nimport turtle\nimport webbrowser\n"
                            "import idlelib.idle\nfrom test import support\nimport ensurepip\nimport venv\n"
                            "tkinter.Tk\nturtle.nothing\nthis.s\nimport json\njson.nothing2\n")
        out, _ = run(t, registries=False)
        assert [x.claim.text for x in out] == ["json.nothing2"]
        # (2) the child refuses on its own, even when asked directly
        qs = [{"imports": [m], "base": m, "attrs": ["x"]} for m in
              ("this", "antigravity", "tkinter", "turtle", "webbrowser", "idlelib", "test", "ensurepip", "venv",
               "json.__main__", "not_stdlib_at_all")]
        res, why = codecheck._run_introspection(qs)
        assert res is not None and all(r == {"s": "skip"} for r in res), (res, why)
    finally:
        codecheck._CHILD_SRC = real
    imported = open(log).read().split()
    for bad in ("this", "antigravity", "tkinter", "turtle", "webbrowser", "idlelib", "test", "ensurepip", "venv",
                "json.__main__", "not_stdlib_at_all"):
        assert bad not in imported, bad
    assert "json" in imported                                   # the audit hook does see real imports


def test_introspection_is_isolated_subprocess_not_in_process():
    before = set(sys.modules)
    t = block("python", "import colorsys\ncolorsys.nothing\n")
    out, _ = run(t, registries=False)
    assert len(out) == 1
    assert "colorsys" not in before and "colorsys" not in sys.modules


def test_child_runs_isolated_cwd_and_env():
    d = tempfile.mkdtemp(prefix="cc-iso-")
    marker = os.path.join(d, "report.json")
    probe = (f"import sys, os, json\n_r = {{'cwd': os.getcwd(), 'env': dict(os.environ), 'stdin': sys.stdin.read(),"
             f" 'flags': [sys.flags.isolated, sys.flags.no_site]}}\n"
             f"open({marker!r}, 'w').write(json.dumps(_r))\n")
    real = codecheck._CHILD_SRC
    os.environ["AIFILTER_SECRET_TEST"] = "hunter2"
    codecheck._CHILD_SRC = probe + real
    try:
        res, why = codecheck._run_introspection([{"imports": ["json"], "base": "json", "attrs": []}])
    finally:
        codecheck._CHILD_SRC = real
        del os.environ["AIFILTER_SECRET_TEST"]
    assert res == [{"s": "ok"}], why
    r = json.load(open(marker))
    assert "AIFILTER_SECRET_TEST" not in r["env"] and r["stdin"] == ""
    assert r["flags"] == [1, 1]
    assert "aifilter-cc-" in r["cwd"] and not os.path.exists(r["cwd"])      # temp dir removed afterwards


def test_subprocess_timeout_is_unverified_not_contradicted():
    real = codecheck._CHILD_SRC
    codecheck._CHILD_SRC = "import time\ntime.sleep(60)\n"
    old = codecheck.INTROSPECT_TIMEOUT
    codecheck.INTROSPECT_TIMEOUT = 0.5
    try:
        t0 = time.time()
        out, _ = run(block("python", "import json\njson.parse('1')\njson.nothing\n"), registries=False)
        assert time.time() - t0 < 5
    finally:
        codecheck._CHILD_SRC = real
        codecheck.INTROSPECT_TIMEOUT = old
    assert len(out) == 1 and out[0].status == U and "timed out" in out[0].reason
    assert "not checked" in out[0].reason


def test_subprocess_failure_is_unverified():
    real = codecheck._PYTHON
    codecheck._PYTHON = "/nonexistent/python-xyz"
    try:
        out, _ = run(block("python", "import json\njson.parse('1')\n"), registries=False)
    finally:
        codecheck._PYTHON = real
    assert len(out) == 1 and out[0].status == U and "could not start" in out[0].reason.lower()
    realsrc = codecheck._CHILD_SRC
    codecheck._CHILD_SRC = "import sys\nsys.exit(3)\n"
    try:
        out, _ = run(block("python", "import json\njson.parse('1')\n"), registries=False)
    finally:
        codecheck._CHILD_SRC = realsrc
    assert len(out) == 1 and out[0].status == U and "exit status 3" in out[0].reason
    codecheck._CHILD_SRC = "open(__import__('sys').argv[2], 'w').write('not json')\n"
    try:
        out, _ = run(block("python", "import json\njson.parse('1')\n"), registries=False)
    finally:
        codecheck._CHILD_SRC = realsrc
    assert len(out) == 1 and out[0].status == U


def test_no_subprocess_when_nothing_to_introspect():
    real = codecheck.subprocess.Popen
    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise AssertionError("no subprocess expected")
    codecheck.subprocess.Popen = boom
    try:
        run(block("python", "import numpy\nprint(1)\n"), {pypi("numpy"): ok(pypi("numpy"))})
        run(block("bash", "pip install x"), {pypi("x"): ok(pypi("x"))})
        run(block("js", "require('x')"), {npm("x"): ok(npm("x"))})
    finally:
        codecheck.subprocess.Popen = real
    assert calls == []


# ------------------------------------------------------------------ registries behaviour

def test_registries_false_makes_zero_fetches():
    t = block("bash", "pip install a b\nnpm i c") + "\n\n" + block("python", "import d\n") + "\n\n" + block("js", "require('e')")
    out, f = run(t, registries=False)
    assert out == [] and f.calls == []


def test_fetch_failures_are_unverified():
    t = block("bash", "pip install a1 b2 c3 d4 e5")
    routes = {
        pypi("a1"): FetchResult(pypi("a1"), pypi("a1"), None, error="timeout", detail="t"),
        pypi("b2"): FetchResult(pypi("b2"), pypi("b2"), 500, b"oops"),
        pypi("c3"): FetchResult(pypi("c3"), pypi("c3"), 429, b""),
        pypi("d4"): FetchResult(pypi("d4"), pypi("d4"), None, error="blocked"),
        pypi("e5"): FetchResult(pypi("e5"), pypi("e5"), None, error="dns"),
    }
    out, _ = run(t, routes)
    assert len(out) == 5 and all(x.status == U for x in out)
    assert all("not a problem with the answer" in x.reason for x in out)


def test_fetcher_that_raises_is_unverified():
    class Boom:
        calls = []

        def fetch(self, *a, **k):
            raise RuntimeError("kaboom")
    out, _ = run(block("bash", "pip install a1"), fetcher=Boom())
    assert len(out) == 1 and out[0].status == U


def test_offline_fetcher_everything_unverified():
    from engine.net import OfflineFetcher
    out, _ = run(block("bash", "pip install a1\nnpm i b2"), fetcher=OfflineFetcher())
    assert len(out) == 2 and all(x.status == U for x in out)


def test_captive_portal_html_and_foreign_redirect_not_trusted():
    u = pypi("zz9")
    routes = {u: FetchResult(u, u, 200, b"<html>login to wifi</html>")}
    out, _ = run(block("bash", "pip install zz9"), routes)
    assert out[0].status == U and "web page" in out[0].reason
    routes = {u: FetchResult(u, "https://evil.example/x", 404, b"")}
    out, _ = run(block("bash", "pip install zz9"), routes)
    assert out[0].status == U and "redirected" in out[0].reason


def test_lookup_cap_beyond_is_unverified_without_fetch():
    names = [f"pkg{i:02d}" for i in range(40)]
    t = block("bash", "pip install " + " ".join(names))
    routes = {pypi(n): nf(pypi(n)) for n in names}
    out, f = run(t, routes)
    assert len(f.calls) == 30 and len(out) == 40
    st = [x.status for x in out]
    assert st[:30] == [C] * 30 and st[30:] == [U] * 10
    assert all("lookup limit reached" in x.reason for x in out[30:])
    # repeated names do not consume the budget
    t2 = block("bash", "pip install " + " ".join(["same"] * 5 + names[:29]))
    out2, f2 = run(t2, {pypi(n): ok(pypi(n)) for n in names + ["same"]})
    assert len(f2.calls) == 30 and all(x.status == V for x in out2)


def test_only_package_names_reach_the_fetcher():
    secret = "SECRET-ANSWER-TEXT-12345"
    t = f"{secret} says so.\n\n" + block("bash", f"pip install lib-x  # {secret}")
    out, f = run(t, {pypi("lib-x"): ok(pypi("lib-x"))})
    assert f.calls == [pypi("lib-x")]
    assert secret not in "".join(f.calls)


def test_adversarial_names_never_reach_the_fetcher():
    long = "a" * 300
    bad = ["../etc/passwd", "foo bar", "café", "ไทย", long, "$(rm -rf /)", "`id`", "a;b", "a\\b",
           "x/../y", "%2e%2e", "a%20b", "pkg?x=1", "@scope", "@/x", "@a/b/c/../../d", "-rf", "http://evil/x", "a" * 214 + "b"]
    t = block("bash", "\n".join(f"pip install {b}\nnpm install {b}\nnpx {b}" for b in bad))
    t += "\n\n" + block("js", "\n".join(f"require({b!r});\nimport x from {json.dumps(b)};" for b in bad))
    t += "\n\n" + block("python", "\n".join(f"import {b}" for b in ("café", "ไทย", "ok_name")))
    t += "\n\n" + block("bash", f'pip install "{long}" \'$(rm -rf /)\' "a b"')
    out, f = run(t, {})
    for c in f.calls:
        assert c.startswith("https://pypi.org/pypi/") or c.startswith("https://registry.npmjs.org/")
        name = c.split("/pypi/")[1][:-5] if "/pypi/" in c else c[len("https://registry.npmjs.org/"):]
        assert len(name) <= 214 and all(ch.isalnum() or ch in "-._~@%" for ch in name), c
        assert "%2e" not in name.lower() or name.startswith("@")
        assert ".." not in name and " " not in name and "$" not in name
    assert not any(long in c for c in f.calls)
    assert not any("rm" in c for c in f.calls)


def test_name_length_boundary():
    n214 = "a" * 214
    out, f = run(block("bash", f"pip install {n214}"), {pypi(n214): ok(pypi(n214))})
    assert len(out) == 1
    out, f = run(block("bash", f"pip install {'a' * 215}"), {})
    assert out == [] and f.calls == []


# ------------------------------------------------------------------ offsets / structure

def test_offsets_exact_with_crlf_thai_and_indented_fences():
    body = "# สวัสดี ก่อน น้ำ\nทดสอบ = 'a็้้' ; import json ; json.parse('กา')\nimport ghost_mod\n"
    prefix = "ทดสอบ พิสูจน์เอาไว้\r\n\r\n"
    for nl in ("\n", "\r\n"):
        for indent in ("", "   ", "\t"):
            raw = fence("python", body, nl)
            raw = "".join(indent + ln for ln in raw.splitlines(True))
            t = prefix.replace("\r\n", nl) + raw + nl + "จบ" + nl
            cl = extract(t)
            assert any(c.kind == ClaimKind.CODE for c in cl), (nl, indent)
            out, _ = run(t, {pypi("ghost-mod"): nf(pypi("ghost-mod"))})
            d = by_name(out)
            assert set(d) == {"json.parse", "ghost_mod"}, (nl, indent, d.keys())
            for x in out:
                assert t[x.claim.start:x.claim.end] == x.claim.text


def test_offsets_with_combining_marks_before_the_token_and_multiline_installs():
    t = ("a\u0301\u0302\u0303 " * 5 + "\n" + block("bash", "echo \u0e01\u0e49\u0e33\u0e48 \u0e17\u0e35\u0e48\npip install \\\r\n  lib-q \\\n  lib-r 2>/dev/null\n"
                                                    "pip install \u0e01\u0e49 lib-s") + "\n")
    out, _ = run(t, {pypi(n): ok(pypi(n)) for n in ("lib-q", "lib-r", "lib-s")})
    assert sorted(x.claim.text for x in out) == ["lib-q", "lib-r", "lib-s"]


def test_ids_are_k_numbers_in_document_order_across_blocks():
    t = ("intro\n\n" + block("js", "require('zeta-js');") + "\n\n" + block("python", "import alpha_py\n")
         + "\n\n" + block("bash", "pip install mid-pkg\nnpm i last-pkg"))
    routes = {pypi("alpha-py"): ok(pypi("alpha-py")), pypi("mid-pkg"): ok(pypi("mid-pkg")),
              npm("zeta-js"): ok(npm("zeta-js")), npm("last-pkg"): ok(npm("last-pkg"))}
    out, _ = run(t, routes)
    assert [x.claim.text for x in out] == ["zeta-js", "alpha_py", "mid-pkg", "last-pkg"]
    assert [x.claim.start for x in out] == sorted(x.claim.start for x in out)


def test_non_code_claims_and_mismatched_claims_ignored():
    t = block("bash", "pip install real-thing")
    good = [c for c in extract(t) if c.kind == ClaimKind.CODE][0]
    other = Claim("c9", ClaimKind.VAGUE, good.text, good.start, good.end, dict(good.data))
    moved = Claim("c8", ClaimKind.CODE, good.text, good.start + 1, good.end + 1, dict(good.data))
    out, f = run(t, {}, claims=[other, moved])
    assert out == [] and f.calls == []
    out, f = run(t, {pypi("real-thing"): ok(pypi("real-thing"))}, claims=[other, good, moved])
    assert len(out) == 1


def test_handmade_claim_without_fences_and_without_code_data():
    code = "import json\njson.parse(1)\n"
    c = Claim("c1", ClaimKind.CODE, code, 0, len(code), {"language": "python", "code": code, "expected_output": None})
    out = codecheck.scan(code, [c], registries=False)
    assert [x.claim.text for x in out] == ["json.parse"]
    c2 = Claim("c1", ClaimKind.CODE, code, 0, len(code), {"language": "py"})
    out = codecheck.scan(code, [c2], registries=False)
    assert [x.claim.text for x in out] == ["json.parse"]


def test_unterminated_fence_still_offsets_exact():
    t = "text\n" + FENCE + "python\nimport json\njson.parse(1)"
    out, _ = run(t, registries=False)
    assert [x.claim.text for x in out] == ["json.parse"]


def test_api_ref_claim_data_shape():
    t = block("bash", "pip install real-thing")
    out, _ = run(t, {pypi("real-thing"): ok(pypi("real-thing"))})
    d = out[0].claim.data
    assert d["language"] == "python" and d["name"] == "real-thing" and d["kind"] == "package"
    assert out[0].evidence == (pypi("real-thing"),)


def test_default_fetcher_is_urllib_fetcher_but_not_used_when_registries_off():
    real = codecheck.UrllibFetcher
    made = []

    class Spy:
        def __init__(self):
            made.append(1)

        def fetch(self, *a, **k):
            return FetchResult(a[0], a[0], None, error="network")
    codecheck.UrllibFetcher = Spy
    try:
        out = codecheck.scan(block("bash", "pip install zz"), extract(block("bash", "pip install zz")))
        assert made == [1] and out[0].status == U
        made.clear()
        t = block("bash", "pip install zz")
        codecheck.scan(t, extract(t), registries=False)
        assert made == []
    finally:
        codecheck.UrllibFetcher = real


def test_fifty_blocks_finish_fast():
    parts = []
    for i in range(50):
        parts.append(block("python", f"import json, os\nimport pkg_{i}\nx = json.loads('1')\nos.path.join('a')\njson.parse{i}('x')\n"))
        parts.append(block("bash", f"pip install lib-{i % 25}"))
    t = "\n\n".join(parts)
    routes = {pypi(f"pkg-{i}"): ok(pypi(f"pkg-{i}")) for i in range(50)}
    routes.update({pypi(f"lib-{i}"): ok(pypi(f"lib-{i}")) for i in range(25)})
    t0 = time.time()
    out, f = run(t, routes)
    dt = time.time() - t0
    assert dt < 10, dt
    assert len(f.calls) == 30
    assert sum(1 for x in out if x.claim.data["origin"] == "stdlib") == 50


def test_huge_single_line_and_deep_nesting_do_not_crash():
    t = block("python", "x = " + "(" * 500 + "1" + ")" * 500 + "\nimport json\njson.nothing\n")
    out, _ = run(t, registries=False)
    assert all(x.status == U for x in out)
    t = block("js", "require('a');" * 5000)
    out, _ = run(t, {npm("a"): ok(npm("a"))})
    assert len(out) == 1


if __name__ == "__main__":
    fails = 0
    t0 = time.time()
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            n += 1
            try:
                fn(); print("ok  ", name)
            except Exception as e:
                import traceback
                fails += 1; print("FAIL", name, repr(e)); traceback.print_exc(limit=3)
    print(f"{n} tests, {time.time() - t0:.1f}s")
    sys.exit(1 if fails else 0)
