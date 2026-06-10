"""Child process for the server-side Python sandbox.

This script is exec'd as an isolated subprocess by `services.sandbox`. It:

  1. Installs hard resource limits (CPU, address space, file size, NOFILE,
     NPROC) via `resource.setrlimit`.
  2. Installs an import allowlist on `sys.meta_path` so user code cannot
     import network / OS / subprocess modules.
  3. Reads `{ "code": str, "df_pickle": <base64> }` from stdin.
  4. Injects `df`, `np`, `pd` into a clean globals dict.
  5. Captures stdout / stderr to StringIO, runs the user code with `exec()`.
  6. After the run, dumps any open matplotlib figures as PNG → base64.
  7. Emits `{stdout, stderr, figures, error}` as a single JSON line on the
     real fd-3 (the parent reads only fd-3 so user `print()` does not pollute
     the protocol).

Run as:
    python -I -u backend/services/code_runner_child.py

Stdin / stdout / stderr are connected to pipes from the parent. fd-3 is the
result channel.
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import resource
import sys
import traceback


# ── Hard resource limits (applied BEFORE importing user-touched libraries) ──

_RLIMITS = {
    "cpu":      ("RLIMIT_CPU",    int(os.environ.get("SANDBOX_CPU_SEC", "30"))),
    "as":       ("RLIMIT_AS",     int(os.environ.get("SANDBOX_MEM_BYTES", str(512 * 1024 * 1024)))),
    "fsize":    ("RLIMIT_FSIZE",  int(os.environ.get("SANDBOX_FSIZE_BYTES", str(10 * 1024 * 1024)))),
    "nofile":   ("RLIMIT_NOFILE", int(os.environ.get("SANDBOX_NOFILE", "64"))),
    "nproc":    ("RLIMIT_NPROC",  int(os.environ.get("SANDBOX_NPROC", "32"))),
}


def _apply_rlimits() -> None:
    """Apply rlimits. Some limits are platform-dependent — silently skip
    unsupported ones."""
    for label, (rname, limit) in _RLIMITS.items():
        const = getattr(resource, rname, None)
        if const is None:
            continue
        try:
            soft, hard = resource.getrlimit(const)
            new_soft = min(limit, hard) if hard != resource.RLIM_INFINITY else limit
            new_hard = min(limit, hard) if hard != resource.RLIM_INFINITY else limit
            resource.setrlimit(const, (new_soft, new_hard))
        except (ValueError, OSError):
            # Some macOS limits (NPROC) can fail when called this way. Best
            # effort — the parent enforces wall-clock timeout regardless.
            pass


# ── Import allowlist hook ───────────────────────────────────────────────────

# Top-level module names that user code is allowed to import. Submodules of
# these are implicitly allowed.
_ALLOWED_TOPLEVEL = {
    # numerics
    "numpy", "pandas", "scipy", "statsmodels", "lifelines", "sklearn",
    "patsy", "joblib", "threadpoolctl",
    # plotting
    "matplotlib", "seaborn",
    # stdlib (read-only / safe)
    "math", "statistics", "datetime", "json", "re", "itertools",
    "collections", "functools", "typing", "dataclasses", "warnings",
    "io", "base64", "decimal", "fractions", "random", "string",
    "operator", "enum", "copy", "bisect", "heapq", "textwrap",
    "csv", "uuid", "abc", "numbers", "array", "struct",
    # required by numpy/pandas plumbing
    "sys", "_io", "encodings", "codecs", "atexit", "weakref",
    "platform", "locale", "calendar",
}

# Explicit deny — overrides any submodule match.
_DENIED = {
    "socket", "ssl", "http", "urllib", "urllib2", "urllib3", "requests",
    "httplib", "ftplib", "smtplib", "telnetlib", "asyncio", "asyncore",
    "subprocess", "multiprocessing", "concurrent",
    "ctypes", "cffi", "ctypes.util",
    "os", "shutil", "tempfile", "pathlib", "glob",
    "importlib", "imp", "pkgutil", "runpy",
    "pty", "termios", "fcntl",
    "pickle", "shelve", "dbm", "sqlite3",
    "xmlrpc", "wsgiref",
    "_socket", "_ssl",
}


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


class _DenyImportFinder:
    """sys.meta_path finder that vetoes blocked imports.

    Listed FIRST in sys.meta_path so that anything matching `_DENIED` raises
    ImportError before the normal finders run. Allowed names pass through —
    the regular finders handle them.
    """

    def find_spec(self, name, path=None, target=None):  # noqa: D401 – matches API
        root = _root_module(name)
        if name in _DENIED or root in _DENIED:
            raise ImportError(f"sandbox: import of '{name}' is blocked")
        if root not in _ALLOWED_TOPLEVEL:
            # Walk the call stack to see if this import was triggered by an
            # already-allowed library (lazy submodule loads). If so, allow it.
            import sys as _sys
            frame = _sys._getframe(1) if hasattr(_sys, "_getframe") else None
            while frame is not None:
                mod = frame.f_globals.get("__name__", "")
                if mod and _root_module(mod) in _ALLOWED_TOPLEVEL:
                    return None
                frame = frame.f_back
            raise ImportError(f"sandbox: import of '{name}' is not in the allowlist")
        return None


def _install_import_guard() -> None:
    sys.meta_path.insert(0, _DenyImportFinder())


# ── Restricted builtins for user code ───────────────────────────────────────

# Removed from the user namespace: the obvious filesystem / dynamic-exec
# escapes. The import allowlist (meta_path) still governs modules; this is
# defense-in-depth ON TOP OF the read-only container filesystem (the real
# boundary — user code runs as a non-root user that cannot write /app).
_BLOCKED_BUILTINS = {"open", "exec", "eval", "compile", "input", "breakpoint", "help"}


def _restricted_builtins() -> dict:
    import builtins as _b
    return {name: getattr(_b, name) for name in dir(_b) if name not in _BLOCKED_BUILTINS}


# ── Main entry ──────────────────────────────────────────────────────────────

def _main() -> int:
    _apply_rlimits()
    _install_import_guard()

    # Force matplotlib non-interactive backend BEFORE any user import.
    os.environ.setdefault("MPLBACKEND", "Agg")

    raw = sys.stdin.buffer.read()
    try:
        msg = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        _emit({"stdout": "", "stderr": "", "figures": [],
               "error": f"sandbox: malformed input: {exc}"})
        return 2

    code = msg.get("code", "")
    df_b64 = msg.get("df_pickle")
    df = None
    if df_b64:
        try:
            df_bytes = base64.b64decode(df_b64.encode("ascii"))
            # Pickle of a pandas DataFrame uses numpy/pandas internals — load
            # AFTER the import guard is in place so the legitimate pandas
            # imports are exercised. pickle itself is denied for user code but
            # the runner uses it through pandas.read_pickle equivalent below.
            import io as _io
            import pandas as _pd  # noqa: F401  – imported into globals later
            df = _pd.read_pickle(_io.BytesIO(df_bytes))
        except Exception as exc:
            _emit({"stdout": "", "stderr": "", "figures": [],
                   "error": f"sandbox: failed to load session df: {exc}"})
            return 3

    # Capture stdout / stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf

    figures: list[str] = []
    err_msg: str | None = None

    # Run from an isolated scratch dir under the system temp dir (/tmp), never
    # the app tree, so a relative-path write can't land in /app/backend. HOME is
    # NOT used: in the container it points at /home/app which is intentionally
    # absent/non-writable, so a HOME-based scratch silently fails and leaves cwd
    # on the read-only app tree. tempfile.gettempdir() is reliably writable.
    try:
        import tempfile
        _scratch = os.path.join(tempfile.gettempdir(), "sandbox_run")
        os.makedirs(_scratch, exist_ok=True)
        os.chdir(_scratch)
    except Exception:
        pass

    user_globals: dict = {"__name__": "__sandbox__", "__builtins__": _restricted_builtins()}
    if df is not None:
        try:
            import numpy as _np
            import pandas as _pd
            user_globals["df"] = df
            user_globals["np"] = _np
            user_globals["pd"] = _pd
        except Exception as exc:
            err_msg = f"sandbox: pandas/numpy unavailable: {exc}"

    if err_msg is None:
        try:
            compiled = compile(code, "<user_code>", "exec")
            exec(compiled, user_globals)
        except SystemExit:
            err_msg = "sandbox: user code called sys.exit()"
        except BaseException:
            err_msg = traceback.format_exc(limit=20)

    # Collect matplotlib figures (if any)
    try:
        import matplotlib.pyplot as _plt
        nums = list(_plt.get_fignums())
        for num in nums:
            try:
                fig = _plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                figures.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            except Exception:
                continue
        _plt.close("all")
    except ImportError:
        # User code did not touch matplotlib — fine.
        pass
    except Exception:
        # Don't let figure collection mask the user error
        pass

    sys.stdout, sys.stderr = old_out, old_err
    _emit({
        "stdout":  stdout_buf.getvalue(),
        "stderr":  stderr_buf.getvalue(),
        "figures": figures,
        "error":   err_msg,
    })
    return 0


def _emit(payload: dict) -> None:
    """Write the result JSON to fd-3 if available, otherwise stdout."""
    data = (json.dumps(payload) + "\n").encode("utf-8")
    try:
        os.write(3, data)
    except OSError:
        sys.__stdout__.buffer.write(data)
        sys.__stdout__.buffer.flush()


if __name__ == "__main__":
    sys.exit(_main())
