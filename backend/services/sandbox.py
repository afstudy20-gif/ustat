"""Server-side Python sandbox runner.

The user-supplied code is executed in a separate subprocess
(`code_runner_child.py`) that installs rlimits and an import allowlist. This
module is the parent-side driver: it serialises the session DataFrame,
spawns the child, enforces a wall-clock timeout, and parses the structured
JSON the child writes to fd-3.

Sandbox guarantees (best-effort, see SECURITY.md):
- Wall-clock timeout (default 30 s, max 60 s).
- CPU / address space / file size / nofile / nproc rlimits in the child.
- Import allowlist via `sys.meta_path` finder.
- `MPLBACKEND=Agg` so figures stay off-screen.
- (Optional, Linux) `unshare -n` for network isolation when available.

Production hardening (containerisation, syscall filtering, dedicated user)
is documented as a roadmap step in `backend/SECURITY.md`.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


CHILD_PATH = Path(__file__).with_name("code_runner_child.py")

DEFAULT_TIMEOUT_S = 30
MAX_TIMEOUT_S     = 60
MAX_CODE_BYTES    = 100 * 1024  # 100 KB


@dataclass
class SandboxResult:
    stdout:    str
    stderr:    str
    figures:   list[str]   # base64 PNGs
    exit_code: int
    time_used_s: float
    error:     Optional[str]


def _serialise_df(df: pd.DataFrame) -> str:
    """Pickle a DataFrame to base64. Pickle is acceptable here because the
    child consumes its own trusted payload — user code can NEVER reach the
    pickle module thanks to the allowlist."""
    buf = io.BytesIO()
    df.to_pickle(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _network_wrapper() -> list[str]:
    """Return a command prefix that strips network access when possible."""
    if sys.platform == "linux":
        unshare = shutil.which("unshare")
        if unshare is not None:
            return [unshare, "--user", "--net", "--map-root-user"]
    return []


def run_python(
    code: str,
    df: Optional[pd.DataFrame] = None,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> SandboxResult:
    """Execute `code` against optional `df` inside the sandbox child.

    Args:
        code: Python source.
        df: optional DataFrame, exposed as `df` in user globals.
        timeout: wall-clock timeout, clamped to [1, MAX_TIMEOUT_S].

    Returns:
        SandboxResult.

    Notes:
        Never raises on user-code errors — captures them via the child's
        stderr / `error` field. Only raises when the sandbox itself cannot
        spawn the subprocess.
    """
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return SandboxResult(stdout="", stderr="", figures=[], exit_code=1,
                             time_used_s=0.0,
                             error=f"sandbox: code exceeds {MAX_CODE_BYTES} bytes")

    timeout = max(1, min(int(timeout or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S))

    payload: dict[str, object] = {"code": code}
    if df is not None:
        try:
            payload["df_pickle"] = _serialise_df(df)
        except Exception as exc:
            return SandboxResult(stdout="", stderr="", figures=[], exit_code=1,
                                 time_used_s=0.0,
                                 error=f"sandbox: failed to serialise session df: {exc}")

    payload_bytes = json.dumps(payload).encode("utf-8")

    # fd-3 is the result channel; the parent reads it after the child exits.
    r_fd, w_fd = os.pipe()

    cmd = _network_wrapper() + [sys.executable, "-I", "-u", str(CHILD_PATH)]

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": "",
        "PYTHONHASHSEED": "0",
        "MPLBACKEND": "Agg",
        "HOME": "/tmp",
        # The result channel is the inherited pipe write-end. pass_fds keeps it
        # open at THIS fd number in the child (it is NOT remapped to fd 3), so
        # the child must learn the real number rather than assume 3.
        "SANDBOX_RESULT_FD": str(w_fd),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "SANDBOX_CPU_SEC":    str(timeout),
        "SANDBOX_MEM_BYTES":  str(int(os.environ.get("SANDBOX_MEM_BYTES", 512 * 1024 * 1024))),
        "SANDBOX_FSIZE_BYTES": str(int(os.environ.get("SANDBOX_FSIZE_BYTES", 10 * 1024 * 1024))),
        "SANDBOX_NOFILE":     str(int(os.environ.get("SANDBOX_NOFILE", 64))),
        "SANDBOX_NPROC":      str(int(os.environ.get("SANDBOX_NPROC", 32))),
    }

    start = time.monotonic()
    proc = None
    timed_out = False

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(w_fd,),
            env=env,
            close_fds=True,
            start_new_session=True,
        )
        # Close our copy of the write end so the child sees EOF when it exits
        os.close(w_fd)
        w_fd = -1

        try:
            stdout_b, stderr_b = proc.communicate(input=payload_bytes, timeout=timeout + 2)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
            try:
                stdout_b, stderr_b = proc.communicate(timeout=2)
            except Exception:
                stdout_b, stderr_b = b"", b""

        elapsed = time.monotonic() - start

        # Drain fd-3 (the result channel)
        chunks: list[bytes] = []
        try:
            while True:
                c = os.read(r_fd, 65536)
                if not c:
                    break
                chunks.append(c)
                # Hard upper bound to keep memory bounded
                if sum(len(x) for x in chunks) > 16 * 1024 * 1024:
                    break
        finally:
            try:
                os.close(r_fd)
            except OSError:
                pass

        raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
        result: dict = {}
        if raw:
            try:
                # The child may write multiple JSON lines; take the last
                # non-empty line as the canonical result.
                for line in raw.splitlines()[::-1]:
                    line = line.strip()
                    if line:
                        result = json.loads(line)
                        break
            except Exception:
                result = {"error": f"sandbox: malformed child output: {raw[:500]}"}

        if timed_out:
            result.setdefault("error", f"sandbox: timeout after {timeout}s")

        return SandboxResult(
            stdout    = result.get("stdout", "") or "",
            stderr    = (result.get("stderr", "") or "") + (stderr_b.decode("utf-8", errors="replace") if stderr_b else ""),
            figures   = result.get("figures", []) or [],
            exit_code = proc.returncode if proc.returncode is not None else -1,
            time_used_s = round(elapsed, 3),
            error     = result.get("error"),
        )
    except FileNotFoundError as exc:
        return SandboxResult(stdout="", stderr="", figures=[], exit_code=1,
                             time_used_s=time.monotonic() - start,
                             error=f"sandbox: failed to spawn interpreter: {exc}")
    finally:
        if w_fd != -1:
            try:
                os.close(w_fd)
            except OSError:
                pass
