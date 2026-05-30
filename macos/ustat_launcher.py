"""macOS launcher for the bundled uSTAT desktop app."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn


def _resource_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def _find_port(preferred: int = 8000) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No available local port found for uSTAT.")


def _open_browser(port: int) -> None:
    time.sleep(1.5)
    webbrowser.open(f"http://127.0.0.1:{port}/")


def main() -> None:
    root = _resource_root()
    backend_dir = root / "backend"
    frontend_dist = root / "frontend" / "dist"

    sys.path.insert(0, str(backend_dir))
    os.environ["USTAT_FRONTEND_DIST"] = str(frontend_dist)

    from main import app  # pylint: disable=import-outside-toplevel

    port = _find_port()
    if os.environ.get("USTAT_NO_BROWSER") != "1":
        threading.Thread(target=_open_browser, args=(port,), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
