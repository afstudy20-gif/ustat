"""HTTP security-headers middleware.

Adds the standard browser-hardening headers to every response:

  - Content-Security-Policy: restricts script/style/connect origins.
  - Strict-Transport-Security: forces HTTPS for one year, includes subdomains.
  - X-Frame-Options + frame-ancestors: prevents clickjacking via iframe embed.
  - X-Content-Type-Options: stops MIME-sniff attacks.
  - Referrer-Policy: strip referrers to third parties.
  - Permissions-Policy: deny powerful APIs (camera, geolocation, etc.).
  - Cross-Origin-Opener-Policy: process isolation for Spectre-class issues.

CSP is **report-only by default** so a deploy never breaks the front-end on
an unexpected inline-script or third-party resource. Flip to enforce by
setting `CSP_ENFORCE=1` once you have observed reports and tuned the policy
(see SECURITY.md for the procedure).
"""

from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


_DEFAULT_CSP = "; ".join([
    "default-src 'self'",
    # Plotly / PWA service worker / inline tailwind styles
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob: https:",
    "font-src 'self' data:",
    "connect-src 'self' https://mapmyvisitors.com",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "object-src 'none'",
])


_DEFAULT_PERMISSIONS = ", ".join([
    "camera=()",
    "microphone=()",
    "geolocation=()",
    "payment=()",
    "usb=()",
    "magnetometer=()",
    "accelerometer=()",
    "gyroscope=()",
])


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply browser-hardening headers to every response."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response: Response = await call_next(request)

        csp = os.environ.get("CSP_POLICY") or _DEFAULT_CSP
        enforce_csp = os.environ.get("CSP_ENFORCE", "").lower() in ("1", "true", "yes", "on")
        csp_header = "Content-Security-Policy" if enforce_csp else "Content-Security-Policy-Report-Only"
        response.headers.setdefault(csp_header, csp)

        # HSTS — one year, subdomains, opt-in to preload list.
        # Render terminates TLS in front of us, so this is safe.
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains; preload",
        )
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", _DEFAULT_PERMISSIONS)
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        # Server header leaks framework name otherwise. starlette's
        # MutableHeaders does not implement .pop, so use del + membership.
        if "server" in response.headers:
            del response.headers["server"]
        return response
