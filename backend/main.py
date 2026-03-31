import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

from routers import upload, stats, charts, models, session, compute, repeated, advanced_anova, categorical, agreement, reliability, missing_data
from services import store

app = FastAPI(title="Wizard Stats API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all localhost ports
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(charts.router, prefix="/api/charts", tags=["charts"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(session.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(compute.router, prefix="/api/compute", tags=["compute"])
app.include_router(repeated.router, prefix="/api/repeated", tags=["repeated"])
app.include_router(advanced_anova.router, prefix="/api/advanced_anova", tags=["advanced_anova"])
app.include_router(categorical.router, prefix="/api/categorical", tags=["categorical"])
app.include_router(agreement.router, prefix="/api/agreement", tags=["agreement"])
app.include_router(reliability.router, prefix="/api/reliability", tags=["reliability"])
app.include_router(missing_data.router, prefix="/api/missing_data", tags=["missing_data"])


@app.get("/api/health")
def health():
    """Lightweight health check — no expensive deep memory scan."""
    result: dict = {"status": "ok", "active_sessions": len(store.list_sessions())}

    if psutil:
        process = psutil.Process()
        mem_info = process.memory_info()
        result["memory"] = {
            "process_rss_mb": round(mem_info.rss / (1024 * 1024), 1),
            "process_percent": round(process.memory_percent(), 1),
        }

    return result


# Serve compiled React frontend (production build).
# Must come AFTER all /api routes so API routes are matched first.
_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
