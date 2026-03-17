import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import upload, stats, charts, models, session, compute

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


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve compiled React frontend (production build).
# Must come AFTER all /api routes so API routes are matched first.
_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
