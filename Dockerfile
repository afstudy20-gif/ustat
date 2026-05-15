FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps
COPY frontend/ ./
RUN npx vite build

FROM python:3.11-slim
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=frontend-builder /app/frontend/dist ../frontend/dist

# Run as non-root for defense-in-depth — addresses the Semgrep
# dockerfile.security.missing-user finding. UID 10001 is a high arbitrary
# UID that doesn't clash with common host users when the image is run with
# --userns=keep-id. /app + /tmp/sandbox need to be writable.
RUN groupadd -r app --gid 10001 \
 && useradd -r -g app --uid 10001 --no-create-home --shell /sbin/nologin app \
 && mkdir -p /tmp/sandbox /app/backend/logs \
 && chown -R app:app /app /tmp/sandbox
USER app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-max-requests", "1000", "--timeout-keep-alive", "30"]
