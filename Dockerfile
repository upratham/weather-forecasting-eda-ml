# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + serve built frontend ────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install only the web-app backend dependencies (not the heavy ML stack)
COPY requirements.txt ./
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    "uvicorn[standard]>=0.29.0" \
    httpx>=0.27.0 \
    sqlalchemy>=2.0.0 \
    aiofiles>=23.2.0 \
    python-multipart>=0.0.9

# Copy backend source
COPY backend/ ./backend/
COPY run.py ./

# Copy built React app from stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# SQLite database will be created here at runtime
VOLUME ["/app/data"]

# Map weather_app.db into the volume so it persists across container restarts
ENV DATABASE_URL=sqlite:////app/data/weather_app.db

EXPOSE 8000

CMD ["python", "run.py"]
