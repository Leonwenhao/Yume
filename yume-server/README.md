# Yume Server

Backend API for the Yume drawing-to-3D-world pipeline.

## Setup

```bash
cd yume-server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in FAL_KEY and MARBLE_API_KEY in .env
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Health check |
| POST | /api/generate | Upload drawing, start pipeline |
| GET | /api/world/{id}/status | Poll pipeline progress |
| GET | /api/world/{id}/assets | Get asset URLs |
| POST | /api/world/{id}/polaroid | Submit viewport screenshot |
| GET | /api/world/{id}/strip | Get final polaroid strip |
