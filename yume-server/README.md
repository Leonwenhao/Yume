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

## Assets Schema

`GET /api/world/{id}/assets` returns:

```json
{
  "world_id": "yume_...",
  "original_drawing": "/assets/{id}/drawing.png",
  "styled_image": "/assets/{id}/styled.png",
  "splat_url": "/assets/{id}/world.spz",
  "splat_ply_url": "/assets/{id}/world.ply",
  "collider_url": "/assets/{id}/collider.glb",
  "panorama_url": "/assets/{id}/panorama.png",
  "thumbnail_url": "/assets/{id}/thumbnail.png",
  "marble_viewer_url": "https://...",
  "cdn_splat_url": "https://...",
  "cdn_splat_500k_url": "https://...",
  "cdn_splat_100k_url": "https://...",
  "cdn_panorama_url": "https://...",
  "cdn_thumbnail_url": "https://...",
  "plushie_photo_url": "/assets/{id}/plushie.png",
  "plushie_styled_url": "/assets/{id}/plushie_styled.png",
  "plushie_glb_url": "/assets/{id}/plushie.glb",
  "plushie_fbx_url": "/assets/{id}/plushie.fbx",
  "plushie_thumbnail_url": "/assets/{id}/plushie_thumbnail.png",
  "cdn_plushie_glb_url": "https://...",
  "cdn_plushie_fbx_url": "https://..."
}
```

Notes:

- World local keys: `original_drawing`, `styled_image`, `splat_url`, `splat_ply_url`, `collider_url`, `panorama_url`, `thumbnail_url`
- World CDN keys: `marble_viewer_url`, `cdn_splat_url`, `cdn_splat_500k_url`, `cdn_splat_100k_url`, `cdn_panorama_url`, `cdn_thumbnail_url`
- Plushie local keys: `plushie_photo_url`, `plushie_styled_url`, `plushie_glb_url`, `plushie_fbx_url`, `plushie_thumbnail_url`
- Plushie CDN keys: `cdn_plushie_glb_url`, `cdn_plushie_fbx_url`
- World CDN keys are `null` when Marble falls back to local bundled assets.
- Plushie keys are `null` when no plushie was provided or plushie generation fails.
