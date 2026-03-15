# Load .env before anything else — fal_client reads FAL_KEY on first import
from dotenv import load_dotenv as _load_dotenv
import os as _os
_load_dotenv()
if _os.getenv("FAL_KEY"):
    _os.environ["FAL_KEY"] = _os.getenv("FAL_KEY")

import asyncio
import base64
import binascii
import logging
import re
from json import JSONDecodeError
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from fastapi.responses import FileResponse

import state
from config import YUME_ASSETS_DIR
from modules.marble import MarbleClient
from modules.meshgen import MeshyClient
from modules.polaroid import create_polaroid, create_strip
from pipeline import persist_original_drawing, persist_plushie_photo, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("yume.main")

ALLOWED_PUBLIC_STATUSES = {
    "stylizing_drawing",
    "generating_world",
    "complete",
    "failed",
}

app = FastAPI(title="Yume API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure assets directory exists
assets_path = Path(YUME_ASSETS_DIR)
assets_path.mkdir(parents=True, exist_ok=True)

app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Shared API clients
marble_client = MarbleClient()
meshy_client = MeshyClient()


def _decode_base64_image(payload: str, field_name: str) -> bytes:
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError(f"Missing '{field_name}' base64 field")

    encoded = payload.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError(f"Invalid '{field_name}' image payload")

    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid '{field_name}' image payload") from exc

    if not decoded:
        raise ValueError(f"Invalid '{field_name}' image payload")

    return decoded


def _parse_capture_number(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("capture_number must be an integer from 1 to 6")

    if isinstance(value, int):
        capture_number = value
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip() or ""):
        capture_number = int(value)
    else:
        raise ValueError("capture_number must be an integer from 1 to 6")

    if not 1 <= capture_number <= 6:
        raise ValueError("capture_number must be an integer from 1 to 6")

    return capture_number


def _public_status(world: dict) -> str:
    status = world["status"]
    if status in ALLOWED_PUBLIC_STATUSES:
        return status
    if world.get("stage", 0) >= 2:
        return "generating_world"
    return "stylizing_drawing"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/generate")
async def generate(
    request: Request,
    drawing: UploadFile | None = File(None),
    plushie: UploadFile | None = File(None),
    mode: str | None = Form(None),
):
    drawing_bytes: bytes | None = None
    plushie_bytes: bytes | None = None
    resolved_mode = "kids"

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        if drawing is None:
            return JSONResponse({"error": "Missing 'drawing' file"}, status_code=400)
        drawing_bytes = await drawing.read()
        if not drawing_bytes:
            return JSONResponse({"error": "Missing 'drawing' file"}, status_code=400)
        if plushie is not None:
            plushie_bytes = await plushie.read()
            if not plushie_bytes:
                plushie_bytes = None
        resolved_mode = mode or "kids"
    elif "application/json" in content_type:
        try:
            body = await request.json()
        except JSONDecodeError:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        b64 = body.get("drawing") or body.get("image")
        if not b64:
            return JSONResponse({"error": "Missing 'drawing' or 'image' base64 field"}, status_code=400)
        try:
            drawing_bytes = _decode_base64_image(b64, "drawing")
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        plushie_b64 = body.get("plushie")
        if plushie_b64:
            try:
                plushie_bytes = _decode_base64_image(plushie_b64, "plushie")
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
        resolved_mode = body.get("mode", "kids")
    else:
        return JSONResponse({"error": "Unsupported content type"}, status_code=400)

    if resolved_mode not in ("dreamy", "spooky"):
        resolved_mode = "dreamy"

    # Create world state
    has_plushie = plushie_bytes is not None
    world = state.create_world(resolved_mode, has_plushie=has_plushie)
    world_id = world["world_id"]

    try:
        drawing_path = persist_original_drawing(world_id, drawing_bytes, assets_path)
    except Exception as exc:
        logger.exception("Failed to persist drawing for %s: %s", world_id, exc)
        state.set_error(world_id, str(exc))
        return JSONResponse({"error": "Failed to persist drawing"}, status_code=500)

    plushie_path = None
    if plushie_bytes:
        try:
            plushie_path = persist_plushie_photo(world_id, plushie_bytes, assets_path)
        except Exception as exc:
            logger.exception("Failed to persist plushie for %s: %s", world_id, exc)
            state.update_plushie_status(world_id, "failed")
            plushie_path = None  # Non-fatal: proceed without plushie

    # Kick off pipeline in background
    asyncio.create_task(run_pipeline(
        world_id, drawing_path, resolved_mode, marble_client,
        plushie_path=plushie_path,
        meshy_client=meshy_client if plushie_path else None,
    ))

    return {
        "world_id": world_id,
        "status": world["status"],
        "created_at": world["created_at"],
    }


@app.get("/api/world/{world_id}/status")
async def world_status(world_id: str):
    world = state.get_world(world_id)
    if world is None:
        return JSONResponse({"error": "World not found"}, status_code=404)
    return {
        "world_id": world["world_id"],
        "status": _public_status(world),
        "stage": world["stage"],
        "total_stages": world["total_stages"],
        "stage_label": world["stage_label"],
        "assets": world["assets"],
        "error": world["error"],
        "plushie_status": world.get("plushie_status"),
    }


@app.get("/api/world/{world_id}/assets")
async def world_assets(world_id: str):
    world = state.get_world(world_id)
    if world is None:
        return JSONResponse({"error": "World not found"}, status_code=404)
    if world["assets"] is None:
        return JSONResponse({"error": "Assets not ready yet", "status": _public_status(world)}, status_code=202)
    return {"world_id": world_id, **world["assets"]}


@app.post("/api/world/{world_id}/polaroid")
async def create_polaroid_endpoint(world_id: str, request: Request):
    world = state.get_world(world_id)
    if world is None:
        return JSONResponse({"error": "World not found"}, status_code=404)

    try:
        body = await request.json()
    except JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    capture_b64 = body.get("capture")
    raw_capture_number = body.get("capture_number")

    if not capture_b64:
        return JSONResponse({"error": "Missing 'capture' base64 field"}, status_code=400)
    try:
        capture_number = _parse_capture_number(raw_capture_number)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    try:
        polaroid_path = create_polaroid(capture_b64, capture_number, world_id, str(assets_path))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as e:
        logger.exception("Polaroid creation failed for %s capture %s: %s", world_id, capture_number, e)
        return JSONResponse({"error": "Failed to create polaroid"}, status_code=500)

    polaroid_url = f"/assets/{world_id}/polaroid_{capture_number}.png"
    count = state.add_polaroid(world_id, polaroid_path)
    remaining = max(0, 6 - count)

    return {"polaroid_url": polaroid_url, "remaining": remaining}


@app.get("/api/world/{world_id}/strip")
async def get_strip(world_id: str):
    world = state.get_world(world_id)
    if world is None:
        return JSONResponse({"error": "World not found"}, status_code=404)

    world_dir = assets_path / world_id
    expected_polaroids = [world_dir / f"polaroid_{i}.png" for i in range(1, 7)]
    missing_polaroids = [path.name for path in expected_polaroids if not path.exists()]

    if missing_polaroids:
        existing_count = 6 - len(missing_polaroids)
        return JSONResponse(
            {
                "error": f"Need 6 polaroids, have {existing_count}",
                "missing": missing_polaroids,
            },
            status_code=400,
        )

    try:
        strip_path = create_strip(world_id, str(assets_path))
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as e:
        logger.exception("Strip creation failed for %s: %s", world_id, e)
        return JSONResponse({"error": "Failed to create strip"}, status_code=500)

    strip_url = f"/assets/{world_id}/strip.png"
    polaroid_urls = [f"/assets/{world_id}/polaroid_{i}.png" for i in range(1, 7)]
    return {
        "strip_url": strip_url,
        "polaroids": polaroid_urls,
        "original_drawing": f"/assets/{world_id}/drawing.png",
    }
