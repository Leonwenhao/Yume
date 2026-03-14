# YUME Infrastructure Development Plan v2

## Overview

This document defines the backend infrastructure for Yume. The system takes a user's drawing, transforms it into a stylized environment image via Fal AI's FLUX.1 [dev] image-to-image model, feeds that image to the Marble API for 3D world generation, and serves the resulting assets to any client (browser frontend or VR headset). The architecture is a modular REST API that any client can integrate with independently.

**Owner:** Leon (PM/orchestrator, prompt craft)
**Execution:** Codex (implementation), Claude Code (task generation, CTO review)
**Runtime:** Python / FastAPI (recommended for async pipeline + Pillow image work)
**Deployment:** Local dev server during hackathon, ngrok or similar for team access

---

## Architecture

The previous plan had three hops: analyze drawing with LLM → generate image from analysis → send image to Marble. The updated pipeline has two hops: transform drawing directly via Fal AI img-to-img → send styled image to Marble. This is simpler, faster, and preserves the spatial composition of the original drawing because the img-to-img model uses the drawing's layout as its structural foundation rather than an LLM's text interpretation of it.

```
Drawing (image file) + mode flag ("kids" | "filmmaker")
       │
       ▼
┌─────────────────────────────────────┐
│  POST /api/generate                 │
│  Yume API Server (FastAPI)          │
│                                     │
│  1. Validate + store drawing        │
│  2. Upload drawing to Fal storage   │
│  3. Call FLUX.1 [dev] img-to-img    │──► Fal AI API
│     (drawing + style prompt)        │
│  4. Download styled image           │
│  5. Submit styled image to Marble   │──► Marble API (async)
│  6. Poll Marble until complete      │
│  7. Download + store Marble assets  │
│  8. Update status → complete        │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  GET /api/world/:id/status          │
│  GET /api/world/:id/assets          │
│  POST /api/world/:id/polaroid       │
│  GET /api/world/:id/strip           │
│                                     │
│  Any client (browser, VR) fetches   │
│  assets and renders independently   │
└─────────────────────────────────────┘
```

**Total pipeline time estimate:** ~35-55 seconds. Fal FLUX.1 img-to-img takes ~3-5 seconds. Marble mini takes ~30-45 seconds. Everything else (upload, download, storage) is under 2 seconds. The bottleneck is Marble.

---

## Module Breakdown

### Module 1: API Server Scaffold

The foundation. A FastAPI server with CORS enabled, static file serving, and all endpoints defined.

**Endpoints:**

`POST /api/generate` — Accepts a drawing image. Kicks off the pipeline as a background task. Returns a world_id immediately so the client can poll.

```json
// Request: multipart/form-data with field "drawing" (image file)
// OR JSON body:
{
  "drawing": "<base64 encoded image>",
  "format": "png",
  "mode": "kids"  // "kids" | "filmmaker"
}

// Response (immediate, 200):
{
  "world_id": "yume_abc123",
  "status": "processing",
  "created_at": "2026-03-15T10:00:00Z"
}
```

`GET /api/world/:id/status` — Returns current pipeline state. Clients poll this every 2-3 seconds.

```json
{
  "world_id": "yume_abc123",
  "status": "stylizing_drawing | generating_world | complete | failed",
  "stage": 1,
  "total_stages": 2,
  "stage_label": "Turning your drawing into a world...",
  "estimated_seconds_remaining": 35,
  "assets": null,
  "error": null
}
```

`GET /api/world/:id/assets` — Returns asset URLs once world generation is complete. This is the primary integration point for both frontend and VR teams.

```json
{
  "world_id": "yume_abc123",
  "original_drawing": "/assets/yume_abc123/drawing.png",
  "styled_image": "/assets/yume_abc123/styled.png",
  "splat_url": "/assets/yume_abc123/world.spz",
  "splat_ply_url": "/assets/yume_abc123/world.ply",
  "collider_url": "/assets/yume_abc123/collider.glb",
  "panorama_url": "/assets/yume_abc123/panorama.png",
  "thumbnail_url": "/assets/yume_abc123/thumbnail.png"
}
```

`POST /api/world/:id/polaroid` — Accepts a viewport screenshot from the client, composites it into a polaroid frame, stores it.

```json
// Request:
{
  "capture": "<base64 encoded screenshot>",
  "capture_number": 1
}

// Response:
{
  "polaroid_url": "/assets/yume_abc123/polaroid_1.png",
  "remaining": 5
}
```

`GET /api/world/:id/strip` — Returns the final polaroid strip (all 6 captures + original drawing composited). Called after the 6th capture.

```json
{
  "strip_url": "/assets/yume_abc123/strip.png",
  "polaroids": [
    "/assets/yume_abc123/polaroid_1.png",
    "/assets/yume_abc123/polaroid_2.png",
    "/assets/yume_abc123/polaroid_3.png",
    "/assets/yume_abc123/polaroid_4.png",
    "/assets/yume_abc123/polaroid_5.png",
    "/assets/yume_abc123/polaroid_6.png"
  ],
  "original_drawing": "/assets/yume_abc123/drawing.png"
}
```

`GET /api/health` — Returns 200.

**Implementation notes:**
- In-memory dict for world state. No database needed for hackathon.
- Serve generated assets from local `/assets` directory via FastAPI static files.
- world_id = `yume_` + 8 random alphanumeric chars.
- Pipeline runs as asyncio background task. POST returns immediately.

---

### Module 2: Fal AI Image Stylization

This is the core transformation: drawing → styled environment image. Uses Fal AI's FLUX.1 [dev] image-to-image endpoint. The model takes the drawing as a structural reference and the prompt as a style directive, producing a detailed environment image that preserves the spatial composition of the original drawing.

**Endpoint:** `fal-ai/flux/dev/image-to-image`

**Key parameters:**
- `image_url` — URL of the uploaded drawing (use Fal's file storage for upload)
- `prompt` — Style directive that controls the output aesthetic
- `strength` — How much the output departs from the input (0.0 = identical to input, 1.0 = fully reimagined). Default 0.95. For Yume, use 0.85-0.90 to preserve spatial layout while fully transforming visual quality.
- `guidance_scale` — How closely the model follows the prompt. Default 3.5. Higher values = more prompt adherence.
- `num_inference_steps` — Quality vs speed tradeoff. Default 40. Can reduce to 28-30 for faster hackathon iteration.
- `output_format` — Use "png" for Marble input quality.

**Implementation:**

```python
import fal_client
import httpx
import os

# --- Fal storage upload ---
# Fal's img-to-img endpoint requires the input image as a URL.
# Upload the drawing to Fal's storage first, get back a URL.
async def upload_to_fal(image_path: str) -> str:
    """Upload a local image file to Fal's CDN storage. Returns a URL."""
    url = fal_client.upload_file(image_path)
    return url


# --- Style prompts ---
# These are the creative prompts that control what the styled image looks like.
# They are the single most important piece of prompt engineering in the pipeline.
# Leon will iterate on these during R&D. The code should load them from
# the /prompts directory so they can be swapped without touching code.

STYLE_PROMPTS = {
    "kids": (
        "A magical, whimsical fantasy landscape environment. Lush and vibrant colors, "
        "soft volumetric lighting, golden hour warmth. Rich environmental detail with "
        "fantastical elements. Wide-angle perspective with atmospheric depth. "
        "Studio Ghibli-inspired warmth and wonder. A single coherent immersive scene "
        "that feels like a place you could walk into. Detailed textures on every surface."
    ),
    "filmmaker": (
        "A cinematic environment with dramatic lighting and rich atmosphere. "
        "Detailed architectural and natural textures, strong depth of field, "
        "volumetric haze. Professional production design quality. Wide-angle "
        "perspective, single coherent scene, immersive and navigable. "
        "Photorealistic materials and surfaces with clear spatial structure."
    ),
}

def load_style_prompt(mode: str) -> str:
    """
    Load style prompt for given mode. First checks /prompts directory
    for a .txt file override, then falls back to hardcoded defaults.
    This lets Leon swap prompts without restarting the server.
    """
    prompt_file = f"prompts/{mode}.txt"
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            return f.read().strip()
    return STYLE_PROMPTS.get(mode, STYLE_PROMPTS["kids"])


# --- Core stylization function ---
async def stylize_drawing(
    image_path: str,
    mode: str = "kids",
    strength: float = 0.88,
    num_inference_steps: int = 35,
    guidance_scale: float = 3.5,
) -> str:
    """
    Transform a drawing into a styled environment image via Fal FLUX.1 [dev].
    
    Args:
        image_path: Local path to the drawing file
        mode: "kids" or "filmmaker" — selects the style prompt
        strength: How much to transform (0.85-0.90 recommended for Yume)
        guidance_scale: Prompt adherence (3.5 default is good)
        num_inference_steps: Quality steps (35 balances speed + quality)
    
    Returns:
        Local path to the downloaded styled image
    """
    # Step 1: Upload drawing to Fal storage
    image_url = await upload_to_fal(image_path)
    
    # Step 2: Load the style prompt for this mode
    prompt = load_style_prompt(mode)
    
    # Step 3: Call FLUX.1 [dev] img-to-img
    result = fal_client.subscribe(
        "fal-ai/flux/dev/image-to-image",
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "output_format": "png",
            "num_images": 1,
        },
    )
    
    # Step 4: Download the generated image
    output_url = result["images"][0]["url"]
    output_path = image_path.replace("drawing", "styled")
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(output_url)
        with open(output_path, "wb") as f:
            f.write(resp.content)
    
    return output_path
```

**Prompt engineering notes for Leon:**

The style prompt has two jobs: (1) make the output look like a real, detailed environment, and (2) make the output produce a good Marble world. These are related but not identical. Marble needs clear depth cues, consistent lighting direction, unambiguous spatial structure, and rich surface texture. A beautiful but flat illustration will produce a worse Marble world than a less pretty but spatially coherent environment image.

Key phrases to test in your prompts:
- "wide-angle perspective" — gives Marble depth information
- "volumetric lighting" — helps Marble understand 3D geometry through light/shadow
- "single coherent scene" — prevents collage/grid outputs that confuse Marble
- "detailed textures on every surface" — gives the Gaussian splat more detail to reconstruct
- "atmospheric depth" — foreground/background separation helps Marble's 3D understanding

Key phrases to avoid:
- "painting" / "illustration" / "artwork" — may produce flat outputs with less depth
- "top-down view" / "aerial view" — Marble works best with eye-level perspectives
- "multiple views" / "collage" — Marble expects one coherent scene

The `strength` parameter is your primary creative control. At 0.90+, a crayon scribble becomes a fully realized environment that happens to share the drawing's spatial layout. At 0.75-0.80, more of the drawing's line quality bleeds through, which is aesthetically interesting but likely worse for Marble. Start at 0.88, test up and down from there.

**Hot-swap prompts without restarting the server:** The `load_style_prompt` function checks the `/prompts` directory first. To change the style, just edit `prompts/kids.txt` or `prompts/filmmaker.txt` and the next generation will use the new prompt. No restart, no redeploy.

---

### Module 3: Marble API Client

Handles the full async lifecycle of Marble world generation. Standalone module that accepts an image path and returns asset URLs.

**Marble API reference:** https://docs.worldlabs.ai/api

**Important: Read the actual Marble docs before implementing.** The pseudocode below is based on the hackathon docs' description. Confirm endpoint paths, auth headers, request/response schemas, and asset download URLs from the live docs. The ML engineer or Leon should verify this against the real API during the first hour.

```python
import asyncio
import httpx
import base64
import os
import logging

logger = logging.getLogger(__name__)


class MarbleError(Exception):
    """Raised when Marble generation fails."""
    pass


class MarbleClient:
    """
    Async client for World Labs Marble API.
    Handles the submit → poll → download lifecycle.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # CONFIRM THIS BASE URL FROM DOCS — may differ
        self.base_url = "https://api.worldlabs.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    async def generate_world(
        self,
        image_path: str,
        text_prompt: str | None = None,
        model: str = "marble-0.1-mini",
        poll_interval: float = 3.0,
    ) -> dict:
        """
        Full lifecycle: submit image → poll for completion → return asset URLs.
        
        Args:
            image_path: Local path to the styled environment image
            text_prompt: Optional text description to accompany the image
            model: Marble model variant. "marble-0.1-mini" = 30-45s, cheaper.
                   "marble-0.1-plus" = higher quality, slower.
            poll_interval: Seconds between status checks
        
        Returns:
            Dict of asset type → URL mappings (splat, collider, panorama, thumbnail)
        """
        # Step 1: Read and encode image
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        # Step 2: Submit generation job
        # CONFIRM PAYLOAD SCHEMA FROM MARBLE DOCS
        payload = {
            "model": model,
            "input": {
                "image": image_b64,
            },
        }
        if text_prompt:
            payload["input"]["text"] = text_prompt
        
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/generations",
                headers=self.headers,
                json=payload,
            )
            resp.raise_for_status()
            job = resp.json()
        
        job_id = job["id"]
        logger.info(f"Marble job submitted: {job_id}")
        
        # Step 3: Poll until complete
        async with httpx.AsyncClient(timeout=15) as client:
            while True:
                resp = await client.get(
                    f"{self.base_url}/generations/{job_id}",
                    headers=self.headers,
                )
                resp.raise_for_status()
                status = resp.json()
                
                if status["status"] == "complete":
                    logger.info(f"Marble job complete: {job_id}")
                    return status["assets"]
                
                if status["status"] == "failed":
                    error_msg = status.get("error", "Unknown Marble error")
                    logger.error(f"Marble job failed: {job_id} — {error_msg}")
                    raise MarbleError(error_msg)
                
                await asyncio.sleep(poll_interval)
    
    async def download_assets(self, assets: dict, output_dir: str) -> dict:
        """
        Download all Marble asset files to a local directory.
        
        Args:
            assets: Dict from generate_world (asset_type → URL)
            output_dir: Local directory to save files into
        
        Returns:
            Dict of asset_type → local_path
        """
        os.makedirs(output_dir, exist_ok=True)
        local_paths = {}
        
        async with httpx.AsyncClient(timeout=60) as client:
            for asset_type, url in assets.items():
                # Determine file extension from URL or asset type
                ext = self._infer_extension(asset_type, url)
                local_path = os.path.join(output_dir, f"{asset_type}{ext}")
                
                resp = await client.get(url)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                
                local_paths[asset_type] = local_path
                logger.info(f"Downloaded {asset_type} → {local_path}")
        
        return local_paths
    
    def _infer_extension(self, asset_type: str, url: str) -> str:
        """Map asset types to file extensions."""
        type_map = {
            "splat": ".spz",
            "splat_ply": ".ply",
            "collider": ".glb",
            "panorama": ".png",
            "thumbnail": ".png",
        }
        if asset_type in type_map:
            return type_map[asset_type]
        # Fallback: extract from URL
        for ext in [".spz", ".ply", ".glb", ".png", ".jpg"]:
            if ext in url:
                return ext
        return ".bin"
```

**Fallback strategy for demo reliability:**

Pre-generate 2-3 worlds before the demo using known-good styled images. If Marble fails or times out during a live demo, return the pre-generated assets instead. This is non-negotiable demo insurance.

```python
# Pre-generated fallback worlds — asset paths stored locally
FALLBACK_WORLDS = {
    "kids": {
        "splat": "assets/fallback_kids/world.spz",
        "collider": "assets/fallback_kids/collider.glb",
        "panorama": "assets/fallback_kids/panorama.png",
        "thumbnail": "assets/fallback_kids/thumbnail.png",
    },
    "filmmaker": {
        "splat": "assets/fallback_filmmaker/world.spz",
        "collider": "assets/fallback_filmmaker/collider.glb",
        "panorama": "assets/fallback_filmmaker/panorama.png",
        "thumbnail": "assets/fallback_filmmaker/thumbnail.png",
    },
}

async def generate_with_fallback(
    marble_client: MarbleClient,
    image_path: str,
    text_prompt: str | None,
    mode: str,
    timeout: float = 90.0,
) -> dict:
    """
    Attempt Marble generation with a timeout. If it fails or times out,
    return pre-generated fallback assets instead.
    """
    try:
        assets = await asyncio.wait_for(
            marble_client.generate_world(image_path, text_prompt),
            timeout=timeout,
        )
        return assets
    except (asyncio.TimeoutError, MarbleError, httpx.HTTPError) as e:
        logger.warning(f"Marble failed ({type(e).__name__}: {e}), using fallback for mode={mode}")
        return FALLBACK_WORLDS.get(mode, FALLBACK_WORLDS["kids"])
```

---

### Module 4: Polaroid Compositor

Takes viewport screenshots from the client and composites them into polaroid-framed images. Also generates the final strip. This module has zero dependencies on the generation pipeline and can be built anytime.

```python
from PIL import Image, ImageDraw, ImageFilter
from io import BytesIO
import base64
import os


async def create_polaroid(
    screenshot_b64: str,
    capture_number: int,
    world_id: str,
    assets_dir: str = "assets",
) -> str:
    """
    Composite a viewport screenshot into a polaroid-framed image.
    
    The polaroid has a white border (thicker on the bottom, classic style),
    with a slight drop shadow for a tactile feel.
    
    Args:
        screenshot_b64: Base64-encoded viewport capture from the client
        capture_number: 1-6
        world_id: The world this polaroid belongs to
        assets_dir: Root directory for asset storage
    
    Returns:
        Local path to the saved polaroid image
    """
    screenshot = Image.open(BytesIO(base64.b64decode(screenshot_b64)))
    
    # Polaroid dimensions
    photo_w, photo_h = 600, 500
    border_side = 40
    border_top = 40
    border_bottom = 100  # Classic polaroid: thick bottom
    
    total_w = photo_w + (border_side * 2)
    total_h = photo_h + border_top + border_bottom
    
    # Create white polaroid frame
    polaroid = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
    
    # Resize screenshot to fit the photo area, preserving aspect ratio
    screenshot = screenshot.convert("RGB")
    screenshot_resized = screenshot.resize((photo_w, photo_h), Image.LANCZOS)
    polaroid.paste(screenshot_resized, (border_side, border_top))
    
    # Save
    output_dir = os.path.join(assets_dir, world_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"polaroid_{capture_number}.png")
    polaroid.save(output_path, "PNG")
    
    return output_path


async def create_strip(
    world_id: str,
    assets_dir: str = "assets",
) -> str:
    """
    Composite all 6 polaroids + the original drawing into a final strip image.
    
    Layout: Original drawing on the left as a larger polaroid, then the 6 captures
    arranged in a 2x3 grid on the right. The whole thing feels like a keepsake
    you'd pin to a wall or stick on a fridge.
    
    Returns:
        Local path to the saved strip image
    """
    world_dir = os.path.join(assets_dir, world_id)
    
    # Load the original drawing and frame it
    drawing_path = os.path.join(world_dir, "drawing.png")
    drawing = Image.open(drawing_path).convert("RGB")
    
    # Load all 6 polaroids
    polaroids = []
    for i in range(1, 7):
        p_path = os.path.join(world_dir, f"polaroid_{i}.png")
        if os.path.exists(p_path):
            polaroids.append(Image.open(p_path))
    
    # Strip layout calculations
    polaroid_w = 680  # width of each polaroid image
    polaroid_h = 640  # height of each polaroid image
    
    # Scale polaroids down for the grid
    thumb_w, thumb_h = 320, 300
    grid_cols, grid_rows = 3, 2
    grid_gap = 16
    
    # Drawing area (left side)
    drawing_display_w = 340
    drawing_display_h = 400
    drawing_border = 30
    drawing_border_bottom = 70
    drawing_total_w = drawing_display_w + (drawing_border * 2)
    drawing_total_h = drawing_display_h + drawing_border + drawing_border_bottom
    
    # Total strip dimensions
    grid_w = (thumb_w * grid_cols) + (grid_gap * (grid_cols - 1))
    strip_w = drawing_total_w + 40 + grid_w + 40  # drawing + gap + grid + margin
    strip_h = max(drawing_total_h, (thumb_h * grid_rows) + (grid_gap * (grid_rows - 1))) + 80
    
    # Create strip canvas with warm off-white background
    strip = Image.new("RGB", (strip_w, strip_h), (252, 250, 245))
    
    # Draw the original drawing as a large polaroid on the left
    drawing_frame = Image.new("RGB", (drawing_total_w, drawing_total_h), (255, 255, 255))
    drawing_resized = drawing.resize((drawing_display_w, drawing_display_h), Image.LANCZOS)
    drawing_frame.paste(drawing_resized, (drawing_border, drawing_border))
    strip.paste(drawing_frame, (40, 40))
    
    # Place the 6 polaroid thumbnails in a 3x2 grid on the right
    grid_x_start = drawing_total_w + 80
    grid_y_start = 40
    
    for idx, pol in enumerate(polaroids):
        row = idx // grid_cols
        col = idx % grid_cols
        x = grid_x_start + (col * (thumb_w + grid_gap))
        y = grid_y_start + (row * (thumb_h + grid_gap))
        pol_resized = pol.resize((thumb_w, thumb_h), Image.LANCZOS)
        strip.paste(pol_resized, (x, y))
    
    # Save the strip
    strip_path = os.path.join(world_dir, "strip.png")
    strip.save(strip_path, "PNG", quality=95)
    
    return strip_path
```

---

## Pipeline Orchestrator

The main function that ties all modules together. Called as a background task from the POST /api/generate endpoint.

```python
import asyncio
import logging
import os
import shutil
import base64
from modules.imagegen import stylize_drawing, load_style_prompt
from modules.marble import MarbleClient, generate_with_fallback
from state import update_status, set_assets

logger = logging.getLogger(__name__)


async def run_pipeline(
    world_id: str,
    drawing_b64: str,
    mode: str,
    marble_client: MarbleClient,
    assets_dir: str = "assets",
):
    """
    Full Yume pipeline: drawing → styled image → Marble world → assets ready.
    
    This runs as a background task. The client polls /status to track progress.
    Two stages:
      Stage 1: Stylize drawing via Fal AI (~3-5 seconds)
      Stage 2: Generate 3D world via Marble (~30-45 seconds)
    """
    world_dir = os.path.join(assets_dir, world_id)
    os.makedirs(world_dir, exist_ok=True)
    
    try:
        # ── Stage 1: Save original drawing + stylize it ────────────────────
        update_status(world_id, "stylizing_drawing", stage=1, total_stages=2,
                      label="Turning your drawing into a world...")
        
        # Save original drawing to disk
        drawing_path = os.path.join(world_dir, "drawing.png")
        with open(drawing_path, "wb") as f:
            f.write(base64.b64decode(drawing_b64))
        
        # Transform drawing → styled environment image via Fal
        styled_path = await stylize_drawing(
            image_path=drawing_path,
            mode=mode,
            strength=0.88,        # Tune during R&D. 0.85-0.90 sweet spot.
            num_inference_steps=35,
            guidance_scale=3.5,
        )
        
        logger.info(f"[{world_id}] Drawing stylized → {styled_path}")
        
        # ── Stage 2: Generate 3D world via Marble ──────────────────────────
        update_status(world_id, "generating_world", stage=2, total_stages=2,
                      label="Building your world...")
        
        # Optionally pass a short text prompt alongside the image
        # to give Marble additional context. Keep it brief.
        style_prompt = load_style_prompt(mode)
        marble_text = style_prompt[:200]  # Marble may have text length limits
        
        assets = await generate_with_fallback(
            marble_client=marble_client,
            image_path=styled_path,
            text_prompt=marble_text,
            mode=mode,
            timeout=90.0,
        )
        
        # Download Marble assets to local storage
        if isinstance(assets, dict) and all(isinstance(v, str) and v.startswith("http") for v in assets.values()):
            # Real Marble response — download URLs
            local_assets = await marble_client.download_assets(assets, world_dir)
        else:
            # Fallback — assets are already local paths, copy them
            local_assets = {}
            for asset_type, src_path in assets.items():
                dst_path = os.path.join(world_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)
                local_assets[asset_type] = dst_path
        
        logger.info(f"[{world_id}] World generation complete. Assets: {list(local_assets.keys())}")
        
        # ── Done ───────────────────────────────────────────────────────────
        set_assets(world_id, local_assets, styled_image_path=styled_path)
        update_status(world_id, "complete", stage=2, total_stages=2,
                      label="Your world is ready.")
    
    except Exception as e:
        logger.error(f"[{world_id}] Pipeline failed: {e}", exc_info=True)
        update_status(world_id, "failed", error=str(e))
```

---

## Task Sequence for Codex

These tasks are ordered by dependency. Each is independently testable. Claude Code generates each task as a detailed spec, reviews the output, and proceeds to the next.

**Task 1: Project scaffold.**
Set up project structure (see Directory Structure below). Create virtual environment. Install dependencies: `fastapi`, `uvicorn`, `httpx`, `Pillow`, `fal-client`, `python-multipart`. Create `main.py` with FastAPI app, CORS middleware (allow all origins for hackathon), static file serving from `/assets`, and `/api/health` endpoint. Verify: `uvicorn main:app --reload` starts and health endpoint returns 200.

**Task 2: World state management.**
Create `state.py` with an in-memory dict storing world state by world_id. Functions: `create_world(world_id, mode) -> dict`, `update_status(world_id, status, stage, total_stages, label, error)`, `get_world(world_id) -> dict | None`, `set_assets(world_id, assets, styled_image_path)`. Write pytest tests for all state transitions including the error case. The dict schema per world:
```python
{
    "world_id": str,
    "mode": str,
    "status": str,  # "stylizing_drawing" | "generating_world" | "complete" | "failed"
    "stage": int,
    "total_stages": int,
    "stage_label": str,
    "assets": dict | None,
    "styled_image": str | None,
    "error": str | None,
    "created_at": str,  # ISO timestamp
}
```

**Task 3: API endpoints (generate + status + assets).**
Implement `POST /api/generate` — accept image upload (both multipart and base64 JSON body), validate image format (png/jpg/webp), generate world_id, create world state, save drawing to `assets/{world_id}/drawing.png`, kick off `run_pipeline` as background task via `asyncio.create_task`, return world_id. Implement `GET /api/world/{world_id}/status` and `GET /api/world/{world_id}/assets` reading from world state. Return 404 for unknown world_id. Verify with curl: upload an image, get world_id back, poll status.

**Task 4: Fal AI image stylization module.**
Implement `modules/imagegen.py` with the `stylize_drawing` function and `load_style_prompt`. Install fal-client (`pip install fal-client`). Set `FAL_KEY` env var. The function should: upload image to Fal storage, call `fal-ai/flux/dev/image-to-image` with image_url + prompt + strength, download result to local path. Create a standalone test script `tests/test_imagegen.py` that takes a drawing path, calls `stylize_drawing`, and saves the output. Verify by running the test with a real drawing image and inspecting the output.

**Task 5: Marble API client.**
Implement `modules/marble.py` with `MarbleClient` class and `generate_with_fallback`. **Before writing code, read https://docs.worldlabs.ai/api and adjust the client to match actual endpoints, auth headers, and response schemas.** The pseudocode in this doc is an approximation. Implement submit, poll, download_assets, and the fallback path. Create `tests/test_marble.py` that generates a world from a test image. Verify the full lifecycle works.

**Task 6: Pipeline orchestrator.**
Implement `pipeline.py` wiring stylize_drawing → Marble generation → asset storage → status updates. End-to-end test: `POST /api/generate` with a drawing image via curl → poll `/status` until complete → verify `/assets` returns valid URLs → verify asset files exist on disk. This is the critical milestone. Once this works, other teams can integrate.

**Task 7: Polaroid endpoints.**
Implement `modules/polaroid.py` with `create_polaroid` and `create_strip`. Implement `POST /api/world/{world_id}/polaroid` and `GET /api/world/{world_id}/strip` in main.py. Test: send a base64 screenshot to the polaroid endpoint, verify a framed polaroid image is saved. Send 6 screenshots, call strip endpoint, verify strip image is saved and contains all 6 + original drawing.

**Task 8: Pre-generate fallback worlds.**
Write a utility script `scripts/generate_fallbacks.py` that takes 2-3 known-good styled images, sends each to Marble, downloads the assets, and stores them in `assets/fallback_kids/` and `assets/fallback_filmmaker/`. Run this script to populate fallback assets. Verify the fallback path works by temporarily making Marble timeout and confirming the API returns fallback assets.

**Task 9: Prompt templates directory.**
Create `prompts/kids.txt` and `prompts/filmmaker.txt` with initial prompt versions. Verify that `load_style_prompt` reads from these files and that changing file contents changes the prompt used in the next generation without restarting the server.

---

## Environment Variables

```bash
FAL_KEY=...                        # Fal AI API key (for FLUX image stylization)
MARBLE_API_KEY=...                 # World Labs Marble API key
YUME_PORT=8000                     # API server port
YUME_ASSETS_DIR=./assets           # Local asset storage
YUME_MODE=development              # development | production
```

Note: Claude API key is no longer needed in the pipeline. The LLM analysis step has been removed. The entire synthesis is now handled by Fal's img-to-img model + prompt engineering.

---

## Directory Structure

```
yume-server/
├── main.py                     # FastAPI app, all endpoints, CORS, static files
├── config.py                   # Env vars, constants, fallback world paths
├── pipeline.py                 # Pipeline orchestrator (run_pipeline)
├── state.py                    # World state management (in-memory dict)
├── modules/
│   ├── imagegen.py             # Fal AI FLUX.1 img-to-img stylization
│   ├── marble.py               # Marble API client + fallback logic
│   └── polaroid.py             # Polaroid frame compositor + strip generator
├── prompts/
│   ├── kids.txt                # Style prompt for kids mode (hot-swappable)
│   └── filmmaker.txt           # Style prompt for filmmaker mode (hot-swappable)
├── assets/                     # Generated assets (gitignored)
│   ├── fallback_kids/          # Pre-generated fallback world for kids mode
│   └── fallback_filmmaker/     # Pre-generated fallback world for filmmaker mode
├── scripts/
│   └── generate_fallbacks.py   # Utility to pre-generate fallback worlds
├── tests/
│   ├── test_imagegen.py        # Standalone: drawing → styled image
│   ├── test_marble.py          # Standalone: styled image → Marble world
│   └── test_pipeline.py        # End-to-end: drawing → complete world
├── requirements.txt
├── .env
└── README.md
```

---

## Integration Contracts for Other Teams

**Frontend team needs to know:**
- Base URL of the API server (http://localhost:8000 or ngrok URL)
- `POST /api/generate` with their drawing image → get `world_id`
- Poll `GET /api/world/{id}/status` every 2-3 seconds until `status = "complete"`
- The status response includes `stage_label` (human-readable) they can display in the loading UI
- `GET /api/world/{id}/assets` returns URLs for `.spz` (splat), `.glb` (collider), panorama, thumbnail
- Load the `.spz` URL into SparkJS viewer
- `POST /api/world/{id}/polaroid` with viewport screenshots as base64
- `GET /api/world/{id}/strip` after all 6 captures for the final keepsake image

**VR team needs to know:**
- Same asset URLs as frontend. `GET /api/world/{id}/assets` returns `.spz` and `.glb` files
- Download the `.spz` file and import into Unity Gaussian splat renderer
- The `.glb` collider mesh provides navigation/collision boundaries
- Polaroid captures work the same way: POST a render texture screenshot as base64 to `/api/world/{id}/polaroid`
- The strip endpoint returns the same composited strip image

**Both teams do NOT need to know:** How the drawing gets stylized, what model is used, how Marble works. The API is a black box. Drawing goes in, world assets come out.

---

## What Claude Code Should Do

Claude Code acts as CTO and task generator. Workflow:

1. Read this entire document.
2. Generate Task 1 as a detailed Codex-ready spec: exact files to create, exact dependencies to install, exact commands to verify.
3. After Codex completes Task 1, review the output. Does the server start? Does health return 200? Is CORS configured?
4. Generate Task 2. Review. Proceed.
5. Continue sequentially through all 9 tasks.
6. After Task 6 (end-to-end pipeline), run a full integration test before moving to Tasks 7-9.
7. Flag decisions needing Leon's input: Fal strength parameter tuning, Marble model selection, fallback world image choices, prompt iterations.

## What Codex Should Do

For each task from Claude Code:

1. Read the task spec.
2. Implement the code with error handling and logging.
3. Write a test or verification step that proves it works.
4. Commit with a descriptive message.
5. Move to the next task only after Claude Code reviews.

---

## Critical Path

Marble generation (30-45s) is the bottleneck. Everything else is fast.

**Revised build order for maximum parallelism:**

- **Hour 1-2:** Task 1 (scaffold) + Task 2 (state). Then immediately Task 4 (Fal module, standalone) + Task 5 (Marble client, standalone). These can be tested independently with scripts before the API is wired up.
- **Hour 2-3:** Task 3 (API endpoints). Then Task 6 (wire pipeline end-to-end). At this point the full flow works: upload drawing via curl → get world assets back.
- **Hour 3-4:** Task 8 (pre-generate fallback worlds). This depends on Task 5 being done. Also Task 9 (prompt templates).
- **Hour 4-5:** Task 7 (polaroid module + endpoints). Zero dependency on the generation pipeline, could be started earlier if someone has bandwidth.
- **Hour 5+:** Integration testing with frontend/VR teams. Prompt tuning. Polish.

Leon's parallel workstream (prompt R&D) should start immediately and continue throughout. Use `tests/test_imagegen.py` to quickly test different prompts and strength values against Fal, then feed the best styled images to `tests/test_marble.py` to see what Marble produces. This loop is the highest-leverage use of Leon's time.
