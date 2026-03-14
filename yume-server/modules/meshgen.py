import asyncio
import base64
import logging
from pathlib import Path

import httpx

from config import MESHY_API_KEY

logger = logging.getLogger("yume.meshgen")

BASE_URL = "https://api.meshy.ai/openapi/v1"
POLL_INTERVAL = 5  # seconds


def _infer_image_mime(image_bytes: bytes, image_path: str) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"

    ext = Path(image_path).suffix.lstrip(".").lower()
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext == "webp":
        return "image/webp"
    return "image/png"


def _task_error_message(task_result: dict) -> str:
    task_error = task_result.get("task_error")
    if isinstance(task_error, dict):
        return task_error.get("message") or task_error.get("detail") or str(task_error)
    if task_error:
        return str(task_error)
    return "Unknown error"


class MeshyClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or MESHY_API_KEY

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Meshy integration unavailable: MESHY_API_KEY is not configured")
        return {"Authorization": f"Bearer {self.api_key}"}

    async def submit_image_to_3d(self, image_url: str) -> str:
        """Submit an image-to-3D task. Returns the task ID."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/image-to-3d",
                headers={**self._auth_headers(), "Content-Type": "application/json"},
                json={
                    "image_url": image_url,
                    "ai_model": "meshy-6",
                    "should_texture": True,
                    "enable_pbr": True,
                    "should_remesh": True,
                    "target_polycount": 20000,
                    "topology": "triangle",
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            task_id = data.get("result") or data.get("id") or data.get("task_id")
            if not task_id:
                raise RuntimeError(f"Meshy submit did not return a task ID: {data}")
            logger.info("Meshy image-to-3D submitted: task_id=%s", task_id)
            return task_id

    async def poll_task(self, task_id: str, timeout: float = 300.0) -> dict:
        """Poll a task until SUCCEEDED or FAILED. Returns the full task response."""
        elapsed = 0.0
        async with httpx.AsyncClient() as client:
            while elapsed < timeout:
                resp = await client.get(
                    f"{BASE_URL}/image-to-3d/{task_id}",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "UNKNOWN")

                if status == "SUCCEEDED":
                    logger.info("Meshy task %s succeeded", task_id)
                    return data
                if status in ("FAILED", "CANCELED"):
                    error = _task_error_message(data)
                    raise RuntimeError(f"Meshy task {task_id} {status}: {error}")

                progress = data.get("progress", 0)
                logger.info("Meshy task %s: %s (%s%% complete, %.0fs elapsed)", task_id, status, progress, elapsed)

                await asyncio.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL

        raise TimeoutError(f"Meshy task {task_id} timed out after {timeout}s")

    async def download_model_assets(self, task_result: dict, output_dir: str) -> dict:
        """Download GLB, FBX, and thumbnail from a completed task. Returns local paths."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_urls = task_result.get("model_urls", {})
        thumbnail_url = task_result.get("thumbnail_url")

        downloads = {}
        if isinstance(model_urls, dict):
            if model_urls.get("glb"):
                downloads["glb_path"] = ("plushie.glb", model_urls["glb"])
            if model_urls.get("fbx"):
                downloads["fbx_path"] = ("plushie.fbx", model_urls["fbx"])
        elif isinstance(model_urls, str):
            # Some responses return a single URL string
            downloads["glb_path"] = ("plushie.glb", model_urls)

        if thumbnail_url:
            downloads["thumbnail_path"] = ("plushie_thumbnail.png", thumbnail_url)

        local_paths = {}
        async with httpx.AsyncClient() as client:
            for key, (filename, url) in downloads.items():
                try:
                    resp = await client.get(url, timeout=60.0, follow_redirects=True)
                    resp.raise_for_status()
                    local_path = out / filename
                    local_path.write_bytes(resp.content)
                    local_paths[key] = str(local_path)
                    logger.info("Downloaded %s → %s (%d bytes)", key, local_path, len(resp.content))
                except Exception as e:
                    logger.error("Failed to download %s: %s", key, e)

        return local_paths


async def generate_plushie_model(
    meshy_client: MeshyClient,
    image_path: str,
    output_dir: str,
    timeout: float = 300.0,
) -> dict:
    """Full flow: convert image to data URI → submit → poll → download."""
    # Convert local image to data URI for Meshy API
    image_bytes = Path(image_path).read_bytes()
    mime = _infer_image_mime(image_bytes, image_path)
    data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    logger.info("Starting Meshy image-to-3D for %s (%d bytes)", image_path, len(image_bytes))

    # Submit
    task_id = await meshy_client.submit_image_to_3d(data_uri)

    # Poll
    task_result = await asyncio.wait_for(
        meshy_client.poll_task(task_id, timeout=timeout),
        timeout=timeout + 30,  # outer timeout slightly longer than inner
    )

    # Download
    local_paths = await meshy_client.download_model_assets(task_result, output_dir)

    # Pass through raw CDN URLs for direct access
    model_urls = task_result.get("model_urls", {})
    if isinstance(model_urls, dict):
        local_paths["cdn_glb_url"] = model_urls.get("glb")
        local_paths["cdn_fbx_url"] = model_urls.get("fbx")
    local_paths["cdn_thumbnail_url"] = task_result.get("thumbnail_url")

    logger.info("Meshy plushie model complete: %s", local_paths)
    return local_paths
