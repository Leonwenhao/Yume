import asyncio
import json
import logging
import shutil
import struct
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

from config import MARBLE_API_KEY, YUME_ASSETS_DIR

logger = logging.getLogger("yume.marble")

BASE_URL = "https://api.worldlabs.ai"
POLL_INTERVAL = 3  # seconds
REQUIRED_WORLD_ASSET_KEYS = ("spz_url", "collider_url", "panorama_url", "thumbnail_url")


class MarbleClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or MARBLE_API_KEY

    def _json_headers(self) -> dict[str, str]:
        api_key = self._require_api_key()
        return {
            "WLT-Api-Key": api_key,
            "Content-Type": "application/json",
        }

    def _auth_headers(self) -> dict[str, str]:
        return {"WLT-Api-Key": self._require_api_key()}

    def _require_api_key(self) -> str:
        if not self.api_key:
            logger.error("Marble integration unavailable: MARBLE_API_KEY is not configured")
            raise RuntimeError("Marble integration unavailable: MARBLE_API_KEY is not configured")
        return self.api_key

    async def _prepare_upload(self, file_name: str, extension: str) -> dict:
        """Prepare a media asset upload and return upload info."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/marble/v1/media-assets:prepare_upload",
                headers=self._json_headers(),
                json={
                    "file_name": file_name,
                    "kind": "image",
                    "extension": extension,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def _upload_file(self, upload_url: str, file_bytes: bytes, required_headers: dict) -> None:
        """PUT file bytes to the signed upload URL."""
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                upload_url,
                content=file_bytes,
                headers=required_headers,
                timeout=60.0,
            )
            resp.raise_for_status()
            logger.info("File uploaded to Marble storage")

    async def _submit_generation(self, media_asset_id: str, display_name: str, model: str = "Marble 0.1-mini") -> dict:
        """Submit a world generation request. Returns operation object."""
        body = {
            "display_name": display_name,
            "model": model,
            "world_prompt": {
                "type": "image",
                "image_prompt": {
                    "source": "media_asset",
                    "media_asset_id": media_asset_id,
                },
            },
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/marble/v1/worlds:generate",
                headers=self._json_headers(),
                json=body,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info("Generation submitted: operation_id=%s", data.get("operation_id"))
            return data

    async def _poll_operation(self, operation_id: str, timeout: float = 120.0) -> dict:
        """Poll operation until done or timeout. Returns the final operation object."""
        elapsed = 0.0
        async with httpx.AsyncClient() as client:
            while elapsed < timeout:
                resp = await client.get(
                    f"{BASE_URL}/marble/v1/operations/{operation_id}",
                    headers=self._auth_headers(),
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("done"):
                    if data.get("error"):
                        raise RuntimeError(f"Marble generation failed: {data['error']}")
                    logger.info("Generation complete for operation %s", operation_id)
                    return data

                status = data.get("metadata", {}).get("progress", {}).get("status", "unknown")
                logger.info("Polling operation %s: %s (%.0fs elapsed)", operation_id, status, elapsed)

                await asyncio.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL

        raise TimeoutError(f"Marble generation timed out after {timeout}s")

    async def _get_world(self, world_id: str) -> dict:
        """Get world details by ID."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/marble/v1/worlds/{world_id}",
                headers=self._auth_headers(),
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def generate_world(self, image_path: str, display_name: str = "Yume World") -> dict:
        """Full flow: upload image → submit generation → poll → return world assets.

        Returns dict with keys: thumbnail_url, spz_url, collider_url, panorama_url
        """
        image_bytes = Path(image_path).read_bytes()
        ext = Path(image_path).suffix.lstrip(".")
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "png"

        # Step 1: Prepare upload
        upload_info = await self._prepare_upload(Path(image_path).name, ext)
        media_asset_id = upload_info["media_asset"]["media_asset_id"]
        upload_url = upload_info["upload_info"]["upload_url"]
        required_headers = upload_info["upload_info"].get("required_headers", {})
        logger.info("Media asset prepared: %s", media_asset_id)

        # Step 2: Upload file
        await self._upload_file(upload_url, image_bytes, required_headers)

        # Step 3: Submit generation
        operation = await self._submit_generation(media_asset_id, display_name)
        operation_id = operation["operation_id"]

        # Step 4: Poll until complete
        result = await self._poll_operation(operation_id)

        # Step 5: Extract assets
        world_data = result.get("response", {}).get("world", {})
        if not world_data:
            # Fall back to fetching world by ID
            world_id = result.get("metadata", {}).get("world_id")
            if world_id:
                world_resp = await self._get_world(world_id)
                world_data = world_resp.get("world", world_resp)

        assets = world_data.get("assets", {})
        return {
            "thumbnail_url": assets.get("thumbnail_url"),
            "spz_url": assets.get("splats", {}).get("spz_urls", {}).get("full_res"),
            "spz_500k_url": assets.get("splats", {}).get("spz_urls", {}).get("500k"),
            "spz_100k_url": assets.get("splats", {}).get("spz_urls", {}).get("100k"),
            "collider_url": assets.get("mesh", {}).get("collider_mesh_url"),
            "panorama_url": assets.get("imagery", {}).get("pano_url"),
            "caption": assets.get("caption"),
            "marble_url": world_data.get("world_marble_url"),
        }

    async def download_assets(self, asset_urls: dict, output_dir: str) -> dict:
        """Download all assets to local directory. Returns dict of local paths."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        file_map = {
            "spz_url": "world.spz",
            "ply_url": "world.ply",
            "spz_500k_url": "world_500k.spz",
            "spz_100k_url": "world_100k.spz",
            "collider_url": "collider.glb",
            "panorama_url": "panorama.png",
            "thumbnail_url": "thumbnail.png",
        }

        local_paths = {}
        async with httpx.AsyncClient() as client:
            for key, filename in file_map.items():
                url = asset_urls.get(key)
                if not url:
                    logger.warning("No URL for %s, skipping", key)
                    continue
                try:
                    resp = await client.get(url, timeout=60.0, follow_redirects=True)
                    resp.raise_for_status()
                    local_path = out / filename
                    _write_asset_file(resp.content, local_path)
                    local_paths[key] = str(local_path)
                    logger.info("Downloaded %s → %s (%d bytes)", key, local_path, len(resp.content))
                except Exception as e:
                    logger.error("Failed to download %s: %s", key, e)

        return local_paths


def _missing_required_assets(assets: dict) -> list[str]:
    missing = []
    for key in REQUIRED_WORLD_ASSET_KEYS:
        if not assets.get(key):
            missing.append(key)
    return missing


def _write_asset_file(content: bytes, local_path: Path) -> None:
    if local_path.suffix.lower() == ".png":
        try:
            image = Image.open(BytesIO(content))
            image.save(local_path, format="PNG")
            return
        except UnidentifiedImageError:
            logger.warning("Downloaded bytes were not an image; writing raw file to %s", local_path)

    local_path.write_bytes(content)


def ensure_placeholder_collider(output_dir: str | Path) -> str:
    """Create a minimal valid GLB so fallback directories always contain collider.glb."""
    collider_path = Path(output_dir) / "collider.glb"
    if collider_path.exists():
        return str(collider_path)

    scene_json = json.dumps(
        {
            "asset": {"version": "2.0", "generator": "yume-server fallback collider"},
            "scene": 0,
            "scenes": [{"nodes": []}],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    padded_json = scene_json + (b" " * ((4 - len(scene_json) % 4) % 4))
    total_length = 12 + 8 + len(padded_json)
    glb = (
        b"glTF"
        + struct.pack("<I", 2)
        + struct.pack("<I", total_length)
        + struct.pack("<I", len(padded_json))
        + b"JSON"
        + padded_json
    )
    collider_path.write_bytes(glb)
    logger.warning("Created placeholder collider GLB at %s", collider_path)
    return str(collider_path)


def _validate_local_assets(local_paths: dict, label: str) -> None:
    missing = []
    for key in REQUIRED_WORLD_ASSET_KEYS:
        local_path = local_paths.get(key)
        if not local_path or not Path(local_path).exists():
            missing.append(key)

    if missing:
        raise FileNotFoundError(f"{label} missing required assets: {', '.join(missing)}")


async def generate_with_fallback(
    marble_client: MarbleClient,
    image_path: str,
    output_dir: str,
    mode: str = "kids",
    timeout: float = 90.0,
) -> dict:
    """Run Marble generation with timeout. On failure, return fallback assets.

    Returns dict with local file paths for each asset.
    """
    try:
        logger.info("Starting Marble generation for %s", image_path)
        asset_urls = await asyncio.wait_for(
            marble_client.generate_world(image_path),
            timeout=timeout,
        )
        local_paths = await marble_client.download_assets(asset_urls, output_dir)

        # Marble mini doesn't return a collider — create placeholder if missing
        if "collider_url" not in local_paths:
            local_paths["collider_url"] = ensure_placeholder_collider(output_dir)

        _validate_local_assets(local_paths, "Marble download")
        local_paths["caption"] = asset_urls.get("caption")
        local_paths["marble_url"] = asset_urls.get("marble_url")
        # Pass through raw CDN URLs for direct access
        local_paths["cdn_spz_url"] = asset_urls.get("spz_url")
        local_paths["cdn_spz_500k_url"] = asset_urls.get("spz_500k_url")
        local_paths["cdn_spz_100k_url"] = asset_urls.get("spz_100k_url")
        local_paths["cdn_panorama_url"] = asset_urls.get("panorama_url")
        local_paths["cdn_thumbnail_url"] = asset_urls.get("thumbnail_url")
        return local_paths
    except (TimeoutError, asyncio.TimeoutError) as e:
        logger.warning("Marble timed out after %.0fs; activating %s fallback: %s", timeout, mode, e)
    except Exception as e:
        logger.warning("Marble generation failed or returned incomplete assets; activating %s fallback: %s", mode, e)

    logger.warning("Using fallback assets for mode '%s'", mode)
    local_paths = _get_fallback_assets(mode, output_dir)
    _validate_local_assets(local_paths, f"{mode} fallback")
    return local_paths


def _get_fallback_assets(mode: str, output_dir: str) -> dict:
    """Copy fallback assets to output directory and return paths."""
    fallback_dir = Path(YUME_ASSETS_DIR) / f"fallback_{mode}"
    if not fallback_dir.exists():
        logger.warning("Fallback directory missing for mode '%s': %s", mode, fallback_dir)
        fallback_dir = Path(YUME_ASSETS_DIR) / "fallback_kids"
        logger.warning("Falling back to default fallback directory: %s", fallback_dir)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    local_paths = {}
    file_map = {
        "spz_url": "world.spz",
        "ply_url": "world.ply",
        "collider_url": "collider.glb",
        "panorama_url": "panorama.png",
        "thumbnail_url": "thumbnail.png",
    }

    for key, filename in file_map.items():
        src = fallback_dir / filename
        dst = out / filename
        if src.exists():
            shutil.copy2(src, dst)
            local_paths[key] = str(dst)
            logger.info("Fallback: copied %s → %s", src, dst)
        else:
            logger.warning("Fallback asset missing: %s", src)

    # Ensure a collider always exists (Marble mini doesn't provide one)
    if "collider_url" not in local_paths:
        local_paths["collider_url"] = ensure_placeholder_collider(out)

    return local_paths
