import asyncio
import logging
from contextlib import suppress
from io import BytesIO
from pathlib import Path

import state
from config import YUME_ASSETS_DIR
from modules.imagegen import stylize_drawing
from modules.marble import MarbleClient, generate_with_fallback
from modules.meshgen import MeshyClient, generate_plushie_model
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("yume.pipeline")


def _plushie_asset_urls(world_id: str, plushie_path: str | None = None) -> dict:
    prefix = f"/assets/{world_id}"
    return {
        "plushie_glb_url": None,
        "plushie_fbx_url": None,
        "plushie_thumbnail_url": None,
        "plushie_photo_url": f"{prefix}/{Path(plushie_path).name}" if plushie_path else None,
        "plushie_styled_url": None,
        "cdn_plushie_glb_url": None,
        "cdn_plushie_fbx_url": None,
    }


def _normalize_png_bytes(image_bytes: bytes) -> bytes:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return image_bytes

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
    except UnidentifiedImageError:
        return image_bytes


def persist_original_drawing(world_id: str, drawing_bytes: bytes, assets_root: str | Path | None = None) -> str:
    """Persist the original drawing before the background pipeline starts."""
    if not drawing_bytes:
        raise ValueError("Drawing payload is empty")

    root = Path(assets_root) if assets_root is not None else Path(YUME_ASSETS_DIR)
    world_dir = root / world_id
    world_dir.mkdir(parents=True, exist_ok=True)

    drawing_path = world_dir / "drawing.png"
    drawing_path.write_bytes(drawing_bytes)
    logger.info("[%s] Persisted original drawing: %s (%d bytes)", world_id, drawing_path, len(drawing_bytes))
    return str(drawing_path)


def persist_plushie_photo(world_id: str, plushie_bytes: bytes, assets_root: str | Path | None = None) -> str:
    """Persist the plushie photo before the background pipeline starts."""
    if not plushie_bytes:
        raise ValueError("Plushie payload is empty")

    root = Path(assets_root) if assets_root is not None else Path(YUME_ASSETS_DIR)
    world_dir = root / world_id
    world_dir.mkdir(parents=True, exist_ok=True)

    plushie_path = world_dir / "plushie.png"
    normalized_bytes = _normalize_png_bytes(plushie_bytes)
    plushie_path.write_bytes(normalized_bytes)
    logger.info("[%s] Persisted plushie photo: %s (%d bytes)", world_id, plushie_path, len(normalized_bytes))
    return str(plushie_path)


async def run_pipeline(
    world_id: str,
    drawing_path: str,
    mode: str,
    marble_client: MarbleClient,
    plushie_path: str | None = None,
    meshy_client: MeshyClient | None = None,
) -> None:
    """Full async pipeline: stylize drawing → generate 3D world (+ optional plushie model in parallel).

    This runs as a background task. Updates world state at each step.
    """
    world_dir = Path(YUME_ASSETS_DIR) / world_id
    styled_path = world_dir / "styled.png"

    try:
        drawing_file = Path(drawing_path)
        if not drawing_file.exists():
            raise FileNotFoundError(f"Original drawing not found: {drawing_file}")

        # Stage 1: Stylize drawing via Fal AI (sequential — Marble needs the styled image)
        state.update_status(world_id, "stylizing_drawing", 1, "Turning your drawing into a world...")
        logger.info("[%s] Stage 1: Stylizing drawing (mode=%s)", world_id, mode)

        await stylize_drawing(str(drawing_file), str(styled_path), mode=mode)
        logger.info("[%s] Stage 1 complete: %s", world_id, styled_path)

        # Stage 2: Generate 3D world (+ plushie model if provided) in parallel
        state.update_status(world_id, "generating_world", 2, "Building your 3D world...")
        logger.info("[%s] Stage 2: Generating world via Marble", world_id)

        async def _world_pipeline() -> dict:
            """Pipeline A: styled drawing → Marble 3D world."""
            local_assets = await generate_with_fallback(
                marble_client, str(styled_path), str(world_dir), mode=mode, timeout=90.0,
            )
            logger.info("[%s] World generation complete: %d assets", world_id, len(local_assets))

            required_local_assets = ("spz_url", "collider_url", "panorama_url", "thumbnail_url")
            missing_assets = [key for key in required_local_assets if not local_assets.get(key)]
            if missing_assets:
                raise RuntimeError(f"Missing required generated assets: {', '.join(missing_assets)}")

            prefix = f"/assets/{world_id}"
            world_assets = {
                "original_drawing": f"{prefix}/drawing.png",
                "styled_image": f"{prefix}/styled.png",
                "splat_url": f"{prefix}/world.spz",
                "splat_ply_url": f"{prefix}/world.ply" if local_assets.get("ply_url") else None,
                "collider_url": f"{prefix}/collider.glb",
                "panorama_url": f"{prefix}/panorama.png",
                "thumbnail_url": f"{prefix}/thumbnail.png",
                # Direct CDN URLs (faster downloads, bypass our server)
                "marble_viewer_url": local_assets.get("marble_url"),
                "cdn_splat_url": local_assets.get("cdn_spz_url"),
                "cdn_splat_500k_url": local_assets.get("cdn_spz_500k_url"),
                "cdn_splat_100k_url": local_assets.get("cdn_spz_100k_url"),
                "cdn_panorama_url": local_assets.get("cdn_panorama_url"),
                "cdn_thumbnail_url": local_assets.get("cdn_thumbnail_url"),
            }
            logger.info(
                "[%s] Pipeline viewer URL resolved: marble_viewer_url=%r panorama_url=%r",
                world_id,
                world_assets["marble_viewer_url"],
                world_assets["panorama_url"],
            )
            return world_assets

        async def _plushie_pipeline() -> dict:
            """Pipeline B: plushie photo → stylize → Meshy 3D model (non-fatal on failure)."""
            state.update_plushie_status(world_id, "generating")
            try:
                # Stylize plushie photo before sending to Meshy
                styled_plushie_path = str(world_dir / "plushie_styled.png")
                logger.info("[%s] Stylizing plushie photo via Fal AI", world_id)
                await stylize_drawing(plushie_path, styled_plushie_path, mode=f"{mode}_plushie")
                logger.info("[%s] Plushie stylization complete: %s", world_id, styled_plushie_path)

                local_paths = await generate_plushie_model(
                    meshy_client, styled_plushie_path, str(world_dir), timeout=300.0,
                )
                state.update_plushie_status(world_id, "complete")
                plushie_assets = _plushie_asset_urls(world_id, plushie_path)
                plushie_assets["plushie_styled_url"] = f"/assets/{world_id}/plushie_styled.png"
                if local_paths.get("glb_path"):
                    plushie_assets["plushie_glb_url"] = f"/assets/{world_id}/plushie.glb"
                if local_paths.get("fbx_path"):
                    plushie_assets["plushie_fbx_url"] = f"/assets/{world_id}/plushie.fbx"
                if local_paths.get("thumbnail_path"):
                    plushie_assets["plushie_thumbnail_url"] = f"/assets/{world_id}/plushie_thumbnail.png"
                # Direct CDN URLs for plushie model
                plushie_assets["cdn_plushie_glb_url"] = local_paths.get("cdn_glb_url")
                plushie_assets["cdn_plushie_fbx_url"] = local_paths.get("cdn_fbx_url")
                return plushie_assets
            except Exception as e:
                logger.warning("[%s] Plushie generation failed (non-fatal): %s", world_id, e)
                state.update_plushie_status(world_id, "failed")
                return _plushie_asset_urls(world_id, plushie_path)

        if plushie_path and meshy_client:
            # Start both tasks concurrently, but cancel plushie generation if world generation fails.
            logger.info("[%s] Running world + plushie pipelines in parallel", world_id)
            world_task = asyncio.create_task(_world_pipeline())
            plushie_task = asyncio.create_task(_plushie_pipeline())
            try:
                world_assets = await world_task
            except Exception:
                plushie_task.cancel()
                with suppress(asyncio.CancelledError):
                    await plushie_task
                if state.get_world(world_id) and state.get_world(world_id).get("plushie_status") == "generating":
                    state.update_plushie_status(world_id, "failed")
                raise

            plushie_assets = await plushie_task
            assets = {**world_assets, **plushie_assets}
        else:
            world_assets = await _world_pipeline()
            assets = {**world_assets, **_plushie_asset_urls(world_id, plushie_path)}

        state.set_assets(world_id, assets)
        state.update_status(world_id, "complete", 2, "Your world is ready!")
        logger.info("[%s] Pipeline complete", world_id)

    except Exception as e:
        logger.exception("[%s] Pipeline failed: %s", world_id, e)
        state.set_error(world_id, str(e))
