import logging
from pathlib import Path

import state
from config import YUME_ASSETS_DIR
from modules.imagegen import stylize_drawing
from modules.marble import MarbleClient, generate_with_fallback

logger = logging.getLogger("yume.pipeline")


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


async def run_pipeline(world_id: str, drawing_path: str, mode: str, marble_client: MarbleClient) -> None:
    """Full async pipeline: stylize drawing → generate 3D world → store assets.

    This runs as a background task. Updates world state at each step.
    """
    world_dir = Path(YUME_ASSETS_DIR) / world_id
    styled_path = world_dir / "styled.png"

    try:
        drawing_file = Path(drawing_path)
        if not drawing_file.exists():
            raise FileNotFoundError(f"Original drawing not found: {drawing_file}")

        # Stage 1: Stylize drawing via Fal AI
        state.update_status(world_id, "stylizing_drawing", 1, "Turning your drawing into a world...")
        logger.info("[%s] Stage 1: Stylizing drawing (mode=%s)", world_id, mode)

        await stylize_drawing(str(drawing_file), str(styled_path), mode=mode)
        logger.info("[%s] Stage 1 complete: %s", world_id, styled_path)

        # Stage 2: Generate 3D world via Marble
        state.update_status(world_id, "generating_world", 2, "Building your 3D world...")
        logger.info("[%s] Stage 2: Generating world via Marble", world_id)

        local_assets = await generate_with_fallback(
            marble_client,
            str(styled_path),
            str(world_dir),
            mode=mode,
            timeout=90.0,
        )
        logger.info("[%s] Stage 2 complete: %d assets", world_id, len(local_assets))

        required_local_assets = ("spz_url", "collider_url", "panorama_url", "thumbnail_url")
        missing_assets = [key for key in required_local_assets if not local_assets.get(key)]
        if missing_assets:
            raise RuntimeError(f"Missing required generated assets: {', '.join(missing_assets)}")

        # Build asset URL paths (relative to /assets mount)
        prefix = f"/assets/{world_id}"
        assets = {
            "original_drawing": f"{prefix}/drawing.png",
            "styled_image": f"{prefix}/styled.png",
            "splat_url": f"{prefix}/world.spz",
            "splat_ply_url": f"{prefix}/world.ply" if local_assets.get("ply_url") else None,
            "collider_url": f"{prefix}/collider.glb",
            "panorama_url": f"{prefix}/panorama.png",
            "thumbnail_url": f"{prefix}/thumbnail.png",
        }

        state.set_assets(world_id, assets)
        state.update_status(world_id, "complete", 2, "Your world is ready!")
        logger.info("[%s] Pipeline complete", world_id)

    except Exception as e:
        logger.exception("[%s] Pipeline failed: %s", world_id, e)
        state.set_error(world_id, str(e))
