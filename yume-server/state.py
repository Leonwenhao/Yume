import logging
import random
import string
from datetime import datetime, timezone

logger = logging.getLogger("yume.state")

_worlds: dict[str, dict] = {}


def _gen_id() -> str:
    return "yume_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def create_world(mode: str, has_plushie: bool = False) -> dict:
    world_id = _gen_id()
    world = {
        "world_id": world_id,
        "mode": mode,
        "status": "stylizing_drawing",
        "stage": 1,
        "total_stages": 2,
        "stage_label": "Turning your drawing into a world...",
        "assets": None,
        "styled_image": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "polaroids": [],
        "has_plushie": has_plushie,
        "plushie_status": "pending" if has_plushie else None,
    }
    _worlds[world_id] = world
    logger.info("Created world %s (mode=%s)", world_id, mode)
    return world


def update_status(world_id: str, status: str, stage: int, stage_label: str) -> None:
    world = _worlds.get(world_id)
    if world is None:
        logger.warning("update_status: unknown world %s", world_id)
        return
    world["status"] = status
    world["stage"] = stage
    world["stage_label"] = stage_label
    logger.info("World %s → %s (stage %d: %s)", world_id, status, stage, stage_label)


def get_world(world_id: str) -> dict | None:
    return _worlds.get(world_id)


def set_assets(world_id: str, assets: dict) -> None:
    world = _worlds.get(world_id)
    if world is None:
        logger.warning("set_assets: unknown world %s", world_id)
        return
    world["assets"] = assets
    logger.info("Assets set for world %s", world_id)


def add_polaroid(world_id: str, polaroid_path: str) -> int:
    """Add a polaroid path. Returns count of polaroids after adding."""
    world = _worlds.get(world_id)
    if world is None:
        logger.warning("add_polaroid: unknown world %s", world_id)
        return 0
    if polaroid_path not in world["polaroids"]:
        world["polaroids"].append(polaroid_path)
    else:
        logger.info("Polaroid already tracked for world %s: %s", world_id, polaroid_path)
    count = len(world["polaroids"])
    logger.info("Polaroid %d added for world %s", count, world_id)
    return count


def update_plushie_status(world_id: str, plushie_status: str, stage_label: str | None = None) -> None:
    """Update plushie generation sub-status."""
    world = _worlds.get(world_id)
    if world is None:
        logger.warning("update_plushie_status: unknown world %s", world_id)
        return
    world["plushie_status"] = plushie_status
    if stage_label:
        world["stage_label"] = stage_label
    logger.info("World %s plushie → %s", world_id, plushie_status)


def set_error(world_id: str, error: str) -> None:
    world = _worlds.get(world_id)
    if world is None:
        return
    world["status"] = "failed"
    world["stage_label"] = "World generation failed."
    world["error"] = error
    logger.error("World %s failed: %s", world_id, error)
