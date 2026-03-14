import asyncio
import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline
import state


def _png_bytes(color: str = "blue") -> bytes:
    image = Image.new("RGB", (24, 24), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def setup_function() -> None:
    state._worlds.clear()


def test_run_pipeline_sets_contract_asset_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "YUME_ASSETS_DIR", str(tmp_path))

    world = state.create_world("kids")
    drawing_path = pipeline.persist_original_drawing(world["world_id"], _png_bytes("green"), tmp_path)

    async def fake_stylize_drawing(_input_path, output_path, mode="kids"):
        Path(output_path).write_bytes(_png_bytes("purple"))
        return output_path

    async def fake_generate_with_fallback(_marble_client, _image_path, output_dir, mode="kids", timeout=90.0):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        local_paths = {
            "spz_url": out / "world.spz",
            "collider_url": out / "collider.glb",
            "panorama_url": out / "panorama.png",
            "thumbnail_url": out / "thumbnail.png",
        }
        for path in local_paths.values():
            path.write_bytes(b"fixture")
        return {key: str(path) for key, path in local_paths.items()}

    monkeypatch.setattr(pipeline, "stylize_drawing", fake_stylize_drawing)
    monkeypatch.setattr(pipeline, "generate_with_fallback", fake_generate_with_fallback)

    asyncio.run(pipeline.run_pipeline(world["world_id"], drawing_path, "kids", marble_client=object()))

    updated = state.get_world(world["world_id"])
    assert updated["status"] == "complete"
    assert updated["assets"] == {
        "original_drawing": f"/assets/{world['world_id']}/drawing.png",
        "styled_image": f"/assets/{world['world_id']}/styled.png",
        "splat_url": f"/assets/{world['world_id']}/world.spz",
        "splat_ply_url": None,
        "collider_url": f"/assets/{world['world_id']}/collider.glb",
        "panorama_url": f"/assets/{world['world_id']}/panorama.png",
        "thumbnail_url": f"/assets/{world['world_id']}/thumbnail.png",
    }


def test_run_pipeline_sets_failed_status_on_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "YUME_ASSETS_DIR", str(tmp_path))

    world = state.create_world("kids")
    drawing_path = pipeline.persist_original_drawing(world["world_id"], _png_bytes("green"), tmp_path)

    async def failing_stylize_drawing(*_args, **_kwargs):
        raise RuntimeError("Fal upload failed: missing key")

    monkeypatch.setattr(pipeline, "stylize_drawing", failing_stylize_drawing)

    asyncio.run(pipeline.run_pipeline(world["world_id"], drawing_path, "kids", marble_client=object()))

    updated = state.get_world(world["world_id"])
    assert updated["status"] == "failed"
    assert updated["error"] == "Fal upload failed: missing key"
