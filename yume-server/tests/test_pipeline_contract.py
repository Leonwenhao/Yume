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


def test_run_pipeline_merges_plushie_assets_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "YUME_ASSETS_DIR", str(tmp_path))

    world = state.create_world("kids", has_plushie=True)
    drawing_path = pipeline.persist_original_drawing(world["world_id"], _png_bytes("green"), tmp_path)
    plushie_path = pipeline.persist_plushie_photo(world["world_id"], _png_bytes("pink"), tmp_path)

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

    async def fake_generate_plushie_model(_meshy_client, _image_path, output_dir, timeout=300.0):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        local_paths = {
            "glb_path": out / "plushie.glb",
            "fbx_path": out / "plushie.fbx",
            "thumbnail_path": out / "plushie_thumbnail.png",
        }
        for path in local_paths.values():
            path.write_bytes(b"fixture")
        return {key: str(path) for key, path in local_paths.items()}

    monkeypatch.setattr(pipeline, "stylize_drawing", fake_stylize_drawing)
    monkeypatch.setattr(pipeline, "generate_with_fallback", fake_generate_with_fallback)
    monkeypatch.setattr(pipeline, "generate_plushie_model", fake_generate_plushie_model)

    asyncio.run(
        pipeline.run_pipeline(
            world["world_id"],
            drawing_path,
            "kids",
            marble_client=object(),
            plushie_path=plushie_path,
            meshy_client=object(),
        )
    )

    updated = state.get_world(world["world_id"])
    assert updated["status"] == "complete"
    assert updated["plushie_status"] == "complete"
    assert updated["assets"]["plushie_glb_url"] == f"/assets/{world['world_id']}/plushie.glb"
    assert updated["assets"]["plushie_fbx_url"] == f"/assets/{world['world_id']}/plushie.fbx"
    assert updated["assets"]["plushie_thumbnail_url"] == f"/assets/{world['world_id']}/plushie_thumbnail.png"
    assert updated["assets"]["plushie_photo_url"] == f"/assets/{world['world_id']}/plushie.png"


def test_run_pipeline_treats_plushie_failure_as_non_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "YUME_ASSETS_DIR", str(tmp_path))

    world = state.create_world("kids", has_plushie=True)
    drawing_path = pipeline.persist_original_drawing(world["world_id"], _png_bytes("green"), tmp_path)
    plushie_path = pipeline.persist_plushie_photo(world["world_id"], _png_bytes("pink"), tmp_path)

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

    async def failing_generate_plushie_model(*_args, **_kwargs):
        raise RuntimeError("Meshy task failed")

    monkeypatch.setattr(pipeline, "stylize_drawing", fake_stylize_drawing)
    monkeypatch.setattr(pipeline, "generate_with_fallback", fake_generate_with_fallback)
    monkeypatch.setattr(pipeline, "generate_plushie_model", failing_generate_plushie_model)

    asyncio.run(
        pipeline.run_pipeline(
            world["world_id"],
            drawing_path,
            "kids",
            marble_client=object(),
            plushie_path=plushie_path,
            meshy_client=object(),
        )
    )

    updated = state.get_world(world["world_id"])
    assert updated["status"] == "complete"
    assert updated["plushie_status"] == "failed"
    assert updated["assets"]["plushie_glb_url"] is None
    assert updated["assets"]["plushie_fbx_url"] is None
    assert updated["assets"]["plushie_thumbnail_url"] is None
    assert updated["assets"]["plushie_photo_url"] == f"/assets/{world['world_id']}/plushie.png"


def test_run_pipeline_cancels_plushie_task_when_world_generation_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "YUME_ASSETS_DIR", str(tmp_path))

    world = state.create_world("kids", has_plushie=True)
    drawing_path = pipeline.persist_original_drawing(world["world_id"], _png_bytes("green"), tmp_path)
    plushie_path = pipeline.persist_plushie_photo(world["world_id"], _png_bytes("pink"), tmp_path)
    cancelled = {"value": False}

    async def fake_stylize_drawing(_input_path, output_path, mode="kids"):
        Path(output_path).write_bytes(_png_bytes("purple"))
        return output_path

    async def failing_generate_with_fallback(*_args, **_kwargs):
        await asyncio.sleep(0.01)
        raise RuntimeError("world boom")

    async def slow_generate_plushie_model(*_args, **_kwargs):
        try:
            await asyncio.sleep(1)
            return {"glb_path": str(tmp_path / "never-written.glb")}
        except asyncio.CancelledError:
            cancelled["value"] = True
            raise

    monkeypatch.setattr(pipeline, "stylize_drawing", fake_stylize_drawing)
    monkeypatch.setattr(pipeline, "generate_with_fallback", failing_generate_with_fallback)
    monkeypatch.setattr(pipeline, "generate_plushie_model", slow_generate_plushie_model)

    asyncio.run(
        pipeline.run_pipeline(
            world["world_id"],
            drawing_path,
            "kids",
            marble_client=object(),
            plushie_path=plushie_path,
            meshy_client=object(),
        )
    )

    updated = state.get_world(world["world_id"])
    assert updated["status"] == "failed"
    assert updated["error"] == "world boom"
    assert updated["plushie_status"] == "failed"
    assert updated["assets"] is None
    assert cancelled["value"] is True


def test_run_pipeline_preserves_plushie_failure_schema_when_generation_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "YUME_ASSETS_DIR", str(tmp_path))

    world = state.create_world("kids", has_plushie=True)
    state.update_plushie_status(world["world_id"], "failed")
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
    assert updated["plushie_status"] == "failed"
    assert updated["assets"]["plushie_glb_url"] is None
    assert updated["assets"]["plushie_fbx_url"] is None
    assert updated["assets"]["plushie_thumbnail_url"] is None
    assert updated["assets"]["plushie_photo_url"] is None
