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
        return {
            **{key: str(path) for key, path in local_paths.items()},
            "marble_url": "https://viewer.example/world",
            "cdn_spz_url": "https://cdn.example/world.spz",
            "cdn_spz_500k_url": "https://cdn.example/world_500k.spz",
            "cdn_spz_100k_url": "https://cdn.example/world_100k.spz",
            "cdn_panorama_url": "https://cdn.example/panorama.png",
            "cdn_thumbnail_url": "https://cdn.example/thumbnail.png",
        }

    monkeypatch.setattr(pipeline, "stylize_drawing", fake_stylize_drawing)
    monkeypatch.setattr(pipeline, "generate_with_fallback", fake_generate_with_fallback)

    asyncio.run(pipeline.run_pipeline(world["world_id"], drawing_path, "kids", marble_client=object()))

    updated = state.get_world(world["world_id"])
    assert updated["status"] == "complete"
    expected_keys = {
        "original_drawing", "styled_image", "splat_url", "splat_ply_url",
        "collider_url", "panorama_url", "thumbnail_url",
        "marble_viewer_url", "cdn_splat_url", "cdn_splat_500k_url",
        "cdn_splat_100k_url", "cdn_panorama_url", "cdn_thumbnail_url",
        "plushie_glb_url", "plushie_fbx_url", "plushie_thumbnail_url",
        "plushie_photo_url", "plushie_styled_url", "cdn_plushie_glb_url", "cdn_plushie_fbx_url",
    }
    assert expected_keys.issubset(updated["assets"].keys())
    wid = world["world_id"]
    assert updated["assets"]["splat_url"] == f"/assets/{wid}/world.spz"
    assert updated["assets"]["collider_url"] == f"/assets/{wid}/collider.glb"
    assert updated["assets"]["panorama_url"] == f"/assets/{wid}/panorama.png"
    assert updated["assets"]["marble_viewer_url"] == "https://viewer.example/world"
    assert updated["assets"]["cdn_splat_url"] == "https://cdn.example/world.spz"
    assert updated["assets"]["cdn_splat_500k_url"] == "https://cdn.example/world_500k.spz"
    assert updated["assets"]["cdn_splat_100k_url"] == "https://cdn.example/world_100k.spz"
    assert updated["assets"]["cdn_panorama_url"] == "https://cdn.example/panorama.png"
    assert updated["assets"]["cdn_thumbnail_url"] == "https://cdn.example/thumbnail.png"
    assert updated["assets"]["plushie_glb_url"] is None
    assert updated["assets"]["plushie_fbx_url"] is None
    assert updated["assets"]["plushie_thumbnail_url"] is None
    assert updated["assets"]["plushie_photo_url"] is None
    assert updated["assets"]["plushie_styled_url"] is None
    assert updated["assets"]["cdn_plushie_glb_url"] is None
    assert updated["assets"]["cdn_plushie_fbx_url"] is None


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
    captured = {}

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
        captured["image_path"] = _image_path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        local_paths = {
            "glb_path": out / "plushie.glb",
            "fbx_path": out / "plushie.fbx",
            "thumbnail_path": out / "plushie_thumbnail.png",
        }
        for path in local_paths.values():
            path.write_bytes(b"fixture")
        return {
            **{key: str(path) for key, path in local_paths.items()},
            "cdn_glb_url": "https://cdn.example/plushie.glb",
            "cdn_fbx_url": "https://cdn.example/plushie.fbx",
            "cdn_thumbnail_url": "https://cdn.example/plushie.png",
        }

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
    assert updated["assets"]["plushie_styled_url"] == f"/assets/{world['world_id']}/plushie_styled.png"
    assert updated["assets"]["cdn_plushie_glb_url"] == "https://cdn.example/plushie.glb"
    assert updated["assets"]["cdn_plushie_fbx_url"] == "https://cdn.example/plushie.fbx"
    assert captured["image_path"] == str(tmp_path / world["world_id"] / "plushie_styled.png")
    assert (tmp_path / world["world_id"] / "plushie_styled.png").exists()


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
    assert updated["assets"]["plushie_styled_url"] is None
    assert updated["assets"]["cdn_plushie_glb_url"] is None
    assert updated["assets"]["cdn_plushie_fbx_url"] is None


def test_run_pipeline_treats_plushie_stylization_failure_as_non_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "YUME_ASSETS_DIR", str(tmp_path))

    world = state.create_world("kids", has_plushie=True)
    drawing_path = pipeline.persist_original_drawing(world["world_id"], _png_bytes("green"), tmp_path)
    plushie_path = pipeline.persist_plushie_photo(world["world_id"], _png_bytes("pink"), tmp_path)

    async def fake_stylize_drawing(_input_path, output_path, mode="dreamy"):
        if "_plushie" in mode:
            raise RuntimeError("Fal image generation failed: plushie boom")
        Path(output_path).write_bytes(_png_bytes("purple"))
        return output_path

    async def fake_generate_with_fallback(_marble_client, _image_path, output_dir, mode="dreamy", timeout=90.0):
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

    async def should_not_generate_plushie(*_args, **_kwargs):
        raise AssertionError("generate_plushie_model should not run after plushie stylization failure")

    monkeypatch.setattr(pipeline, "stylize_drawing", fake_stylize_drawing)
    monkeypatch.setattr(pipeline, "generate_with_fallback", fake_generate_with_fallback)
    monkeypatch.setattr(pipeline, "generate_plushie_model", should_not_generate_plushie)

    asyncio.run(
        pipeline.run_pipeline(
            world["world_id"],
            drawing_path,
            "dreamy",
            marble_client=object(),
            plushie_path=plushie_path,
            meshy_client=object(),
        )
    )

    updated = state.get_world(world["world_id"])
    assert updated["status"] == "complete"
    assert updated["plushie_status"] == "failed"
    assert updated["assets"]["plushie_photo_url"] == f"/assets/{world['world_id']}/plushie.png"
    assert updated["assets"]["plushie_styled_url"] is None
    assert updated["assets"]["plushie_glb_url"] is None
    assert not (tmp_path / world["world_id"] / "plushie_styled.png").exists()


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
    assert updated["assets"]["plushie_styled_url"] is None
    assert updated["assets"]["cdn_plushie_glb_url"] is None
    assert updated["assets"]["cdn_plushie_fbx_url"] is None


def test_run_pipeline_sets_world_cdn_keys_to_none_for_fallback_assets(tmp_path, monkeypatch):
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
        glb_path = out / "plushie.glb"
        glb_path.write_bytes(b"fixture")
        return {"glb_path": str(glb_path)}

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
    assert updated["assets"]["marble_viewer_url"] is None
    assert updated["assets"]["cdn_splat_url"] is None
    assert updated["assets"]["cdn_splat_500k_url"] is None
    assert updated["assets"]["cdn_splat_100k_url"] is None
    assert updated["assets"]["cdn_panorama_url"] is None
    assert updated["assets"]["cdn_thumbnail_url"] is None
