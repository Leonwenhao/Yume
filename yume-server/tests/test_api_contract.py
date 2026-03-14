import asyncio
import base64
import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import state


def _png_bytes(color: str = "blue") -> bytes:
    return _image_bytes(color=color, image_format="PNG")


def _image_bytes(color: str = "blue", image_format: str = "PNG") -> bytes:
    image = Image.new("RGB", (32, 32), color)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _write_png(path: Path, color: str = "blue") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(color))


def setup_function() -> None:
    state._worlds.clear()


def _capture_pipeline_invocation(monkeypatch):
    captured = {}
    scheduled = []

    async def fake_run_pipeline(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return None

    def fake_create_task(coro):
        scheduled.append(coro)
        return object()

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(main.asyncio, "create_task", fake_create_task)
    return captured, scheduled


def test_generate_accepts_json_base64(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)

    async def fake_run_pipeline(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)

    drawing_bytes = _png_bytes("green")

    with TestClient(main.app) as client:
        response = client.post(
            "/api/generate",
            json={
                "drawing": base64.b64encode(drawing_bytes).decode("ascii"),
                "mode": "kids",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "stylizing_drawing"
    assert (tmp_path / body["world_id"] / "drawing.png").read_bytes() == drawing_bytes


def test_generate_accepts_json_data_uri(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)

    async def fake_run_pipeline(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)

    drawing_bytes = _png_bytes("purple")
    data_uri = f"data:image/png;base64,{base64.b64encode(drawing_bytes).decode('ascii')}"

    with TestClient(main.app) as client:
        response = client.post(
            "/api/generate",
            json={
                "image": data_uri,
                "mode": "filmmaker",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "stylizing_drawing"
    assert (tmp_path / body["world_id"] / "drawing.png").read_bytes() == drawing_bytes


def test_generate_rejects_invalid_json_image_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)

    async def fake_run_pipeline(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/generate",
            json={"drawing": "not-valid-base64"},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "Invalid 'drawing' image payload"


def test_generate_accepts_multipart_plushie_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    captured, scheduled = _capture_pipeline_invocation(monkeypatch)

    drawing_bytes = _png_bytes("green")
    plushie_bytes = _png_bytes("pink")

    with TestClient(main.app) as client:
        response = client.post(
            "/api/generate",
            files={
                "drawing": ("drawing.png", drawing_bytes, "image/png"),
                "plushie": ("plushie.png", plushie_bytes, "image/png"),
            },
            data={"mode": "kids"},
        )

    assert response.status_code == 200
    body = response.json()
    asyncio.run(scheduled.pop())

    world_dir = tmp_path / body["world_id"]
    assert (world_dir / "drawing.png").read_bytes() == drawing_bytes
    assert (world_dir / "plushie.png").read_bytes() == plushie_bytes
    assert captured["kwargs"]["plushie_path"] == str(world_dir / "plushie.png")
    assert captured["kwargs"]["meshy_client"] is main.meshy_client
    assert state.get_world(body["world_id"])["plushie_status"] == "pending"


def test_generate_accepts_json_plushie_base64(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    captured, scheduled = _capture_pipeline_invocation(monkeypatch)

    drawing_bytes = _png_bytes("green")
    plushie_bytes = _png_bytes("pink")

    with TestClient(main.app) as client:
        response = client.post(
            "/api/generate",
            json={
                "drawing": base64.b64encode(drawing_bytes).decode("ascii"),
                "plushie": base64.b64encode(plushie_bytes).decode("ascii"),
            },
        )

    assert response.status_code == 200
    body = response.json()
    asyncio.run(scheduled.pop())

    world_dir = tmp_path / body["world_id"]
    assert (world_dir / "plushie.png").read_bytes() == plushie_bytes
    assert captured["kwargs"]["plushie_path"] == str(world_dir / "plushie.png")


def test_generate_accepts_json_plushie_data_uri_and_normalizes_photo(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    captured, scheduled = _capture_pipeline_invocation(monkeypatch)

    drawing_bytes = _png_bytes("green")
    plushie_bytes = _image_bytes("pink", image_format="JPEG")
    plushie_data_uri = f"data:image/jpeg;base64,{base64.b64encode(plushie_bytes).decode('ascii')}"

    with TestClient(main.app) as client:
        response = client.post(
            "/api/generate",
            json={
                "drawing": base64.b64encode(drawing_bytes).decode("ascii"),
                "plushie": plushie_data_uri,
            },
        )

    assert response.status_code == 200
    body = response.json()
    asyncio.run(scheduled.pop())

    persisted = (tmp_path / body["world_id"] / "plushie.png").read_bytes()
    assert persisted.startswith(b"\x89PNG\r\n\x1a\n")
    assert captured["kwargs"]["plushie_path"] == str(tmp_path / body["world_id"] / "plushie.png")


def test_generate_treats_empty_plushie_upload_as_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    captured, scheduled = _capture_pipeline_invocation(monkeypatch)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/generate",
            files={
                "drawing": ("drawing.png", _png_bytes("green"), "image/png"),
                "plushie": ("plushie.png", b"", "image/png"),
            },
        )

    assert response.status_code == 200
    body = response.json()
    asyncio.run(scheduled.pop())

    world_dir = tmp_path / body["world_id"]
    assert not (world_dir / "plushie.png").exists()
    assert captured["kwargs"]["plushie_path"] is None
    assert captured["kwargs"]["meshy_client"] is None
    assert state.get_world(body["world_id"])["plushie_status"] is None


def test_generate_rejects_invalid_json_plushie_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    captured, _scheduled = _capture_pipeline_invocation(monkeypatch)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/generate",
            json={
                "drawing": base64.b64encode(_png_bytes("green")).decode("ascii"),
                "plushie": "not-valid-base64",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "Invalid 'plushie' image payload"
    assert captured == {}


def test_generate_marks_plushie_failed_when_persisting_optional_photo_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    captured, scheduled = _capture_pipeline_invocation(monkeypatch)

    def fail_persist(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(main, "persist_plushie_photo", fail_persist)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/generate",
            json={
                "drawing": base64.b64encode(_png_bytes("green")).decode("ascii"),
                "plushie": base64.b64encode(_png_bytes("pink")).decode("ascii"),
            },
        )

    assert response.status_code == 200
    body = response.json()
    asyncio.run(scheduled.pop())

    world = state.get_world(body["world_id"])
    assert world["plushie_status"] == "failed"
    assert captured["kwargs"]["plushie_path"] is None
    assert captured["kwargs"]["meshy_client"] is None


def test_status_endpoint_does_not_leak_processing(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    world = state.create_world("kids")
    state._worlds[world["world_id"]]["status"] = "processing"
    state._worlds[world["world_id"]]["stage"] = 0

    with TestClient(main.app) as client:
        response = client.get(f"/api/world/{world['world_id']}/status")

    assert response.status_code == 200
    assert response.json()["status"] == "stylizing_drawing"


def test_assets_endpoint_returns_complete_asset_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    world = state.create_world("kids", has_plushie=True)
    assets = {
        "original_drawing": f"/assets/{world['world_id']}/drawing.png",
        "styled_image": f"/assets/{world['world_id']}/styled.png",
        "splat_url": f"/assets/{world['world_id']}/world.spz",
        "splat_ply_url": f"/assets/{world['world_id']}/world.ply",
        "collider_url": f"/assets/{world['world_id']}/collider.glb",
        "panorama_url": f"/assets/{world['world_id']}/panorama.png",
        "thumbnail_url": f"/assets/{world['world_id']}/thumbnail.png",
        "marble_viewer_url": "https://viewer.example/world",
        "cdn_splat_url": "https://cdn.example/world.spz",
        "cdn_splat_500k_url": "https://cdn.example/world_500k.spz",
        "cdn_splat_100k_url": "https://cdn.example/world_100k.spz",
        "cdn_panorama_url": "https://cdn.example/panorama.png",
        "cdn_thumbnail_url": "https://cdn.example/thumbnail.png",
        "plushie_photo_url": f"/assets/{world['world_id']}/plushie.png",
        "plushie_styled_url": f"/assets/{world['world_id']}/plushie_styled.png",
        "plushie_glb_url": f"/assets/{world['world_id']}/plushie.glb",
        "plushie_fbx_url": f"/assets/{world['world_id']}/plushie.fbx",
        "plushie_thumbnail_url": f"/assets/{world['world_id']}/plushie_thumbnail.png",
        "cdn_plushie_glb_url": "https://cdn.example/plushie.glb",
        "cdn_plushie_fbx_url": "https://cdn.example/plushie.fbx",
    }
    state.set_assets(world["world_id"], assets)

    with TestClient(main.app) as client:
        response = client.get(f"/api/world/{world['world_id']}/assets")

    assert response.status_code == 200
    assert response.json() == {"world_id": world["world_id"], **assets}


def test_polaroid_accepts_data_uri_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    world = state.create_world("kids")

    capture_bytes = _png_bytes("orange")
    data_uri = f"data:image/png;base64,{base64.b64encode(capture_bytes).decode('ascii')}"

    with TestClient(main.app) as client:
        response = client.post(
            f"/api/world/{world['world_id']}/polaroid",
            json={"capture": data_uri, "capture_number": "1"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "polaroid_url": f"/assets/{world['world_id']}/polaroid_1.png",
        "remaining": 5,
    }
    assert (tmp_path / world["world_id"] / "polaroid_1.png").exists()


def test_polaroid_rejects_non_numeric_capture_number(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    world = state.create_world("kids")

    with TestClient(main.app) as client:
        response = client.post(
            f"/api/world/{world['world_id']}/polaroid",
            json={"capture": base64.b64encode(_png_bytes()).decode("ascii"), "capture_number": "first"},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "capture_number must be an integer from 1 to 6"


def test_polaroid_rejects_invalid_capture_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    world = state.create_world("kids")

    with TestClient(main.app) as client:
        response = client.post(
            f"/api/world/{world['world_id']}/polaroid",
            json={"capture": "not-valid-base64", "capture_number": 1},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "Invalid 'capture' image payload"


def test_strip_requires_all_six_polaroids(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    world = state.create_world("kids")
    world_dir = tmp_path / world["world_id"]
    _write_png(world_dir / "drawing.png", "white")

    for capture_number in range(1, 6):
        path = world_dir / f"polaroid_{capture_number}.png"
        _write_png(path, "red")
        state.add_polaroid(world["world_id"], str(path))

    with TestClient(main.app) as client:
        response = client.get(f"/api/world/{world['world_id']}/strip")

    assert response.status_code == 400
    assert response.json()["error"] == "Need 6 polaroids, have 5"
    assert response.json()["missing"] == ["polaroid_6.png"]


def test_strip_returns_controlled_400_when_expected_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "assets_path", tmp_path)
    world = state.create_world("kids")
    world_dir = tmp_path / world["world_id"]
    _write_png(world_dir / "drawing.png", "white")

    for capture_number in range(1, 7):
        path = world_dir / f"polaroid_{capture_number}.png"
        _write_png(path, "yellow")
        state.add_polaroid(world["world_id"], str(path))

    (world_dir / "polaroid_4.png").unlink()

    with TestClient(main.app) as client:
        response = client.get(f"/api/world/{world['world_id']}/strip")

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "Need 6 polaroids, have 5"
    assert body["missing"] == ["polaroid_4.png"]
