import asyncio
import io
import sys
from pathlib import Path

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.meshgen as meshgen


class FakeResponse:
    def __init__(self, json_data=None, status_code: int = 200, content: bytes = b""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.content = content

    def json(self):
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.com")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)


def _jpeg_bytes(color: str = "pink") -> bytes:
    image = Image.new("RGB", (24, 24), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_auth_headers_requires_api_key(monkeypatch):
    monkeypatch.setattr(meshgen, "MESHY_API_KEY", "")
    client = meshgen.MeshyClient()

    with pytest.raises(RuntimeError, match="MESHY_API_KEY is not configured"):
        client._auth_headers()


def test_submit_image_to_3d_uses_expected_payload_and_parses_result(monkeypatch):
    captured = {}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse({"result": "task_123"})

    monkeypatch.setattr(meshgen.httpx, "AsyncClient", FakeAsyncClient)

    task_id = asyncio.run(meshgen.MeshyClient("mesh-key").submit_image_to_3d("data:image/png;base64,abc"))

    assert task_id == "task_123"
    assert captured["url"] == f"{meshgen.BASE_URL}/image-to-3d"
    assert captured["headers"]["Authorization"] == "Bearer mesh-key"
    assert captured["json"] == {
        "image_url": "data:image/png;base64,abc",
        "ai_model": "meshy-6",
        "should_texture": True,
        "enable_pbr": True,
        "should_remesh": True,
        "target_polycount": 20000,
        "topology": "triangle",
    }


def test_poll_task_raises_runtime_error_for_failed_status(monkeypatch):
    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None, timeout=None):
            return FakeResponse({"status": "FAILED", "task_error": {"message": "bad input"}})

    monkeypatch.setattr(meshgen.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(RuntimeError, match=r"Meshy task task_123 FAILED: bad input"):
        asyncio.run(meshgen.MeshyClient("mesh-key").poll_task("task_123", timeout=5.0))


def test_poll_task_raises_timeout(monkeypatch):
    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None, timeout=None):
            return FakeResponse({"status": "PENDING", "progress": 0})

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(meshgen.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(meshgen, "POLL_INTERVAL", 1)
    monkeypatch.setattr(meshgen.asyncio, "sleep", fake_sleep)

    with pytest.raises(TimeoutError, match=r"Meshy task task_123 timed out after 2.5s"):
        asyncio.run(meshgen.MeshyClient("mesh-key").poll_task("task_123", timeout=2.5))


def test_download_model_assets_survives_individual_failures(tmp_path, monkeypatch):
    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, timeout=None, follow_redirects=False):
            if url.endswith(".glb"):
                return FakeResponse(content=b"glb-bytes")
            raise httpx.ConnectError("download failed")

    monkeypatch.setattr(meshgen.httpx, "AsyncClient", FakeAsyncClient)

    local_paths = asyncio.run(
        meshgen.MeshyClient("mesh-key").download_model_assets(
            {
                "model_urls": {
                    "glb": "https://example.com/plushie.glb",
                    "fbx": "https://example.com/plushie.fbx",
                },
                "thumbnail_url": "https://example.com/plushie.png",
            },
            str(tmp_path),
        )
    )

    assert local_paths == {"glb_path": str(tmp_path / "plushie.glb")}
    assert (tmp_path / "plushie.glb").read_bytes() == b"glb-bytes"


def test_generate_plushie_model_uses_actual_image_mime_not_path_suffix(tmp_path):
    image_path = tmp_path / "plushie.png"
    image_path.write_bytes(_jpeg_bytes())
    captured = {}

    class FakeMeshyClient(meshgen.MeshyClient):
        async def submit_image_to_3d(self, image_url: str) -> str:
            captured["image_url"] = image_url
            return "task_123"

        async def poll_task(self, task_id: str, timeout: float = 300.0) -> dict:
            return {"model_urls": {}}

        async def download_model_assets(self, task_result: dict, output_dir: str) -> dict:
            return {}

    asyncio.run(
        meshgen.generate_plushie_model(
            FakeMeshyClient("mesh-key"),
            str(image_path),
            str(tmp_path / "out"),
        )
    )

    assert captured["image_url"].startswith("data:image/jpeg;base64,")
