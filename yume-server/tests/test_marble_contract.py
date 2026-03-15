import asyncio
import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.marble as marble


def _write_required_fallback_assets(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "world.spz").write_bytes(b"spz")
    (directory / "collider.glb").write_bytes(b"glb")
    (directory / "panorama.png").write_bytes(b"png")
    (directory / "thumbnail.png").write_bytes(b"png")


def _write_png(path: Path, color: str = "blue") -> None:
    image = Image.new("RGB", (24, 24), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())


class OperationResponseClient(marble.MarbleClient):
    def __init__(self, operation_result: dict, fetched_world: dict | None = None):
        super().__init__(api_key="test-key")
        self.operation_result = operation_result
        self.fetched_world = fetched_world or {}
        self.fetched_world_ids: list[str] = []

    async def _prepare_upload(self, file_name: str, extension: str) -> dict:
        return {
            "media_asset": {"media_asset_id": "media_asset_123"},
            "upload_info": {"upload_url": "https://upload.example/world", "required_headers": {}},
        }

    async def _upload_file(self, upload_url: str, file_bytes: bytes, required_headers: dict) -> None:
        return None

    async def _submit_generation(self, media_asset_id: str, display_name: str, model: str = "Marble 0.1-mini") -> dict:
        return {"operation_id": "operation_123"}

    async def _poll_operation(self, operation_id: str, timeout: float = 120.0) -> dict:
        return self.operation_result

    async def _get_world(self, world_id: str) -> dict:
        self.fetched_world_ids.append(world_id)
        return self.fetched_world


class IncompleteDownloadClient:
    async def generate_world(self, _image_path: str) -> dict:
        return {
            "spz_url": "https://example.com/world.spz",
            "collider_url": "https://example.com/collider.glb",
            "panorama_url": "https://example.com/panorama.png",
            "thumbnail_url": "https://example.com/thumbnail.png",
        }

    async def download_assets(self, _asset_urls: dict, output_dir: str) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        world = out / "world.spz"
        world.write_bytes(b"partial")
        return {"spz_url": str(world)}


class TimeoutClient:
    async def generate_world(self, _image_path: str) -> dict:
        raise asyncio.TimeoutError("timed out")

    async def download_assets(self, _asset_urls: dict, output_dir: str) -> dict:
        raise AssertionError("download_assets should not be called after timeout")


class ClientErrorClient:
    async def generate_world(self, _image_path: str) -> dict:
        raise marble.MarbleClientError("Marble world generation request failed (402): Payment Required")

    async def download_assets(self, _asset_urls: dict, output_dir: str) -> dict:
        raise AssertionError("download_assets should not be called after a non-retryable client error")


def test_generate_with_fallback_uses_fallback_when_downloaded_assets_are_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(marble, "YUME_ASSETS_DIR", str(tmp_path))
    _write_required_fallback_assets(tmp_path / "fallback_kids")

    output_dir = tmp_path / "world_output"
    local_paths = asyncio.run(
        marble.generate_with_fallback(
            IncompleteDownloadClient(),
            image_path=str(tmp_path / "styled.png"),
            output_dir=str(output_dir),
            mode="kids",
        )
    )

    assert set(("spz_url", "collider_url", "panorama_url", "thumbnail_url")).issubset(local_paths)
    assert (output_dir / "world.spz").exists()
    assert (output_dir / "collider.glb").exists()
    assert (output_dir / "panorama.png").exists()
    assert (output_dir / "thumbnail.png").exists()


def test_generate_with_fallback_raises_when_required_fallback_assets_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(marble, "YUME_ASSETS_DIR", str(tmp_path))
    (tmp_path / "fallback_kids").mkdir(parents=True, exist_ok=True)

    output_dir = tmp_path / "world_output"

    try:
        asyncio.run(
            marble.generate_with_fallback(
                TimeoutClient(),
                image_path=str(tmp_path / "styled.png"),
                output_dir=str(output_dir),
                mode="kids",
            )
        )
    except FileNotFoundError as exc:
        assert "fallback missing required assets" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing fallback assets")


def test_generate_with_fallback_reraises_non_retryable_client_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(marble, "YUME_ASSETS_DIR", str(tmp_path))
    _write_required_fallback_assets(tmp_path / "fallback_kids")

    output_dir = tmp_path / "world_output"

    try:
        asyncio.run(
            marble.generate_with_fallback(
                ClientErrorClient(),
                image_path=str(tmp_path / "styled.png"),
                output_dir=str(output_dir),
                mode="kids",
            )
        )
    except marble.MarbleClientError as exc:
        assert "402" in str(exc)
    else:
        raise AssertionError("Expected MarbleClientError for non-retryable Marble API client errors")

    assert not output_dir.exists()


def test_generate_world_extracts_viewer_url_from_operation_response(tmp_path):
    image_path = tmp_path / "styled.png"
    _write_png(image_path)

    world_id = "world_abc123"
    client = OperationResponseClient(
        operation_result={
            "response": {
                "id": world_id,
                "world_marble_url": f"https://marble.worldlabs.ai/world/{world_id}",
                "assets": {
                    "thumbnail_url": "https://cdn.example/thumbnail.png",
                    "splats": {"spz_urls": {"full_res": "https://cdn.example/world.spz"}},
                    "imagery": {"pano_url": "https://cdn.example/panorama.png"},
                },
            },
            "metadata": {"world_id": world_id},
        }
    )

    asset_urls = asyncio.run(client.generate_world(str(image_path)))

    assert asset_urls["marble_url"] == f"https://marble.worldlabs.ai/world/{world_id}"
    assert asset_urls["spz_url"] == "https://cdn.example/world.spz"
    assert asset_urls["panorama_url"] == "https://cdn.example/panorama.png"
    assert client.fetched_world_ids == []


def test_generate_world_derives_viewer_url_from_world_id_when_field_is_missing(tmp_path):
    image_path = tmp_path / "styled.png"
    _write_png(image_path, color="green")

    world_id = "world_xyz789"
    client = OperationResponseClient(
        operation_result={
            "response": {
                "world": {
                    "world_id": world_id,
                    "assets": {
                        "thumbnail_url": "https://cdn.example/thumbnail.png",
                        "splats": {"spz_urls": {"full_res": "https://cdn.example/world.spz"}},
                        "imagery": {"pano_url": "https://cdn.example/panorama.png"},
                    },
                }
            },
            "metadata": {"world_id": world_id},
        },
        fetched_world={
            "world": {
                "world_id": world_id,
                "assets": {
                    "thumbnail_url": "https://cdn.example/thumbnail.png",
                    "splats": {"spz_urls": {"full_res": "https://cdn.example/world.spz"}},
                    "imagery": {"pano_url": "https://cdn.example/panorama.png"},
                },
            }
        },
    )

    asset_urls = asyncio.run(client.generate_world(str(image_path)))

    assert asset_urls["marble_url"] == f"https://marble.worldlabs.ai/world/{world_id}"
    assert client.fetched_world_ids == []


def test_download_assets_preserves_marble_url_metadata(tmp_path):
    client = marble.MarbleClient(api_key="test-key")

    local_paths = asyncio.run(
        client.download_assets(
            {"marble_url": "https://viewer.example/world"},
            str(tmp_path / "world_output"),
        )
    )

    assert local_paths["marble_url"] == "https://viewer.example/world"


def test_write_asset_file_transcodes_webp_bytes_to_real_png(tmp_path):
    image = Image.new("RGB", (12, 12), "teal")
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP")

    output_path = tmp_path / "thumbnail.png"
    marble._write_asset_file(buffer.getvalue(), output_path)

    assert output_path.exists()
    with Image.open(output_path) as written:
        assert written.format == "PNG"


def test_ensure_placeholder_collider_creates_valid_glb_header(tmp_path):
    collider_path = Path(marble.ensure_placeholder_collider(tmp_path))
    assert collider_path.exists()
    assert collider_path.read_bytes().startswith(b"glTF")
