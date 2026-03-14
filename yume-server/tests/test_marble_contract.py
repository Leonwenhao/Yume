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
