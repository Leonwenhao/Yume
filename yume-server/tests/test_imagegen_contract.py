import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.imagegen as imagegen


def test_load_style_prompt_reads_plushie_prompt_file():
    prompt_path = imagegen._PROMPTS_DIR / "plushie.txt"
    assert imagegen.load_style_prompt("plushie") == prompt_path.read_text().strip()


def test_load_style_prompt_uses_plushie_default_when_prompt_missing_or_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(imagegen, "_PROMPTS_DIR", tmp_path)

    assert imagegen.load_style_prompt("plushie") == imagegen._DEFAULT_PROMPTS["plushie"]

    (tmp_path / "plushie.txt").write_text("   \n")
    assert imagegen.load_style_prompt("plushie") == imagegen._DEFAULT_PROMPTS["plushie"]


def test_stylize_drawing_uses_async_fal_client_calls(tmp_path, monkeypatch):
    input_path = tmp_path / "drawing.png"
    output_path = tmp_path / "styled.png"
    input_path.write_bytes(b"input")
    calls = []

    async def fake_upload_file_async(path):
        calls.append(("upload_async", path))
        await asyncio.sleep(0)
        return "https://storage.example/input.png"

    async def fake_subscribe_async(application, arguments, **_kwargs):
        calls.append(
            (
                "subscribe_async",
                application,
                arguments["image_urls"][0],
                arguments["prompt"],
                arguments["output_format"],
            )
        )
        await asyncio.sleep(0)
        return {"images": [{"url": "https://cdn.example/styled.png"}]}

    def fail_sync(*_args, **_kwargs):
        raise AssertionError("synchronous Fal helpers should not be used by stylize_drawing")

    class FakeResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, timeout):
            calls.append(("download", url, timeout))
            return FakeResponse(b"styled-bytes")

    monkeypatch.setattr(imagegen, "_require_fal_key", lambda: "fal-test-key")
    monkeypatch.setattr(imagegen.fal_client, "upload_file_async", fake_upload_file_async)
    monkeypatch.setattr(imagegen.fal_client, "subscribe_async", fake_subscribe_async)
    monkeypatch.setattr(imagegen.fal_client, "upload_file", fail_sync)
    monkeypatch.setattr(imagegen.fal_client, "subscribe", fail_sync)
    monkeypatch.setattr(imagegen.httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(imagegen.stylize_drawing(str(input_path), str(output_path), mode="plushie"))

    assert result == str(output_path)
    assert output_path.read_bytes() == b"styled-bytes"
    assert calls[0] == ("upload_async", str(input_path))
    assert calls[1] == (
        "subscribe_async",
        "fal-ai/flux-2-pro/edit",
        "https://storage.example/input.png",
        imagegen.load_style_prompt("plushie"),
        "png",
    )
    assert calls[2] == ("download", "https://cdn.example/styled.png", 30.0)
