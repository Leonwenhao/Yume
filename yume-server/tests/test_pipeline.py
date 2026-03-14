#!/usr/bin/env python3
"""End-to-end pipeline test. Requires FAL_KEY and MARBLE_API_KEY.

Usage: FAL_KEY=... MARBLE_API_KEY=... python tests/test_pipeline.py [drawing_path]

If no drawing_path is given, uses a generated test image.
"""
import asyncio
import base64
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from httpx import AsyncClient, ASGITransport

VALID_STATUSES = {
    "stylizing_drawing",
    "generating_world",
    "complete",
    "failed",
}
REQUIRED_ASSET_KEYS = {
    "original_drawing",
    "styled_image",
    "splat_url",
    "splat_ply_url",
    "collider_url",
    "panorama_url",
    "thumbnail_url",
}


def png_bytes(color: str) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (512, 512), color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


async def submit_generate(client: AsyncClient, drawing_path: str, mode: str, use_json: bool) -> str:
    if use_json:
        drawing_bytes = Path(drawing_path).read_bytes()
        drawing_data_uri = f"data:image/png;base64,{base64.b64encode(drawing_bytes).decode('ascii')}"
        response = await client.post(
            "/api/generate",
            json={"image": drawing_data_uri, "mode": mode},
        )
        source = "json"
    else:
        with open(drawing_path, "rb") as handle:
            response = await client.post(
                "/api/generate",
                files={"drawing": ("drawing.png", handle, "image/png")},
                data={"mode": mode},
            )
        source = "multipart"

    assert response.status_code == 200, f"{source} generate failed: {response.text}"
    data = response.json()
    assert data["status"] in VALID_STATUSES, f"Unexpected generate status from {source}: {data}"
    print(f"{source.title()} submit ({mode}) -> {data['world_id']} [{data['status']}]")
    return data["world_id"]


async def poll_until_complete(client: AsyncClient, world_id: str, timeout_seconds: float = 180.0) -> dict:
    start = time.time()
    last_status = None

    while True:
        await asyncio.sleep(2)
        response = await client.get(f"/api/world/{world_id}/status")
        assert response.status_code == 200, f"Status failed: {response.text}"
        data = response.json()
        status = data["status"]
        assert status in VALID_STATUSES, f"Unexpected public status: {data}"

        if status != last_status:
            elapsed = time.time() - start
            print(f"  [{elapsed:.0f}s] {world_id}: {status} — {data['stage_label']}")
            last_status = status

        if status == "complete":
            return data
        if status == "failed":
            raise AssertionError(f"Pipeline failed for {world_id}: {data['error']}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Pipeline timed out for {world_id} after {timeout_seconds}s")


async def verify_assets(client: AsyncClient, world_id: str) -> dict:
    response = await client.get(f"/api/world/{world_id}/assets")
    assert response.status_code == 200, f"Assets failed: {response.text}"
    assets = response.json()
    assert REQUIRED_ASSET_KEYS.issubset(assets.keys()), f"Missing asset keys: {assets}"

    for key in sorted(REQUIRED_ASSET_KEYS):
        print(f"  {world_id} asset {key}: {assets[key]}")

    for key in ("original_drawing", "styled_image", "splat_url", "collider_url", "panorama_url", "thumbnail_url"):
        assert assets[key], f"Expected non-empty asset value for {key}"

    return assets


async def submit_polaroids_and_strip(client: AsyncClient, world_id: str) -> None:
    print("Submitting six polaroids...")
    for capture_number in range(1, 7):
        capture = png_bytes(["red", "orange", "yellow", "green", "blue", "purple"][capture_number - 1])
        capture_data_uri = f"data:image/png;base64,{base64.b64encode(capture).decode('ascii')}"
        response = await client.post(
            f"/api/world/{world_id}/polaroid",
            json={"capture": capture_data_uri, "capture_number": capture_number},
        )
        assert response.status_code == 200, f"Polaroid {capture_number} failed: {response.text}"
        print(f"  polaroid {capture_number}: {response.json()}")

    response = await client.get(f"/api/world/{world_id}/strip")
    assert response.status_code == 200, f"Strip failed: {response.text}"
    print(f"Strip response: {response.json()}")


async def main():
    from main import app

    drawing_path = sys.argv[1] if len(sys.argv) > 1 else None

    if drawing_path and not Path(drawing_path).exists():
        print(f"File not found: {drawing_path}")
        sys.exit(1)

    # Create test image if none provided
    if not drawing_path:
        from PIL import Image
        img = Image.new("RGB", (512, 512), "green")
        drawing_path = "test_drawing_e2e.png"
        img.save(drawing_path)
        print(f"Using generated test image: {drawing_path}")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        print("Step 1: Submit JSON/data-URI request...")
        json_world_id = await submit_generate(client, drawing_path, mode="kids", use_json=True)

        print("Step 2: Submit multipart request...")
        multipart_world_id = await submit_generate(client, drawing_path, mode="filmmaker", use_json=False)

        print("Step 3: Polling both worlds...")
        start = time.time()
        await asyncio.gather(
            poll_until_complete(client, json_world_id),
            poll_until_complete(client, multipart_world_id),
        )

        print("Step 4: Fetching assets...")
        await verify_assets(client, json_world_id)
        await verify_assets(client, multipart_world_id)

        print("Step 5: Verifying polaroids and strip...")
        await submit_polaroids_and_strip(client, json_world_id)

        total = time.time() - start
        print(f"\nPipeline checks complete in {total:.1f}s")
        print(f"Kids world: {json_world_id}")
        print(f"Filmmaker world: {multipart_world_id}")
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
