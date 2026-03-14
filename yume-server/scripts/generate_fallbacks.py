#!/usr/bin/env python3
"""Generate fallback assets from a known-good styled image via Marble API.

Usage: MARBLE_API_KEY=... python scripts/generate_fallbacks.py <styled_image> <mode>

Saves assets to assets/fallback_{mode}/
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from modules.marble import MarbleClient, ensure_placeholder_collider


async def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/generate_fallbacks.py <styled_image_path> <mode>")
        print("  mode: kids or filmmaker")
        sys.exit(1)

    image_path = sys.argv[1]
    mode = sys.argv[2]

    if mode not in ("kids", "filmmaker"):
        print(f"Invalid mode: {mode}. Must be 'kids' or 'filmmaker'.")
        sys.exit(1)

    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    output_dir = f"assets/fallback_{mode}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    client = MarbleClient()
    print(f"Generating fallback assets for mode '{mode}' from {image_path}...")
    print(f"Output: {output_dir}/")

    asset_urls = await client.generate_world(image_path, display_name=f"Yume Fallback ({mode})")
    print("Asset URLs received:")
    for k, v in asset_urls.items():
        print(f"  {k}: {v}")

    local_paths = await client.download_assets(asset_urls, output_dir)
    if "collider_url" not in local_paths:
        local_paths["collider_url"] = ensure_placeholder_collider(output_dir)

    required = {
        "spz_url": Path(output_dir) / "world.spz",
        "collider_url": Path(output_dir) / "collider.glb",
        "panorama_url": Path(output_dir) / "panorama.png",
        "thumbnail_url": Path(output_dir) / "thumbnail.png",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        print(f"\nFallback generation incomplete; missing required files: {', '.join(missing)}")
        sys.exit(1)

    print(f"\nFallback assets saved:")
    for k, v in local_paths.items():
        size = Path(v).stat().st_size
        print(f"  {k}: {v} ({size:,} bytes)")

    print(f"\nDone! Fallback assets for '{mode}' are ready in {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
