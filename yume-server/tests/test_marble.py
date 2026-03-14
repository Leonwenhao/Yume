#!/usr/bin/env python3
"""Standalone test for Marble API client. Hits real API.

Usage: MARBLE_API_KEY=... python tests/test_marble.py path/to/styled_image.png
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from modules.marble import MarbleClient


async def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_marble.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    output_dir = "test_marble_output"

    client = MarbleClient()
    print(f"Input: {image_path}")
    print("Submitting to Marble API...")

    start = time.time()
    asset_urls = await client.generate_world(image_path)
    gen_elapsed = time.time() - start

    print(f"Generation elapsed: {gen_elapsed:.1f}s")
    print("Asset URLs:")
    for k, v in asset_urls.items():
        print(f"  {k}: {v}")

    print(f"\nDownloading assets to {output_dir}/...")
    local_paths = await client.download_assets(asset_urls, output_dir)

    total_elapsed = time.time() - start
    print(f"\nLocal files:")
    for k, v in local_paths.items():
        size = Path(v).stat().st_size
        print(f"  {k}: {v} ({size:,} bytes)")
    print(f"Total elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
