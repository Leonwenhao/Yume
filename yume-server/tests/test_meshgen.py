#!/usr/bin/env python3
"""Standalone test for Meshy AI client. Hits real API.

Usage: MESHY_API_KEY=... python tests/test_meshgen.py path/to/plushie_photo.png
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from modules.meshgen import MeshyClient, generate_plushie_model


async def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_meshgen.py <plushie_image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    output_dir = "test_meshy_output"

    print(f"Input: {image_path}")
    print("Submitting to Meshy API...")

    start = time.time()
    local_paths = await generate_plushie_model(
        MeshyClient(), image_path, output_dir, timeout=300.0
    )
    elapsed = time.time() - start

    print(f"\nElapsed: {elapsed:.1f}s")
    print("Local files:")
    for k, v in local_paths.items():
        size = Path(v).stat().st_size
        print(f"  {k}: {v} ({size:,} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
