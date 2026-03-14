#!/usr/bin/env python3
"""Standalone test for Fal AI image stylization. Hits real API.

Usage: FAL_KEY=... python tests/test_imagegen.py path/to/drawing.png [mode]
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from modules.imagegen import stylize_drawing, load_style_prompt


async def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_imagegen.py <drawing_path> [mode]")
        sys.exit(1)

    input_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "kids"

    if not Path(input_path).exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    output_path = f"test_output_styled_{mode}.png"

    print(f"Input: {input_path}")
    print(f"Mode: {mode}")
    print(f"Prompt: {load_style_prompt(mode)[:80]}...")
    print("Stylizing...")

    start = time.time()
    result = await stylize_drawing(input_path, output_path, mode=mode)
    elapsed = time.time() - start

    print(f"Output: {result}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"File size: {Path(result).stat().st_size:,} bytes")


if __name__ == "__main__":
    asyncio.run(main())
