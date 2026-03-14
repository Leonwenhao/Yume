import logging
import os
from pathlib import Path

import fal_client
import httpx

from config import FAL_KEY

logger = logging.getLogger("yume.imagegen")

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_DEFAULT_PROMPTS = {
    "kids": (
        "A magical, whimsical fantasy landscape environment. Lush and vibrant colors, "
        "soft volumetric lighting, golden hour warmth. Rich environmental detail with "
        "fantastical elements. Wide-angle perspective with atmospheric depth. Studio "
        "Ghibli-inspired warmth and wonder. A single coherent immersive scene that feels "
        "like a place you could walk into. Detailed textures on every surface."
    ),
    "filmmaker": (
        "A cinematic environment with dramatic lighting and rich atmosphere. Detailed "
        "architectural and natural textures, strong depth of field, volumetric haze. "
        "Professional production design quality. Wide-angle perspective, single coherent "
        "scene, immersive and navigable. Photorealistic materials and surfaces with clear "
        "spatial structure."
    ),
}


def load_style_prompt(mode: str) -> str:
    """Load prompt from prompts/{mode}.txt if it exists, else use hardcoded default."""
    prompt_file = _PROMPTS_DIR / f"{mode}.txt"
    if prompt_file.exists():
        text = prompt_file.read_text().strip()
        if text:
            logger.info("Loaded prompt from %s", prompt_file)
            return text
    default = _DEFAULT_PROMPTS.get(mode, _DEFAULT_PROMPTS["kids"])
    logger.info("Using default prompt for mode '%s'", mode)
    return default


def _require_fal_key() -> str:
    fal_key = os.getenv("FAL_KEY") or FAL_KEY
    if not fal_key:
        logger.error("Fal integration unavailable: FAL_KEY is not configured")
        raise RuntimeError("Fal integration unavailable: FAL_KEY is not configured")
    os.environ["FAL_KEY"] = fal_key
    return fal_key


async def stylize_drawing(
    input_path: str,
    output_path: str,
    mode: str = "kids",
) -> str:
    """Upload image to Fal, run FLUX 2 Pro edit, download result.

    Returns the output_path on success.
    """
    prompt = load_style_prompt(mode)
    logger.info("Stylizing %s (mode=%s)", input_path, mode)
    logger.info("Prompt: %s", prompt[:100] + "...")

    _require_fal_key()

    try:
        logger.info("Uploading source image to Fal storage")
        url = fal_client.upload_file(input_path)
        logger.info("Uploaded to Fal storage: %s", url)
    except Exception as exc:
        logger.exception("Fal upload failed for %s: %s", input_path, exc)
        raise RuntimeError(f"Fal upload failed: {exc}") from exc

    try:
        logger.info("Submitting Fal FLUX 2 Pro edit request")
        result = fal_client.subscribe(
            "fal-ai/flux-2-pro/edit",
            arguments={
                "image_urls": [url],
                "prompt": prompt,
                "output_format": "png",
            },
        )
    except Exception as exc:
        logger.exception("Fal image generation failed for %s: %s", input_path, exc)
        raise RuntimeError(f"Fal image generation failed: {exc}") from exc

    try:
        output_url = result["images"][0]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.error("Fal response missing image URL: %s", result)
        raise RuntimeError("Fal image generation failed: missing output URL") from exc

    logger.info("Fal returned image: %s", output_url)

    try:
        logger.info("Downloading stylized image from Fal")
        async with httpx.AsyncClient() as client:
            resp = await client.get(output_url, timeout=30.0)
            resp.raise_for_status()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(resp.content)
            logger.info("Styled image saved to %s (%d bytes)", output_path, len(resp.content))
    except Exception as exc:
        logger.exception("Fal download failed for %s: %s", output_url, exc)
        raise RuntimeError(f"Fal download failed: {exc}") from exc

    return output_path
