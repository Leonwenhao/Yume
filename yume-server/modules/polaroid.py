import base64
import binascii
import io
import logging
from pathlib import Path

from PIL import Image, ImageDraw, UnidentifiedImageError

logger = logging.getLogger("yume.polaroid")

# Polaroid frame dimensions
BORDER_SIDE = 40
BORDER_TOP = 40
BORDER_BOTTOM = 100
POLAROID_BG = (255, 255, 255)

# Strip layout
STRIP_BG = (245, 235, 220)  # warm off-white
STRIP_PADDING = 40


def _decode_image_payload(payload: str) -> bytes:
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("Missing 'capture' base64 field")

    encoded = payload.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("Invalid 'capture' image payload")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid 'capture' image payload") from exc

    if not image_bytes:
        raise ValueError("Invalid 'capture' image payload")

    return image_bytes


def create_polaroid(screenshot_b64: str, capture_number: int, world_id: str, assets_dir: str) -> str:
    """Take a base64 viewport screenshot and composite it into a polaroid frame.

    Returns the local file path of the saved polaroid PNG.
    """
    try:
        img_data = _decode_image_payload(screenshot_b64)
        screenshot = Image.open(io.BytesIO(img_data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid 'capture' image payload") from exc
    sw, sh = screenshot.size

    # Create polaroid frame
    frame_w = sw + BORDER_SIDE * 2
    frame_h = sh + BORDER_TOP + BORDER_BOTTOM
    frame = Image.new("RGB", (frame_w, frame_h), POLAROID_BG)
    frame.paste(screenshot, (BORDER_SIDE, BORDER_TOP))

    # Add subtle shadow line under the photo
    draw = ImageDraw.Draw(frame)
    shadow_y = BORDER_TOP + sh + 2
    draw.line(
        [(BORDER_SIDE, shadow_y), (BORDER_SIDE + sw, shadow_y)],
        fill=(220, 220, 220),
        width=1,
    )

    # Save
    out_dir = Path(assets_dir) / world_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"polaroid_{capture_number}.png"
    frame.save(str(out_path), "PNG")
    logger.info("Polaroid %d saved: %s (%dx%d)", capture_number, out_path, frame_w, frame_h)
    return str(out_path)


def create_strip(world_id: str, assets_dir: str) -> str:
    """Compose all 6 polaroids + original drawing into a single strip image.

    Layout: drawing on left (larger, framed), 6 polaroids in 3x2 grid on right.
    Returns the local file path of the saved strip PNG.
    """
    world_dir = Path(assets_dir) / world_id

    # Load original drawing
    drawing_path = world_dir / "drawing.png"
    if not drawing_path.exists():
        raise FileNotFoundError(f"Drawing not found: {drawing_path}")
    drawing = Image.open(str(drawing_path)).convert("RGB")

    # Load all 6 polaroids
    polaroids = []
    for i in range(1, 7):
        p_path = world_dir / f"polaroid_{i}.png"
        if not p_path.exists():
            raise FileNotFoundError(f"Polaroid {i} not found: {p_path}")
        polaroids.append(Image.open(str(p_path)).convert("RGB"))

    # Determine sizes
    # Normalize polaroid sizes (use the first one as reference)
    pol_w, pol_h = polaroids[0].size

    # Drawing frame: scale drawing to match height of 2 polaroids + gap
    grid_h = pol_h * 3 + STRIP_PADDING * 2  # 3 rows with 2 gaps
    draw_scale = grid_h / drawing.height
    draw_w = int(drawing.width * draw_scale)
    draw_h = grid_h
    drawing_resized = drawing.resize((draw_w, draw_h), Image.LANCZOS)

    # Drawing frame (polaroid-style)
    framed_draw_w = draw_w + BORDER_SIDE * 2
    framed_draw_h = draw_h + BORDER_TOP + BORDER_BOTTOM
    framed_drawing = Image.new("RGB", (framed_draw_w, framed_draw_h), POLAROID_BG)
    framed_drawing.paste(drawing_resized, (BORDER_SIDE, BORDER_TOP))

    # Strip dimensions
    grid_w = pol_w * 2 + STRIP_PADDING
    strip_w = STRIP_PADDING + framed_draw_w + STRIP_PADDING + grid_w + STRIP_PADDING
    strip_h = STRIP_PADDING * 2 + framed_draw_h

    strip = Image.new("RGB", (strip_w, strip_h), STRIP_BG)

    # Paste framed drawing on left
    strip.paste(framed_drawing, (STRIP_PADDING, STRIP_PADDING))

    # Paste polaroids in 3x2 grid on right
    grid_x_start = STRIP_PADDING + framed_draw_w + STRIP_PADDING
    grid_y_start = STRIP_PADDING + (framed_draw_h - (pol_h * 3 + STRIP_PADDING * 2)) // 2

    for idx, pol in enumerate(polaroids):
        col = idx % 2
        row = idx // 2
        x = grid_x_start + col * (pol_w + STRIP_PADDING)
        y = grid_y_start + row * (pol_h + STRIP_PADDING)
        strip.paste(pol, (x, y))

    out_path = world_dir / "strip.png"
    strip.save(str(out_path), "PNG")
    logger.info("Strip saved: %s (%dx%d)", out_path, strip_w, strip_h)
    return str(out_path)
