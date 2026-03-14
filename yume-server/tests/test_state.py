import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from state import _worlds, add_polaroid, create_world, get_world, set_assets, set_error, update_plushie_status, update_status


def setup_function():
    _worlds.clear()


def test_create_world():
    w = create_world("kids")
    assert w["world_id"].startswith("yume_")
    assert len(w["world_id"]) == 13  # "yume_" + 8 chars
    assert w["mode"] == "kids"
    assert w["status"] == "stylizing_drawing"
    assert w["stage"] == 1
    assert w["stage_label"] == "Turning your drawing into a world..."
    assert w["polaroids"] == []
    assert w["assets"] is None


def test_update_status():
    w = create_world("filmmaker")
    update_status(w["world_id"], "stylizing_drawing", 1, "Turning your drawing into a world...")
    world = get_world(w["world_id"])
    assert world["status"] == "stylizing_drawing"
    assert world["stage"] == 1
    assert world["stage_label"] == "Turning your drawing into a world..."


def test_update_status_through_all_stages():
    w = create_world("kids")
    wid = w["world_id"]

    update_status(wid, "stylizing_drawing", 1, "Stylizing...")
    assert get_world(wid)["status"] == "stylizing_drawing"

    update_status(wid, "generating_world", 2, "Generating world...")
    assert get_world(wid)["status"] == "generating_world"
    assert get_world(wid)["stage"] == 2

    update_status(wid, "complete", 2, "Done!")
    assert get_world(wid)["status"] == "complete"


def test_set_assets():
    w = create_world("kids")
    assets = {"splat_url": "/assets/test/world.spz", "panorama_url": "/assets/test/panorama.png"}
    set_assets(w["world_id"], assets)
    world = get_world(w["world_id"])
    assert world["assets"] == assets


def test_get_world_unknown():
    assert get_world("yume_nonexist") is None


def test_add_polaroid():
    w = create_world("kids")
    wid = w["world_id"]

    count = add_polaroid(wid, "/assets/test/polaroid_1.png")
    assert count == 1

    count = add_polaroid(wid, "/assets/test/polaroid_2.png")
    assert count == 2

    world = get_world(wid)
    assert len(world["polaroids"]) == 2
    assert world["polaroids"][0] == "/assets/test/polaroid_1.png"


def test_add_all_six_polaroids():
    w = create_world("kids")
    wid = w["world_id"]
    for i in range(1, 7):
        count = add_polaroid(wid, f"/assets/test/polaroid_{i}.png")
        assert count == i
    assert len(get_world(wid)["polaroids"]) == 6


def test_add_polaroid_does_not_duplicate_path():
    w = create_world("kids")
    wid = w["world_id"]
    add_polaroid(wid, "/assets/test/polaroid_1.png")
    count = add_polaroid(wid, "/assets/test/polaroid_1.png")
    assert count == 1
    assert get_world(wid)["polaroids"] == ["/assets/test/polaroid_1.png"]


def test_create_world_with_plushie():
    w = create_world("kids", has_plushie=True)
    assert w["has_plushie"] is True
    assert w["plushie_status"] == "pending"


def test_create_world_without_plushie_backward_compatible():
    w = create_world("kids")
    assert w["has_plushie"] is False
    assert w["plushie_status"] is None


def test_update_plushie_status():
    w = create_world("kids", has_plushie=True)
    wid = w["world_id"]
    update_plushie_status(wid, "generating")
    assert get_world(wid)["plushie_status"] == "generating"
    update_plushie_status(wid, "complete")
    assert get_world(wid)["plushie_status"] == "complete"


def test_update_plushie_status_with_label():
    w = create_world("kids", has_plushie=True)
    wid = w["world_id"]
    update_plushie_status(wid, "generating", "Making your plushie 3D model...")
    world = get_world(wid)
    assert world["plushie_status"] == "generating"
    assert world["stage_label"] == "Making your plushie 3D model..."


def test_set_error():
    w = create_world("kids")
    set_error(w["world_id"], "Marble API timeout")
    world = get_world(w["world_id"])
    assert world["status"] == "failed"
    assert world["stage_label"] == "World generation failed."
    assert world["error"] == "Marble API timeout"
