import io
import random

import pytest
from PIL import Image

from archivetrust.htr.benchmark.imaging import (
    crop_line,
    hamming,
    near_duplicate_pairs,
    open_page_rgb,
    polygon_bbox,
    prepare_supplied_line_image,
    probe_image_bytes,
)
from tests.htr.benchmark._fixtures import line_png, page_jpg


def test_near_duplicate_pairs_matches_brute_force():
    rng = random.Random(7)
    base = [rng.getrandbits(64) for _ in range(40)]
    hashes = [(f"k{i}", f"{v:016x}") for i, v in enumerate(base)]
    for i in range(20):  # plant near-duplicates at distances 0..6
        flipped = base[i]
        for bit in rng.sample(range(64), i % 7):
            flipped ^= 1 << bit
        hashes.append((f"n{i}", f"{flipped:016x}"))
    expected = sorted(
        (a, b, hamming(ha, hb)) if a < b else (b, a, hamming(ha, hb))
        for idx, (a, ha) in enumerate(hashes) for b, hb in hashes[idx + 1:] if hamming(ha, hb) <= 4
    )
    assert sorted(near_duplicate_pairs(hashes, max_distance=4)) == expected


def test_probe_reports_basic_properties_and_blank():
    probe = probe_image_bytes(line_png("abc"))
    assert probe.ok and probe.format == "PNG" and (probe.width, probe.height) == (240, 40) and not probe.blank
    blank = io.BytesIO()
    Image.new("L", (100, 30), 250).save(blank, format="PNG")
    assert probe_image_bytes(blank.getvalue()).blank
    assert not probe_image_bytes(b"not an image").ok


def test_polygon_bbox_rounds_and_clamps():
    assert polygon_bbox([(-3.2, 5.4), (120.6, 30.5)], width=100, height=100) == (0, 5, 100, 30)
    assert polygon_bbox([(200, 5), (300, 30)], width=100, height=100) is None


def test_crop_is_deterministic_and_mask_whitens_outside():
    page = open_page_rgb(page_jpg())
    a = crop_line(page, (40, 90, 720, 150))
    assert a == crop_line(page, (40, 90, 720, 150))
    masked = crop_line(page, (40, 90, 720, 150), polygon=[(60, 95), (700, 95), (700, 145), (60, 145)], mask_polygon=True)
    image = Image.open(io.BytesIO(masked))
    assert image.getpixel((2, 2)) == (255, 255, 255)
    assert image.size == (680, 60)


def test_prepare_supplied_line_image_passthrough_convert_refuse():
    png = line_png()
    assert prepare_supplied_line_image(png, probe_image_bytes(png)) == (png, ".png", None)
    rgba = io.BytesIO()
    Image.new("RGBA", (50, 20), (0, 0, 0, 255)).save(rgba, format="PNG")
    data, ext, conversion = prepare_supplied_line_image(rgba.getvalue(), probe_image_bytes(rgba.getvalue()))
    assert ext == ".png" and conversion == "PNG/RGBA->PNG/RGB" and Image.open(io.BytesIO(data)).mode == "RGB"
    transparent = io.BytesIO()
    Image.new("RGBA", (50, 20), (0, 0, 0, 0)).save(transparent, format="PNG")
    with pytest.raises(ValueError, match="alpha"):
        prepare_supplied_line_image(transparent.getvalue(), probe_image_bytes(transparent.getvalue()))
