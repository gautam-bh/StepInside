"""Sanity checks for the TIP pipeline. Run with `python tests/test_tip.py`
(or `pytest` if installed)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tip import build_scene, load_annotation, project  # noqa: E402
from tip.render import render_view  # noqa: E402
from tip.textures import extract_textures, load_image, polygon_mask  # noqa: E402

ANN = ROOT / "data" / "annotations" / "hallway.json"


def _setup(scale_img=0.25):
    ann = load_annotation(ANN)
    image = load_image(ann.image)
    h, w = image.shape[:2]
    scene = build_scene(ann, w, h)
    tex = extract_textures(scene, image, [f.polygon for f in ann.foreground], max_res=768)
    return ann, image, scene, tex


def test_back_wall_projects_to_annotation():
    ann = load_annotation(ANN)
    scene = build_scene(ann, 1200, 1600)
    by = {p.name: p for p in scene.planes}
    x1, y1, x2, y2 = ann.back_wall
    # Back wall of box 0 is split into bands around the doorway; its outer
    # corners still land on the annotated rectangle.
    u, v = project(by["s0_back_top"].corners[:2], scene.focal, scene.principal)
    np.testing.assert_allclose(u, [x1, x2], atol=1e-6)
    np.testing.assert_allclose(v, [y1, y1], atol=1e-6)
    # The doorway reaches the floor, so there's no bottom band; the floor's far
    # edge is the back wall's bottom edge instead.
    u, v = project(by["s0_floor"].corners[:2], scene.focal, scene.principal)
    np.testing.assert_allclose(u, [x1, x2], atol=1e-6)
    np.testing.assert_allclose(v, [y2, y2], atol=1e-6)
    # Every further box's far end projects onto its annotated rectangle.
    for k, r in enumerate(ann.segments, start=1):
        b = scene.boxes[k]
        c = np.array([[b.xl, b.yc, b.z1], [b.xr, b.yf, b.z1]])
        u, v = project(c, scene.focal, scene.principal)
        np.testing.assert_allclose([u[0], v[0], u[1], v[1]], r, atol=1e-6)
    assert abs(scene.yf + ann.camera_height) < 1e-9


def test_side_planes_meet_back_wall_edges():
    ann = load_annotation(ANN)
    scene = build_scene(ann, 1200, 1600)
    by = {p.name: p for p in scene.planes}
    # The far edge of the floor is the back wall's bottom edge.
    # The side walls' bottom edges sit on the floor's side edges.
    np.testing.assert_allclose(by["s0_left"].corners[3], by["s0_floor"].corners[3])
    np.testing.assert_allclose(by["s0_right"].corners[2], by["s0_floor"].corners[2])
    # The back-wall bands meet the box's top corners.
    np.testing.assert_allclose(by["s0_back_top"].corners[0], by["s0_left"].corners[1])
    # Consecutive boxes share a depth.
    for a, b in zip(scene.boxes, scene.boxes[1:]):
        assert abs(a.z1 - b.z0) < 1e-9


def test_original_view_reproduces_photo():
    ann, image, scene, tex = _setup()
    scale = 0.25
    rgb, hole = render_view(scene, tex, position=(0, 0, 0), scale=scale)
    h, w = rgb.shape[:2]
    from PIL import Image

    ref = np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    # Ignore a thin band around the foreground cutout (mask dilation).
    fg = polygon_mask([np.asarray(f.polygon) * scale for f in ann.foreground], w, h, dilate=3)
    keep = ~fg & ~hole
    err = np.abs(rgb - ref).mean(axis=-1)[keep]
    assert hole.mean() < 0.01, f"too many holes in original view: {hole.mean():.3f}"
    assert err.mean() < 0.03, f"mean abs error {err.mean():.4f}"


def test_sideways_move_reveals_holes_and_parallax():
    _, _, scene, tex = _setup()
    _, hole0 = render_view(scene, tex, position=(0, 0, 0), scale=0.2)
    _, hole1 = render_view(scene, tex, position=(0.3, 0, 0), scale=0.2)
    assert hole1.mean() > hole0.mean() + 0.02


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
