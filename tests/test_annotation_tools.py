"""Tests for automatic annotation and the interactive editor.

Run with `python tests/test_annotation_tools.py` (or `pytest`).

Automatic detection is checked against synthetic rooms with known geometry,
rendered with our own TIP renderer, and against the hand-annotated hallway.
The editor is driven with simulated clicks and key presses (no window).
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tip import build_scene, load_annotation  # noqa: E402
from tip.annotations import Annotation  # noqa: E402
from tip.auto_annotate import auto_annotate_array, find_vanishing_point, to_gray  # noqa: E402
from tip.editor import AnnotationEditor  # noqa: E402
from tip.render import render_view  # noqa: E402
from tip.textures import load_image  # noqa: E402


# --------------------------------------------------------------------------- synthetic rooms
def synthetic_room(vp, back_wall, size=(600, 800), seed=0):
    """Render a box room with plain-coloured, slightly noisy surfaces and a
    tiled floor, with known VP and back wall.

    Noise is kept fine and faint: coarse texture noise gets magnified near
    the camera into blocky bilinear patterns that real photos don't have."""
    w, h = size
    ann = Annotation(image="synthetic", vanishing_point=vp, back_wall=back_wall)
    scene = build_scene(ann, w, h)
    rng = np.random.default_rng(seed)
    shade = {"floor": 0.55, "ceiling": 0.22, "left": 0.85, "right": 0.75, "back": 0.66}
    tex = {}
    for p in scene.planes:
        kind = p.name.split("_", 1)[1]
        base = shade[kind]
        n = 1024
        t = np.full((n, n, 3), base, np.float32)
        t += rng.normal(0, 0.005, t.shape).astype(np.float32)
        if kind == "floor":  # tile grout lines
            t[::128, :, :] -= 0.12
            t[:, ::128, :] -= 0.12
        tex[p.name] = np.concatenate([np.clip(t, 0, 1), np.ones((n, n, 1), np.float32)], axis=-1)
    # Supersample 3x and box-downsample: anti-aliased edges like a real
    # camera's (a one-ray-per-pixel render has staircase edges whose gradient
    # directions are biased by several degrees).
    ss = 3
    rgb, _ = render_view(scene, tex, scale=ss)
    rgb = rgb.reshape(h, ss, w, ss, 3).mean(axis=(1, 3))
    return rgb.astype(np.float32)


CASES = [
    # (vp, back_wall)
    ((300.0, 420.0), (200.0, 300.0, 420.0, 560.0)),  # centred
    ((260.0, 380.0), (150.0, 240.0, 380.0, 520.0)),  # off-centre, up-left
    ((340.0, 450.0), (250.0, 330.0, 470.0, 600.0)),  # off-centre, down-right
]


def test_vanishing_point_on_synthetic_rooms():
    for vp, bw in CASES:
        img = synthetic_room(vp, bw)
        est = find_vanishing_point(to_gray(img))
        err = np.hypot(est[0] - vp[0], est[1] - vp[1])
        assert err < 4.0, f"VP {est} vs truth {vp} (err {err:.1f}px)"


def test_back_wall_on_synthetic_rooms():
    for vp, bw in CASES:
        img = synthetic_room(vp, bw)
        r = auto_annotate_array(img)
        vp_err = np.hypot(r.vanishing_point[0] - vp[0], r.vanishing_point[1] - vp[1])
        bw_err = np.abs(np.array(r.back_wall) - np.array(bw)).max()
        assert vp_err < 6.0, f"VP {r.vanishing_point} vs {vp}"
        assert bw_err < 10.0, f"back wall {r.back_wall} vs {bw} (max err {bw_err:.1f}px)"


def test_auto_annotation_builds_a_valid_scene():
    vp, bw = CASES[1]
    img = synthetic_room(vp, bw)
    r = auto_annotate_array(img)
    ann = Annotation(image="synthetic", vanishing_point=r.vanishing_point, back_wall=tuple(r.back_wall))
    scene = build_scene(ann, img.shape[1], img.shape[0])
    assert scene.depth > scene.z_near > 0


def test_hallway_matches_hand_annotation():
    """The real photo: within lens-distortion slack of the hand annotation."""
    ann = load_annotation(ROOT / "data" / "annotations" / "hallway.json")
    r = auto_annotate_array(load_image(ann.image))
    bw_err = np.abs(np.array(r.back_wall) - np.array(ann.back_wall)).max()
    vp_err = np.hypot(r.vanishing_point[0] - ann.vanishing_point[0], r.vanishing_point[1] - ann.vanishing_point[1])
    assert bw_err < 25, f"back wall {r.back_wall} vs hand {ann.back_wall}"
    # Ceiling lines meet near y~610 and floor lines near y~660 in this photo
    # (lens distortion), so any VP in between is defensible.
    assert vp_err < 40, f"VP {r.vanishing_point} vs hand {ann.vanishing_point}"


# --------------------------------------------------------------------------- editor
def _editor(tmp=None):
    ann = Annotation(image="x.jpg", vanishing_point=(300, 420), back_wall=(200, 300, 420, 560))
    return AnnotationEditor(ann, (600, 800), out_path=tmp)


def test_drag_vanishing_point_and_corner():
    ed = _editor()
    ed.press(302, 418)  # grab the VP
    ed.motion(310, 430)
    ed.release()
    assert ed.state.vanishing_point == [310, 430]
    ed.press(199, 301)  # back wall top-left corner
    ed.motion(180, 290)
    ed.release()
    assert ed.state.back_wall[:2] == [180, 290]
    # A click far from any handle doesn't move anything.
    ed.press(10, 10)
    ed.motion(50, 50)
    ed.release()
    assert ed.state.vanishing_point == [310, 430]


def test_dragging_past_opposite_corner_keeps_rect_ordered():
    ed = _editor()
    ed.press(420, 560)  # bottom-right corner
    ed.motion(150, 250)  # past the top-left
    ed.release()
    x1, y1, x2, y2 = ed.state.back_wall
    assert x1 < x2 and y1 < y2


def test_add_boxes_and_validation():
    ed = _editor()
    ed.key("b")
    ed.press(240, 360)
    ed.press(380, 500)
    assert ed.state.segments == [[240, 360, 380, 500]]
    # Must contain the VP.
    ed.key("b")
    ed.press(10, 10)
    ed.press(100, 100)
    assert len(ed.state.segments) == 1 and "vanishing point" in ed.message
    # Must be farther than the previous box (bottom edge higher).
    ed.key("b")
    ed.press(250, 380)
    ed.press(360, 520)
    assert len(ed.state.segments) == 1 and "farther" in ed.message
    ed.key("b")
    ed.press(270, 390)
    ed.press(340, 470)
    assert len(ed.state.segments) == 2
    # Box corners are draggable handles too.
    assert ed.handle_at(340, 470) == ("seg", 1, 1)


def test_foreground_outline_and_undo():
    ed = _editor()
    ed.key("f")
    for p in [(450, 600), (500, 600), (500, 700)]:
        ed.press(*p)
    ed.key("backspace")  # remove last point
    assert len(ed.pending) == 2
    ed.press(480, 720)
    ed.press(450, 700)
    ed.key("enter")
    assert len(ed.state.foreground) == 1 and len(ed.state.foreground[0]["polygon"]) == 4
    assert ed.mode == "edit"
    # Too few points is refused.
    ed.key("f")
    ed.press(1, 1)
    ed.key("enter")
    assert len(ed.state.foreground) == 1
    ed.key("backspace")
    assert ed.state.foreground == []


def test_save_roundtrip_builds_scene():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "a.json"
        ed = _editor(out)
        ed.key("b")
        ed.press(240, 360)
        ed.press(380, 500)
        ed.key("f")
        for p in [(450, 600), (500, 600), (500, 700), (450, 700)]:
            ed.press(*p)
        ed.key("enter")
        ed.key("s")
        assert out.exists(), ed.message
        data = json.loads(out.read_text())
        assert data["segments"] == [[240, 360, 380, 500]]
        assert data["foreground"][0]["polygon"][0] == [450, 600]
        ann = load_annotation(out)
        scene = build_scene(ann, 600, 800)
        assert len(scene.boxes) == 2
        assert any(p.kind == "foreground" for p in scene.planes)


def test_invalid_annotation_is_not_saved():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "a.json"
        ed = _editor(out)
        ed.press(300, 420)  # drag the VP outside the back wall
        ed.motion(100, 100)
        ed.release()
        ed.key("s")
        assert not out.exists()
        assert "not saved" in ed.message


def test_editor_draws_with_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ed = _editor()
    fig, ax = plt.subplots()
    ed.attach(fig, ax, np.zeros((800, 600, 3)))
    ed.key("b")
    ed.press(240, 360)
    ed.press(380, 500)  # triggers redraw with a box
    fig.canvas.draw()
    assert "[edit]" in ax.get_title()
    plt.close(fig)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
