"""Collect every built scene into site/examples.json (and thumbnails) for the gallery.

    python scripts/build_site.py      # run automatically by scripts/serve.py

An example is any folder in scenes/ with a scene.json. Its title and blurb come
from EXAMPLES below when listed there; anything else is discovered from disk:
the annotation (data/annotations/<scene>.json), the annotation drawing, the
novel views rendered by scripts/render_views.py, and extra figures.
"""

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"

# Shown in this order; scenes not listed here follow, alphabetically.
EXAMPLES = {
    "st_augustine": dict(
        title="Saint Augustine in His Study",
        medium="Vittore Carpaccio, c. 1502 · tempera on canvas",
        kind="Painting",
        annotated="Automatic",
        outputs="st_augustine",
        blurb=(
            "A Renaissance painting turned into a room you can step into, with no hand input. "
            "The vanishing point lands on the saint's desk. The back wall is the far wall around the "
            "altar niche, found by checking that each room surface keeps one colour along rays from "
            "the vanishing point. The saint, his chair, the lectern and the dog stand up as cutouts on the floor, "
            "so moving the camera shows him in front of the cabinet and back wall, not painted on them."
        ),
    ),
    "hallway_auto": dict(
        title="Hallway",
        medium="iPhone photograph",
        kind="Photograph",
        annotated="Automatic",
        outputs="hallway_auto",
        annotation_image="outputs/hallway/auto_annotation.jpg",
        figures=[
            ("outputs/hallway/solids_auto.jpg",
             "Round objects: the empty-room background, what differs from it, the traced silhouettes, "
             "and a moved camera where both extinguishers stand off the wall."),
            ("outputs/hallway/wall_things_auto.jpg",
             "Wall boxes and ceiling pipes: the sign and strobe become thin boxes, six pipes become cylinders."),
        ],
        blurb=(
            "The whole pipeline, untouched by hand. Vanishing point, back wall, the doorway and three "
            "corridor stretches behind it, both fire extinguishers as 3D solids, the sign and strobe "
            "as boxes on the wall, and six pipes hanging under the ceiling."
        ),
    ),
    "hallway": dict(
        title="Hallway, by hand",
        medium="iPhone photograph",
        kind="Photograph",
        annotated="By hand",
        outputs="hallway",
        figures=[
            ("outputs/hallway/faces_unfolded.jpg",
             "Every face of the room, perspective-corrected and unfolded at true scale. Holes (no photo data) in magenta."),
            ("outputs/hallway/trash_bin_object.jpg",
             "The trash bin 27 m down the corridor, added in the editor, standing out from the wall as a box."),
        ],
        blurb=(
            "The reference annotation, made in the editor: seven boxes from the camera through the "
            "doorway to the corridor's end wall, about 49 m away. The extinguishers are flat cutouts "
            "here, which is how classic Tour Into the Picture handles foreground."
        ),
    ),
    "furnished_room": dict(
        title="Furnished room",
        medium="Synthetic test room, known geometry",
        kind="Synthetic",
        annotated="Automatic",
        outputs="furnished_room",
        figures=[
            ("outputs/furnished_room/objects_demo.jpg",
             "Furniture square to the room becomes solid boxes; the poster and the door correctly do not."),
        ],
        blurb=(
            "A rendered room whose true geometry is known, used to test the detectors. Both pieces of "
            "furniture are found as 3D boxes; moving the camera shows their sides and the floor they hid."
        ),
    ),
}

VIEWS = [  # (file stem, label) as written by scripts/render_views.py
    ("original", "Original camera"),
    ("left_0.3m", "0.3 m left"),
    ("right_0.3m", "0.3 m right"),
    ("up_0.3m", "0.3 m up"),
    ("forward_1.5m", "1.5 m forward"),
    ("through_door_9m", "9 m forward"),
]


def rel(p: Path) -> str:
    return "../" + p.relative_to(ROOT).as_posix()


def stats(scene: dict, ann: dict | None) -> list:
    kinds = [p["kind"] for p in scene["planes"]]
    boxes = scene["boxes"]
    out = [("Room boxes", len(boxes))]
    if ann is not None:
        for key, label in (("solids", "Round solids"), ("reliefs", "Wall boxes"), ("tubes", "Pipes"),
                           ("objects", "Furniture boxes"), ("foreground", "Cutouts")):
            if ann.get(key):
                out.append((label, len(ann[key])))
    elif kinds.count("tube"):
        out.append(("Pipes", kinds.count("tube")))
    depth = -boxes[-1]["z"][1]
    out.append(("Depth", f"{depth:.1f} m"))
    b0 = boxes[0]
    out.append(("Width × height", f"{b0['x'][1] - b0['x'][0]:.1f} × {b0['y'][1] - b0['y'][0]:.1f} m"))
    if ann is not None and ann.get("focal_35mm"):
        out.append(("Lens", f"{ann['focal_35mm']:.0f} mm equiv."))
    return [[k, str(v)] for k, v in out]


def thumbnail(src: Path, name: str) -> str:
    dst = SITE / "thumbs" / f"{name}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        im = Image.open(src).convert("RGB")
        im.thumbnail((480, 480))
        im.save(dst, quality=84)
    return "thumbs/" + dst.name


def main():
    scenes = sorted(p.parent.name for p in (ROOT / "scenes").glob("*/scene.json"))
    order = [s for s in EXAMPLES if s in scenes] + [s for s in scenes if s not in EXAMPLES]
    examples = []
    for name in order:
        meta = EXAMPLES.get(name, {})
        sdir = ROOT / "scenes" / name
        scene = json.loads((sdir / "scene.json").read_text())
        ann_path = ROOT / "data" / "annotations" / f"{name}.json"
        ann = json.loads(ann_path.read_text()) if ann_path.exists() else None
        odir = ROOT / "outputs" / meta.get("outputs", name)

        annotation = meta.get("annotation_image")
        if annotation is None:
            for cand in ("annotation.jpg", "auto_annotation.jpg"):
                if (odir / cand).exists():
                    annotation = (odir / cand).relative_to(ROOT).as_posix()
                    break
        views = []
        for stem, label in VIEWS:
            if stem == "through_door_9m" and len(scene["boxes"]) < 2:
                continue  # one room, nothing to walk through
            h, f = odir / f"{stem}_holes.jpg", odir / f"{stem}_filled.jpg"
            if h.exists() and f.exists():
                views.append({"label": label, "holes": rel(h), "filled": rel(f)})
        figures = [{"src": "../" + src, "caption": cap} for src, cap in meta.get("figures", [])
                   if (ROOT / src).exists()]
        sweep = odir / "sweep.gif"

        examples.append({
            "id": name,
            "title": meta.get("title", name.replace("_", " ").capitalize()),
            "medium": meta.get("medium", ""),
            "kind": meta.get("kind", "Scene"),
            "annotated": meta.get("annotated", "Automatic" if name.endswith("_auto") else ""),
            "blurb": meta.get("blurb", ""),
            "width": scene["width"],
            "height": scene["height"],
            "input": rel(sdir / "input.jpg"),
            "thumb": thumbnail(sdir / "input.jpg", name),
            "annotation": "../" + annotation if annotation else None,
            "views": views,
            "sweep": rel(sweep) if sweep.exists() else None,
            "figures": figures,
            "stats": stats(scene, ann),
        })
    SITE.mkdir(exist_ok=True)
    (SITE / "examples.json").write_text(json.dumps(examples, indent=1))
    print(f"wrote site/examples.json ({len(examples)} examples: {', '.join(order)})")


if __name__ == "__main__":
    sys.exit(main())
