"""Annotate a photo for TIP: automatic detection + interactive editor.

    python scripts/annotate.py data/images/my_room.jpg
        -> auto-detects the vanishing point and back wall, opens the editor,
           and saves data/annotations/my_room.json

    python scripts/annotate.py data/images/my_room.jpg --auto-only
        -> just run detection and save (no window)

    python scripts/annotate.py data/annotations/hallway.json
        -> open an existing annotation in the editor

Editor controls: drag the red VP / magenta corners · b add a box behind the
current last one (click 2 corners) · o add a room-aligned object (click its
front face's 2 corners, drag the orange back handle) · f outline a foreground
object (Enter to finish) · Backspace undo · a re-run detection · s save ·
q save and quit.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tip.annotations import Annotation, RoomObject, load_annotation, save_annotation  # noqa: E402
from tip.auto_annotate import auto_annotate_array  # noqa: E402
from tip.textures import load_image  # noqa: E402


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="an image, or an existing annotation .json")
    ap.add_argument("-o", "--out", help="annotation path (default data/annotations/<image name>.json)")
    ap.add_argument("--auto-only", action="store_true", help="run detection and save without opening a window")
    ap.add_argument("--no-solids", action="store_true", help="skip the solid (vase, extinguisher) detector")
    ap.add_argument("--focal-35mm", type=float, default=26.0)
    ap.add_argument("--camera-height", type=float, default=1.6)
    a = ap.parse_args()

    src = Path(a.input)
    if src.suffix.lower() == ".json":
        ann = load_annotation(src)
        image = load_image(ann.image)
        out = Path(a.out or src)
    else:
        image = load_image(src)
        print("auto-detecting vanishing point and back wall ...")
        r = auto_annotate_array(image)
        print(f"  vanishing point {r.vanishing_point}, back wall {r.back_wall}")
        for i, seg in enumerate(r.segments or [], start=1):
            print(f"  deeper box {i}: {seg}")
        for i, (front, back_y) in enumerate(r.objects or [], start=1):
            print(f"  object {i}: front {front}, back edge at y={back_y}")
        ann = Annotation(
            image=rel(src),
            vanishing_point=r.vanishing_point,
            back_wall=tuple(r.back_wall),
            focal_35mm=a.focal_35mm,
            camera_height=a.camera_height,
            segments=r.segments or [],
            objects=[RoomObject(f"object{i + 1}", list(f), b) for i, (f, b) in enumerate(r.objects or [])],
        )
        out = Path(a.out or ROOT / "data" / "annotations" / (src.stem + ".json"))
        if not a.no_solids:
            from tip.auto_annotate import add_solids

            print("detecting solids (surfaces of revolution) ...")
            add_solids(ann, image)
            for sol in ann.solids:
                print(f"  {sol.name}: {sol.mount}-mounted, rows {sol.profile[0][0]:.0f}-{sol.profile[-1][0]:.0f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    if a.auto_only:
        save_annotation(ann, out)
        print("wrote", out)
        return

    import matplotlib.pyplot as plt

    from tip.editor import AnnotationEditor

    # matplotlib binds s/q/f/Backspace etc. to its own toolbar actions.
    for k in list(plt.rcParams):
        if k.startswith("keymap."):
            plt.rcParams[k] = []

    def auto():
        r = auto_annotate_array(image)
        return r.vanishing_point, r.back_wall, r.segments or [], r.objects or []

    h, w = image.shape[:2]
    editor = AnnotationEditor(ann, (w, h), out_path=out, auto_fn=auto)
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93)
    editor.attach(fig, ax, image)
    plt.show()
    print(editor.message or f"done ({out})")


if __name__ == "__main__":
    main()
