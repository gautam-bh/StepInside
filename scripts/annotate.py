"""Click-to-annotate tool for new TIP scenes (matplotlib window).

    python scripts/annotate.py data/images/my_room.jpg
    -> data/annotations/my_room.json

Steps (instructions also show in the window title):
  1. Click the back wall's TOP-LEFT corner, then its BOTTOM-RIGHT corner.
  2. Click the vanishing point (where floor/ceiling/wall edges converge).
  3. Optional foreground objects: click points around the outline and press
     Enter to close it. Press Enter on an empty outline to finish.
Then check the result with scripts/show_annotation.py.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tip.annotations import Annotation, Foreground, save_annotation  # noqa: E402


def ask(ax, msg, n):
    ax.set_title(msg, fontsize=10)
    plt.draw()
    return plt.ginput(n, timeout=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--out")
    ap.add_argument("--focal-35mm", type=float, default=26.0)
    ap.add_argument("--camera-height", type=float, default=1.6)
    a = ap.parse_args()

    img = Image.open(a.image).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img)
    ax.axis("off")

    (x1, y1), (x2, y2) = ask(ax, "Back wall: click TOP-LEFT then BOTTOM-RIGHT corner", 2)
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="m", lw=2))

    (vx, vy), = ask(ax, "Click the vanishing point", 1)
    ax.plot([vx], [vy], "ro")

    fgs = []
    while True:
        pts = ask(ax, f"Foreground #{len(fgs) + 1}: click outline, Enter to close (Enter now = done)", -1)
        if len(pts) < 3:
            break
        xs, ys = zip(*(pts + [pts[0]]))
        ax.plot(xs, ys, "g-", lw=2)
        fgs.append(Foreground(name=f"object{len(fgs) + 1}", polygon=[[round(x, 1), round(y, 1)] for x, y in pts]))
    plt.close(fig)

    try:
        image_ref = str(Path(a.image).resolve().relative_to(ROOT))
    except ValueError:
        image_ref = str(Path(a.image).resolve())
    ann = Annotation(
        image=image_ref,
        vanishing_point=(round(vx, 1), round(vy, 1)),
        back_wall=(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
        focal_35mm=a.focal_35mm,
        camera_height=a.camera_height,
        foreground=fgs,
    )
    out = Path(a.out or ROOT / "data" / "annotations" / (Path(a.image).stem + ".json"))
    save_annotation(ann, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
