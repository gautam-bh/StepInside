"""Draw an annotation over its photo: back wall, vanishing-point rays to the
image borders (the room edges TIP assumes) and foreground polygons.

    python scripts/show_annotation.py data/annotations/hallway.json -o outputs/hallway_annotation.png
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tip import load_annotation  # noqa: E402


def border_hit(vp, corner, w, h):
    """Extend the ray from vp through corner until it reaches the image border."""
    (vx, vy), (cx, cy) = vp, corner
    dx, dy = cx - vx, cy - vy
    ts = []
    if dx:
        ts += [(0 - vx) / dx, (w - vx) / dx]
    if dy:
        ts += [(0 - vy) / dy, (h - vy) / dy]
    t = min(t for t in ts if t > 0)
    return vx + t * dx, vy + t * dy


def draw(ann_path, out_path, crop=None):
    ann = load_annotation(ann_path)
    im = Image.open(ann.image).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = ann.back_wall
    vp = ann.vanishing_point
    for c in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        d.line([c, border_hit(vp, c, w, h)], fill=(0, 255, 255), width=3)
    d.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=4)
    r = 8
    d.ellipse([vp[0] - r, vp[1] - r, vp[0] + r, vp[1] + r], outline=(255, 0, 0), width=3)
    for fg in ann.foreground:
        pts = [tuple(p) for p in fg.polygon]
        d.line(pts + [pts[0]], fill=(0, 255, 0), width=3)
    if crop:
        im = im.crop(crop)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("L", "T", "R", "B"))
    a = ap.parse_args()
    draw(a.annotation, a.out, a.crop)
