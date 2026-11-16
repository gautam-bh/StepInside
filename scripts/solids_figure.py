"""Figure of the automatic solid detector's stages on one photo.

    python scripts/solids_figure.py data/annotations/hallway_auto.json
    -> outputs/<name>/solids_auto.jpg
       (empty-room background · outliers + proposals · traced solids · a novel view)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tip import build_scene, load_annotation  # noqa: E402
from tip.auto_annotate import downscale  # noqa: E402
from tip.render import render_view  # noqa: E402
from tip.revolve import outlier_proposals, room_background  # noqa: E402
from tip.textures import extract_textures, load_image  # noqa: E402


def lab_to_rgb(lab):
    import cv2

    return cv2.cvtColor(np.nan_to_num(lab).astype(np.uint8), cv2.COLOR_LAB2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation")
    ap.add_argument("--out")
    ap.add_argument("--view", type=float, nargs=3, default=(-0.8, -0.3, 1.6), help="camera offset for panel 4")
    ap.add_argument("--yaw", type=float, default=25.0)
    a = ap.parse_args()

    ann = load_annotation(a.annotation)
    image = load_image(ann.image)
    h, w = image.shape[:2]
    solids = ann.solids
    ann.solids = []
    room = build_scene(ann, w, h)
    small, s = downscale(image, 800)
    bg, bg_alt, pid = room_background(small, room, s)
    outlier, boxes = outlier_proposals(small, bg, bg_alt, pid)

    to8 = lambda x: Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))  # noqa: E731
    p1 = Image.fromarray(lab_to_rgb(bg))
    p2 = to8(small * np.where(outlier[..., None], 0.35, 1.0) + outlier[..., None] * np.array([0.6, 0.0, 0.0]))
    d2 = ImageDraw.Draw(p2)
    for x, y, bw, bh in boxes:
        d2.rectangle([x, y, x + bw, y + bh], outline=(0, 220, 0))
    p3 = to8(small)
    d3 = ImageDraw.Draw(p3)
    for sol in solids:
        prof = np.asarray(sol.profile) * s
        d3.line([(xl, y) for y, xl, _ in prof], fill=(0, 255, 0), width=2)
        d3.line([(xr, y) for y, _, xr in prof], fill=(0, 255, 0), width=2)
        c = (prof[:, 1] + prof[:, 2]).mean() / 2
        d3.line([(c, prof[0, 0]), (c, prof[-1, 0])], fill=(0, 120, 255), width=1)
        d3.text((prof[:, 2].max() + 4, prof[0, 0]), f"{sol.name}\n{sol.mount}", fill=(0, 255, 0))

    ann.solids = solids
    scene = build_scene(ann, w, h)
    tex = extract_textures(scene, image, [f.polygon for f in ann.foreground])
    rgb, _ = render_view(scene, tex, position=tuple(a.view), yaw=a.yaw, scale=small.shape[0] / h,
                         background=(1.0, 0.0, 1.0))
    p4 = to8(rgb).resize(p1.size)

    titles = ["empty-room background", "outliers + proposals", "traced solids (GrabCut + symmetry)",
              "novel view (holes magenta)"]
    tw, th = p1.size
    sheet = Image.new("RGB", (tw * 4, th + 22), "white")
    d = ImageDraw.Draw(sheet)
    for i, (p, t) in enumerate(zip((p1, p2, p3, p4), titles)):
        sheet.paste(p, (i * tw, 22))
        d.text((i * tw + 6, 5), t, fill=(0, 0, 0))
    out = Path(a.out or ROOT / "outputs" / Path(a.annotation).stem.replace("_auto", "") / "solids_auto.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    print("wrote", out)


if __name__ == "__main__":
    main()
