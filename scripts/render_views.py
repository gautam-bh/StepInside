"""Render novel views of a TIP scene to images (for figures and evaluation).

    python scripts/render_views.py data/annotations/hallway.json
    -> outputs/hallway/  one image per viewpoint, holes vs. simple fill,
                         plus a contact sheet and a GIF sweep.

Camera offsets are in meters relative to the input camera:
x = right, y = up, z = forward.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tip import build_scene, load_annotation  # noqa: E402
from tip.render import render_view  # noqa: E402
from tip.textures import extract_textures, load_image, push_pull_fill  # noqa: E402

VIEWS = {
    "original": (0.0, 0.0, 0.0),
    "left_0.3m": (-0.3, 0.0, 0.0),
    "right_0.3m": (0.3, 0.0, 0.0),
    "forward_1.5m": (0.0, 0.0, 1.5),
    "up_0.3m": (0.0, 0.3, 0.0),
}


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(text), 20], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 255))
    return img


def to_img(rgb):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation")
    ap.add_argument("--scale", type=float, default=0.5, help="output resolution relative to input")
    ap.add_argument("--out", help="output dir (default outputs/<annotation name>)")
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args()

    ann = load_annotation(a.annotation)
    image = load_image(ann.image)
    h, w = image.shape[:2]
    scene = build_scene(ann, w, h)
    raw = extract_textures(scene, image, [f.polygon for f in ann.foreground])
    filled = {k: (push_pull_fill(v) if k in {"back", "floor", "ceiling", "left", "right"} else v) for k, v in raw.items()}

    name = Path(a.annotation).stem
    out = Path(a.out or ROOT / "outputs" / name)
    out.mkdir(parents=True, exist_ok=True)

    tiles_holes, tiles_fill = [], []
    for view, pos in VIEWS.items():
        rgb_h, hole = render_view(scene, raw, position=pos, scale=a.scale, background=(1.0, 0.0, 1.0))
        rgb_f, _ = render_view(scene, filled, position=pos, scale=a.scale)
        to_img(rgb_h).save(out / f"{view}_holes.jpg", quality=90)
        to_img(rgb_f).save(out / f"{view}_filled.jpg", quality=90)
        print(f"{view:<14} pos={pos}  hole pixels {hole.mean():5.1%}")
        tiles_holes.append(label(to_img(rgb_h), f"{view} (holes magenta)"))
        tiles_fill.append(label(to_img(rgb_f), f"{view} (push-pull fill)"))

    tw, th = tiles_holes[0].size
    sheet = Image.new("RGB", (tw * len(VIEWS), th * 2), "white")
    for i, (t1, t2) in enumerate(zip(tiles_holes, tiles_fill)):
        sheet.paste(t1, (i * tw, 0))
        sheet.paste(t2, (i * tw, th))
    sheet.save(out / "contact_sheet.jpg", quality=88)
    print("wrote", out / "contact_sheet.jpg")

    if not a.no_gif:
        frames = []
        n = 24
        for i in range(n):
            ph = 2 * np.pi * i / n
            pos = (0.35 * np.sin(ph), 0.08 * np.sin(2 * ph), 0.6 * (1 - np.cos(ph)) / 2)
            rgb, _ = render_view(scene, filled, position=pos, scale=a.scale * 0.6)
            frames.append(to_img(rgb))
        frames[0].save(out / "sweep.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)
        print("wrote", out / "sweep.gif")


if __name__ == "__main__":
    main()
