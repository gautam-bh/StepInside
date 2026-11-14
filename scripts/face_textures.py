"""Figure: the room's faces rectified (perspective removed) and unfolded.

    python scripts/face_textures.py data/annotations/hallway.json
    -> outputs/hallway/faces_unfolded.jpg   input photo + unfolded box, holes in magenta
       outputs/hallway/faces/<face>.png     each rectified face on its own

Every face of box 0 (the room around the camera) is drawn at the same
pixels-per-meter scale and laid out as an unfolded box:

             ceiling
    left  |  back   |  right
              floor

Each face touches the back wall along their shared edge. Magenta marks
texels with no photo data: outside the frame, or behind the foreground cutout.
Here the floor tiles become a regular grid and the missing regions become
clean strips along each face's near edge. That's why the proposal completes
holes in this rectified space rather than in the photo.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tip import build_scene, load_annotation  # noqa: E402
from tip.geometry import Plane  # noqa: E402
from tip.textures import box_texture, extract_textures, load_image, polygon_mask  # noqa: E402

HOLE = np.array([1.0, 0.0, 1.0])


def flatten(rgba: np.ndarray) -> np.ndarray:
    """RGBA texture -> RGB with holes painted magenta."""
    a = rgba[..., 3:4] >= 0.5
    return np.where(a, rgba[..., :3], HOLE)


def resize(rgb: np.ndarray, w: int, h: int) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).resize((w, h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation")
    ap.add_argument("--ppm", type=float, default=160, help="pixels per meter in the unfolded figure")
    ap.add_argument("--out", help="output dir (default outputs/<annotation name>)")
    a = ap.parse_args()

    ann = load_annotation(a.annotation)
    image = load_image(ann.image)
    H, W = image.shape[:2]
    scene = build_scene(ann, W, H)
    polys = [f.polygon for f in ann.foreground]
    tex = extract_textures(scene, image, polys, max_res=1536)
    b = scene.boxes[0]

    # The back wall as one full rectangle (the doorway included), sampled
    # straight from the photo, the way classic single-box TIP sees it.
    back = Plane("back_full", np.array([b.xl, b.yc, b.z1]), np.array([b.xr - b.xl, 0, 0]),
                 np.array([0, b.yf - b.yc, 0]))
    fg_mask = polygon_mask(polys, W, H, dilate=3)
    faces = {
        "back": box_texture(back, scene, image, fg_mask, max_res=1536),
        "floor": tex["s0_floor"],
        "ceiling": tex["s0_ceiling"],
        "left": tex["s0_left"],
        "right": tex["s0_right"],
    }

    name = Path(a.annotation).stem
    out = Path(a.out or ROOT / "outputs" / name)
    (out / "faces").mkdir(parents=True, exist_ok=True)
    for k, t in faces.items():
        resize(flatten(t), t.shape[1], t.shape[0]).save(out / "faces" / f"{k}.png")

    # Physical sizes (meters) -> pixels.
    ppm = a.ppm
    bw, bh, d = (b.xr - b.xl) * ppm, (b.yc - b.yf) * ppm, (b.z1 - b.z0) * ppm
    bw, bh, d = int(round(bw)), int(round(bh)), int(round(d))
    pad = 24
    cw, ch = d + bw + d + 2 * pad, d + bh + d + 2 * pad
    canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
    ox, oy = pad + d, pad + d  # back wall top-left
    canvas.paste(resize(flatten(faces["back"]), bw, bh), (ox, oy))
    canvas.paste(resize(flatten(faces["floor"]), bw, d), (ox, oy + bh))  # far edge on top
    canvas.paste(resize(flatten(faces["ceiling"]), bw, d), (ox, oy - d))  # far edge at bottom
    canvas.paste(resize(flatten(faces["left"]), d, bh), (ox - d, oy))  # far edge on the right
    canvas.paste(resize(flatten(faces["right"]), d, bh), (ox + bw, oy))  # far edge on the left
    dr = ImageDraw.Draw(canvas)
    for (x0, y0, x1, y1) in [(ox, oy, ox + bw, oy + bh), (ox, oy + bh, ox + bw, oy + bh + d),
                             (ox, oy - d, ox + bw, oy), (ox - d, oy, ox, oy + bh), (ox + bw, oy, ox + bw + d, oy + bh)]:
        dr.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(40, 40, 40), width=2)
    for text, (x, y) in {
        "back wall": (ox + 8, oy + 8), "floor": (ox + 8, oy + bh + 8), "ceiling": (ox + 8, oy - d + 8),
        "left wall": (ox - d + 8, oy + 8), "right wall": (ox + bw + 8, oy + 8),
    }.items():
        dr.rectangle([x - 3, y - 2, x + 7 * len(text) + 3, y + 13], fill=(0, 0, 0))
        dr.text((x, y), text, fill=(255, 255, 255))
    meters = f"1 m = {ppm:.0f} px   (box {b.xr - b.xl:.1f} x {b.yc - b.yf:.1f} x {b.z1 - b.z0:.1f} m)"
    dr.text((pad, ch - pad + 4), meters, fill=(80, 80, 80))
    canvas.save(out / "faces_unfolded_only.png")

    # Input photo with the box edges, at the unfolded figure's height.
    photo = Image.fromarray((image * 255).astype(np.uint8))
    pd = ImageDraw.Draw(photo)
    vx, vy = ann.vanishing_point
    x1, y1, x2, y2 = ann.back_wall
    for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        dx, dy = cx - vx, cy - vy
        ts = [t for t in ([(-vx) / dx, (W - vx) / dx] if dx else []) + ([(-vy) / dy, (H - vy) / dy] if dy else [])
              if t > 0]
        t = min(ts)
        pd.line([(cx, cy), (vx + t * dx, vy + t * dy)], fill=(0, 255, 255), width=5)
    pd.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=6)
    ph = ch
    photo = photo.resize((int(W * ph / H), ph), Image.LANCZOS)

    fig = Image.new("RGB", (photo.width + 20 + cw, ch), (255, 255, 255))
    fig.paste(photo, (0, 0))
    fig.paste(canvas, (photo.width + 20, 0))
    fig.save(out / "faces_unfolded.jpg", quality=90)

    for k, t in faces.items():
        print(f"  {k:<8} {t.shape[1]}x{t.shape[0]}  holes {(t[..., 3] < 0.5).mean():5.1%}")
    print("wrote", out / "faces_unfolded.jpg")


if __name__ == "__main__":
    main()
