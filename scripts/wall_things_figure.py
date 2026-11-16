"""Figure of the automatic wall-box and ceiling-pipe detector on one photo.

    python scripts/wall_things_figure.py data/annotations/hallway_auto.json
    -> outputs/<name>/wall_things_auto.jpg
       (detections · a view of the wall boxes from the side · a view up at the pipes)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tip import build_scene, load_annotation, project  # noqa: E402
from tip.render import render_view  # noqa: E402
from tip.textures import extract_textures, load_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation")
    ap.add_argument("--out")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()

    ann = load_annotation(a.annotation)
    image = load_image(ann.image)
    h, w = image.shape[:2]
    scene = build_scene(ann, w, h)
    k = a.scale

    p1 = Image.fromarray((image * 255).astype(np.uint8)).resize((int(w * k), int(h * k)))
    d = ImageDraw.Draw(p1)
    for p in scene.planes:
        if p.name.startswith("w_"):
            u, v = project(p.corners, scene.focal, scene.principal)
            d.polygon(list(zip(u * k, v * k)), outline=(0, 255, 0))
    vx, vy = scene.principal
    for t in ann.tubes:
        n, f = np.array(t.near) - [vx, vy], np.array(t.far) - [vx, vy]
        th = np.arctan2(n[1], n[0])
        for sg in (-1, 1):
            ang = th + sg * t.half_angle
            ends = [(vx + np.linalg.norm(x) * np.cos(ang), vy + np.linalg.norm(x) * np.sin(ang)) for x in (f, n)]
            d.line([(x * k, y * k) for x, y in ends], fill=(255, 0, 255), width=2)

    tex = extract_textures(scene, image, [f.polygon for f in ann.foreground])
    views = [((0.2, 0.25, 2.0), 40.0, 8.0), ((-0.3, 0.35, 1.2), 8.0, 20.0)]
    panels = [p1]
    for pos, yaw, pitch in views:
        rgb, _ = render_view(scene, tex, position=pos, yaw=yaw, pitch=pitch, scale=k, background=(1.0, 0.0, 1.0))
        panels.append(Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)))

    titles = ["wall boxes (green) and ceiling pipes (magenta), all automatic",
              "moved camera: the signs stand off the wall", "looking up: the pipes hang below the ceiling"]
    tw, th_ = p1.size
    sheet = Image.new("RGB", (tw * 3, th_ + 22), "white")
    dd = ImageDraw.Draw(sheet)
    for i, (p, t) in enumerate(zip(panels, titles)):
        sheet.paste(p, (i * tw, 22))
        dd.text((i * tw + 6, 5), t, fill=(0, 0, 0))
    out = Path(a.out or ROOT / "outputs" / Path(a.annotation).stem.replace("_auto", "") / "wall_things_auto.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    print("wrote", out)


if __name__ == "__main__":
    main()
