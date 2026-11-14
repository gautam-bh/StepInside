"""Build a TIP scene from an annotated photo and export it for the viewer.

    python scripts/build_scene.py data/annotations/hallway.json
    -> scenes/hallway/{scene.json, *.png, *_filled.jpg}
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tip import build_scene, load_annotation  # noqa: E402
from tip.export import write_scene  # noqa: E402
from tip.textures import extract_textures, load_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation")
    ap.add_argument("--out", help="output dir (default scenes/<annotation name>)")
    ap.add_argument("--max-res", type=int, default=1536, help="max texture side in pixels")
    a = ap.parse_args()

    ann = load_annotation(a.annotation)
    image = load_image(ann.image)
    h, w = image.shape[:2]
    scene = build_scene(ann, w, h)
    textures = extract_textures(scene, image, [f.polygon for f in ann.foreground], max_res=a.max_res)

    name = Path(a.annotation).stem
    out = write_scene(scene, textures, a.out or ROOT / "scenes" / name, Path(ann.image).name, image)

    print(f"scene '{name}': {w}x{h}, f={scene.focal:.0f}px, vp={scene.principal}")
    for k, b in enumerate(scene.boxes):
        print(
            f"  box {k}  x[{b.xl:6.2f}, {b.xr:5.2f}]  y[{b.yf:5.2f}, {b.yc:5.2f}]  "
            f"z[{b.z0:6.2f}, {b.z1:6.2f}] m"
        )
    for p in scene.planes:
        t = textures[p.name]
        holes = (t[..., 3] < 0.5).mean()
        print(f"  {p.name:<18} {t.shape[1]}x{t.shape[0]}  holes {holes:5.1%}")
    print("wrote", out)


if __name__ == "__main__":
    main()
