"""Write a TIP scene to disk for the three.js viewer.

Layout of scenes/<name>/:
    scene.json            geometry and camera (three.js coordinates)
    <plane>.png           RGBA texture, alpha=0 where the photo has no data
    <plane>_filled.png    the same texture with push-pull fill (baseline)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import Scene
from .textures import push_pull_fill, to_pil


def to_three(p: np.ndarray) -> list[float]:
    """TIP camera frame (Z forward) to three.js (camera looks down -Z)."""
    return [round(float(p[0]), 5), round(float(p[1]), 5), round(float(-p[2]), 5)]


def write_scene(scene: Scene, textures: dict, out_dir, image_name: str, source_image=None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    planes = []
    for plane in scene.planes:
        tex = textures[plane.name]
        to_pil(tex).save(out / f"{plane.name}.png", optimize=True)
        entry = {
            "name": plane.name,
            "kind": plane.kind,
            # Order: top-left, top-right, bottom-right, bottom-left of the texture.
            "corners": [to_three(c) for c in plane.corners],
            "texture": f"{plane.name}.png",
            "hole_fraction": round(float((tex[..., 3] < 0.5).mean()), 4),
        }
        if plane.kind == "box":
            to_pil(push_pull_fill(tex)).convert("RGB").save(out / f"{plane.name}_filled.jpg", quality=92)
            entry["texture_filled"] = f"{plane.name}_filled.jpg"
        planes.append(entry)

    if source_image is not None:
        to_pil(source_image).save(out / "input.jpg", quality=92)

    meta = {
        "image": image_name,
        "width": scene.width,
        "height": scene.height,
        "focal": scene.focal,
        "principal": list(scene.principal),
        "box": {
            "x": [scene.xl, scene.xr],
            "y": [scene.yf, scene.yc],
            "z": [-scene.z_near, -scene.depth],
        },
        "planes": planes,
    }
    (out / "scene.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out
