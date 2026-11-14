"""User annotations for one TIP scene.

All image coordinates are in pixels of the original image, x to the right
and y downward, origin at the top-left corner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Foreground:
    """A foreground object to pop out as a flat billboard.

    polygon: outline of the object in image pixels.
    contact_y: image row where the object touches the floor. Defaults to the
        lowest polygon point. Its depth comes from where that row meets the floor.
    """

    name: str
    polygon: list[list[float]]
    contact_y: float | None = None


@dataclass
class Annotation:
    image: str
    vanishing_point: tuple[float, float]
    # Back wall rectangle: [x_left, y_top, x_right, y_bottom].
    back_wall: tuple[float, float, float, float]
    # Camera intrinsics: 35mm-equivalent focal length (iPhone main camera ~26mm).
    focal_35mm: float = 26.0
    # Real-world eye height in meters; sets the scene's scale.
    camera_height: float = 1.6
    foreground: list[Foreground] = field(default_factory=list)

    def focal_px(self, width: int, height: int) -> float:
        """Focal length in pixels (35mm frame long side = 36mm)."""
        return self.focal_35mm / 36.0 * max(width, height)


def load_annotation(path: str | Path) -> Annotation:
    path = Path(path)
    raw = json.loads(path.read_text())
    fgs = [Foreground(**f) for f in raw.pop("foreground", [])]
    ann = Annotation(**raw, foreground=fgs)
    ann.vanishing_point = tuple(ann.vanishing_point)
    ann.back_wall = tuple(ann.back_wall)
    # Resolve the image path relative to the repo root (annotation's grandparent).
    img = Path(ann.image)
    if not img.is_absolute() and not img.exists():
        candidate = path.parent.parent.parent / img
        if candidate.exists():
            ann.image = str(candidate)
    return ann


def save_annotation(ann: Annotation, path: str | Path) -> None:
    data = {
        "image": ann.image,
        "vanishing_point": list(ann.vanishing_point),
        "back_wall": list(ann.back_wall),
        "focal_35mm": ann.focal_35mm,
        "camera_height": ann.camera_height,
        "foreground": [
            {k: v for k, v in f.__dict__.items() if v is not None} for f in ann.foreground
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
