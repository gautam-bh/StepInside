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
class RoomObject:
    """A box-shaped object aligned with the room (cabinet, desk, shelf, bed),
    standing on the floor with its front face toward the camera.

    front: [x1, y1, x2, y2] image rectangle of its front face. The bottom
        edge y2 is where it meets the floor, which gives its depth.
    back_y: image row where its far bottom edge meets the floor (between the
        VP and y2). Its depth sets how far back the object extends.
    """

    name: str
    front: list[float]
    back_y: float


@dataclass
class Solid:
    """A surface of revolution with a vertical axis: vase, bin, lamp,
    extinguisher, hydrant. Found automatically by tip.revolve.

    profile: [[y, x_left, x_right], ...] its silhouette, one entry per image
        row, top to bottom.
    mount: "floor" if it stands on the floor (its lowest row gives its
        depth) or "wall" if it hangs against the side wall behind it (the
        wall fixes its depth).
    """

    name: str
    profile: list[list[float]]
    mount: str = "floor"


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
    # Further boxes behind the back wall (e.g. a corridor beyond a doorway),
    # each given as the image rectangle of its far end, nearest first.
    segments: list[list[float]] = field(default_factory=list)
    # Room-aligned 3D boxes standing on the floor.
    objects: list[RoomObject] = field(default_factory=list)
    # Surfaces of revolution (vertical axis).
    solids: list[Solid] = field(default_factory=list)

    def focal_px(self, width: int, height: int) -> float:
        """Focal length in pixels (35mm frame long side = 36mm)."""
        return self.focal_35mm / 36.0 * max(width, height)


def load_annotation(path: str | Path) -> Annotation:
    path = Path(path)
    raw = json.loads(path.read_text())
    fgs = [Foreground(**f) for f in raw.pop("foreground", [])]
    objs = [RoomObject(**o) for o in raw.pop("objects", [])]
    sols = [Solid(**o) for o in raw.pop("solids", [])]
    ann = Annotation(**raw, foreground=fgs, objects=objs, solids=sols)
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
        "segments": [list(r) for r in ann.segments],
        "objects": [{"name": o.name, "front": list(o.front), "back_y": o.back_y} for o in ann.objects],
        "solids": [{"name": o.name, "mount": o.mount, "profile": [[round(float(v), 1) for v in r] for r in o.profile]}
                   for o in ann.solids],
        "foreground": [
            {k: v for k, v in f.__dict__.items() if v is not None} for f in ann.foreground
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
