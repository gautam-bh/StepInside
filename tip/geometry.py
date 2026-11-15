"""Box geometry for Tour Into the Picture, as a chain of boxes.

Camera frame: the input camera sits at the origin, X points right, Y up and
Z forward into the scene. As in the original TIP paper, the vanishing point
is the principal point, so the room's depth axis is the optical axis:

    u = vx + f * X / Z
    v = vy - f * Y / Z

Box 0 is the classic TIP room. It runs from near the camera to the annotated
back wall. Each further box k (annotation `segments`) continues behind an
opening in the previous box's back wall. Box k is annotated by the image
rectangle of its far end. Its depth comes from where its floor edge
(rectangle bottom) meets the image, assuming the floor stays camera_height
below the camera. Box k then spans from the previous box's far depth to its
own, with constant cross-section. Each back wall becomes up to four bands
around the opening into the next box, and the last box keeps a full back wall.

Room-aligned objects (annotation `objects`) are small axis-aligned boxes
standing on the floor: front, two sides, top and back. Their faces are
ordinary box planes, so they hide the walls behind them (those texels become
holes) and get parallax for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .annotations import Annotation, Foreground, RoomObject


@dataclass
class Plane:
    """A textured rectangle in 3D: point(s, t) = origin + s*u_axis + t*v_axis.

    s runs along texture columns (left to right) and t along texture rows
    (top to bottom), both in [0, 1].
    """

    name: str
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    kind: str = "box"  # "box" or "foreground"
    segment: int = 0

    def point(self, s, t):
        s = np.asarray(s)[..., None]
        t = np.asarray(t)[..., None]
        return self.origin + s * self.u_axis + t * self.v_axis

    @property
    def corners(self) -> np.ndarray:
        """Corners in order: top-left, top-right, bottom-right, bottom-left."""
        o, u, v = self.origin, self.u_axis, self.v_axis
        return np.stack([o, o + u, o + u + v, o + v])

    @property
    def normal_axis(self) -> int:
        """Index of the coordinate axis the plane is perpendicular to."""
        n = np.cross(self.u_axis, self.v_axis)
        return int(np.argmax(np.abs(n)))


@dataclass
class Box:
    """Axis-aligned room segment: X in [xl, xr], Y in [yf, yc], Z in [z0, z1]."""

    xl: float
    xr: float
    yf: float
    yc: float
    z0: float
    z1: float


@dataclass
class Scene:
    width: int
    height: int
    focal: float
    principal: tuple[float, float]  # (vx, vy)
    boxes: list[Box]
    planes: list[Plane] = field(default_factory=list)

    # Box 0 (the room around the camera), kept for convenience.
    @property
    def xl(self): return self.boxes[0].xl
    @property
    def xr(self): return self.boxes[0].xr
    @property
    def yf(self): return self.boxes[0].yf
    @property
    def yc(self): return self.boxes[0].yc
    @property
    def z_near(self): return self.boxes[0].z0
    @property
    def depth(self): return self.boxes[0].z1


def project(points: np.ndarray, focal: float, principal) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame 3D points (..., 3) to image pixels (u, v)."""
    vx, vy = principal
    z = points[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = vx + focal * points[..., 0] / z
        v = vy - focal * points[..., 1] / z
    return u, v


def backproject(u, v, z, focal: float, principal) -> np.ndarray:
    """Lift image pixel(s) at depth z back to camera-frame 3D points."""
    vx, vy = principal
    u, v, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(z, float))
    return np.stack([(u - vx) * z / focal, (vy - v) * z / focal, z], axis=-1)


def floor_depth(v_contact: float, scene: Scene) -> float:
    """Depth of the floor point that appears in image row v_contact."""
    vx, vy = scene.principal
    if v_contact <= vy + 1e-6:
        raise ValueError(
            f"contact row {v_contact:.1f} is at or above the horizon (vy={vy:.1f}); "
            "it can't sit on the floor"
        )
    return scene.focal * scene.yf / (vy - v_contact)


def _rect_box(rect, depth, f, vx, vy, z0) -> Box:
    x1, y1, x2, y2 = rect
    return Box(
        xl=(x1 - vx) * depth / f,
        xr=(x2 - vx) * depth / f,
        yf=(vy - y2) * depth / f,
        yc=(vy - y1) * depth / f,
        z0=z0,
        z1=depth,
    )


def _box_planes(b: Box, k: int) -> list[Plane]:
    W, Hh, Dz = b.xr - b.xl, b.yc - b.yf, b.z1 - b.z0
    P = lambda *a: np.array(a, dtype=float)  # noqa: E731
    p = f"s{k}_"
    return [
        # Floor as seen from above: top row far, bottom row near.
        Plane(p + "floor", P(b.xl, b.yf, b.z1), P(W, 0, 0), P(0, 0, -Dz), segment=k),
        # Ceiling: top row near, bottom row far.
        Plane(p + "ceiling", P(b.xl, b.yc, b.z0), P(W, 0, 0), P(0, 0, Dz), segment=k),
        # Left wall: left column near, right column far.
        Plane(p + "left", P(b.xl, b.yc, b.z0), P(0, 0, Dz), P(0, -Hh, 0), segment=k),
        # Right wall: left column far, right column near.
        Plane(p + "right", P(b.xr, b.yc, b.z1), P(0, 0, -Dz), P(0, -Hh, 0), segment=k),
    ]


def _back_wall_planes(b: Box, opening: Box | None, k: int) -> list[Plane]:
    """Back wall of box b at z1, minus the opening into the next box (if any)."""
    z = b.z1
    P = lambda *a: np.array(a, dtype=float)  # noqa: E731

    def rect(name, xl, xr, yb, yt):
        if xr - xl < 1e-6 or yt - yb < 1e-6:
            return None
        return Plane(f"s{k}_{name}", P(xl, yt, z), P(xr - xl, 0, 0), P(0, yb - yt, 0), segment=k)

    if opening is None:
        return [rect("back", b.xl, b.xr, b.yf, b.yc)]
    # Clip the opening to this box's cross-section.
    ol, orr = max(opening.xl, b.xl), min(opening.xr, b.xr)
    of, oc = max(opening.yf, b.yf), min(opening.yc, b.yc)
    bands = [
        rect("back_top", b.xl, b.xr, oc, b.yc),
        rect("back_bottom", b.xl, b.xr, b.yf, of),
        rect("back_left", b.xl, ol, of, oc),
        rect("back_right", orr, b.xr, of, oc),
    ]
    return [p for p in bands if p is not None]


def build_scene(ann: Annotation, width: int, height: int) -> Scene:
    vx, vy = ann.vanishing_point
    f = ann.focal_px(width, height)
    rects = [tuple(ann.back_wall)] + [tuple(r) for r in ann.segments]
    for i, (x1, y1, x2, y2) in enumerate(rects):
        if not (x1 < vx < x2 and y1 < vy < y2):
            raise ValueError(
                f"the vanishing point {ann.vanishing_point} must be inside every box rectangle; "
                f"box {i} is {rects[i]}"
            )

    # Box 0: depth set so the floor is camera_height below the camera.
    x1, y1, x2, y2 = rects[0]
    depth0 = ann.camera_height * f / (y2 - vy)
    b0 = _rect_box(rects[0], depth0, f, vx, vy, z0=0.0)
    # Each side plane leaves the image at the matching border. Box 0 runs to
    # the nearest of those depths. Texels nearer than a plane's own exit fall
    # outside the photo and become holes.
    near_exits = [
        f * b0.yf / (vy - height),
        f * b0.yc / vy,
        f * b0.xl / (0.0 - vx),
        f * b0.xr / (width - vx),
    ]
    b0.z0 = float(np.clip(min(near_exits), 0.02 * depth0, 0.95 * depth0))
    boxes = [b0]

    # Further boxes: depth from the bottom of their far-end rectangle.
    for k, r in enumerate(rects[1:], start=1):
        prev = boxes[-1]
        z = ann.camera_height * f / (r[3] - vy)
        if z <= prev.z1 + 1e-6:
            raise ValueError(
                f"box {k} {r} must be farther than box {k - 1}: its bottom edge (y={r[3]}) "
                f"must be closer to the vanishing point than the previous box's"
            )
        boxes.append(_rect_box(r, z, f, vx, vy, z0=prev.z1))

    scene = Scene(width, height, f, (vx, vy), boxes)
    for k, b in enumerate(boxes):
        scene.planes += _box_planes(b, k)
        scene.planes += _back_wall_planes(b, boxes[k + 1] if k + 1 < len(boxes) else None, k)
    for obj in ann.objects:
        scene.planes += object_planes(obj, scene)
    for fg in ann.foreground:
        scene.planes.append(foreground_plane(fg, scene))
    return scene


def object_planes(obj: RoomObject, scene: Scene) -> list[Plane]:
    """Five faces (front, left, right, top, back) of a room-aligned object."""
    x1, y1, x2, y2 = obj.front
    vy = scene.principal[1]
    if not vy < obj.back_y < y2:
        raise ValueError(f"object {obj.name}: need vanishing point row {vy:.0f} < back_y {obj.back_y} < "
                         f"front bottom {y2} (it must stand on the floor and extend away from the camera)")
    z0 = floor_depth(y2, scene)
    z1 = floor_depth(obj.back_y, scene)
    tl = backproject(x1, y1, z0, scene.focal, scene.principal)
    br = backproject(x2, y2, z0, scene.focal, scene.principal)
    xl, xr, yb, yt = tl[0], br[0], scene.yf, tl[1]
    W, Hh, D = xr - xl, yt - yb, z1 - z0
    seg = next((k for k, b in enumerate(scene.boxes) if b.z0 <= z0 <= b.z1), 0)
    P = lambda *a: np.array(a, dtype=float)  # noqa: E731
    p = f"o_{obj.name}_"
    return [
        Plane(p + "front", P(xl, yt, z0), P(W, 0, 0), P(0, -Hh, 0), segment=seg),
        # Sides as seen from outside the object.
        Plane(p + "left", P(xl, yt, z1), P(0, 0, -D), P(0, -Hh, 0), segment=seg),
        Plane(p + "right", P(xr, yt, z0), P(0, 0, D), P(0, -Hh, 0), segment=seg),
        Plane(p + "top", P(xl, yt, z1), P(W, 0, 0), P(0, 0, -D), segment=seg),
        Plane(p + "back", P(xr, yt, z1), P(-W, 0, 0), P(0, -Hh, 0), segment=seg),
    ]


def foreground_plane(fg: Foreground, scene: Scene) -> Plane:
    """Billboard parallel to the image plane, standing where it meets the floor."""
    poly = np.asarray(fg.polygon, dtype=float)
    u0, v0 = poly.min(axis=0)
    u1, v1 = poly.max(axis=0)
    contact = fg.contact_y if fg.contact_y is not None else v1
    z = floor_depth(contact, scene)
    tl = backproject(u0, v0, z, scene.focal, scene.principal)
    br = backproject(u1, v1, z, scene.focal, scene.principal)
    return Plane(
        f"fg_{fg.name}",
        tl,
        np.array([br[0] - tl[0], 0.0, 0.0]),
        np.array([0.0, br[1] - tl[1], 0.0]),
        kind="foreground",
    )
