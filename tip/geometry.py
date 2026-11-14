"""Box geometry for Tour Into the Picture.

Camera frame: the input camera sits at the origin, X points right, Y up and
Z forward into the scene. As in the original TIP paper, the vanishing point
is the principal point, so the room's depth axis is the optical axis:

    u = vx + f * X / Z
    v = vy - f * Y / Z

The room is the box X in [xl, xr], Y in [yf, yc], Z in [z_near, depth].
Its scale is set so the floor lies `camera_height` meters below the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotations import Annotation, Foreground

BOX_PLANES = ("back", "floor", "ceiling", "left", "right")


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
class Scene:
    width: int
    height: int
    focal: float
    principal: tuple[float, float]  # (vx, vy)
    xl: float
    xr: float
    yf: float
    yc: float
    z_near: float
    depth: float
    planes: list[Plane]


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


def build_scene(ann: Annotation, width: int, height: int) -> Scene:
    vx, vy = ann.vanishing_point
    x1, y1, x2, y2 = ann.back_wall
    if not (x1 < vx < x2 and y1 < vy < y2):
        raise ValueError(
            "the vanishing point must be inside the back-wall rectangle "
            f"(vp={ann.vanishing_point}, back_wall={ann.back_wall})"
        )
    f = ann.focal_px(width, height)

    # Fix scale so the floor is camera_height below the camera.
    depth = ann.camera_height * f / (y2 - vy)
    xl = (x1 - vx) * depth / f
    xr = (x2 - vx) * depth / f
    yc = (vy - y1) * depth / f
    yf = (vy - y2) * depth / f  # == -camera_height

    # Each side plane leaves the image at the matching border. The box runs to
    # the nearest of those depths. Texels nearer than a plane's own exit fall
    # outside the photo and become holes.
    near_exits = {
        "floor": f * yf / (vy - height),
        "ceiling": f * yc / vy,
        "left": f * xl / (0.0 - vx),
        "right": f * xr / (width - vx),
    }
    z_near = max(min(near_exits.values()), 0.02 * depth)
    z_near = min(z_near, 0.95 * depth)

    W = xr - xl
    Hh = yc - yf
    Dz = depth - z_near
    P = lambda *a: np.array(a, dtype=float)  # noqa: E731

    planes = [
        # Back wall, seen head-on.
        Plane("back", P(xl, yc, depth), P(W, 0, 0), P(0, -Hh, 0)),
        # Floor as seen from above: top row far, bottom row near.
        Plane("floor", P(xl, yf, depth), P(W, 0, 0), P(0, 0, -Dz)),
        # Ceiling: top row near, bottom row far.
        Plane("ceiling", P(xl, yc, z_near), P(W, 0, 0), P(0, 0, Dz)),
        # Left wall: left column near, right column far.
        Plane("left", P(xl, yc, z_near), P(0, 0, Dz), P(0, -Hh, 0)),
        # Right wall: left column far, right column near.
        Plane("right", P(xr, yc, depth), P(0, 0, -Dz), P(0, -Hh, 0)),
    ]

    scene = Scene(width, height, f, (vx, vy), xl, xr, yf, yc, z_near, depth, planes)
    for fg in ann.foreground:
        scene.planes.append(foreground_plane(fg, scene))
    return scene


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
