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

Solids (annotation `solids`) are surfaces of revolution about a vertical
axis, rebuilt from their image silhouette. A circle of radius r whose axis is
a horizontal distance D from the camera, in direction beta, is seen between
the two rays at beta +- alpha with sin(alpha) = r / D. So each silhouette row
gives alpha, and the radius follows once D is known. D comes from the floor
(the lowest silhouette point is the near edge of the base circle) or from the
side wall the object hangs on (its outer edge touches the wall).

Reliefs (annotation `reliefs`) are boxes fixed to a side wall or the
ceiling. Their near face is parallel to the image, so it is seen undistorted.
Its outer edge lies on the wall, which gives its depth: z0 = f * W / p for a
wall at distance W, seen p pixels from the vanishing point. Its inner edge
then gives how far it sticks out, and the inner edge of its far end gives
its length along the wall. A sign sticking out of the wall is a relief
that is long out of the wall and short along it; an alarm box is the reverse.

Tubes (annotation `tubes`) are pipes along the depth axis under the ceiling.
Each cross-section circle lies in a plane z = const, so it projects to a
circle, and the pipe is seen as a wedge between two lines through the
vanishing point, with sin(half angle) = r / rho for radius r at distance rho
from the axis. The pipe touching the ceiling gives rho.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .annotations import Annotation, Foreground, Relief, RoomObject, Solid, Tube


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
class Lathe:
    """A surface of revolution about the vertical line X = ax, Z = az.

    Profile points (radius, Y) run bottom to top and start and end at radius
    0, so the caps are part of the surface. Texture coordinates match
    three.js's LatheGeometry: s = phi / 2pi around the axis (phi = 0 faces
    the input camera's side, -Z) and t from the top profile point (t = 0) to
    the bottom one (t = 1), linear in profile index.
    """

    name: str
    ax: float
    az: float
    radius: np.ndarray  # (n,)
    ys: np.ndarray  # (n,) non-decreasing
    kind: str = "lathe"
    segment: int = 0

    def _profile(self, t):
        n = len(self.ys)
        k = np.clip((1 - np.asarray(t, float)) * (n - 1), 0, n - 1)
        j = np.minimum(np.floor(k).astype(int), n - 2)
        a = k - j
        r = self.radius[j] * (1 - a) + self.radius[j + 1] * a
        y = self.ys[j] * (1 - a) + self.ys[j + 1] * a
        return r, y, j

    def point(self, s, t):
        phi = 2 * np.pi * np.asarray(s, float)
        r, y, _ = self._profile(t)
        phi, r, y = np.broadcast_arrays(phi, r, y)
        return np.stack([self.ax + r * np.sin(phi), y, self.az - r * np.cos(phi)], axis=-1)

    def normal(self, s, t):
        """Outward surface normal (not normalised)."""
        phi = 2 * np.pi * np.asarray(s, float)
        _, _, j = self._profile(t)
        dr = self.radius[j + 1] - self.radius[j]
        dy = self.ys[j + 1] - self.ys[j]
        phi, dr, dy = np.broadcast_arrays(phi, dr, dy)
        # Profile tangent (dr, dy) turned outward: (dy, -dr) in (radial, Y).
        return np.stack([dy * np.sin(phi), -dr, -dy * np.cos(phi)], axis=-1)

    def hit(self, cam: np.ndarray, dirs: np.ndarray):
        """Ray/surface intersection, like render._hit: (t, s, tt, valid)."""
        shape = dirs.shape[:-1]
        d = dirs.reshape(-1, 3)
        ox, oy, oz = cam[0] - self.ax, cam[1], cam[2] - self.az
        best = np.full(len(d), np.inf)
        frac = np.zeros(len(d))
        # Bounding cylinder test first: most rays miss.
        rmax = float(self.radius.max())
        a2 = d[:, 0] ** 2 + d[:, 2] ** 2
        b2 = ox * d[:, 0] + oz * d[:, 2]
        c2 = ox * ox + oz * oz - rmax * rmax
        near = (b2 * b2 - a2 * c2 >= 0) | (c2 <= 0)
        idx = np.nonzero(near)[0]
        if len(idx):
            dx, dy, dz = d[idx, 0], d[idx, 1], d[idx, 2]
            bt = best[idx]
            bf = frac[idx]
            n = len(self.ys)
            with np.errstate(divide="ignore", invalid="ignore"):
                for j in range(n - 1):
                    y0, y1 = self.ys[j], self.ys[j + 1]
                    r0, r1 = self.radius[j], self.radius[j + 1]
                    if abs(y1 - y0) < 1e-9:  # flat ring (a cap): plane Y = y0
                        t = (y0 - oy) / dy
                        rr = np.hypot(ox + t * dx, oz + t * dz)
                        lo, hi = min(r0, r1), max(r0, r1)
                        ok = np.isfinite(t) & (t > 1e-6) & (rr >= lo - 1e-9) & (rr <= hi + 1e-9) & (t < bt)
                        a = np.where(abs(r1 - r0) > 1e-12, (rr - r0) / (r1 - r0 + 1e-300), 0.0)
                        bt = np.where(ok, t, bt)
                        bf = np.where(ok, j + a, bf)
                        continue
                    # r(Y) = r0 + m (Y - y0); solve x^2 + z^2 = r(Y)^2 along the ray.
                    m = (r1 - r0) / (y1 - y0)
                    q0 = r0 + m * (oy - y0)
                    qa = dx * dx + dz * dz - (m * dy) ** 2
                    qb = 2 * (ox * dx + oz * dz - q0 * m * dy)
                    qc = ox * ox + oz * oz - q0 * q0
                    disc = qb * qb - 4 * qa * qc
                    sq = np.sqrt(np.maximum(disc, 0))
                    for sign in (-1.0, 1.0):
                        t = (-qb + sign * sq) / (2 * qa)
                        y = oy + t * dy
                        ok = (disc >= 0) & np.isfinite(t) & (t > 1e-6) & (y >= y0 - 1e-9) & (y <= y1 + 1e-9) & (t < bt)
                        ok &= (q0 + m * t * dy) >= -1e-9  # the real cone, not its mirror
                        bt = np.where(ok, t, bt)
                        bf = np.where(ok, j + (y - y0) / (y1 - y0), bf)
            best[idx] = bt
            frac[idx] = bf
        valid = np.isfinite(best)
        t = np.where(valid, best, 0.0)
        p = cam + t[:, None] * d
        phi = np.arctan2(p[:, 0] - self.ax, -(p[:, 2] - self.az)) % (2 * np.pi)
        s = phi / (2 * np.pi)
        tt = 1 - frac / (len(self.ys) - 1)
        out = lambda a: a.reshape(shape)  # noqa: E731
        return out(np.where(valid, best, np.inf)), out(s), out(tt), out(valid)


@dataclass
class Cylinder:
    """A pipe along the depth axis: X = cx + r sin(phi), Y = cy + r cos(phi),
    Z in [z0, z1]. Texture: s = phi / 2pi (phi = 0 on top) and t = 0 at the
    far end, 1 at the near end, as three.js's CylinderGeometry turned to lie
    along -Z."""

    name: str
    cx: float
    cy: float
    r: float
    z0: float
    z1: float
    kind: str = "tube"
    segment: int = 0

    def point(self, s, t):
        phi = 2 * np.pi * np.asarray(s, float)
        z = self.z1 - np.asarray(t, float) * (self.z1 - self.z0)
        phi, z = np.broadcast_arrays(phi, z)
        return np.stack([self.cx + self.r * np.sin(phi), self.cy + self.r * np.cos(phi), z], axis=-1)

    def normal(self, s, t):
        phi = 2 * np.pi * np.asarray(s, float)
        phi, _ = np.broadcast_arrays(phi, np.asarray(t, float))
        return np.stack([np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)

    def mirror_cols(self, cols: int) -> np.ndarray:
        """Texture column seen in place of each column on the hidden side:
        the mirror image through the plane holding the axis and the camera."""
        phi_c = np.arctan2(-self.cx, -self.cy)
        phi = 2 * np.pi * (np.arange(cols) + 0.5) / cols
        return np.floor(((2 * phi_c - phi) % (2 * np.pi)) / (2 * np.pi) * cols).astype(int) % cols

    def hit(self, cam: np.ndarray, dirs: np.ndarray):
        shape = dirs.shape[:-1]
        d = dirs.reshape(-1, 3)
        ox, oy = cam[0] - self.cx, cam[1] - self.cy
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2 * (ox * d[:, 0] + oy * d[:, 1])
        c = ox * ox + oy * oy - self.r ** 2
        disc = b * b - 4 * a * c
        sq = np.sqrt(np.maximum(disc, 0))
        best = np.full(len(d), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            for sign in (1.0, -1.0):  # far root first, so the near one wins
                t = (-b - sign * sq) / (2 * a)
                z = cam[2] + t * d[:, 2]
                ok = (disc >= 0) & np.isfinite(t) & (t > 1e-6) & (z >= self.z0) & (z <= self.z1)
                best = np.where(ok & (t < best), t, best)
        valid = np.isfinite(best)
        t = np.where(valid, best, 0.0)
        p = cam + t[:, None] * d
        s = (np.arctan2(p[:, 0] - self.cx, p[:, 1] - self.cy) % (2 * np.pi)) / (2 * np.pi)
        tt = (self.z1 - p[:, 2]) / max(self.z1 - self.z0, 1e-9)
        out = lambda x: x.reshape(shape)  # noqa: E731
        return out(np.where(valid, best, np.inf)), out(s), out(tt), out(valid)


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
    for sol in ann.solids:
        scene.planes.append(solid_surface(sol, scene))
    for rel in ann.reliefs:
        scene.planes += relief_planes(rel, scene)
    for tube in ann.tubes:
        scene.planes.append(tube_surface(tube, scene))
    for fg in ann.foreground:
        scene.planes.append(foreground_plane(fg, scene))
    return scene


def _wall_distance(scene: Scene, beta: float, sin_a: float, right: bool):
    """Horizontal distance D to the axis of a solid in direction beta whose
    outer silhouette edge touches the left/right wall. Each box has its own
    walls, so try them in turn and keep the one the solid lands in."""
    for b in scene.boxes:
        xw = b.xr if right else b.xl
        den = np.sin(beta) + (sin_a if right else -sin_a)
        if abs(den) < 1e-9 or xw * den <= 0:
            continue
        D = xw / den
        z = D * np.cos(beta)
        if b.z0 - 1e-6 <= z <= b.z1 + 1e-6:
            return D
    return None


def solid_geometry(sol: Solid, scene: Scene, mount: str | None = None):
    """(D, beta, rows) for a solid: horizontal distance and direction of its
    axis, and per profile row (Y, radius). None if it doesn't fit the room."""
    f = scene.focal
    vx, vy = scene.principal
    prof = np.asarray(sol.profile, float)
    y, xl, xr = prof[:, 0], prof[:, 1], prof[:, 2]
    bl = np.arctan((xl - vx) / f)
    br = np.arctan((xr - vx) / f)
    beta = float(np.median((bl + br) / 2))
    alpha = np.clip((br - bl) / 2, 1e-4, 1.2)
    mount = mount or sol.mount
    if mount == "wall":
        D = _wall_distance(scene, beta, float(np.sin(alpha.max())), right=beta > 0)
    else:
        yb = y.max()
        if yb <= vy + 1:
            return None
        sin_b = float(np.sin(alpha[np.argmax(y)]))
        D = f * (-scene.yf) / ((yb - vy) * np.cos(beta)) / (1 - sin_b)
    if D is None or not np.isfinite(D) or D <= 0:
        return None
    radius = D * np.sin(alpha)
    # Height of each row's tangent points (average depth of the two sides).
    z = D * np.cos(alpha) * np.cos(beta) * np.cos(alpha)
    ys = (vy - y) * z / f
    return D, beta, ys, radius


def solid_surface(sol: Solid, scene: Scene) -> Lathe:
    g = solid_geometry(sol, scene)
    if g is None:
        raise ValueError(f"solid {sol.name}: its silhouette doesn't fit the room ({sol.mount} mount)")
    D, beta, ys, radius = g
    order = np.argsort(ys)
    ys, radius = ys[order], radius[order]
    if len(ys) > 48:  # a smooth profile needs few rings
        k = np.linspace(0, len(ys) - 1, 48)
        ys, radius = np.interp(k, np.arange(len(ys)), ys), np.interp(k, np.arange(len(ys)), radius)
    if sol.mount == "floor":
        ys = ys - (ys[0] - scene.yf)  # rest exactly on the floor
    # Caps: close the profile at both ends.
    ys = np.concatenate([[ys[0]], ys, [ys[-1]]])
    radius = np.concatenate([[0.0], radius, [0.0]])
    ax, az = D * np.sin(beta), D * np.cos(beta)
    seg = next((k for k, b in enumerate(scene.boxes) if b.z0 <= az <= b.z1), len(scene.boxes) - 1)
    return Lathe(f"r_{sol.name}", float(ax), float(az), radius, ys, segment=seg)


def _segment_at(scene: Scene, z: float) -> int:
    return next((k for k, b in enumerate(scene.boxes) if b.z0 - 1e-6 <= z <= b.z1 + 1e-6), len(scene.boxes) - 1)


def relief_box(rel: Relief, scene: Scene):
    """3D extent (lo, hi) of a relief, as two corners (X, Y, Z), or None if
    its contact edge doesn't land on the wall it names."""
    f = scene.focal
    vx, vy = scene.principal
    # Signed distance from the vanishing point across the wall, in pixels.
    across = (lambda q: vy - q) if rel.wall == "ceiling" else (lambda q: q - vx)
    pc, pf, p1 = across(rel.contact), across(rel.front), across(rel.far)
    z0 = None
    for b in scene.boxes:
        W = {"left": b.xl, "right": b.xr, "ceiling": b.yc}[rel.wall]
        if abs(pc) < 1e-6 or W * pc <= 0:
            continue
        z = f * W / pc
        if b.z0 - 1e-6 <= z <= b.z1 + 1e-6:
            z0 = z
            break
    if z0 is None or pf * pc <= 0 or abs(pf) > abs(pc) + 1e-6:
        return None
    Wf = pf * z0 / f  # the front face's distance from the depth axis
    z1 = f * Wf / p1 if p1 * pf > 0 else z0
    z1 = max(z1, z0 + 0.01)  # a sign facing the camera still has some thickness
    q0, q1 = sorted(rel.span)
    if rel.wall == "ceiling":
        xs = ((q0 - vx) * z0 / f, (q1 - vx) * z0 / f)
        lo, hi = (xs[0], Wf, z0), (xs[1], W, z1)
    else:
        ys = ((vy - q1) * z0 / f, (vy - q0) * z0 / f)
        xs = sorted((Wf, W))
        lo, hi = (xs[0], ys[0], z0), (xs[1], ys[1], z1)
    return np.array(lo, float), np.array(hi, float)


def cuboid_planes(prefix: str, lo, hi, skip=(), segment: int = 0) -> list[Plane]:
    """The faces of an axis-aligned box, each oriented to be seen from
    outside (u x v points out; vertical faces have v pointing down, top and
    bottom faces have u along +X). skip: faces to leave out, from
    "-x", "+x", "-y", "+y", "-z", "+z"."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ext = hi - lo
    planes = []
    names = {"-x": "left", "+x": "right", "-y": "bottom", "+y": "top", "-z": "front", "+z": "back"}
    E = np.eye(3)
    for key, nm in names.items():
        if key in skip:
            continue
        a = "xyz".index(key[1])
        n = E[a] * (1.0 if key[0] == "+" else -1.0)
        if a == 1:  # top / bottom
            u = E[0] * ext[0]
            v = E[2] * ext[2] * (-1.0 if key[0] == "+" else 1.0)
        else:
            v = -E[1] * ext[1]
            u_dir = np.cross(-E[1], n)
            u = u_dir * ext[int(np.argmax(np.abs(u_dir)))]
        centre = (lo + hi) / 2 + n * ext[a] / 2
        planes.append(Plane(prefix + nm, centre - u / 2 - v / 2, u, v, segment=segment))
    return planes


def relief_planes(rel: Relief, scene: Scene) -> list[Plane]:
    box = relief_box(rel, scene)
    if box is None:
        raise ValueError(f"relief {rel.name}: its contact edge doesn't meet the {rel.wall} wall")
    lo, hi = box
    skip = {"left": "-x", "right": "+x", "ceiling": "+y"}[rel.wall]
    return cuboid_planes(f"w_{rel.name}_", lo, hi, skip=(skip,), segment=_segment_at(scene, lo[2]))


def tube_geometry(tube: Tube, scene: Scene):
    """(cx, cy, r, z0, z1) of a pipe touching the ceiling, or None."""
    f = scene.focal
    vx, vy = scene.principal
    n = np.array([tube.near[0] - vx, vy - tube.near[1]], float)
    fa = np.array([tube.far[0] - vx, vy - tube.far[1]], float)
    theta = np.arctan2(n[1], n[0])
    sa = np.sin(tube.half_angle)
    den = np.sin(theta) + sa
    if den <= 1e-6:
        return None
    rn, rf = np.linalg.norm(n), np.linalg.norm(fa)
    for k, b in enumerate(scene.boxes):
        rho = b.yc / den
        z0, z1 = f * rho / rn, f * rho / rf
        # Its near end may be in front of the first box (a pipe below the
        # ceiling leaves the image nearer than the ceiling does).
        if (b.z0 - 1e-6 <= z0 or k == 0) and z0 <= b.z1 + 1e-6:
            z1 = min(z1, b.z1)  # it ends at this box's back wall at the latest
            if z1 <= z0 or z0 <= 0:
                return None
            return rho * np.cos(theta), rho * np.sin(theta), rho * sa, z0, z1
    return None


def tube_surface(tube: Tube, scene: Scene) -> Cylinder:
    g = tube_geometry(tube, scene)
    if g is None:
        raise ValueError(f"tube {tube.name}: it doesn't fit under the ceiling")
    cx, cy, r, z0, z1 = g
    return Cylinder(f"t_{tube.name}", float(cx), float(cy), float(r), float(z0), float(z1),
                    segment=_segment_at(scene, z0))


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
