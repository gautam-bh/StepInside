"""Automatic TIP annotation: vanishing point and back wall from image edges.

Works for one-point-perspective interiors (camera looking down the room).

1. Vanishing point. Every edge pixel lies on a line along its edge
   direction. In one-point perspective, the edges running into the room meet
   at the VP. Candidate points are scored by how many (gradient-weighted) edge
   lines pass close to them: first on a coarse grid, then refined. Near
   horizontal and vertical edges are skipped, since they are parallel to the
   image plane and don't converge.
2. Back wall. Its four corners fix the four room-edge rays (floor/wall and
   ceiling/wall junctions), which run from the corners out to the image
   border. We search every rectangle around the VP on a grid and score:
   how consistently the image changes across each of the four rays (from
   its corner outward), plus edge strength along the rectangle's own sides.
   Running sums make the exhaustive 4D search cheap.

The result is a starting point. scripts/annotate.py opens it for adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotations import Annotation


# --------------------------------------------------------------------------- image helpers
def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return (image[..., :3] @ np.array([0.299, 0.587, 0.114])).astype(np.float32)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (edge-padded)."""
    if sigma <= 0:
        return img
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(img, r, mode="edge")
    tmp = sum(k[i] * p[:, i : i + img.shape[1]] for i in range(2 * r + 1))
    return sum(k[i] * tmp[i : i + img.shape[0], :] for i in range(2 * r + 1)).astype(np.float32)


def gradients(gray: np.ndarray, sigma: float = 1.5):
    g = blur(gray, sigma)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    gy[1:-1, :] = 0.5 * (g[2:, :] - g[:-2, :])
    return gx, gy


def downscale(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Box-downscale so the longer side is <= max_side. Returns (img, scale)."""
    h, w = image.shape[:2]
    f = int(np.ceil(max(h, w) / max_side))
    if f <= 1:
        return image, 1.0
    h2, w2 = h // f, w // f
    img = image[: h2 * f, : w2 * f].reshape(h2, f, w2, f, *image.shape[2:]).mean(axis=(1, 3))
    return img.astype(np.float32), 1.0 / f


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    ax, ay = x - x0, y - y0
    return (
        img[y0, x0] * (1 - ax) * (1 - ay)
        + img[y0, x0 + 1] * ax * (1 - ay)
        + img[y0 + 1, x0] * (1 - ax) * ay
        + img[y0 + 1, x0 + 1] * ax * ay
    )


# --------------------------------------------------------------------------- vanishing point
@dataclass
class EdgePixels:
    x: np.ndarray
    y: np.ndarray
    dx: np.ndarray  # unit edge direction
    dy: np.ndarray
    w: np.ndarray  # weight (gradient magnitude)


def converging_edges(gx, gy, max_pixels: int = 25000, axis_margin_deg: float = 12.0, seed: int = 0) -> EdgePixels:
    """Strong edge pixels whose direction isn't near horizontal/vertical."""
    mag = np.hypot(gx, gy)
    # Top 10% of gradients, but never weaker than a fifth of the strong edges,
    # so flat, noisy surfaces don't flood the vote with random directions.
    thresh = max(np.percentile(mag, 90), 0.2 * np.percentile(mag, 99.5))
    yy, xx = np.nonzero(mag > thresh)
    m = mag[yy, xx]
    # Edge direction is perpendicular to the gradient.
    ex, ey = -gy[yy, xx] / m, gx[yy, xx] / m
    ang = np.degrees(np.arctan2(np.abs(ey), np.abs(ex)))  # 0 = horizontal, 90 = vertical
    keep = (ang > axis_margin_deg) & (ang < 90 - axis_margin_deg)
    xx, yy, ex, ey, m = xx[keep], yy[keep], ex[keep], ey[keep], m[keep]
    if len(xx) > max_pixels:
        idx = np.random.default_rng(seed).choice(len(xx), max_pixels, replace=False, p=m / m.sum())
        xx, yy, ex, ey, m = xx[idx], yy[idx], ex[idx], ey[idx], m[idx]
    return EdgePixels(xx.astype(np.float32), yy.astype(np.float32), ex, ey, m)


def vp_score(cx: np.ndarray, cy: np.ndarray, e: EdgePixels, tol_deg: float = 2.0, chunk: int = 256) -> np.ndarray:
    """Score candidate VPs: weighted count of edge lines passing within tol_deg."""
    cx = np.asarray(cx, np.float32).ravel()
    cy = np.asarray(cy, np.float32).ravel()
    s = np.sin(np.radians(tol_deg))
    out = np.empty(len(cx), np.float32)
    for i in range(0, len(cx), chunk):
        vx = cx[i : i + chunk, None] - e.x[None]
        vy = cy[i : i + chunk, None] - e.y[None]
        r = np.hypot(vx, vy) + 1e-6
        # |sin| of the angle between the edge direction and the direction to the candidate.
        sin = np.abs(vx * e.dy - vy * e.dx) / r
        # Edge pixels very close to the candidate say little about it.
        out[i : i + chunk] = (e.w * (sin < s) * (r > 15)).sum(axis=1)
    return out


def find_vanishing_point(gray: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """VP in pixel coords of `gray`; searched inside the image's central region."""
    h, w = gray.shape
    gx, gy = gradients(gray)
    e = converging_edges(gx, gy, seed=seed)
    # Coarse grid over the middle of the image (a one-point-perspective VP is
    # inside the frame).
    xs = np.linspace(0.1 * w, 0.9 * w, 41)
    ys = np.linspace(0.1 * h, 0.9 * h, 41)
    X, Y = np.meshgrid(xs, ys)
    sc = vp_score(X, Y, e, tol_deg=2.0)
    i = int(np.argmax(sc))
    best = np.array([X.ravel()[i], Y.ravel()[i]])
    step = xs[1] - xs[0]
    # Refine on finer grids around the best point.
    for _ in range(4):
        off = np.linspace(-step, step, 9)
        X, Y = np.meshgrid(best[0] + off, best[1] + off)
        sc = vp_score(X, Y, e, tol_deg=1.0)
        i = int(np.argmax(sc))
        best = np.array([X.ravel()[i], Y.ravel()[i]])
        step = off[1] - off[0]
    return float(best[0]), float(best[1])


# --------------------------------------------------------------------------- back wall
def _ray_length(vp, dx, dy, w, h):
    """Distance from vp along unit direction (dx, dy) to the image border."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 1e-9, (w - 1 - vp[0]) / dx, np.where(dx < -1e-9, -vp[0] / dx, np.inf))
        ty = np.where(dy > 1e-9, (h - 1 - vp[1]) / dy, np.where(dy < -1e-9, -vp[1] / dy, np.inf))
    return np.minimum(tx, ty)


def ray_support(gx, gy, vp, cx, cy, n: int = 48, slack: int = 2) -> np.ndarray:
    """Score room-edge rays that start at corners (cx, cy) and run away from
    the VP to the image border.

    The score is the median edge strength across the ray along that outer
    segment, times sqrt(segment length / full ray length). A real room edge
    is a consistent boundary from the back-wall corner all the way out. A
    corner placed too close to the VP adds edge-free samples, which drag the
    median down, and one placed too far out is penalised by the shorter
    segment.
    """
    h, w = gx.shape
    cx, cy = np.broadcast_arrays(np.asarray(cx, np.float32), np.asarray(cy, np.float32))
    shape = cx.shape
    cx, cy = cx.ravel(), cy.ravel()
    ddx, ddy = cx - vp[0], cy - vp[1]
    d0 = np.hypot(ddx, ddy) + 1e-6
    ux, uy = ddx / d0, ddy / d0
    L = _ray_length(vp, ux, uy, w, h)
    frac = np.clip((L - d0) / L, 0, 1)
    t = d0[:, None] + np.linspace(0, 1, n)[None, :] * np.maximum(L - d0, 0)[:, None]
    x, y = vp[0] + t * ux[:, None], vp[1] + t * uy[:, None]
    nx, ny = -uy[:, None], ux[:, None]
    # Allow the true edge to sit up to `slack` px to either side of the ray
    # (absorbs small vanishing-point errors and lens distortion).
    g = np.zeros_like(x)
    for o in range(-slack, slack + 1):
        xo, yo = x + o * nx, y + o * ny
        g = np.maximum(g, np.abs(_sample(gx, xo, yo) * nx + _sample(gy, xo, yo) * ny))
    return (np.median(g, axis=1) * np.sqrt(frac)).reshape(shape)


def _band_max(a: np.ndarray, axis: int, r: int = 1) -> np.ndarray:
    """Max over +-r neighbours along an axis (tolerates off-by-one edges)."""
    out = a.copy()
    for k in range(1, r + 1):
        out = np.maximum(out, np.roll(a, k, axis=axis))
        out = np.maximum(out, np.roll(a, -k, axis=axis))
    return out


def find_back_wall(gx, gy, vp, step: int = 4, min_frac: float = 0.06):
    """Exhaustive search for the back-wall rectangle [x1, y1, x2, y2].

    score = support of the four room-edge rays leaving its corners
          + mean horizontal-edge strength along its top and bottom edges
          + 0.5 * mean vertical-edge strength along its sides.
    """
    h, w = gx.shape
    vx, vy = vp
    m = int(min_frac * min(h, w))
    X1 = np.arange(2, int(vx) - m, step)
    X2 = np.arange(int(vx) + m, w - 2, step)
    Y1 = np.arange(2, int(vy) - m, step)
    Y2 = np.arange(int(vy) + m, h - 2, step)
    if min(len(X1), len(X2), len(Y1), len(Y2)) == 0:
        raise ValueError("vanishing point too close to the image border for a back wall")

    # Ray terms on 2D grids of corners.
    R_bl = ray_support(gx, gy, vp, X1[:, None], Y2[None, :])  # (x1, y2)
    R_br = ray_support(gx, gy, vp, X2[:, None], Y2[None, :])  # (x2, y2)
    R_tl = ray_support(gx, gy, vp, X1[:, None], Y1[None, :])  # (x1, y1)
    R_tr = ray_support(gx, gy, vp, X2[:, None], Y1[None, :])  # (x2, y1)

    # Edge terms by running sums along rows (horizontal edges) and columns.
    agy = _band_max(np.abs(gy), axis=0)
    agx = _band_max(np.abs(gx), axis=1)
    Ch = np.concatenate([np.zeros((h, 1)), np.cumsum(agy, axis=1)], axis=1)  # along x
    Cv = np.concatenate([np.zeros((1, w)), np.cumsum(agx, axis=0)], axis=0)  # along y

    def h_edge(Y):  # (x1, x2, y) mean |gy| on row y between x1 and x2
        return (Ch[Y[None, None, :], X2[None, :, None]] - Ch[Y[None, None, :], X1[:, None, None]]) / (
            X2[None, :, None] - X1[:, None, None])

    E_b = h_edge(Y2)  # (x1, x2, y2)
    E_t = h_edge(Y1)  # (x1, x2, y1)
    # Vertical sides: mean |gx| on column x between y1 and y2 -> (x, y1, y2)
    V_l = (Cv[Y2[None, None, :], X1[:, None, None]] - Cv[Y1[None, :, None], X1[:, None, None]]) / (
        Y2[None, None, :] - Y1[None, :, None])
    V_r = (Cv[Y2[None, None, :], X2[:, None, None]] - Cv[Y1[None, :, None], X2[:, None, None]]) / (
        Y2[None, None, :] - Y1[None, :, None])

    # Total score over (x1, x2, y1, y2).
    S = (
        R_bl[:, None, None, :] + R_br[None, :, None, :] + R_tl[:, None, :, None] + R_tr[None, :, :, None]
        + E_b[:, :, None, :] + E_t[:, :, :, None]
        + 0.5 * (V_l[:, None, :, :] + V_r[None, :, :, :])
    )
    k = int(np.argmax(S))
    i1, i2, j1, j2 = np.unravel_index(k, S.shape)
    return [float(X1[i1]), float(Y1[j1]), float(X2[i2]), float(Y2[j2])], float(S.ravel()[k])


def fit_edge_line(gx, gy, corner, theta0, span_deg: float = 6.0, n: int = 96):
    """Best line through `corner` within +-span_deg of direction theta0
    (pointing away from the VP), by mean edge strength across the line out
    to the image border. Returns the refined angle."""
    h, w = gx.shape
    thetas = theta0 + np.radians(np.linspace(-span_deg, span_deg, 121))
    ux, uy = np.cos(thetas), np.sin(thetas)
    L = _ray_length(corner, ux, uy, w, h)
    t = np.linspace(0, 1, n)[None, :] * L[:, None]
    x, y = corner[0] + t * ux[:, None], corner[1] + t * uy[:, None]
    nx, ny = -uy[:, None], ux[:, None]
    g = np.abs(_sample(gx, x, y) * nx + _sample(gy, x, y) * ny)
    return float(thetas[np.argmax(np.median(g, axis=1))])


def intersect_lines(points, thetas):
    """Least-squares intersection of lines (point, direction angle)."""
    A, b = [], []
    for (px, py), th in zip(points, thetas):
        n = np.array([-np.sin(th), np.cos(th)])
        A.append(n)
        b.append(n @ np.array([px, py]))
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return float(sol[0]), float(sol[1])


def refine_vp_from_corners(gx, gy, vp, rect):
    """Fit the four room-edge lines through the back-wall corners and return
    their intersection, a sharper VP than edge voting."""
    x1, y1, x2, y2 = rect
    corners = {"bl": (x1, y2), "br": (x2, y2), "tl": (x1, y1), "tr": (x2, y1)}
    pts, ths = [], []
    for c in corners.values():
        th0 = np.arctan2(c[1] - vp[1], c[0] - vp[0])
        pts.append(c)
        ths.append(fit_edge_line(gx, gy, c, th0))
    new = intersect_lines(pts, ths)
    # Keep the VP inside the back wall.
    return (float(np.clip(new[0], x1 + 2, x2 - 2)), float(np.clip(new[1], y1 + 2, y2 - 2)))


def rays_from_rect(vp, rect) -> dict:
    """Angles (radians, image coords) of the four room-edge rays through the corners."""
    x1, y1, x2, y2 = rect
    a = lambda x, y: float(np.arctan2(y - vp[1], x - vp[0]))  # noqa: E731
    return {"bl": a(x1, y2), "br": a(x2, y2), "tl": a(x1, y1), "tr": a(x2, y1)}


# --------------------------------------------------------------------------- main entry
@dataclass
class AutoResult:
    vanishing_point: tuple[float, float]
    back_wall: list[float]
    rays: dict  # quadrant -> angle in radians (image coords)


def auto_annotate_array(image: np.ndarray, max_side: int = 640, seed: int = 0) -> AutoResult:
    """Run the detector on an RGB/gray float image; returns full-res coords."""
    small, s = downscale(image, max_side)
    gray = to_gray(small)
    vp = find_vanishing_point(gray, seed=seed)
    gx, gy = gradients(gray, sigma=1.0)
    # The edge-vote VP leans toward wherever most texture is (often the far
    # end of a corridor). Refine it jointly with the back wall: the room's
    # four big junction lines are the most reliable evidence.
    best = None
    for r in (6, 2):
        center = vp if best is None else best[0]
        for dx in range(-r * 2, r * 2 + 1, r):
            for dy in range(-r * 2, r * 2 + 1, r):
                cand = (center[0] + dx, center[1] + dy)
                bw, score = find_back_wall(gx, gy, cand)
                if best is None or score > best[2]:
                    best = (cand, bw, score)
    vp, bw, _ = best
    vp = refine_vp_from_corners(gx, gy, vp, bw)
    rays = rays_from_rect(vp, bw)
    inv = 1.0 / s
    # Map back to full-res pixel coordinates (pixel centers).
    to_full = lambda v: (v + 0.5) * inv - 0.5  # noqa: E731
    vp_full = (round(to_full(vp[0]), 1), round(to_full(vp[1]), 1))
    bw_full = [round(to_full(v), 1) for v in bw]
    return AutoResult(vp_full, bw_full, rays)


def auto_annotate(image_path: str, image: np.ndarray | None = None, **kw) -> Annotation:
    if image is None:
        from .textures import load_image

        image = load_image(image_path)
    r = auto_annotate_array(image, **kw)
    return Annotation(image=str(image_path), vanishing_point=r.vanishing_point, back_wall=tuple(r.back_wall))
