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
   The VP is then sharpened by tracking the four room edges out from the
   back-wall corners and intersecting straight lines RANSAC-fit to them.
3. Deeper boxes (find_deeper_boxes). Dynamic programming over where each
   box ends, with no fixed spacing. A box gains the edge evidence along its
   four floor/ceiling/wall junctions (over the exact stretch where they'd
   run) plus the edges of its end rectangle, minus a fixed cost per box. So
   the boxes split only where a real boundary is visible.
4. Room-aligned objects (find_room_objects): cabinets, desks and other
   boxes on the floor, square to the room. Their front face is an upright
   rectangle and their side/top edges run toward the VP, ending where the
   object does.

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
def _ray_length(vp, dx, dy, w, h, bound=None):
    """Distance from vp along unit direction (dx, dy) to the image border, or
    to the edge of `bound` = (x1, y1, x2, y2) if given."""
    bx1, by1, bx2, by2 = bound if bound is not None else (0, 0, w - 1, h - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 1e-9, (bx2 - vp[0]) / dx, np.where(dx < -1e-9, (bx1 - vp[0]) / dx, np.inf))
        ty = np.where(dy > 1e-9, (by2 - vp[1]) / dy, np.where(dy < -1e-9, (by1 - vp[1]) / dy, np.inf))
    return np.minimum(tx, ty)


def ray_support(gx, gy, vp, cx, cy, n: int = 48, slack: int = 2, bound=None) -> np.ndarray:
    """Score room-edge rays that start at corners (cx, cy) and run away from
    the VP to the image border (or to the edge of `bound`).

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
    L = _ray_length(vp, ux, uy, w, h, bound)
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


def _edge_terms(gx, gy, X1, X2, Y1, Y2):
    """Mean edge strength along candidate rectangle sides, via running sums.

    Returns E_b (x1, x2, y2) and E_t (x1, x2, y1), the mean |d/dy| along the
    bottom and top edges, and V_l (x1, y1, y2) and V_r (x2, y1, y2), the mean
    |d/dx| along the sides."""
    h, w = gx.shape
    agy = _band_max(np.abs(gy), axis=0)
    agx = _band_max(np.abs(gx), axis=1)
    Ch = np.concatenate([np.zeros((h, 1)), np.cumsum(agy, axis=1)], axis=1)  # along x
    Cv = np.concatenate([np.zeros((1, w)), np.cumsum(agx, axis=0)], axis=0)  # along y

    def h_edge(Y):
        return (Ch[Y[None, None, :], X2[None, :, None]] - Ch[Y[None, None, :], X1[:, None, None]]) / (
            X2[None, :, None] - X1[:, None, None])

    def v_edge(X):
        return (Cv[Y2[None, None, :], X[:, None, None]] - Cv[Y1[None, :, None], X[:, None, None]]) / (
            Y2[None, None, :] - Y1[None, :, None])

    return h_edge(Y2), h_edge(Y1), v_edge(X1), v_edge(X2)


def _search_rect(gx, gy, vp, X1, X2, Y1, Y2, bound=None, valid=None, extra=None):
    """Exhaustive search over rectangles with x1 in X1, x2 in X2, y1 in Y1 and
    y2 in Y2:

        score = support of the four room-edge rays leaving its corners
              + mean horizontal-edge strength along its top and bottom edges
              + 0.5 * mean vertical-edge strength along its sides.

    valid(x1, x2, y1, y2) -> bool array (broadcast 4D) can exclude candidates.
    Returns (rect, score), or (None, -inf) if nothing is allowed.
    """
    h, w = gx.shape
    if min(len(X1), len(X2), len(Y1), len(Y2)) == 0:
        return None, -np.inf
    # Ray terms on 2D grids of corners.
    R_bl = ray_support(gx, gy, vp, X1[:, None], Y2[None, :], bound=bound)  # (x1, y2)
    R_br = ray_support(gx, gy, vp, X2[:, None], Y2[None, :], bound=bound)  # (x2, y2)
    R_tl = ray_support(gx, gy, vp, X1[:, None], Y1[None, :], bound=bound)  # (x1, y1)
    R_tr = ray_support(gx, gy, vp, X2[:, None], Y1[None, :], bound=bound)  # (x2, y1)

    E_b, E_t, V_l, V_r = _edge_terms(gx, gy, X1, X2, Y1, Y2)

    # Total score over (x1, x2, y1, y2).
    S = (
        R_bl[:, None, None, :] + R_br[None, :, None, :] + R_tl[:, None, :, None] + R_tr[None, :, :, None]
        + E_b[:, :, None, :] + E_t[:, :, :, None]
        + 0.5 * (V_l[:, None, :, :] + V_r[None, :, :, :])
    )
    if extra is not None:  # further per-rectangle terms, same 4D shape
        S = S + extra
    if valid is not None:
        ok = valid(X1[:, None, None, None], X2[None, :, None, None], Y1[None, None, :, None], Y2[None, None, None, :])
        S = np.where(ok, S, -np.inf)
    k = int(np.argmax(S))
    if not np.isfinite(S.ravel()[k]):
        return None, -np.inf
    i1, i2, j1, j2 = np.unravel_index(k, S.shape)
    return [float(X1[i1]), float(Y1[j1]), float(X2[i2]), float(Y2[j2])], float(S.ravel()[k])


def split_gain_terms(lab: np.ndarray, vp, X1, X2, Y1, Y2, n_theta: int = 720) -> np.ndarray:
    """How well each candidate back wall splits the image into room surfaces.

    Every room surface but the back wall runs along the depth axis (a wall's
    paint and baseboard, the floor, the ceiling), so along a ray from the VP
    its colour changes little, until the ray crosses the back wall's edge.
    So each ray is fitted with two constant pieces, back wall up to the
    rectangle and side surface beyond, and the score is the colour variance
    that this split explains, averaged over rays (Lab^2 units). A rectangle
    hugging a piece of furniture splits the rays where nothing changes, and
    puts real wall corners in the middle of a piece.

    Each ray's split depends only on the side of the rectangle it hits and
    that side's position, so the gains are tabulated per ray and side
    position, and summed over the angular sector of each side with
    cumulative sums. Returns a (len(X1), len(X2), len(Y1), len(Y2)) array."""
    import cv2

    h, w = lab.shape[:2]
    vx, vy = vp
    rmax = float(np.hypot(max(vx, w - vx), max(vy, h - vy))) + 2
    th = -np.pi + (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta
    rr = np.arange(0.0, rmax, 1.0)
    U = (vx + np.cos(th)[:, None] * rr[None]).astype(np.float32)
    V = (vy + np.sin(th)[:, None] * rr[None]).astype(np.float32)
    P = cv2.remap(lab.astype(np.float32), U, V, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (U >= 0) & (U <= w - 1) & (V >= 0) & (V <= h - 1)
    P = np.where(valid[..., None], P, 0.0)
    cs = lambda a: np.concatenate([np.zeros((n_theta, 1) + a.shape[2:]), np.cumsum(a, axis=1)], axis=1)  # noqa: E731
    N, S, Q = cs(valid.astype(float)), cs(P.astype(float)), cs((P.astype(float) ** 2).sum(-1))
    R = N.shape[1] - 1
    k = np.arange(n_theta)[:, None]

    def sse(a, b):
        n = N[k, b] - N[k, a]
        sm = S[k, b] - S[k, a]
        return Q[k, b] - Q[k, a] - (sm ** 2).sum(-1) / np.maximum(n, 1)

    whole = sse(np.zeros((n_theta, 1), int), np.full((n_theta, 1), R))
    n_all = np.maximum(N[:, -1:], 1)
    c, sn = np.cos(th)[:, None], np.sin(th)[:, None]

    def gains(pos, along_x: bool):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (pos[None, :] - (vx if along_x else vy)) / (c if along_x else sn)
        rb = np.clip(np.round(np.where(np.isfinite(r) & (r > 0), r, R)), 0, R).astype(int)
        g = (whole - sse(np.zeros_like(rb), rb) - sse(rb, np.full_like(rb, R))) / n_all
        return np.concatenate([np.zeros((1, len(pos))), np.cumsum(g, axis=0)], axis=0)  # cumulative over angle

    CL, CR, CT, CB = gains(X1.astype(float), True), gains(X2.astype(float), True), gains(Y1.astype(float), False), gains(Y2.astype(float), False)

    def idx(x, y):  # number of ray angles below the corner's angle
        a = np.arctan2(y - vy, x - vx)
        return np.clip(np.floor((a + np.pi) / (2 * np.pi) * n_theta + 0.5).astype(int), 0, n_theta)

    x1, x2 = X1[:, None, None, None].astype(float), X2[None, :, None, None].astype(float)
    y1, y2 = Y1[None, None, :, None].astype(float), Y2[None, None, None, :].astype(float)
    i1 = np.arange(len(X1))[:, None, None, None]
    i2 = np.arange(len(X2))[None, :, None, None]
    j1 = np.arange(len(Y1))[None, None, :, None]
    j2 = np.arange(len(Y2))[None, None, None, :]
    tl, tr, br, bl = idx(x1, y1), idx(x2, y1), idx(x2, y2), idx(x1, y2)
    # Angles grow clockwise on screen (y down): TL (~-3pi/4) -> TR -> BR -> BL (~+3pi/4).
    top = CT[tr, j1] - CT[tl, j1]
    right = CR[br, i2] - CR[tr, i2]
    bottom = CB[bl, j2] - CB[br, j2]
    left = CL[n_theta, i1] - CL[bl, i1] + CL[tl, i1]
    return ((top + right + bottom + left) / n_theta).astype(np.float32)


def find_back_wall(gx, gy, vp, step: int = 4, min_frac: float = 0.06, lab: np.ndarray | None = None,
                   split_weight: float = 3e-4, max_bottom: float | None = None):
    """Best back-wall rectangle [x1, y1, x2, y2] anywhere in the image, and its score.

    With lab (the Lab image), the edge score is joined by split_weight times
    split_gain_terms: how well the rectangle splits the rays from the VP into
    back wall and room surfaces. Edges place the wall precisely where the
    room's corners are crisp; the split finds it where they are soft (paintings),
    against crisp furniture outlines.

    max_bottom: the bottom edge must be at or above this row (the floor is
    seen up to there, e.g. its tiles, and the back wall can't hold floor)."""
    h, w = gx.shape
    vx, vy = vp
    m = int(min_frac * min(h, w))
    X1 = np.arange(2, int(vx) - m, step)
    X2 = np.arange(int(vx) + m, w - 2, step)
    Y1 = np.arange(2, int(vy) - m, step)
    Y2 = np.arange(int(vy) + m, h - 2, step)
    if max_bottom is not None and (Y2 <= max_bottom).any():
        Y2 = Y2[Y2 <= max_bottom]
    extra = None
    if lab is not None and min(len(X1), len(X2), len(Y1), len(Y2)) > 0:
        extra = split_weight * split_gain_terms(lab, vp, X1, X2, Y1, Y2)
    rect, score = _search_rect(gx, gy, vp, X1, X2, Y1, Y2, extra=extra)
    if rect is None:
        raise ValueError("vanishing point too close to the image border for a back wall")
    return rect, score


# --------------------------------------------------------------------------- deeper boxes
def _box_shape(rect, vp):
    """3D width and height of a box ending at `rect`, up to the common scale
    f / camera_height: a rectangle whose bottom is d px below the VP sits at
    depth proportional to 1/d, so its size is (pixels / d)."""
    x1, y1, x2, y2 = rect
    d = y2 - vp[1]
    return (x2 - x1) / d, (y2 - y1) / d


def _ray_prefix(gx, gy, vp, xs, ys, slack: int = 2):
    """Running sums of edge evidence along the outward ray from every corner
    (x, y) on the grid xs x ys, in 1 px steps out to the image border.

    Evidence at a sample is the edge strength across the ray (best within
    +-slack px), divided by a high percentile and capped at 1, so a solid
    room edge contributes about 1 per pixel of length. Returns
    (prefix[ix, iy, k] = evidence over the first k px, d0 = corner distance
    from the VP, room = px available before the border)."""
    h, w = gx.shape
    cap = np.percentile(np.hypot(gx, gy), 97) + 1e-6
    X, Y = np.meshgrid(np.asarray(xs, np.float32), np.asarray(ys, np.float32), indexing="ij")
    dx, dy = X - vp[0], Y - vp[1]
    d0 = np.hypot(dx, dy) + 1e-6
    ux, uy = dx / d0, dy / d0
    room = np.maximum(_ray_length(vp, ux, uy, w, h) - d0, 0)
    t = np.arange(int(np.ceil(room.max())) + 1, dtype=np.float32)
    x, y = X[..., None] + t * ux[..., None], Y[..., None] + t * uy[..., None]
    nx, ny = -uy[..., None], ux[..., None]
    g = np.zeros_like(x)
    for o in range(-slack, slack + 1):
        xo, yo = x + o * nx, y + o * ny
        g = np.maximum(g, np.abs(_sample(gx, xo, yo) * nx + _sample(gy, xo, yo) * ny))
    g = np.minimum(g / cap, 1.0) * (t <= room[..., None])
    prefix = np.concatenate([np.zeros(g.shape[:2] + (1,), np.float32), np.cumsum(g, axis=-1)], axis=-1)
    return prefix, d0, room


def find_deeper_boxes(gx, gy, vp, back_wall, step: int = 2, min_px: int = 6, ratio_range=(1.05, 12.0),
                      shape_range=(0.7, 1.1), box_cost: float = 0.6, edge_norm: float = 50.0,
                      bottom_min: float = 0.3, side_min: float = 0.2, min_inset: int = 3):
    """Chain of further boxes behind the back wall, nearest first, by dynamic
    programming over where the boxes end. No fixed spacing between boxes.

    A box ending at depth Z_k starts at the previous box's depth Z_{k-1}, so
    each of its four floor/ceiling/wall junctions runs from its far corner
    outward to Z_k / Z_{k-1} times as far from the VP. A box's gain is

        junction evidence summed along those four segments / edge_norm
      + edge evidence along its end rectangle (as for the back wall)
      - box_cost.

    Junction evidence is a sum over length, so splitting a uniform corridor
    gains nothing. A split pays off only where a real boundary (door frame,
    step, end wall) adds edge evidence above box_cost.

    DP over candidate bottom rows y (near to far): best[y] = the best total
    gain of a chain whose last box ends at row y. It comes from any nearer
    box end whose depth ratio is within ratio_range, and that box's
    cross-section is fitted given the previous box's. Each box must contain
    the VP, lie inside the previous box, and keep a 3D width, height and
    aspect within shape_range of it (at most 30% smaller or 10% bigger:
    rooms seen through a doorway are rarely bigger than the room in front).
    That rules out door panels and posters. A box must also end at a visible
    floor boundary (mean edge strength >= bottom_min along its bottom edge)
    and visible sides (>= side_min),
    be at least 5% deeper than the previous box, and be inset at least
    min_inset px from it on every side. That drops paper-thin boxes, which
    can't be told apart from the edge of an opening.
    The answer is the best chain overall, which may be empty.
    """
    vx, vy = vp
    bx1, by1, bx2, by2 = back_wall
    xs = np.arange(int(np.ceil(bx1)) + 1, int(bx2), step)
    ys = np.arange(int(np.ceil(by1)) + 1, int(by2), step)
    if len(xs) < 2 or len(ys) < 2:
        return [], []
    prefix, d0, room = _ray_prefix(gx, gy, vp, xs, ys)
    ix_of = {int(v): i for i, v in enumerate(xs)}
    iy_of = {int(v): i for i, v in enumerate(ys)}
    agy = _band_max(np.abs(gy), axis=0)
    agx = _band_max(np.abs(gx), axis=1)
    cap = np.percentile(np.hypot(gx, gy), 97) + 1e-6
    Ch = np.concatenate([np.zeros((gy.shape[0], 1)), np.cumsum(agy, axis=1)], axis=1) / cap
    Cv = np.concatenate([np.zeros((1, gx.shape[1])), np.cumsum(agx, axis=0)], axis=0) / cap

    def seg(ix, iy, ratio):
        """Junction evidence from corner (xs[ix], ys[iy]) out to `ratio` x its VP distance."""
        k = np.clip(np.round((ratio - 1) * d0[ix, iy]), 0, room[ix, iy]).astype(int)
        return prefix[ix, iy, k]

    def best_box(prev, y2):
        """Best (gain, rect) for a box ending at row y2 behind box `prev`."""
        ratio = (prev[3] - vy) / (y2 - vy)
        if not ratio_range[0] <= ratio <= ratio_range[1] or prev[3] - y2 < min_inset:
            return -np.inf, None
        # Seen through the previous box's opening, the next box must be
        # visibly smaller on every side (by more than an edge's blur width),
        # so near-duplicate boxes can't re-claim the same edges.
        X1 = np.array([x for x in xs if prev[0] + min_inset <= x <= vx - min_px])
        X2 = np.array([x for x in xs if vx + min_px <= x <= prev[2] - min_inset])
        Y1 = np.array([y for y in ys if prev[1] + min_inset <= y <= vy - min_px])
        if min(len(X1), len(X2), len(Y1)) == 0:
            return -np.inf, None
        i1 = np.array([ix_of[int(x)] for x in X1])
        i2 = np.array([ix_of[int(x)] for x in X2])
        j1 = np.array([iy_of[int(y)] for y in Y1])
        j2 = iy_of[int(y2)]
        x1, x2, y1 = X1[:, None, None], X2[None, :, None], Y1[None, None, :]
        junction = (seg(i1, j2, ratio)[:, None, None] + seg(i2, j2, ratio)[None, :, None]
                    + seg(i1[:, None], j1[None, :], ratio)[:, None, :]
                    + seg(i2[:, None], j1[None, :], ratio)[None, :, :])
        bottom = (Ch[y2, x2] - Ch[y2, x1]) / (x2 - x1)
        sides = 0.5 * ((Cv[y2, x1] - Cv[y1, x1]) + (Cv[y2, x2] - Cv[y1, x2])) / (y2 - y1)
        edges = bottom + (Ch[y1, x2] - Ch[y1, x1]) / (x2 - x1) + sides
        gain = junction / edge_norm + edges - box_cost
        # A box end must show up as a rectangle: a visible floor boundary
        # (threshold, step, baseboard or end wall) and visible side edges
        # (frame or wall corners). The bottom is the only edge tied to depth,
        # since it's a floor row. Without it, a box's sides and top could
        # borrow deeper frames' edges at any depth. Without sides, a tile
        # grout line on the floor looks like a box end.
        gain = np.where((bottom >= bottom_min) & (sides >= side_min), gain, -np.inf)
        # Shape relative to the previous box (3D size ~ pixels / depth-row offset).
        pw, ph = _box_shape(prev, vp)
        d = y2 - vy
        w3, h3 = (x2 - x1) / d, (y2 - y1) / d
        lo, hi = shape_range
        ok = (w3 > lo * pw) & (w3 < hi * pw) & (h3 > lo * ph) & (h3 < hi * ph)
        ok &= (h3 / w3 > 0.75 * ph / pw) & (h3 / w3 < 1.33 * ph / pw)
        gain = np.where(ok, gain, -np.inf)
        k = int(np.argmax(gain))
        if not np.isfinite(gain.ravel()[k]):
            return -np.inf, None
        a, b, c = np.unravel_index(k, gain.shape)
        return float(gain.ravel()[k]), [float(X1[a]), float(Y1[c]), float(X2[b]), float(y2)]

    # States: (total gain, rect, index of previous state). State 0 is the back wall.
    states = [(0.0, list(back_wall), -1)]
    for y2 in sorted((int(y) for y in ys if vy + min_px <= y < by2), reverse=True):
        best = (-np.inf, None, -1)
        for i, (total, rect, _) in enumerate(states):
            if rect[3] <= y2:
                continue
            g, r = best_box(rect, y2)
            if r is not None and total + g > best[0]:
                best = (total + g, r, i)
        if best[1] is not None:
            states.append(best)

    # Best chain overall (state 0 = no deeper boxes), traced back.
    i = int(np.argmax([s[0] for s in states]))
    chain = []
    while i > 0:
        chain.append(states[i])
        i = states[i][2]
    chain.reverse()
    gains = [c[0] - (states[c[2]][0]) for c in chain]
    return [c[1] for c in chain], gains


def find_opening(gx, gy, vp, back_wall, inner=None, step: int = 2, min_px: int = 6, min_lintel: float = 0.05,
                 top_min: float = 0.35, side_min: float = 0.25, thickness: float = 0.03,
                 near_best: float = 0.9):
    """A doorway in the back wall: the opening the deeper boxes are seen
    through, when it is clearly smaller than the wall (a lintel above it,
    possibly jambs beside it). The deeper-box search can't find it: a door
    frame is only a wall's thickness deep, and boxes that thin are dropped.

    The opening is a rectangle [x1, y1, x2, by2] that contains the VP and
    the first deeper box (inner), and whose bottom is the back wall's
    bottom (the threshold). Its top must sit at least min_lintel of the
    wall's height below the wall's top, with a strong horizontal edge along
    it (mean >= top_min, as in find_deeper_boxes), and its sides need
    vertical edges from the top down to the floor (mean >= side_min).
    Among valid rectangles scoring at least near_best x the best edge
    evidence, the largest wins (the frame, not door leaves inside it).

    Returns the door-frame box as a rect for `segments` (its bottom moved
    toward the VP so the box is `thickness` x the room's depth deep), and
    its score; or (None, 0) if there is no clear opening."""
    vx, vy = vp
    bx1, by1, bx2, by2 = back_wall
    ix1, iy1, ix2 = (inner[0], inner[1], inner[2]) if inner is not None else (vx - min_px, vy - min_px, vx + min_px)
    # Jambs stand clear of the wall's own corners (a corner is no jamb).
    margin = 0.03 * (bx2 - bx1)
    X1 = np.arange(int(np.ceil(bx1 + margin)), int(min(vx - min_px, ix1)) + 1, step)
    X2 = np.arange(int(np.ceil(max(vx + min_px, ix2))), int(bx2 - margin) + 1, step)
    Y1 = np.arange(int(np.ceil(by1 + min_lintel * (by2 - by1))), int(min(vy - min_px, iy1)) + 1, step)
    if min(len(X1), len(X2), len(Y1)) == 0:
        return None, 0.0
    y2 = int(round(by2))
    agy = _band_max(np.abs(gy), axis=0)
    agx = _band_max(np.abs(gx), axis=1)
    cap = np.percentile(np.hypot(gx, gy), 97) + 1e-6
    Ch = np.concatenate([np.zeros((gy.shape[0], 1)), np.cumsum(agy, axis=1)], axis=1) / cap
    Cv = np.concatenate([np.zeros((1, gx.shape[1])), np.cumsum(agx, axis=0)], axis=0) / cap
    x1, x2, y1 = X1[:, None, None], X2[None, :, None], Y1[None, None, :]
    top = (Ch[y1, x2] - Ch[y1, x1]) / (x2 - x1)
    left = (Cv[y2, x1] - Cv[y1, x1]) / (y2 - y1)
    right = (Cv[y2, x2] - Cv[y1, x2]) / (y2 - y1)
    score = top + 0.5 * (left + right)
    score = np.where((top >= top_min) & (left >= side_min) & (right >= side_min), score, -np.inf)
    best = score.max()
    if not np.isfinite(best):
        return None, 0.0
    # Several nested rectangles can qualify (a frame's inner edge, open door
    # leaves standing in it): take the largest one that scores nearly as well.
    area = np.broadcast_to((x2 - x1) * (y2 - y1), score.shape).astype(float)
    k = int(np.argmax(np.where(score >= near_best * best, area, -1.0)))
    a, b, c = np.unravel_index(k, score.shape)
    # A box thickness x the room's depth deep ends at this floor row.
    bottom = vy + (by2 - vy) / (1.0 + thickness)
    return [float(X1[a]), float(Y1[c]), float(X2[b]), float(bottom)], float(score.ravel()[k])


def _seen_through_opening(back_wall, first, min_band: float = 0.08) -> bool:
    """True if the first deeper box leaves a clear band of back wall around
    it (at least min_band of the wall's width or height on some side
    besides the bottom), so the back wall really has an opening in it."""
    bx1, by1, bx2, by2 = back_wall
    x1, y1, x2, _ = first
    W, H = bx2 - bx1, by2 - by1
    return max((x1 - bx1) / W, (bx2 - x2) / W, (y1 - by1) / H) >= min_band


# --------------------------------------------------------------------------- room-aligned objects
def _closing(m: np.ndarray, axis: int, r: int) -> np.ndarray:
    """1D morphological closing (fills gaps up to 2r px) along an axis."""
    d = m.copy()
    for k in range(1, r + 1):
        d |= np.roll(m, k, axis=axis) | np.roll(m, -k, axis=axis)
    e = d.copy()
    for k in range(1, r + 1):
        e &= np.roll(d, k, axis=axis) & np.roll(d, -k, axis=axis)
    return e


def _vertical_segments(vmask: np.ndarray, min_len: int):
    """Runs of True down each column: arrays (x, y_top, y_bottom)."""
    m = np.pad(vmask.T, ((0, 0), (1, 1))).astype(np.int8)  # (w, h+2), column-major order
    d = np.diff(m, axis=1)
    xs, ys = np.nonzero(d == 1)
    _, ye = np.nonzero(d == -1)
    keep = ye - ys >= min_len
    return xs[keep], ys[keep], ye[keep] - 1


def find_room_objects(gx, gy, vp, rects, min_size: int = 10, end_tol: int = 4, cover_min: float = 0.7,
                      recede_min: float = 0.7, min_recede_px: float = 6.0, max_objects: int = 8):
    """Room-aligned boxes (cabinets, desks, shelves) standing on the floor.

    In one-point perspective such an object's front face is an axis-aligned
    rectangle (vertical sides, horizontal top and bottom), and the edges of
    its visible side face (and top, if below eye level) run toward the VP,
    stopping where the object ends. That separates it from:
      - things on the side walls (posters, doors): their top and bottom
        edges slant toward the VP instead of being horizontal;
      - things flat on the back wall: no receding edges;
      - the room's own boxes: they contain the VP.

    1. Vertical edge runs, paired into front-face candidates with matching
       ends and horizontal edges along the top and bottom.
    2. The bottom must be below the VP (on the floor) and inside the room
       box at that depth.
    3. Receding edges (only edges running along the ray count) are followed
       from the front corners toward the VP. Each edge ends where its
       evidence runs out, and the object ends at the median of those ends.
       Every edge must be mostly present up to there (with 3+ edges, one may
       be weaker, e.g. against a background of similar colour).
    rects: the room boxes' far-end rectangles, nearest first.
    Returns [(front [x1, y1, x2, y2], back_y, score)], best first.
    """
    h, w = gx.shape
    vx, vy = vp
    agx, agy = np.abs(gx), np.abs(gy)
    mag = np.hypot(gx, gy)
    # Half the VP detector's threshold: furniture edges are often low-contrast,
    # and the structure required below (paired verticals, horizontal top and
    # bottom, receding edges) keeps noise out.
    thresh = 0.5 * max(np.percentile(mag, 90), 0.2 * np.percentile(mag, 99.5))
    # Edge pixels by orientation, within ~18 deg (vertical edges have a horizontal gradient).
    vm = (agx > thresh) & (agx > 3 * agy)
    hm = (agy > thresh) & (agy > 3 * agx)
    vm = _closing(_band_max(vm, axis=1), axis=0, r=1)
    hm = _closing(_band_max(hm, axis=0), axis=1, r=1)
    cum0 = lambda a: np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)], axis=0)  # noqa: E731
    cum1 = lambda a: np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)  # noqa: E731
    Cv, Ch, Gv, Gh = cum0(vm), cum1(hm), cum0(agx), cum1(agy)
    vcov = lambda x, a, b: (Cv[b + 1, x] - Cv[a, x]) / np.maximum(b + 1 - a, 1)  # noqa: E731
    hcov = lambda y, a, b: (Ch[y, b + 1] - Ch[y, a]) / np.maximum(b + 1 - a, 1)  # noqa: E731

    X, YT, YB = _vertical_segments(vm, min_size)
    if len(X) < 2:
        return []
    # Pair every two runs that overlap over most of their length.
    i, j = np.triu_indices(len(X), k=1)
    a, b = np.where(X[i] < X[j], i, j), np.where(X[i] < X[j], j, i)
    top, bot = np.maximum(YT[a], YT[b]), np.minimum(YB[a], YB[b])
    ok = ((X[b] - X[a] >= min_size) & (bot - top >= min_size)
          & (np.abs(YT[a] - YT[b]) <= 3 * end_tol) & (np.abs(YB[a] - YB[b]) <= 3 * end_tol))
    x1, x2, y1, y2 = X[a][ok], X[b][ok], top[ok], bot[ok]
    if len(x1) == 0:
        return []
    r = np.arange(-2 * end_tol, 2 * end_tol + 1)

    def snap_row(y, xa, xb):
        """Move y to the row nearby with the strongest horizontal edge along [xa, xb]."""
        c = np.clip(y[:, None] + r[None, :], 0, h - 1)
        k = np.argmax(Gh[c, xb[:, None] + 1] - Gh[c, xa[:, None]], axis=1)
        return c[np.arange(len(y)), k]

    def snap_col(x, ya, yb):
        c = np.clip(x[:, None] + r[None, :] // 2, 0, w - 1)
        k = np.argmax(Gv[yb[:, None] + 1, c] - Gv[ya[:, None], c], axis=1)
        return c[np.arange(len(x)), k]

    for _ in range(2):
        y2, y1 = snap_row(y2, x1, x2), snap_row(y1, x1, x2)
        x1, x2 = snap_col(x1, y1, y2), snap_col(x2, y1, y2)
    ok = ((y2 > vy + min_size) & (y2 - y1 >= min_size) & (x2 - x1 >= min_size)
          & ~((x1 < vx) & (vx < x2) & (y1 < vy) & (vy < y2)))
    x1, x2, y1, y2 = x1[ok], x2[ok], y1[ok], y2[ok]
    cb, ct, cl, cr = hcov(y2, x1, x2), hcov(y1, x1, x2), vcov(x1, y1, y2), vcov(x2, y1, y2)
    ok = (np.minimum(cb, ct) >= cover_min) & (np.minimum(cl, cr) >= cover_min)
    cands = np.stack([x1, y1, x2, y2], axis=1)[ok]
    fcov = ((cb + ct + cl + cr) / 4)[ok]
    if len(cands) == 0:
        return []
    cands, idx = np.unique(cands, axis=0, return_index=True)
    fcov = fcov[idx]

    # Room box at each candidate's floor contact, and its walls at that row.
    rows = np.array([r[3] for r in rects], float)
    out = []
    for (x1, y1, x2, y2), fc in zip(cands.astype(float), fcov):
        k = int(np.sum(rows >= y2))  # box index whose floor contains row y2
        if k >= len(rects):
            continue
        R = rects[k]
        s = (y2 - vy) / (R[3] - vy)
        wl, wr = vx + (R[0] - vx) * s, vx + (R[2] - vx) * s
        # Loose: the room box may itself be a few px off.
        tol = max(end_tol, 0.15 * (wr - wl))
        if x1 < wl - tol or x2 > wr + tol:
            continue
        # Visible receding edges: the side facing the VP, and the top if below eye level.
        corners = []
        if x2 < vx:
            corners += [(x2, y1), (x2, y2)]
        elif x1 > vx:
            corners += [(x1, y1), (x1, y2)]
        if y1 > vy:
            corners += [(x1, y1), (x2, y1)]
        corners = list(dict.fromkeys(corners))
        if not corners:
            continue
        # Sample the receding edges at common depth ratios: point = c + u (vp - c),
        # u up to the end of the room box.
        u_max = 1 - (R[3] - vy) / (y2 - vy)
        L = max(np.hypot(vx - cx, vy - cy) for cx, cy in corners) * u_max
        n = int(L)
        if n < min_recede_px:
            continue
        u = (np.arange(n) + 0.5) / n * u_max
        ev = []
        for cx, cy in corners:
            ddx, ddy = vx - cx, vy - cy
            nl = np.hypot(ddx, ddy)
            nx, ny = -ddy / nl, ddx / nl
            px, py = cx + u * ddx, cy + u * ddy
            hit = np.zeros(n, bool)
            for o in (-1.5, -0.75, 0, 0.75, 1.5):
                sx, sy = _sample(gx, px + o * nx, py + o * ny), _sample(gy, px + o * nx, py + o * ny)
                across = np.abs(sx * nx + sy * ny)
                along = np.abs(sx * ddx + sy * ddy) / nl
                # An edge running along the ray (within ~14 deg), not just crossing it.
                hit |= (across > thresh) & (along < 0.25 * across)
            ev.append(hit.astype(float))
        ev = np.array(ev)
        # Each edge ends where its evidence runs out (best running sum of
        # evidence - 0.5); the object ends at the median of those ends.
        # Within a blur radius of a back corner the gradient mixes two edges
        # and fails the orientation test, so add one sample back.
        ends = np.argmax(np.cumsum(ev - 0.5, axis=1), axis=1)
        m = min(int(np.median(ends)) + 1, n - 1)
        u_end = u[m]
        lens = [np.hypot(vx - cx, vy - cy) * u_end for cx, cy in corners]
        cov = ev[:, : m + 1].mean(axis=1)
        # With 3+ edges, one may fade (e.g. against a background of similar colour).
        cov = np.sort(cov)
        weak_ok = cov[0] >= 0.4 if len(cov) >= 3 else cov[0] >= recede_min
        if min(lens) < min_recede_px or cov[-2] < recede_min or not weak_ok:
            continue
        back_y = vy + (y2 - vy) * (1 - u_end)
        out.append(([x1, y1, x2, y2], float(back_y), float(fc + cov.mean())))
    # Non-maximum suppression: drop candidates mostly overlapping a better one.
    out.sort(key=lambda o: -o[2])
    kept = []
    for o in out:
        a = o[0]
        area = (a[2] - a[0]) * (a[3] - a[1])
        if all(max(0, min(a[2], b[0][2]) - max(a[0], b[0][0])) * max(0, min(a[3], b[0][3]) - max(a[1], b[0][1]))
               < 0.3 * min(area, (b[0][2] - b[0][0]) * (b[0][3] - b[0][1])) for b in kept):
            kept.append(o)
    return kept[:max_objects]


def object_footprint(front, back_y, vp) -> np.ndarray:
    """Image outline (convex polygon) of a room-aligned object: its front
    rectangle plus the same rectangle shrunk toward the VP to its back depth."""
    x1, y1, x2, y2 = front
    k = (back_y - vp[1]) / (y2 - vp[1])
    pts = np.array([(x, y) for x in (x1, x2) for y in (y1, y2)], float)
    pts = np.concatenate([pts, vp + (pts - vp) * k])
    # Monotone-chain convex hull.
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(ps):
        out = []
        for q in ps:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], q - out[-2]) <= 0:
                out.pop()
            out.append(q)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def _rect_score(gx, gy, vp, rect, bound=None):
    """Score of one given rectangle (same formula as _search_rect)."""
    a = lambda v: np.array([int(round(v))])  # noqa: E731
    _, s = _search_rect(gx, gy, vp, a(rect[0]), a(rect[2]), a(rect[1]), a(rect[3]), bound=bound)
    return s


@dataclass
class EdgeLine:
    point: np.ndarray  # a point on the fitted line
    theta: float  # direction angle (image coords)
    inliers: float  # fraction of tracked samples within `tol` of the line
    span: float  # length covered by the inliers, px


def track_edge_line(gx, gy, corner, vp, search: float = 8.0, tol: float = 0.75, iters: int = 400, seed: int = 0):
    """Follow the room edge leaving `corner` (away from the VP) to the image
    border and fit a straight line to it.

    At every pixel step along the ray we take the strongest edge across it
    (within +-search px), then RANSAC a line through those samples, so clutter
    on the edge (an extinguisher on the floor line, a cable tray on the
    ceiling line) is ignored. Returns None if no clear edge is found.
    """
    h, w = gx.shape
    c = np.asarray(corner, float)
    d = c - np.asarray(vp, float)
    d /= np.linalg.norm(d) + 1e-9
    n = np.array([-d[1], d[0]])
    L = float(_ray_length(c, d[0:1], d[1:2], w, h, bound=(search + 2, search + 2, w - search - 3, h - search - 3))[0])
    if not L > 20:
        return None
    t = np.arange(2.0, L)
    offs = np.arange(-search, search + 0.01, 0.25)
    P = c + t[:, None, None] * d + offs[None, :, None] * n
    g = np.abs(_sample(gx, P[..., 0], P[..., 1]) * n[0] + _sample(gy, P[..., 0], P[..., 1]) * n[1])
    q = c + t[:, None] * d + offs[np.argmax(g, axis=1)][:, None] * n
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(q), 2, replace=False)
        dd = q[j] - q[i]
        nd = np.linalg.norm(dd)
        if nd < 0.25 * L:
            continue
        k = np.abs((q - q[i]) @ np.array([-dd[1], dd[0]]) / nd) < tol
        if best is None or k.sum() > best.sum():
            best = k
    if best is None or best.sum() < max(10, 0.15 * len(q)):
        return None
    m = q[best].mean(axis=0)
    _, _, vt = np.linalg.svd(q[best] - m)
    span = float(np.ptp((q[best] - m) @ vt[0]))
    return EdgeLine(m, float(np.arctan2(vt[0][1], vt[0][0])), float(best.mean()), span)


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
    """Track the four room-edge lines leaving the back-wall corners, fit a
    straight line to each, and return their least-squares intersection (a
    sharper VP than edge voting) along with the fitted lines."""
    x1, y1, x2, y2 = rect
    corners = {"bl": (x1, y2), "br": (x2, y2), "tl": (x1, y1), "tr": (x2, y1)}
    lines = {k: track_edge_line(gx, gy, c, vp) for k, c in corners.items()}
    good = [ln for ln in lines.values() if ln is not None]
    if len(good) < 2:
        return tuple(vp), lines
    new = intersect_lines([ln.point for ln in good], [ln.theta for ln in good])
    # Keep the VP inside the back wall.
    return (float(np.clip(new[0], x1 + 2, x2 - 2)), float(np.clip(new[1], y1 + 2, y2 - 2))), lines


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
    segments: list = None  # deeper boxes, nearest first (full-res rects)
    segment_scores: list = None
    objects: list = None  # [(front [x1, y1, x2, y2], back_y)] room-aligned boxes, full-res
    focal_px: float | None = None  # measured from distance points, if the floor shows them


def _lab(img: np.ndarray):
    """Lab image (float) of an RGB image in [0, 1], or None for gray / no OpenCV."""
    if img.ndim != 3:
        return None
    try:
        import cv2
    except ImportError:
        return None
    # 8-bit Lab (L, a, b each 0-255), the same scale as tip.revolve.
    return cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


# --------------------------------------------------------------------------- distance points
def _segments(image: np.ndarray, max_side: int = 1000, min_frac: float = 0.02, margin_deg: float = 12.0):
    """Straight line segments (OpenCV's LSD), longer than min_frac of the
    image diagonal and not near horizontal/vertical. Full-res (N, 4) and lengths."""
    import cv2

    h, w = image.shape[:2]
    k = min(1.0, max_side / max(h, w))
    small = cv2.resize(image.astype(np.float32), (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    g = (np.clip(to_gray(small), 0, 1) * 255).astype(np.uint8)
    found = cv2.createLineSegmentDetector().detect(g)[0]
    if found is None:
        return np.zeros((0, 4)), np.zeros(0)
    L = found.reshape(-1, 4).astype(float) / k
    d = L[:, 2:] - L[:, :2]
    ln = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
    keep = (ln > min_frac * np.hypot(h, w)) & (ang > margin_deg) & (ang < 90 - margin_deg)
    return L[keep], ln[keep]


def _ransac_point(L, ln, iters: int = 2000, tol_deg: float = 1.0, seed: int = 0):
    """The point most (length-weighted) segments point at: (point, inlier
    mask, inlier length)."""
    rng = np.random.default_rng(seed)
    hom = lambda P: np.c_[P, np.ones(len(P))]  # noqa: E731
    lines = np.cross(hom(L[:, :2]), hom(L[:, 2:]))
    mid = (L[:, :2] + L[:, 2:]) / 2
    d = (L[:, 2:] - L[:, :2]) / ln[:, None]
    best = (None, None, 0.0)
    for _ in range(iters):
        i, j = rng.choice(len(L), 2, replace=False, p=ln / ln.sum())
        v = np.cross(lines[i], lines[j])
        if abs(v[2]) < 1e-9:
            continue
        v = v[:2] / v[2]
        to = v - mid
        to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-12
        inl = np.abs(to[:, 0] * d[:, 1] - to[:, 1] * d[:, 0]) < np.sin(np.radians(tol_deg))
        if ln[inl].sum() > best[2]:
            best = (v, inl, float(ln[inl].sum()))
    if best[0] is None:
        return best
    # Refine: the point closest (length-weighted least squares, distances to
    # the lines) to the inlier lines, then re-collect the inliers.
    nrm = lines[:, :2] / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    c = -lines[:, 2] / np.linalg.norm(lines[:, :2], axis=1)
    v, inl = best[0], best[1]
    for _ in range(3):
        if inl.sum() < 2:
            break
        wgt = ln[inl]
        A = nrm[inl] * wgt[:, None]
        v = np.linalg.lstsq(A.T @ nrm[inl], A.T @ c[inl], rcond=None)[0]
        to = v - mid
        to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-12
        inl = np.abs(to[:, 0] * d[:, 1] - to[:, 1] * d[:, 0]) < np.sin(np.radians(tol_deg))
    return v, inl, float(ln[inl].sum())


def find_vp_segments(image: np.ndarray, tol_deg: float = 1.0, n: int = 161):
    """VP voted by straight line segments (LSD), each weighted by its length.
    Long room edges dominate, and short texture edges (a painting's brush
    strokes and cracks, a carpet's pattern) count for little. Searched over
    the central 80% of the image. Full-res pixels, or None."""
    try:
        L, ln = _segments(image)
    except ImportError:
        return None
    if len(L) < 4:
        return None
    h, w = image.shape[:2]
    X, Y = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, n), np.linspace(0.1 * h, 0.9 * h, n))
    mid = (L[:, :2] + L[:, 2:]) / 2
    d = (L[:, 2:] - L[:, :2]) / ln[:, None]
    sc = np.zeros(X.shape)
    sin_tol = np.sin(np.radians(tol_deg))
    for (mx, my), (dx, dy), length in zip(mid, d, ln):
        vx, vy = X - mx, Y - my
        sc += length * (np.abs(vx * dy - vy * dx) < sin_tol * (np.hypot(vx, vy) + 1e-9))
    i = int(np.argmax(sc))
    return float(X.ravel()[i]), float(Y.ravel()[i])


def find_distance_points(image: np.ndarray, min_inliers: int = 6, min_share: float = 0.3, seed: int = 0):
    """The painter's distance-point construction, run backwards. Square tiles
    laid at 45 degrees to the room have edges that converge to two points
    on the horizon, at VP -+ f. So the VP is their midpoint, and half their
    separation is the focal length in pixels. Returns (vp, f, support,
    floor_top) in full-res pixels: support is the total length of the two
    families' segments that meet at their points, floor_top the highest row
    the tiles reach (the floor's far edge; None if none below the horizon).
    None if the photo shows no such
    pair of line families."""
    try:
        L, ln = _segments(image)
    except ImportError:
        return None
    h, w = image.shape[:2]
    d = L[:, 2:] - L[:, :2]
    rising = d[:, 0] * d[:, 1] > 0  # image y grows downward
    pts, support, tiles = [], 0.0, []
    for fam in (rising, ~rising):
        if fam.sum() < min_inliers:
            return None
        v, inl, length = _ransac_point(L[fam], ln[fam], seed=seed)
        if v is None or inl.sum() < min_inliers or length < min_share * ln[fam].sum():
            return None
        pts.append(v)
        support += length
        tiles.append(L[fam][inl])
    (x1, y1), (x2, y2) = sorted(pts, key=lambda p: p[0])
    # Each family alone is a bundle of nearly parallel lines, whose meeting
    # point slides along the bundle. So fit the whole construction jointly:
    # (vx, vy, f) with distance points (vx -+ f, vy) and the room's receding
    # lines at (vx, vy), by a soft inlier count of all segments weighted by
    # length squared (short segments, e.g. small tiles near the horizon,
    # have biased directions).
    vx, vy, f = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2
    mid = (L[:, :2] + L[:, 2:]) / 2
    dirs = d / np.linalg.norm(d, axis=1, keepdims=True)
    sin_tol = np.sin(np.radians(0.5))
    wgt = ln ** 2

    def score(prm):
        cx, cy, ff = prm
        best = np.zeros(len(L))
        for px in (cx - ff, cx, cx + ff):
            to = np.array([px, cy]) - mid
            sn = np.abs(to[:, 0] * dirs[:, 1] - to[:, 1] * dirs[:, 0]) / (np.linalg.norm(to, axis=1) + 1e-9)
            best = np.maximum(best, np.clip(1 - (sn / sin_tol) ** 2, 0, None))
        return float((wgt * best).sum())

    # Coarse grid around the per-family estimate, then local refinement.
    h_, w_ = image.shape[:2]
    best_grid = (score([vx, vy, f]), [vx, vy, f])
    for cx in vx + np.arange(-0.04, 0.0401, 0.01) * w_:
        for cy in vy + np.arange(-0.06, 0.0601, 0.005) * h_:
            for ff in f * np.arange(0.85, 1.151, 0.01):
                sc_ = score([cx, cy, ff])
                if sc_ > best_grid[0]:
                    best_grid = (sc_, [cx, cy, ff])
    cur, prm = best_grid
    for step in (32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5):
        moved = True
        while moved:
            moved = False
            for i in range(3):
                for sg in (-1, 1):
                    cand = list(prm)
                    cand[i] += sg * step
                    sc_ = score(cand)
                    if sc_ > cur + 1e-9:
                        prm, cur, moved = cand, sc_, True
    vx, vy, f = prm
    x1, x2, y1, y2 = vx - f, vx + f, vy, vy
    vp = (vx, vy)
    f35 = f / max(w, h) * 36
    if (abs(y1 - y2) > 0.05 * h or x1 > 0.1 * w or x2 < 0.9 * w or not (0.1 * w < vp[0] < 0.9 * w)
            or not (0.1 * h < vp[1] < 0.9 * h) or not (15 < f35 < 120)):
        return None
    # The tiled floor's far edge: the highest tile-edge point below the horizon.
    T = np.concatenate(tiles)
    ys = np.concatenate([T[:, 1], T[:, 3]])
    below = ys[ys > vp[1] + 0.02 * h]
    floor_top = float(below.min()) if len(below) else None
    return vp, float(f), support, floor_top


def explained_length(L, ln, vp, tol_deg: float = 1.0) -> float:
    """Total length of the segments that point at vp."""
    mid = (L[:, :2] + L[:, 2:]) / 2
    d = (L[:, 2:] - L[:, :2]) / ln[:, None]
    v = np.asarray(vp, float) - mid
    ok = np.abs(v[:, 0] * d[:, 1] - v[:, 1] * d[:, 0]) < np.sin(np.radians(tol_deg)) * np.linalg.norm(v, axis=1)
    return float(ln[ok].sum())


def auto_annotate_array(image: np.ndarray, max_side: int = 640, seed: int = 0, deeper: bool = True,
                        objects: bool = True) -> AutoResult:
    """Run the detector on an RGB/gray float image; returns full-res coords."""
    small, s = downscale(image, max_side)
    gray = to_gray(small)
    gx, gy = gradients(gray, sigma=1.0)
    lab = _lab(small)
    # VP candidates: the edge vote, the line-segment vote, and the midpoint of the distance points
    # if the floor (tiles laid diagonally) shows them. Each is refined
    # jointly with the back wall, since the room's four big junction lines
    # are the most reliable evidence (the edge vote leans toward wherever
    # most texture is, often the far end of a corridor), and the best
    # back-wall fit wins. Candidates that explain much less of the image's
    # long straight lines than the best one (a painting's brush strokes can
    # mislead the edge vote) are dropped first.
    starts = [(find_vanishing_point(gray, seed=seed), None)]
    sv = find_vp_segments(image)
    if sv is not None:
        starts.append(((sv[0] * s, sv[1] * s), None))
    dp = find_distance_points(image, seed=seed)
    if dp is not None:
        starts.append(((dp[0][0] * s, dp[0][1] * s), dp[1]))
    floor_top = dp[3] * s if dp is not None and dp[3] is not None else None
    evidence = lambda v, f: 0.0  # noqa: E731
    if len(starts) > 1:
        try:
            L, ln = _segments(image)
        except ImportError:
            L = None
        if L is not None and len(L):
            def evidence(v, f, tol=2.0):
                """Line length explained by VP v (small-image px), plus the two
                distance-point families if f (full-res px) is given."""
                x, y = v[0] / s, v[1] / s
                e = explained_length(L, ln, (x, y), tol)
                if f is not None:
                    e += explained_length(L, ln, (x - f, y), tol) + explained_length(L, ln, (x + f, y), tol)
                return e
            expl = [evidence(v, f) for v, f in starts]
            starts = [st for st, e in zip(starts, expl) if e >= 0.5 * max(expl)]
    best = None
    for vp0, focal in starts:
        b = None
        e0 = evidence(vp0, focal)
        for r in (6, 2):
            center = vp0 if b is None else b[0]
            for dx in range(-r * 2, r * 2 + 1, r):
                for dy in range(-r * 2, r * 2 + 1, r):
                    cand = (center[0] + dx, center[1] + dy)
                    # Don't drift off what the long lines say.
                    if (dx or dy) and evidence(cand, focal) < 0.7 * e0:
                        continue
                    try:
                        bw, score = find_back_wall(gx, gy, cand, lab=lab,
                                                   max_bottom=floor_top if focal is not None else None)
                    except ValueError:  # too close to the border for a back wall
                        continue
                    if b is None or score > b[2]:
                        b = (cand, bw, score)
        if b is not None and (best is None or b[2] > best[2]):
            best = (*b, focal)
    if best is None:
        raise ValueError("no vanishing point with a plausible back wall")
    vp, bw, _, focal_px = best
    e_vp = evidence(vp, focal_px)
    vp_r, _ = refine_vp_from_corners(gx, gy, vp, bw)
    if evidence(vp_r, focal_px) >= 0.7 * e_vp:
        vp = vp_r
    # Furniture edges can pose as room edges. If there are room-aligned
    # objects, find the back wall again with their image footprints masked out.
    rgx, rgy = gx, gy
    first = find_room_objects(gx, gy, vp, [bw]) if objects else []
    if first:
        from .textures import polygon_mask

        polys = [object_footprint(f, b, vp) for f, b, _ in first]
        keep = ~polygon_mask(polys, gx.shape[1], gx.shape[0], dilate=2)
        rgx, rgy = gx * keep, gy * keep
        bw, _ = find_back_wall(rgx, rgy, vp, lab=lab, max_bottom=floor_top if focal_px is not None else None)
    rays = rays_from_rect(vp, bw)
    segs, seg_scores = (find_deeper_boxes(rgx, rgy, vp, bw) if deeper else ([], []))
    if segs:
        # A doorway (lintel, jambs) in front of the deeper boxes becomes a
        # thin door-frame box, so the lintel is a wall face, not ceiling.
        frame, frame_score = find_opening(rgx, rgy, vp, bw, inner=segs[0])
        if frame is not None and frame[3] > segs[0][3]:
            segs, seg_scores = [frame] + segs, [frame_score] + list(seg_scores)
        elif frame is None and not _seen_through_opening(bw, segs[0]):
            # Further boxes are seen through an opening in the back wall: a
            # doorway (found above), or at least a clear band of wall around
            # the first box. A "box" that nearly fills the back wall, with no
            # doorway, is furniture or a niche (paintings), not a corridor.
            segs, seg_scores = [], []
    objs = find_room_objects(gx, gy, vp, [bw] + segs) if objects else []
    inv = 1.0 / s
    # Map back to full-res pixel coordinates (pixel centers).
    to_full = lambda v: (v + 0.5) * inv - 0.5  # noqa: E731
    vp_full = (round(to_full(vp[0]), 1), round(to_full(vp[1]), 1))
    bw_full = [round(to_full(v), 1) for v in bw]
    segs_full = [[round(to_full(v), 1) for v in r] for r in segs]
    objs_full = [([round(float(to_full(v)), 1) for v in front], round(float(to_full(back_y)), 1)) for front, back_y, _ in objs]
    return AutoResult(vp_full, bw_full, rays, segs_full, seg_scores, objs_full, focal_px)


def add_solids(ann: Annotation, image: np.ndarray) -> Annotation:
    """Detect surfaces of revolution (revolve.find_solids) in the room that
    ann describes and store them as ann.solids."""
    from .geometry import build_scene
    from .revolve import find_solids

    ann.solids = []
    scene = build_scene(ann, image.shape[1], image.shape[0])
    ann.solids = find_solids(image, scene)
    return ann


def add_wall_things(ann: Annotation, image: np.ndarray) -> Annotation:
    """Detect boxes fixed to the walls or ceiling (signs, alarm boxes) and
    pipes under the ceiling (tip.relief), and store them as ann.reliefs and
    ann.tubes. Shares the room background and proposals between the two."""
    from .geometry import build_scene
    from .relief import find_reliefs, find_tubes
    from .revolve import outlier_proposals, room_background

    ann.reliefs, ann.tubes = [], []
    solids, ann.solids = ann.solids, []
    scene = build_scene(ann, image.shape[1], image.shape[0])
    ann.solids = solids
    small, s = downscale(image, 800)
    bg, bg_alt, pid = room_background(small, scene, s)
    outlier, boxes, labels = outlier_proposals(small, bg, bg_alt, pid, labels=True)
    ann.reliefs = find_reliefs(image, scene, proposals=(outlier, boxes, labels, pid), solids=solids)
    ann.tubes = find_tubes(image, scene, proposals=(bg, pid))
    return ann


def add_cutouts(ann: Annotation, image: np.ndarray) -> Annotation:
    """Detect things standing on the floor that are neither boxes nor solids
    (people, chairs) and store them as foreground billboards (tip.cutouts).
    Needs Segment Anything and its checkpoint; a no-op without them."""
    from .cutouts import find_cutouts, sam_available
    from .geometry import build_scene

    if not sam_available():
        return ann
    import cv2

    H, W = image.shape[:2]
    avoid = []
    for sol in ann.solids:  # silhouettes of the solids already found
        prof = np.asarray(sol.profile, float)
        poly = np.concatenate([prof[:, [1, 0]], prof[::-1][:, [2, 0]]])
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
        avoid.append(m.astype(bool))
    keep = ann.solids, ann.reliefs, ann.tubes, ann.foreground
    ann.solids, ann.reliefs, ann.tubes, ann.foreground = [], [], [], []
    scene = build_scene(ann, W, H)
    ann.solids, ann.reliefs, ann.tubes, ann.foreground = keep
    cutouts, replaced = find_cutouts(image, scene, solids=avoid)
    ann.foreground = list(ann.foreground) + cutouts
    if replaced:  # solids that only covered part of a cutout's object
        ann.solids = [sol for i, sol in enumerate(ann.solids) if i not in replaced]
    return ann


def auto_annotate(image_path: str, image: np.ndarray | None = None, solids: bool = True, cutouts: bool = True,
                  **kw) -> Annotation:
    if image is None:
        from .textures import load_image

        image = load_image(image_path)
    r = auto_annotate_array(image, **kw)
    from .annotations import RoomObject

    if r.focal_px is not None:
        kw_f = {"focal_35mm": round(r.focal_px / max(image.shape[:2]) * 36, 2)}
    else:
        kw_f = {}
    ann = Annotation(image=str(image_path), vanishing_point=r.vanishing_point, back_wall=tuple(r.back_wall), **kw_f,
                     segments=r.segments or [],
                     objects=[RoomObject(f"object{i + 1}", list(f), b) for i, (f, b) in enumerate(r.objects or [])])
    if solids:
        try:
            import cv2  # noqa: F401  (the solid detector's default path needs OpenCV)
        except ImportError:
            return ann
        add_solids(ann, image)
        add_wall_things(ann, image)
    if cutouts:
        add_cutouts(ann, image)
    return ann
