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


def _search_rect(gx, gy, vp, X1, X2, Y1, Y2, bound=None, valid=None):
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
    if valid is not None:
        ok = valid(X1[:, None, None, None], X2[None, :, None, None], Y1[None, None, :, None], Y2[None, None, None, :])
        S = np.where(ok, S, -np.inf)
    k = int(np.argmax(S))
    if not np.isfinite(S.ravel()[k]):
        return None, -np.inf
    i1, i2, j1, j2 = np.unravel_index(k, S.shape)
    return [float(X1[i1]), float(Y1[j1]), float(X2[i2]), float(Y2[j2])], float(S.ravel()[k])


def find_back_wall(gx, gy, vp, step: int = 4, min_frac: float = 0.06):
    """Best back-wall rectangle [x1, y1, x2, y2] anywhere in the image, and its score."""
    h, w = gx.shape
    vx, vy = vp
    m = int(min_frac * min(h, w))
    X1 = np.arange(2, int(vx) - m, step)
    X2 = np.arange(int(vx) + m, w - 2, step)
    Y1 = np.arange(2, int(vy) - m, step)
    Y2 = np.arange(int(vy) + m, h - 2, step)
    rect, score = _search_rect(gx, gy, vp, X1, X2, Y1, Y2)
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


def auto_annotate_array(image: np.ndarray, max_side: int = 640, seed: int = 0, deeper: bool = True,
                        objects: bool = True) -> AutoResult:
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
    vp, _ = refine_vp_from_corners(gx, gy, vp, bw)
    # Furniture edges can pose as room edges. If there are room-aligned
    # objects, find the back wall again with their image footprints masked out.
    rgx, rgy = gx, gy
    first = find_room_objects(gx, gy, vp, [bw]) if objects else []
    if first:
        from .textures import polygon_mask

        polys = [object_footprint(f, b, vp) for f, b, _ in first]
        keep = ~polygon_mask(polys, gx.shape[1], gx.shape[0], dilate=2)
        rgx, rgy = gx * keep, gy * keep
        bw, _ = find_back_wall(rgx, rgy, vp)
    rays = rays_from_rect(vp, bw)
    segs, seg_scores = (find_deeper_boxes(rgx, rgy, vp, bw) if deeper else ([], []))
    objs = find_room_objects(gx, gy, vp, [bw] + segs) if objects else []
    inv = 1.0 / s
    # Map back to full-res pixel coordinates (pixel centers).
    to_full = lambda v: (v + 0.5) * inv - 0.5  # noqa: E731
    vp_full = (round(to_full(vp[0]), 1), round(to_full(vp[1]), 1))
    bw_full = [round(to_full(v), 1) for v in bw]
    segs_full = [[round(to_full(v), 1) for v in r] for r in segs]
    objs_full = [([round(float(to_full(v)), 1) for v in front], round(float(to_full(back_y)), 1)) for front, back_y, _ in objs]
    return AutoResult(vp_full, bw_full, rays, segs_full, seg_scores, objs_full)


def add_solids(ann: Annotation, image: np.ndarray) -> Annotation:
    """Detect surfaces of revolution (revolve.find_solids) in the room that
    ann describes and store them as ann.solids."""
    from .geometry import build_scene
    from .revolve import find_solids

    ann.solids = []
    scene = build_scene(ann, image.shape[1], image.shape[0])
    ann.solids = find_solids(image, scene)
    return ann


def auto_annotate(image_path: str, image: np.ndarray | None = None, solids: bool = True, **kw) -> Annotation:
    if image is None:
        from .textures import load_image

        image = load_image(image_path)
    r = auto_annotate_array(image, **kw)
    from .annotations import RoomObject

    ann = Annotation(image=str(image_path), vanishing_point=r.vanishing_point, back_wall=tuple(r.back_wall),
                     segments=r.segments or [],
                     objects=[RoomObject(f"object{i + 1}", list(f), b) for i, (f, b) in enumerate(r.objects or [])])
    if solids:
        try:
            import cv2  # noqa: F401  (the solid detector's default path needs OpenCV)
        except ImportError:
            return ann
        add_solids(ann, image)
    return ann
