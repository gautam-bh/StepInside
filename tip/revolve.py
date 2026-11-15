"""Automatic detection of solids: surfaces of revolution with a vertical axis
(vases, bins, lamps, fire extinguishers, hydrants).

Everything here is classical, using the room geometry that TIP has already
recovered:

1. Mirror-symmetric edge pairs. In this camera model (no roll; the VP is the
   principal point) a vertical axis stays vertical in the image, and the
   silhouette of a surface of revolution is mirror-symmetric about it. Its
   two sides are the same colour step taken in opposite directions, so their
   colour gradients are opposite vectors. Every pair of vertical-ish edge
   points on a row with near-opposite colour gradients votes for its midpoint.
2. Seeds. Runs of rows whose pairs share a midpoint and change width slowly
   are candidate axes (short runs are fine; labels, straps and hoses
   interrupt the silhouette).
3. Profile. For each seed, a Viterbi pass down its axis column picks the
   half-width on every row: it rewards symmetric edge evidence, penalises
   width changes, and decides where the object starts and ends. So gaps
   (a strap, a hose crossing) are bridged by the smoothness prior.
4. Room checks. The silhouette is lifted to 3D (geometry.solid_geometry)
   with two hypotheses: standing on the floor, or with its outer side
   touching the wall behind it. If the wall hypothesis puts the bottom above
   the floor, it hangs on the wall; otherwise it stands on the floor. It must
   then have a plausible real size, fit inside the room, and both of its
   sides must end on the same rows (a doorway or panel on a side wall
   recedes, so its near side is taller than its far side).
"""

from __future__ import annotations

import numpy as np

from .annotations import Solid
from .auto_annotate import blur, downscale


# --------------------------------------------------------------------------- edges
def color_gradients(image: np.ndarray, sigma: float = 1.0):
    """Per-channel x/y gradients (H, W, 3) of a blurred RGB image."""
    b = np.stack([blur(image[..., c], sigma) for c in range(3)], axis=-1)
    gx = np.zeros_like(b)
    gy = np.zeros_like(b)
    gx[:, 1:-1] = (b[:, 2:] - b[:, :-2]) / 2
    gy[1:-1] = (b[2:] - b[:-2]) / 2
    return gx, gy


def symmetric_pairs(gx, gy, thr: float = 0.02, w_range=(6, 120), min_anti: float = 0.8):
    """All mirror-edge pairs: rows of (y, xl + xr, xr - xl, score).

    Edge points are horizontal maxima of the colour gradient along x that are
    more vertical than horizontal. A pair scores sqrt(mL * mR) * (-cos),
    where cos is between their colour gradients (-1 for a perfect mirror)."""
    mx = np.linalg.norm(gx, axis=-1)
    my = np.linalg.norm(gy, axis=-1)
    peak = (mx > thr) & (mx > 1.2 * my)
    peak[:, 1:-1] &= (mx[:, 1:-1] >= mx[:, :-2]) & (mx[:, 1:-1] >= mx[:, 2:])
    peak[:, [0, -1]] = False
    out = []
    for y in range(gx.shape[0]):
        xs = np.nonzero(peak[y])[0]
        if len(xs) < 2:
            continue
        g, m = gx[y, xs], mx[y, xs]
        i, j = np.triu_indices(len(xs), 1)
        w = xs[j] - xs[i]
        ok = (w >= w_range[0]) & (w <= w_range[1])
        i, j = i[ok], j[ok]
        cos = (g[i] * g[j]).sum(-1) / (m[i] * m[j])
        ok = cos < -min_anti
        i, j = i[ok], j[ok]
        if len(i):
            out.append(np.stack([np.full(len(i), y), xs[i] + xs[j], xs[j] - xs[i],
                                 np.sqrt(m[i] * m[j]) * -cos[ok]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 4))


def axis_seeds(pairs: np.ndarray, width: int, min_rows: int = 10, max_gap: int = 5):
    """Chains of pairs with (nearly) the same midpoint whose width changes
    slowly from row to row. Returns [(c2, y_top, y_bottom, n_rows, rows)] with
    c2 the doubled axis column and rows an array of (y, full width), longest
    first, one per axis column."""
    if not len(pairs):
        return []
    pairs = pairs[np.lexsort((pairs[:, 0], pairs[:, 1]))]
    c2 = pairs[:, 1].astype(int)
    starts = np.searchsorted(c2, np.arange(2 * width + 3))
    seeds = []
    for c in range(1, 2 * width + 1):
        q = pairs[starts[c - 1]: starts[c + 2]]
        if len(q) < min_rows:
            continue
        q = q[np.argsort(q[:, 0], kind="stable")]
        y, w = q[:, 0], q[:, 2]
        n = len(q)
        best = np.ones(n)
        prev = -np.ones(n, int)
        lo = np.searchsorted(y, y - max_gap)
        hi = np.searchsorted(y, y, "left")
        for k in range(n):
            if hi[k] > lo[k]:
                cand = np.arange(lo[k], hi[k])
                cand = cand[np.abs(w[cand] - w[k]) <= max(1.5, 0.05 * w[k])]
                if len(cand):
                    p = cand[np.argmax(best[cand])]
                    best[k], prev[k] = best[p] + 1, p
        k = int(np.argmax(best))
        if best[k] < min_rows:
            continue
        chain = [k]
        while prev[chain[-1]] >= 0:
            chain.append(prev[chain[-1]])
        ch = q[chain[::-1]]
        seeds.append((c, int(ch[0, 0]), int(ch[-1, 0]), int(best[k]), ch[:, [0, 2]]))
    seeds.sort(key=lambda s: -s[3])
    return seeds


# --------------------------------------------------------------------------- profile
def _interp_cols(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation of (N, W, C) rows at float columns x: (N, len(x), C)."""
    x = np.clip(x, 0, a.shape[1] - 1.001)
    i = np.floor(x).astype(int)
    f = (x - i)[None, :, None]
    return a[:, i] * (1 - f) + a[:, i + 1] * f


def symmetric_evidence(gx, c: float, rows: np.ndarray, half_widths: np.ndarray) -> np.ndarray:
    """E[row, k]: mirror-edge evidence for half-width half_widths[k] about
    column c: sqrt(|gL| |gR|) * max(0, -cos(gL, gR))."""
    g = gx[rows]
    gl = _interp_cols(g, c - half_widths)
    gr = _interp_cols(g, c + half_widths)
    ml = np.linalg.norm(gl, axis=-1)
    mr = np.linalg.norm(gr, axis=-1)
    cos = (gl * gr).sum(-1) / np.maximum(ml * mr, 1e-9)
    return np.sqrt(ml * mr) * np.maximum(0.0, -cos)


def _kmeans(X: np.ndarray, k: int, iters: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = min(k, len(X))
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
    a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    return C, a


class ColorModel:
    """A small Gaussian mixture (k-means clusters with isotropic variances)."""

    def __init__(self, X: np.ndarray, k: int = 3):
        if len(X) > 4000:
            X = X[np.random.default_rng(0).choice(len(X), 4000, replace=False)]
        self.C, a = _kmeans(X, k)
        self.w = np.array([(a == j).mean() for j in range(len(self.C))]) + 1e-3
        self.var = np.array([max(((X[a == j] - self.C[j]) ** 2).sum(-1).mean() / 3 if (a == j).any() else 1e-3, 4e-4)
                             for j in range(len(self.C))])

    def loglik(self, X: np.ndarray) -> np.ndarray:
        d2 = ((X[..., None, :] - self.C) ** 2).sum(-1)
        ll = np.log(self.w) - 1.5 * np.log(2 * np.pi * self.var) - d2 / (2 * self.var)
        m = ll.max(-1)
        return m + np.log(np.exp(ll - m[..., None]).sum(-1))


def symmetric_grabcut(img: np.ndarray, gx: np.ndarray, c: float, seed: np.ndarray, w_cap: float,
                      iters: int = 3, step: float = 0.5, smooth: float = 1.0, edge_weight: float = 20.0):
    """Silhouette of a mirror-symmetric object about column c, GrabCut style:
    alternately fit colour models to the object and its surroundings, and
    re-segment. The segmentation step is a Viterbi pass down the rows (not a
    graph cut): each row is either outside the object (before or after it) or
    inside with half-width w, scoring the colour log-likelihood ratio summed
    over [c - w, c + w], plus symmetric edge evidence at c +- w, minus a cost
    for width changes. So the result is a symmetric profile by construction.

    seed: (y, full width) rows from axis_seeds. Returns (rows, half_widths)."""
    H, W = img.shape[:2]
    y0, y1 = int(seed[0, 0]), int(seed[-1, 0])
    w0 = float(np.median(seed[:, 1])) / 2
    span = 2 * (y1 - y0) + 20
    rows = np.arange(max(0, y0 - span), min(H, y1 + span + 1))
    w_lim = min(w_cap, 3 * w0 + 8, c - 1, W - 2 - c)
    hw = np.arange(1, int(w_lim / step) + 1) * step
    if len(hw) < 4:
        return None
    # Pixels sampled symmetrically about the axis: (rows, offsets, side, rgb).
    win = img[rows]
    left = _interp_cols(win, c - hw)
    right = _interp_cols(win, c + hw)
    E = symmetric_evidence(gx, c, rows, hw)
    seed_rows = {int(y): w / 2 for y, w in seed}
    prof = np.zeros(len(rows))
    for i, y in enumerate(rows):  # initial profile: the seed
        if y in seed_rows:
            prof[i] = seed_rows[y]
    for i in range(1, len(rows)):  # bridge gaps inside the seed
        if y0 <= rows[i] <= y1 and prof[i] == 0:
            prof[i] = prof[i - 1]
    result = None
    for _ in range(iters):
        inside = (hw[None, :] <= prof[:, None] - 1) & (prof[:, None] > 0)
        rim = (np.abs(hw[None, :] - prof[:, None]) > 2) & ~inside
        on = prof > 0
        if on.any():  # background: beside the object, and above/below it
            lo, hi = np.nonzero(on)[0][[0, -1]]
            near = np.zeros(len(rows), bool)
            near[max(0, lo - (hi - lo)):hi + (hi - lo) + 1] = True
            rim &= near[:, None]
        fg = np.concatenate([left[inside], right[inside]])
        bg = np.concatenate([left[rim], right[rim]])
        if len(fg) < 20 or len(bg) < 20:
            return None
        L = np.clip(ColorModel(fg).loglik(np.stack([left, right])) - ColorModel(bg).loglik(np.stack([left, right])),
                    -4, 4)
        R = np.cumsum((L[0] + L[1]) * step, axis=1)
        score = R + edge_weight * E
        new = _viterbi_profile(score, rows, (y0, y1), smooth)
        if new is None:
            return None
        first, k = new
        prof = np.zeros(len(rows))
        prof[first:first + len(k)] = hw[k]
        result = (rows[first:first + len(k)], hw[k])
    return result


def _viterbi_profile(score: np.ndarray, rows: np.ndarray, forced_rows, smooth: float, max_dk: int = 2):
    """Best path through rows: before (0 per row) -> inside at state k
    (score[i, k], width changes cost smooth * |dk|) -> after (0 per row).
    Rows in forced_rows must be inside. Returns (first row index, k per row)."""
    n, K = score.shape
    NEG = -1e18
    forced = (rows >= forced_rows[0]) & (rows <= forced_rows[1])
    inside = np.full((n, K), NEG)
    back = np.zeros((n, K), int)
    before = 0.0
    after, after_last = NEG, -1
    ar = np.arange(K)
    for i in range(n):
        if forced[i]:
            after, after_last = NEG, -1
        prev = inside[i - 1] if i else np.full(K, NEG)
        best = np.full(K, before)
        arg = np.full(K, -1)
        for d in range(-max_dk, max_dk + 1):
            sh = np.full(K, NEG)
            if d > 0:
                sh[d:] = prev[:-d]
            elif d < 0:
                sh[:d] = prev[-d:]
            else:
                sh = prev.copy()
            sh -= smooth * abs(d)
            better = sh > best
            best = np.where(better, sh, best)
            arg = np.where(better, ar - d, arg)
        inside[i] = best + score[i]
        back[i] = arg
        if forced[i]:
            before = NEG
        if inside[i].max() > after and not forced[i + 1:].any():
            after, after_last = inside[i].max(), i
    if after_last < 0:
        return None
    i = after_last
    k = int(np.argmax(inside[i]))
    ks = [k]
    while back[i, ks[-1]] >= 0 and i > 0:
        ks.append(back[i, ks[-1]])
        i -= 1
    return i, np.array(ks[::-1])


def side_extent(gx, x: float, y_top: int, y_bottom: int, thr: float, reach: int):
    """How many rows beyond the top and bottom the vertical edge at column x
    continues (strong horizontal colour gradient within a pixel of x)."""
    H, W = gx.shape[:2]
    xi = np.clip(np.round(x + np.arange(-1, 2)).astype(int), 0, W - 1)
    m = np.linalg.norm(gx[:, xi], axis=-1).max(axis=1)

    def run(rng):
        n = 0
        for y in rng:
            if m[y] < thr:
                break
            n += 1
        return n

    up = run(range(y_top - 1, max(-1, y_top - 1 - reach), -1))
    down = run(range(y_bottom + 1, min(H, y_bottom + 1 + reach)))
    return up, down


def curve_support(gy, xs, ys, thr: float, shifts: int = 3) -> float:
    """Fraction of the points (xs, ys) on a strong vertical-gradient edge
    (within a pixel), at the best vertical shift up to +-shifts pixels."""
    H, W = gy.shape[:2]
    if len(xs) < 3:
        return 0.0
    mag = np.linalg.norm(gy, axis=-1)
    X = np.clip(np.round(xs).astype(int), 0, W - 1)
    best = 0.0
    for sh in range(-shifts, shifts + 1):
        Y = np.round(ys).astype(int) + sh
        g = np.zeros(len(xs))
        for d in (-1, 0, 1):
            g = np.maximum(g, mag[np.clip(Y + d, 0, H - 1), X])
        best = max(best, float((g > thr).mean()))
    return best


def projected_outline(lathe, scene, s: float, shape):
    """Top and bottom outline points (x, y arrays) of the solid ray-cast from
    the input camera at the working resolution (scale s of the full image):
    the parts of its outline that run roughly horizontally (rims, caps, a
    rounded base). The silhouette's sides found the candidate, so they are
    not tested again."""
    H, W = shape
    f, (vx, vy) = scene.focal, scene.principal
    pts = lathe.point(*np.meshgrid(np.linspace(0, 1, 24), np.linspace(0, 1, 24)))
    u = (vx + f * pts[..., 0] / pts[..., 2] + 0.5) * s - 0.5
    v = (vy - f * pts[..., 1] / pts[..., 2] + 0.5) * s - 0.5
    x0, x1 = int(max(0, np.floor(u.min()) - 2)), int(min(W, np.ceil(u.max()) + 3))
    y0, y1 = int(max(0, np.floor(v.min()) - 2)), int(min(H, np.ceil(v.max()) + 3))
    empty = (np.zeros(0), np.zeros(0))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return empty, empty
    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    uf, vf = (xs + 0.5) / s - 0.5, (ys + 0.5) / s - 0.5
    dirs = np.stack([(uf - vx) / f, -(vf - vy) / f, np.ones_like(uf)], axis=-1)
    _, _, _, mask = lathe.hit(np.zeros(3), dirs)
    my_, mx_ = np.gradient(mask.astype(float))
    horiz = (np.hypot(mx_, my_) > 0.25) & mask & (np.abs(my_) > np.abs(mx_))
    top = horiz & (my_ > 0)  # mask starts below this pixel's upper neighbour
    bottom = horiz & (my_ < 0)
    pick = lambda m: (xs[m].astype(float), ys[m].astype(float))  # noqa: E731
    return pick(top), pick(bottom)


def straight_edge_support(gy, vp, c: float, hw: float, y: float, thr: float, search: int) -> float:
    """Best support for a straight edge crossing the axis column c near row
    y, over [c - hw, c + hw], that is either horizontal or runs to the
    vanishing point: the top or bottom edge of a flat rectangle facing the
    camera (a poster, a box front) or on a side wall (a door, a panel)."""
    xs = np.linspace(c - hw, c + hw, max(6, int(2 * hw)))
    best = 0.0
    for y_c in np.arange(y - search, y + search + 1):
        slopes = [0.0]
        if abs(c - vp[0]) > 1e-6:
            slopes.append((y_c - vp[1]) / (c - vp[0]))
        for slope in slopes:
            ys = y_c + slope * (xs - c)
            best = max(best, curve_support(gy, xs, ys, thr, shifts=0))
    return best


# --------------------------------------------------------------------------- main
def find_solids(image: np.ndarray, scene, max_side: int = 800, min_height_px: int = 24,
                size_range=(0.12, 2.2), max_radius: float = 0.6, max_candidates: int = 60, min_outline: float = 0.5,
                debug: list | None = None) -> list[Solid]:
    """Detect upright surfaces of revolution in a photo whose room geometry
    (scene, from geometry.build_scene) is known. Returns full-res Solids."""
    from .geometry import solid_geometry

    small, s = downscale(image, max_side)
    Hs, Ws = small.shape[:2]
    gx, gy = color_gradients(small, 1.0)
    pairs = symmetric_pairs(gx, gy, w_range=(6, int(0.25 * Ws)))
    seeds = axis_seeds(pairs, Ws)
    to_full = lambda v: (np.asarray(v, float) + 0.5) / s - 0.5  # noqa: E731
    floor_y = scene.yf

    cands = []
    taken = []  # (c, y0, y1) of traced profiles, to skip duplicate seeds
    for c2, y0, y1, nrows, chain in seeds:
        c = c2 / 2
        if any(abs(c - tc) <= 1.5 and y0 >= ty0 - 2 and y1 <= ty1 + 2 for tc, ty0, ty1 in taken):
            continue
        seg = pairs[(np.abs(pairs[:, 1] - c2) <= 1) & (pairs[:, 0] >= y0) & (pairs[:, 0] <= y1)]
        level = float(np.median(seg[:, 3]))
        w_max = min(c, Ws - 1 - c, 0.15 * Ws)
        if w_max < 4:
            continue
        r = symmetric_grabcut(small, gx, c, chain, w_max)
        if r is None:
            continue
        rows, hw = r
        taken.append((c, rows[0], rows[-1]))
        if len(hw) < max(8, min_height_px * s):
            continue
        cands.append(dict(c=c, rows=rows, hw=hw, score=float(len(rows)), level=level))

    found = []
    for cd in cands:
        c, rows, hw = cd["c"], cd["rows"], cd["hw"]
        why = None
        # Both sides must end together (receding wall rectangles don't).
        thr = 0.5 * cd["level"]
        reach = max(6, len(rows) // 3)
        tol = max(3, len(rows) // 10)
        lu, ld = side_extent(gx, c - hw[0], rows[0], rows[-1], thr, reach)
        ru, rd = side_extent(gx, c + hw[0], rows[0], rows[-1], thr, reach)
        lu_b, ld_b = side_extent(gx, c - hw[-1], rows[0], rows[-1], thr, reach)
        ru_b, rd_b = side_extent(gx, c + hw[-1], rows[0], rows[-1], thr, reach)
        if abs(lu - ru) > tol or abs(ld_b - rd_b) > tol:
            why = f"sides end unevenly (top {lu}/{ru}, bottom {ld_b}/{rd_b})"
        prof = [[float(to_full(y)), float(to_full(c - w)), float(to_full(c + w))] for y, w in zip(rows, hw)]
        sol = Solid("solid", prof, "floor")
        mount = None
        if why is None:
            wall = solid_geometry(sol, scene, "wall")
            if wall is not None and wall[2].min() > floor_y + 0.05:
                mount, g = "wall", wall
            else:
                mount, g = "floor", solid_geometry(sol, scene, "floor")
            if g is None:
                why = "doesn't fit the room"
            else:
                D, beta, ys, radius = g
                height = ys.max() - ys.min()
                ax = D * np.sin(beta)
                az = D * np.cos(beta)
                box = next((b for b in scene.boxes if b.z0 <= az <= b.z1), None)
                if not size_range[0] <= height <= size_range[1]:
                    why = f"height {height:.2f} m"
                elif radius.max() > max_radius:
                    why = f"radius {radius.max():.2f} m"
                elif box is None:
                    why = "outside the room's depth"
                elif ax - radius.max() < box.xl - 0.05 or ax + radius.max() > box.xr + 0.05:
                    why = "pokes through a wall"
                elif ys.max() > box.yc:
                    why = "above the ceiling"
                else:
                    sol.mount = mount
                    from .geometry import solid_surface

                    thr = 0.35 * cd["level"]
                    top, bottom = projected_outline(solid_surface(sol, scene), scene, s, (Hs, Ws))
                    round_ = [curve_support(gy, *top, thr), curve_support(gy, *bottom, thr)]
                    # The rival explanation: a flat rectangle, whose top and
                    # bottom edges are straight (horizontal, or receding to the VP).
                    vp_s = ((scene.principal[0] + 0.5) * s - 0.5, (scene.principal[1] + 0.5) * s - 0.5)
                    q = max(2, len(hw) // 4)  # span the object's width near each end
                    flat = [straight_edge_support(gy, vp_s, c, part.max() + 1, y_end, thr, search=4)
                            for part, y_end in ((hw[:q], rows[0]), (hw[-q:], rows[-1]))]
                    cd.update(height=height, D=D, outline=round_, flat=flat)
                    if max(round_) < min_outline:
                        why = f"top/bottom outline not in the image ({max(round_):.0%})"
                    elif not any(r_ > f_ + 0.15 and r_ >= min_outline for r_, f_ in zip(round_, flat)):
                        why = f"flat on the wall (rims {round_[0]:.0%}/{round_[1]:.0%} vs lines {flat[0]:.0%}/{flat[1]:.0%})"
        cd["why"] = why
        cd["solid"] = sol
        if why is None:
            found.append(cd)
    if debug is not None:
        debug.extend(cands)

    # Overlapping detections: keep the one with the higher score.
    found.sort(key=lambda d: -d["score"])
    keep = []
    for d in found:
        x0, x1 = d["c"] - d["hw"].max(), d["c"] + d["hw"].max()
        clash = False
        for k in keep:
            kx0, kx1 = k["c"] - k["hw"].max(), k["c"] + k["hw"].max()
            ox = min(x1, kx1) - max(x0, kx0)
            oy = min(d["rows"][-1], k["rows"][-1]) - max(d["rows"][0], k["rows"][0])
            if ox > 0.3 * min(x1 - x0, kx1 - kx0) and oy > 0.3 * min(len(d["rows"]), len(k["rows"])):
                clash = True
                break
        if not clash:
            keep.append(d)
    keep.sort(key=lambda d: d["c"])
    out = []
    for i, d in enumerate(keep, start=1):
        d["solid"].name = f"solid{i}"
        out.append(d["solid"])
    return out
