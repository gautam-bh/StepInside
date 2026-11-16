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


def _to_lab(img: np.ndarray) -> np.ndarray:
    import cv2

    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(u8), cv2.COLOR_RGB2LAB).astype(np.float32)


def room_background(img: np.ndarray, scene, s: float, cell=(0.02, 0.05), window: float = 2.0,
                    passes: int = 2, outlier: float = 15.0):
    """What each pixel would look like if the room were empty, from the room
    geometry alone. Room surfaces are made of stripes that run along the room:
    a side wall's paint, baseboard and rails depend only on height, and so
    do a back wall's bands. So on a wall the background colour at a point is
    the median of that wall's colours over +-window metres along it, at the
    same height. On the floor and ceiling the median runs across the room
    (along X, at the same depth): an object standing on the floor hides a
    long stretch of floor in depth behind it, but only its own width across. A second pass
    repeats the median without the cells the first pass found to be objects,
    so large objects don't leak into the background around them.

    img is the downscaled photo (factor s of the scene's resolution). Returns
    (background, alternative background, plane index per pixel), the
    backgrounds as Lab images (NaN where there is none). The alternative is
    the floor and ceiling median in depth, which knows about the baseboards
    where the room's fitted seams are a little off."""
    from .render import _hit

    H, W = img.shape[:2]
    lab = _to_lab(img)
    u, v = np.meshgrid((np.arange(W) + 0.5) / s - 0.5, (np.arange(H) + 0.5) / s - 0.5)
    vx, vy = scene.principal
    dirs = np.stack([(u - vx) / scene.focal, -(v - vy) / scene.focal, np.ones_like(u)], axis=-1)
    best = np.full((H, W), np.inf)
    pid = np.full((H, W), -1)
    for i, p in enumerate(scene.planes):
        if p.kind != "box":
            continue
        t, _, _, ok = _hit(p, np.zeros(3), dirs)
        closer = ok & (t < best)
        best[closer] = t[closer]
        pid[closer] = i
    pts = dirs * np.where(np.isfinite(best), best, 0)[..., None]
    bg = np.full_like(lab, np.nan)
    bg_alt = np.full_like(lab, np.nan)
    for i, p in enumerate(scene.planes):
        m = pid == i
        if m.sum() < 50:
            continue
        ax = p.normal_axis
        if ax == 1:  # floor / ceiling: across the room, and (alt) in depth
            bg[m] = _stripe_median(lab[m], pts[m][:, 0], pts[m][:, 2], cell, window, passes, outlier)
            bg_alt[m] = _stripe_median(lab[m], pts[m][:, 2], pts[m][:, 0], cell, window, passes, outlier)
        else:  # walls: along the wall at the same height
            along = 2 if ax == 0 else 0
            bg[m] = _stripe_median(lab[m], pts[m][:, along], pts[m][:, 1], cell, window, passes, outlier)
    return bg, bg_alt, pid


def _stripe_median(vals: np.ndarray, A: np.ndarray, C: np.ndarray, cell, window: float, passes: int,
                   outlier: float) -> np.ndarray:
    """Robust background for points with colours vals at plane coordinates
    A (along the stripes) and C (across them): colours are averaged in cells
    (cell[0] across x cell[1] along), and each cell's background is the
    median of the cells within +-window along, at the same C. Repeated
    without the cells that the previous pass found to be objects."""
    import warnings

    from numpy.lib.stride_tricks import sliding_window_view

    ci = np.floor((C - C.min()) / cell[0]).astype(int)
    ai = np.floor((A - A.min()) / cell[1]).astype(int)
    nC, nA = ci.max() + 1, ai.max() + 1
    cnt = np.zeros((nC, nA))
    acc = np.zeros((nC, nA, 3))
    np.add.at(cnt, (ci, ai), 1)
    np.add.at(acc, (ci, ai), vals)
    G = np.full((nC, nA, 3), np.nan)
    G[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0][:, None]
    win = max(1, int(window / cell[1]))
    for _ in range(passes):
        pad = np.pad(G, ((0, 0), (win, win), (0, 0)), constant_values=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            B = np.nanmedian(sliding_window_view(pad, 2 * win + 1, axis=1), axis=-1)
        G = np.where((np.linalg.norm(G - B, axis=-1) > outlier)[..., None], np.nan, G)
    return B[ci, ai]


def outlier_proposals(img: np.ndarray, bg: np.ndarray, bg_alt: np.ndarray, pid: np.ndarray, thr: float = 20.0,
                      min_area: int = 100, core: float = 6.0, seam: int = 8, labels: bool = False):
    """Regions that don't look like the empty room: connected components of
    the pixels far (Lab distance) from the room background, cleaned with a
    morphological open and close. Within seam px of a boundary between
    planes, a floor or ceiling pixel must also differ from the alternative
    (in-depth) background. Returns (mask, [(x, y, w, h)]), and with
    labels=True also the label image (component i + 1 is boxes[i])."""
    import cv2

    lab = _to_lab(img)
    d = np.nan_to_num(np.linalg.norm(lab - bg, axis=-1))
    d_alt = np.linalg.norm(lab - bg_alt, axis=-1)
    edge = np.zeros(pid.shape, np.uint8)
    edge[:, 1:] |= pid[:, 1:] != pid[:, :-1]
    edge[1:, :] |= pid[1:, :] != pid[:-1, :]
    near = cv2.dilate(edge, np.ones((2 * seam + 1, 2 * seam + 1), np.uint8)) > 0
    use_alt = near & np.isfinite(d_alt)
    d = np.where(use_alt, np.minimum(d, np.nan_to_num(d_alt)), d)
    m = (d > thr).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    # Split regions joined by thin bridges (door frames, baseboards): seed
    # from the thick cores (distance to the background > core px), then grow
    # each core back out inside the mask.
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents((dt > core).astype(np.uint8), connectivity=8)
    lab = lab.astype(np.int32)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(int(3 * core)):
        grown = cv2.dilate(lab.astype(np.float32), k3).astype(np.int32)
        lab = np.where((lab == 0) & (m > 0), grown, lab)
    boxes = []
    out = np.zeros_like(lab)
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= min_area:
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
            out[ys, xs] = len(boxes)
    if labels:
        return m.astype(bool), boxes, out
    return m.astype(bool), boxes


def grabcut_region(img: np.ndarray, outlier: np.ndarray, box, iters: int = 5, pad: float = 0.15):
    """GrabCut a proposal: inside its box the outlier pixels are probably
    object and the rest probably background; a margin around the box is
    definitely background. Returns (mask over the window, (X0, Y0))."""
    import cv2

    H, W = img.shape[:2]
    x, y, w, h = box
    px, py = int(pad * w) + 4, int(pad * h) + 4
    X0, Y0 = max(0, x - 2 * px), max(0, y - 2 * py)
    X1, Y1 = min(W, x + w + 2 * px), min(H, y + h + 2 * py)
    crop = np.ascontiguousarray((np.clip(img[Y0:Y1, X0:X1], 0, 1) * 255).astype(np.uint8)[..., ::-1])
    mask = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    a0, b0 = max(0, y - py) - Y0, min(H, y + h + py) - Y0
    a1, b1 = max(0, x - px) - X0, min(W, x + w + px) - X0
    mask[a0:b0, a1:b1] = cv2.GC_PR_BGD
    sub = outlier[Y0:Y1, X0:X1]
    inner = np.zeros_like(sub)
    inner[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = True
    mask[sub & inner] = cv2.GC_PR_FGD
    if (mask == cv2.GC_PR_FGD).sum() < 20 or (mask == cv2.GC_BGD).sum() < 20:
        return None, (X0, Y0)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(crop, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None, (X0, Y0)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), (X0, Y0)


def mirror_widths(lab: np.ndarray, fg: np.ndarray, c2: int, max_hw: int, tol: float = 18.0, gap: int = 2):
    """Half-widths of the mirror-symmetric shape about the axis at column
    c2 / 2 (c2 odd: between two pixels). On every row, walk outwards in both
    directions while both pixels are object (fg) and their colours match
    (Lab distance, lightness at half weight so one-sided lighting is
    tolerated); stop at a mismatch that lasts more than gap pixels."""
    H, W = fg.shape
    d = np.arange(max_hw)
    xl = (c2 - 1) // 2 - d if c2 % 2 else c2 // 2 - d
    xr = (c2 + 1) // 2 + d if c2 % 2 else c2 // 2 + d
    inb = (xl >= 0) & (xr < W)
    d, xl, xr = d[inb], xl[inb], xr[inb]
    if not len(d):
        return np.zeros(H)
    diff = lab[:, xl] - lab[:, xr]
    diff[..., 0] *= 0.5
    ok = fg[:, xl] & fg[:, xr] & (np.linalg.norm(diff, axis=-1) < tol)
    # The shape ends at the first step that starts gap+1 consecutive failures
    # (steps beyond the window count as failures).
    bad = ~ok
    run = bad.copy()
    for k in range(1, gap + 1):
        run[:, :-k] &= bad[:, k:]
    run[:, -1] = True
    first = np.argmax(run, axis=1)
    hw = first.astype(float) + (0.5 if c2 % 2 else 0.0)
    hw[~ok[:, 0]] = 0
    return hw


def symmetry_axes(lab: np.ndarray, fg: np.ndarray, max_axes: int = 3, min_frac: float = 0.35, max_hw: int = 120):
    """Candidate vertical symmetry axes inside a mask: the local maxima over
    the axis position of S = the area of the mirror-symmetric shape about it
    (mirror_widths). Colour matters, so two touching objects of different
    colours give two peaks. Returns [(axis column, half-width per row)]."""
    H, W = fg.shape
    cols = np.nonzero(fg.any(axis=0))[0]
    if len(cols) < 3:
        return []
    c2s = np.arange(2 * cols[0], 2 * cols[-1] + 1)
    widths = [mirror_widths(lab, fg, c2, max_hw) for c2 in c2s]
    S = np.array([w.sum() for w in widths])
    if S.max() <= 0:
        return []
    n = len(S)
    peaks = [i for i in range(n) if S[i] > 0 and S[i] >= S[max(0, i - 1)] and S[i] >= S[min(n - 1, i + 1)]]
    peaks.sort(key=lambda i: -S[i])
    out = []
    for i in peaks:
        if S[i] < min_frac * S.max():
            break
        # A separate axis needs a real dip in S between it and the others.
        if any(abs(i - o) < 6 or S[min(i, o):max(i, o) + 1].min() > 0.7 * min(S[i], S[o]) for o in out):
            continue
        out.append(i)
        if len(out) == max_axes:
            break
    return [(c2s[i] / 2, widths[i]) for i in out]


def mask_widths(fg: np.ndarray, c: float) -> np.ndarray:
    """Per row, the half-width of the mask's run through column c, taking
    the nearer of its two ends: something merged into one side (a hose,
    a neighbouring object) doesn't widen the profile."""
    H, W = fg.shape
    xl, xr = int(np.floor(c)), int(np.ceil(c))
    if xl < 0 or xr >= W:
        return np.zeros(H)
    left = np.zeros(H)
    on = fg[:, xl].copy()
    for x in range(xl, -1, -1):
        on &= fg[:, x]
        left += on
    right = np.zeros(H)
    on = fg[:, xr].copy()
    for x in range(xr, W):
        on &= fg[:, x]
        right += on
    hw = np.minimum(left, right) - (xr - xl) * 0.0
    return np.where(fg[:, xl] & fg[:, xr], hw - 0.5 + (c - xl if xl == xr else 0.5), 0.0)


def profile_block(hw: np.ndarray, min_rows: int = 8):
    """The longest run of rows with a shape (half-width > 1), median
    smoothed. Returns (rows, half_widths) or None."""
    ok = hw > 1
    best, start = (0, 0), None
    for y in range(len(hw) + 1):
        if y < len(hw) and ok[y]:
            start = y if start is None else start
        elif start is not None:
            if y - start > best[1] - best[0]:
                best = (start, y)
            start = None
    a, b = best
    if b - a < min_rows:
        return None
    h = hw[a:b]
    pad = np.pad(h, 2, mode="edge")
    h = np.median(np.stack([pad[i:i + len(h)] for i in range(5)]), axis=0)
    return np.arange(a, b), h


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


def straight_edge_support(gy, vp, c: float, hw: float, y: float, thr: float, search: int,
                          continued: bool = True) -> float:
    """Best support for a straight edge crossing the axis column c near row
    y, over [c - hw, c + hw], that is either horizontal or runs to the
    vanishing point: the top or bottom edge of a flat rectangle facing the
    camera (a poster, a box front) or on a side wall (a door, a panel).
    Support counts only where the line does not carry on past the object's
    sides (continued=True subtracts the support of its continuation)."""
    n = max(6, int(2 * hw))
    xs = np.linspace(c - hw, c + hw, n)
    # The same line continued past both sides: a flat object's edge stops at
    # its sides, a background line (a baseboard, a shelf) runs on behind it.
    gap = max(2.0, 0.15 * hw)
    out_l = np.linspace(c - 2 * hw - gap, c - hw - gap, n // 2)
    out_r = np.linspace(c + hw + gap, c + 2 * hw + gap, n // 2)
    best = 0.0
    for y_c in np.arange(y - search, y + search + 1):
        slopes = [0.0]
        if abs(c - vp[0]) > 1e-6:
            slopes.append((y_c - vp[1]) / (c - vp[0]))
        for slope in slopes:
            on = curve_support(gy, xs, y_c + slope * (xs - c), thr, shifts=0)
            if continued:
                beyond = max(curve_support(gy, o, y_c + slope * (o - c), thr, shifts=0) for o in (out_l, out_r))
                on -= beyond
            best = max(best, on)
    return best


# --------------------------------------------------------------------------- main
def find_solids(image: np.ndarray, scene, max_side: int = 800, min_height_px: int = 24,
                size_range=(0.12, 2.2), max_radius: float = 0.6, max_candidates: int = 60, min_outline: float = 0.5,
                min_sides: float = 0.5, min_halfwidth_px: float = 8.0, debug: list | None = None, proposals: str | None = None) -> list[Solid]:
    """Detect upright surfaces of revolution in a photo whose room geometry
    (scene, from geometry.build_scene) is known. Returns full-res Solids.

    proposals: "room" (default when OpenCV is installed): regions that differ
    from the empty-room background, traced with OpenCV's GrabCut, axes from
    the mask's symmetry. "seeds" (pure numpy): mirror-symmetric edge pairs,
    traced with symmetric_grabcut."""
    from .geometry import solid_geometry

    if proposals is None:
        try:
            import cv2  # noqa: F401
            proposals = "room"
        except ImportError:
            proposals = "seeds"

    small, s = downscale(image, max_side)
    Hs, Ws = small.shape[:2]
    gx, gy = color_gradients(small, 1.0)
    to_full = lambda v: (np.asarray(v, float) + 0.5) / s - 0.5  # noqa: E731
    floor_y = scene.yf
    gmag = np.linalg.norm(gx, axis=-1)

    def side_strength(c, rows, hw):
        """Per row, the weaker of the two sides' edge strengths (best within
        a pixel of the traced side)."""
        out = []
        for side in (-1, 1):
            xs = c + side * hw
            xi = np.clip(np.round(xs[:, None] + np.arange(-1, 2)).astype(int), 0, Ws - 1)
            out.append(gmag[rows[:, None], xi].max(axis=1))
        return np.minimum(*out)

    cands = []
    if proposals == "room":
        outlier, boxes = outlier_proposals(small, *room_background(small, scene, s))
        for box in boxes:
            fg, (X0, Y0) = grabcut_region(small, outlier, box)
            if fg is None:
                continue
            lab = _to_lab(small[Y0:Y0 + fg.shape[0], X0:X0 + fg.shape[1]])
            for a, _ in symmetry_axes(lab, fg):
                r = profile_block(mask_widths(fg, a))
                if r is None:
                    continue
                rows, hw = r
                c, rows = a + X0, rows + Y0
                if len(hw) < max(8, min_height_px * s) or hw.max() < min_halfwidth_px:
                    continue  # too few pixels to judge roundness
                strength = side_strength(c, rows, hw)
                level = float(np.median(strength))
                cands.append(dict(c=c, rows=rows, hw=hw, score=float(len(rows)), level=level,
                                  sides=float(np.mean(strength > 0.05))))
    else:
        pairs = symmetric_pairs(gx, gy, w_range=(6, int(0.25 * Ws)))
        taken = []  # (c, y0, y1) of traced profiles, to skip duplicate seeds
        for c2, y0, y1, nrows, chain in axis_seeds(pairs, Ws):
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
            cands.append(dict(c=c, rows=rows, hw=hw, score=float(len(rows)), level=level, sides=1.0))

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
        if cd["sides"] < min_sides:
            why = f"sides not on image edges ({cd['sides']:.0%})"
        # A solid's sides end together. An attachment may run on past one end
        # (a hose below an extinguisher), but not both; a rectangle receding
        # along a wall, or the edge of a door, is uneven at both ends.
        elif abs(lu - ru) > tol and abs(ld_b - rd_b) > tol:
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
