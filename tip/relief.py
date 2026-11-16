"""Find boxes fixed to the walls or ceiling, and pipes along the ceiling,
with classical CV only.

Both start from the same proposals as the solid detector (tip.revolve):
the empty-room background and the regions that don't match it.

Reliefs (signs, alarm boxes, speakers, hanging signs)
-----------------------------------------------------
Work in a frame tied to the wall: p is the distance from the vanishing
point across the wall (columns for a side wall, rows for the ceiling) and q
runs along it. A box fixed to the wall shows two faces:

* its near face, parallel to the image, so it is a true rectangle: from
  p = contact (where it meets the wall) in to p = front, over q in [qa, qb];
* its front face, parallel to the wall, so its top and bottom edges run
  toward the vanishing point: from p = front in to p = far.

A flat poster is the same shape with contact = front, so no near face. The
detector fits both hypotheses to the colour gradients of each proposal (edge
support along the predicted outline, refined by coordinate ascent) and keeps
the box only if the near face's edges are there: a vertical edge at front,
and top and bottom edges parallel to the image rows rather than receding.
The contact edge must border the wall it names. Flat things stay on the
wall, which already shows them correctly.

Tubes (pipes under the ceiling)
-------------------------------
A pipe along the room's depth axis is seen as a wedge between two lines
through the vanishing point (tip.geometry). In a polar resampling of the
ceiling about the vanishing point its two edges are rays of constant angle.
So each ray gets an edge score (how much of the ceiling along it has a
strong gradient across it), and a pipe is a pair of neighbouring edge rays,
narrow, both supported over the same long stretch, with a filling whose
colour differs from the empty ceiling's (the room background).
"""

from __future__ import annotations

import numpy as np

from .annotations import Relief, Tube


# ---------------------------------------------------------------- frames ----

def _to_image(wall: str, vx: float, vy: float):
    """(p, q) in the wall's frame -> image (u, v); and the vanishing point's q."""
    if wall == "right":
        return (lambda p, q: (vx + p, q)), vy
    if wall == "left":
        return (lambda p, q: (vx - p, q)), vy
    return (lambda p, q: (q, vy - p)), vx  # ceiling


def _to_frame(wall: str, vx: float, vy: float, u, v):
    if wall == "right":
        return u - vx, v
    if wall == "left":
        return vx - u, v
    return vy - v, u


def _sample(g: np.ndarray, u, v) -> np.ndarray:
    """Nearest-pixel samples of g (H, W, C) at float coords (NaN outside)."""
    H, W = g.shape[:2]
    ui = np.round(np.asarray(u)).astype(int)
    vi = np.round(np.asarray(v)).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    out = np.full(ui.shape + g.shape[2:], np.nan)
    out[ok] = g[vi[ok], ui[ok]]
    return out


def segment_support(gx, gy, a, b, thr: float, shifts: int = 1) -> tuple[float, float]:
    """(fraction of points with an edge, length) along the image segment a-b.
    An edge point has a colour gradient across the segment above thr, at
    the segment or up to `shifts` px either side of it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.hypot(*(b - a)))
    if L < 1:
        return 0.0, 0.0
    n = max(3, int(L))
    t = (np.arange(n) + 0.5) / n
    pts = a + t[:, None] * (b - a)
    d = (b - a) / L
    nrm = np.array([-d[1], d[0]])
    best = np.zeros(n)
    for k in range(-shifts, shifts + 1):
        q = pts + k * nrm
        g = np.abs(_sample(gx, q[:, 0], q[:, 1]) * nrm[0] + _sample(gy, q[:, 0], q[:, 1]) * nrm[1])
        best = np.maximum(best, np.nan_to_num(np.max(g, axis=-1)))
    return float((best > thr).mean()), L


def _ascend_free(score, prm, bounds, steps=(4, 2, 1), iters=4):
    """Coordinate ascent on the parameters prm within bounds."""
    prm = list(prm)
    best = score(prm)
    for step in steps:
        for _ in range(iters):
            moved = False
            for i in range(len(prm)):
                for dv in (-step, step):
                    cand = list(prm)
                    cand[i] = float(np.clip(cand[i] + dv, *bounds[i]))
                    s_ = score(cand)
                    if s_ > best + 1e-9:
                        best, prm, moved = s_, cand, True
            if not moved:
                break
    return prm, best


def bounds_ok(prm) -> bool:
    pc, pf, p1, qa, qb = prm
    return p1 <= pf <= pc and qb - qa >= 4 and p1 > 2


def hull_mask(rel: Relief, scene, s: float, X0: int, Y0: int, w: int, h: int):
    """Silhouette of a relief in a window of the downscaled photo: the convex
    hull of its box's projected corners (a box is convex). None if it
    doesn't meet its wall."""
    import cv2

    from .geometry import project, relief_box

    box = relief_box(rel, scene)
    if box is None:
        return None
    lo, hi = box
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    u, v = project(c, scene.focal, scene.principal)
    pts = np.stack([u * s - X0, v * s - Y0], -1)
    if not np.isfinite(pts).all():
        return None
    hull = cv2.convexHull(np.round(pts * 4).astype(np.int32))
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, hull, 1, lineType=cv2.LINE_8, shift=2)
    return m.astype(bool)


def fit_relief(fg, X0, Y0, wall, scene, s, min_thick_px=2.0, grads=None):
    """Fit the flat and the box hypothesis to a traced object (mask fg over
    the window at (X0, Y0) of the downscaled photo) by the overlap (IoU) of
    their silhouettes with it. Returns {"flat": (prm, iou), "box": (prm,
    iou)} with prm = (pc, pf, p1, qa, qb) in the wall frame (downscaled px),
    or None if the object isn't on that side of the vanishing point.

    The silhouette pins the front corner (pf) only loosely, since near the
    vanishing point a receding edge looks much like a straight one. So with
    grads = (gx, gy, thr), pf is then moved to the strongest edge between
    near and front face nearby, among placements whose IoU is within 0.01
    of the best."""
    vx, vy = (c * s for c in scene.principal)
    h, w = fg.shape
    ys, xs = np.nonzero(fg)
    if len(xs) < 20:
        return None
    p, q = _to_frame(wall, vx, vy, xs + X0 + 0.5, ys + Y0 + 0.5)
    if np.percentile(p, 2) <= 3:
        return None
    pc, p1 = np.percentile(p, 99.5), np.percentile(p, 0.5)
    rim = p >= pc - 3
    qa, qb = np.percentile(q[rim], 1), np.percentile(q[rim], 99)
    pad = max(4.0, 0.25 * (pc - p1))
    qpad = max(4.0, 0.25 * (qb - qa))
    area = fg.sum()

    def iou(prm):
        if not bounds_ok(prm):
            return -1.0
        c_, f_, r_, sp_ = _frame_to_image_params(wall, (vx, vy), prm)
        rel = Relief("fit", wall, c_ / s, f_ / s, r_ / s, [sp_[0] / s, sp_[1] / s])
        m = hull_mask(rel, scene, s, X0, Y0, w, h)
        if m is None:
            return -1.0
        inter = (m & fg).sum()
        return inter / (area + m.sum() - inter)

    bounds = [(pc - pad, pc + pad), (p1 - pad, pc + pad), (p1 - pad, p1 + pad), (qa - qpad, qa + qpad),
              (qb - qpad, qb + qpad)]
    # Flat: pf tied to pc.
    flat_score = lambda f4: iou([f4[0], f4[0], f4[1], f4[2], f4[3]])  # noqa: E731
    f4, f_iou = _ascend_free(flat_score, [pc, p1, qa, qb], [bounds[0], bounds[2], bounds[3], bounds[4]])
    flat_prm = [f4[0], f4[0], f4[1], f4[2], f4[3]]
    out = {"flat": (flat_prm, f_iou)}
    # Box: scan the front edge, then refine everything with pf <= pc - min_thick.
    # (From the traced rows at the contact edge: the flat fit tilts its top
    # and bottom to mimic a near face, which would mislead the scan.)
    best = None
    for pf in np.arange(p1, pc - min_thick_px + 0.5, 1.0):
        prm = [pc, float(pf), p1, qa, qb]
        sc_ = iou(prm)
        if best is None or sc_ > best[1]:
            best = (prm, sc_)
    if best is not None:
        box_score = lambda prm: iou(prm) if prm[0] - prm[1] >= min_thick_px else -1.0  # noqa: E731
        bp, b_iou = _ascend_free(box_score, best[0], bounds)
        if b_iou > 0 and grads is not None:
            gx, gy, thr = grads
            to_img, _ = _to_image(wall, vx, vy)
            reach = max(3.0, 0.3 * (bp[0] - bp[2]))
            options = []
            for pf in np.arange(max(bp[2], bp[1] - reach), min(bp[0] - min_thick_px, bp[1] + reach) + 0.01, 0.5):
                prm = [bp[0], float(pf), bp[2], bp[3], bp[4]]
                v = iou(prm)
                if v >= b_iou - 0.01:
                    e, _ = segment_support(gx, gy, to_img(pf, bp[3]), to_img(pf, bp[4]), thr)
                    options.append((e, v, prm))
            if options:
                _, b_iou, bp = max(options, key=lambda o: (round(o[0], 2), o[1]))
        if b_iou > 0:
            out["box"] = (bp, b_iou)
    return out


# ------------------------------------------------------------- detector ----

def find_reliefs(image: np.ndarray, scene, max_side: int = 800, min_px: float = 8.0, min_thick_px: float = 2.0,
                 min_iou: float = 0.75, min_gain: float = 0.02, min_edge: float = 0.5, max_area: float = 0.05,
                 max_out: float = 0.6, max_len: float = 2.0, debug: list | None = None,
                 proposals=None, solids=()) -> list[Relief]:
    """Boxes fixed to the side walls or ceiling of the room `scene`
    (built without reliefs). Proposals that are mostly one of `solids`
    (already found) are skipped. Returns Relief annotations in full-image px."""
    from .auto_annotate import downscale
    from .geometry import relief_box
    from .revolve import color_gradients, grabcut_region, outlier_proposals, room_background

    small, s = downscale(image, max_side)
    H, W = small.shape[:2]
    vx, vy = scene.principal
    vps = (vx * s, vy * s)
    if proposals is None:
        bg, bg_alt, pid = room_background(small, scene, s)
        outlier, boxes, labels = outlier_proposals(small, bg, bg_alt, pid, labels=True)
    else:
        outlier, boxes, labels, pid = proposals
    gx, gy = color_gradients(small, 1.0)
    gmag = np.linalg.norm(gx, axis=-1) + np.linalg.norm(gy, axis=-1)
    names = [p.name for p in scene.planes]
    taken = np.zeros((H, W), bool)
    for sol in solids:
        for y, xl, xr in np.asarray(sol.profile, float) * s:
            yi = int(round(y))
            if 0 <= yi < H:
                taken[yi, max(0, int(xl)):min(W, int(np.ceil(xr)) + 1)] = True
    found = []
    for i, (bx, by, bw, bh) in enumerate(boxes):
        comp = labels == i + 1
        if taken[comp].mean() > 0.3:
            continue
        if bw < min_px * 0.75 or bh < min_px * 0.75 or bw * bh > max_area * W * H:
            continue
        fg, (X0, Y0) = grabcut_region(small, outlier, (bx, by, bw, bh))
        fg = _main_blob(fg, comp[Y0:Y0 + fg.shape[0], X0:X0 + fg.shape[1]])
        if fg is None:
            continue
        win = gmag[by:by + bh, bx:bx + bw]
        thr = max(0.03, 0.25 * float(np.percentile(win, 95))) / 2
        best = None
        for wall in ("right", "left", "ceiling"):
            why = None
            fit = fit_relief(fg, X0, Y0, wall, scene, s, min_thick_px, grads=(gx, gy, 0.5 * thr))
            if fit is None or "box" not in fit:
                continue
            (bp, b_iou), (fp, f_iou) = fit["box"], fit["flat"]
            pc, pf, p1, qa, qb = bp
            to_img, q0 = _to_image(wall, *vps)
            # The contact edge borders the wall it names (a few px outside it).
            qs = np.linspace(qa, qb, 12)
            ou, ov = to_img(np.full_like(qs, pc + 3), qs)
            pids = _sample(pid[..., None].astype(float), ou, ov)[:, 0]
            on_wall = np.mean([np.isfinite(x) and names[int(x)].endswith("_" + wall) for x in pids])
            front_edge, _ = segment_support(gx, gy, to_img(pf, qa), to_img(pf, qb), 0.5 * thr)
            near_edges = min(segment_support(gx, gy, to_img(pf, q_), to_img(pc, q_), 0.5 * thr, shifts=0)[0]
                             for q_ in (qa, qb)) if pc - pf >= 3 else 0.0
            ends = [to_img(pc, qa), to_img(pc, qb), to_img(p1, qa), to_img(p1, qb)]
            if qb - qa < min_px or pc - p1 < min_px * 0.75:
                why = "too small"
            elif any(not (2 <= u_ <= W - 3 and 2 <= v_ <= H - 3) for u_, v_ in ends):
                why = "touches the image border"
            elif on_wall < 0.6:
                why = f"contact edge not on the {wall} wall ({on_wall:.0%})"
            elif b_iou < min_iou:
                why = f"silhouette doesn't fit (IoU {b_iou:.2f})"
            # Better than flat: a clearly better silhouette, or (for a box only a
            # little thicker than flat, whose silhouettes are alike) a near
            # face whose own outline is in the photo: its corner edge and its
            # top and bottom edges, which run straight across, not receding.
            elif b_iou < f_iou + min_gain and not (b_iou >= f_iou - 0.01 and min(front_edge, near_edges) >= 0.8):
                why = f"no better than flat (IoU {b_iou:.2f} vs {f_iou:.2f}, near face edges {near_edges:.2f})"
            elif front_edge < min_edge:
                why = f"no front corner edge ({front_edge:.2f})"
            c_, f_, r_, sp_ = _frame_to_image_params(wall, vps, bp)
            rel = Relief("relief", wall, c_ / s, f_ / s, r_ / s, [float(sp_[0]) / s, float(sp_[1]) / s])
            if why is None:
                box = relief_box(rel, scene)
                if box is None:
                    why = "doesn't meet the wall"
                else:
                    lo, hi = box
                    out = (hi - lo)[{"left": 0, "right": 0, "ceiling": 1}[wall]]
                    b0 = scene.boxes[0]
                    if out > max_out:
                        why = f"sticks out {out:.2f} m"
                    elif hi[2] - lo[2] > max_len:
                        why = f"too long ({hi[2] - lo[2]:.2f} m)"
                    elif lo[1] < b0.yf - 0.05 or hi[1] > max(b.yc for b in scene.boxes) + 0.05:
                        why = "outside the room"
            if debug is not None:
                debug.append(dict(box=(bx, by, bw, bh), wall=wall, box_iou=round(float(b_iou), 3),
                                  flat_iou=round(float(f_iou), 3), front_edge=round(front_edge, 2),
                                  on_wall=round(float(on_wall), 2), why=why, relief=rel))
            if why is None and (best is None or b_iou > best[0]):
                best = (b_iou, rel)
        if best is not None:
            found.append(best)
    found.sort(key=lambda x: -x[0])
    out = []
    for k, (_, rel) in enumerate(found):
        rel.name = f"relief{k + 1}"
        out.append(rel)
    return out


def _main_blob(fg: np.ndarray, seed: np.ndarray):
    """The connected part of the GrabCut mask fg that overlaps the proposal
    most (GrabCut can pick up neighbouring things in its window)."""
    import cv2

    n, lab = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    hits = np.bincount(lab[seed & (lab > 0)], minlength=n)
    k = int(np.argmax(hits))
    if hits[k] == 0:
        return None
    return lab == k


def _frame_to_image_params(wall, vp, prm):
    """Wall-frame fit (pc, pf, p1, qa, qb) -> Relief fields (contact, front,
    far, span) in image coordinates of the same scale."""
    vx, vy = vp
    pc, pf, p1, qa, qb = prm
    if wall == "right":
        r = vx + pc, vx + pf, vx + p1
    elif wall == "left":
        r = vx - pc, vx - pf, vx - p1
    else:
        r = vy - pc, vy - pf, vy - p1
    return (*(float(x) for x in r), [float(qa), float(qb)])


# ---------------------------------------------------------------- tubes ----

def find_tubes(image: np.ndarray, scene, max_side: int = 800, d_theta: float = 0.0015, min_half_px: float = 1.0,
               max_half: float = 0.05, min_edge: float = 0.35, min_run: float = 0.35, min_fill: float = 10.0, max_rungs: float = 0.25,
               debug: list | None = None, proposals=None) -> list[Tube]:
    """Pipes along the depth axis on the first box's ceiling. Returns Tube
    annotations in full-image px."""
    from .auto_annotate import downscale
    from .geometry import tube_geometry
    from .revolve import color_gradients, outlier_proposals, room_background

    small, s = downscale(image, max_side)
    H, W = small.shape[:2]
    vx, vy = (c * s for c in scene.principal)
    from .revolve import _to_lab

    if proposals is None:
        bg, bg_alt, pid = room_background(small, scene, s)
    else:
        bg, pid = proposals
    lab = _to_lab(small)
    names = [p.name for p in scene.planes]
    ceil = np.zeros((H, W), bool)
    for i, nm in enumerate(names):
        if nm.endswith("_ceiling"):
            ceil |= pid == i
    ceil_ids = [i for i, nm in enumerate(names) if nm.endswith("_ceiling")]
    # Pipes run in front of the ceiling, and the ceiling's edges with the
    # walls are themselves rays from the VP: keep a margin from them.
    import cv2

    ceil_in = cv2.erode(ceil.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    gx, gy = color_gradients(small, 1.0)

    # Polar grid: angles over the ceiling, radii out to the image corner.
    ys, xs = np.nonzero(ceil_in)
    if len(xs) < 200:
        return []
    ang = np.arctan2(-(ys + 0.5 - vy), xs + 0.5 - vx)  # 0 = right, pi/2 = up
    th = np.arange(ang.min(), ang.max(), d_theta)
    rmax = float(np.hypot(max(vx, W - vx), max(vy, H - vy)))
    rr = np.arange(4.0, rmax, 1.0)
    TH, RR = np.meshgrid(th, rr, indexing="ij")
    U, V = vx + RR * np.cos(TH), vy - RR * np.sin(TH)
    inside = _sample(ceil_in[..., None].astype(float), U, V)[..., 0] > 0.5
    # Gradient across the ray (along the tangent (-sin, -cos) in image coords).
    tu, tv = -np.sin(TH), -np.cos(TH)
    g = np.abs(_sample(gx, U, V) * tu[..., None] + _sample(gy, U, V) * tv[..., None])
    g = np.nan_to_num(np.max(g, axis=-1))
    # Gradient along the ray: a pipe is smooth along its length (a grating
    # or a cable tray is not).
    g_r = np.nan_to_num(np.max(np.abs(_sample(gx, U, V) * np.cos(TH)[..., None]
                                      - _sample(gy, U, V) * np.sin(TH)[..., None]), axis=-1))
    thr = max(0.02, float(np.percentile(g[inside], 85))) if inside.any() else 0.02
    edge = (g > thr) & inside
    edge = edge | np.roll(edge, 1, axis=0) | np.roll(edge, -1, axis=0)  # +-1 angle step jitter
    # How far each sample is from the empty ceiling's colour.
    diff = np.linalg.norm(_sample(lab, U, V) - _sample(bg, U, V), axis=-1)
    diff_in = np.where(inside, diff, np.nan)
    merged_lo = lambda a_, K_: a_ - K_  # noqa: E731
    n_in = inside.sum(axis=1)
    score = np.where(n_in > 0, edge.sum(axis=1) / np.maximum(n_in, 1), 0)
    # Edge rays: local maxima of the score.
    sm = np.convolve(score, np.ones(3) / 3, mode="same")
    peaks = [k for k in range(1, len(th) - 1) if sm[k] >= min_edge and sm[k] >= sm[k - 1] and sm[k] >= sm[k + 1]]
    # Merge plateaus.
    merged = []
    for k in peaks:
        if merged and k - merged[-1] <= 2:
            if sm[k] > sm[merged[-1]]:
                merged[-1] = k
        else:
            merged.append(k)
    cands = []
    for a, b in zip(merged, merged[1:]):
        half = (th[b] - th[a]) / 2
        if half > max_half or half < 1e-4:
            continue
        both = edge[a] & edge[b] & inside[a] & inside[b]
        run = _longest_run(both, gap=6)
        if run is None:
            continue
        r0, r1 = run
        mid = (a + b) // 2
        valid = inside[mid]
        if valid.sum() < 20:
            continue
        run_frac = (r1 - r0) / valid.sum()
        inner = diff[a + 1:b, r0:r1] if b - a > 1 else diff[a:b + 1, r0:r1]
        fill = float(np.nanmedian(inner)) if np.isfinite(inner).any() else 0.0
        rungs = float((g_r[a + 1:b, r0:r1] > thr).mean()) if b - a > 1 else 0.0
        # The two edge rays can be a highlight inside the pipe, and a real
        # pipe drifts a little in angle along its length (it isn't exactly
        # parallel to the depth axis). So measure its angular width in four
        # stretches: the contiguous rays, around the brightest-differing one,
        # whose colour still differs from the empty ceiling.
        K = int(max_half / d_theta)
        lim = max(min_fill, 0.35 * fill)
        lo_k, hi_k = max(0, a - K), min(len(th), b + K + 1)
        halves, centres = [], []
        for c0 in np.array_split(np.arange(r0, r1), 4):
            if len(c0) < 5:
                continue
            blk = diff_in[lo_k:hi_k, c0[0]:c0[-1] + 1]
            ok_rows = np.isfinite(blk).mean(1) > 0.5
            rf = np.full(len(blk), -1.0)
            rf[ok_rows] = np.nanmedian(blk[ok_rows], axis=1)
            ks = np.arange(lo_k, hi_k)
            k0 = int(np.argmax(np.where((ks >= a) & (ks <= b), rf, -np.inf)))
            if rf[k0] < lim:
                continue
            i0 = i1 = k0
            while i0 > 0 and rf[i0 - 1] >= lim:
                i0 -= 1
            while i1 < len(rf) - 1 and rf[i1 + 1] >= lim:
                i1 += 1
            halves.append((i1 - i0 + 1) * d_theta / 2)
            centres.append(th[lo_k + (i0 + i1) // 2])
        if halves:
            half_w = float(np.median(halves))
            mid_k = int(np.argmin(np.abs(th - np.median(centres))))
            w_k = max(1, int(round(half_w / d_theta)))
            a, b = max(0, mid_k - w_k), min(len(th) - 1, mid_k + w_k)
        half = (th[b] - th[a]) / 2
        r_px = rr[r1 - 1] * np.sin(half)  # half-width at the near end
        why = None
        if r_px < min_half_px:
            why = f"too thin ({r_px:.1f} px)"
        elif run_frac < min_run or r1 - r0 < 40:
            why = f"short ({run_frac:.0%})"
        elif fill < min_fill:
            why = f"looks like the empty ceiling (Lab {fill:.0f})"
        elif min(sm[max(0, a - 2):a + 3].max(), sm[max(0, b - 2):b + 3].max()) < 0.8 * min_edge:
            why = "its full width isn't bounded by edges on both sides"
        elif rungs > max_rungs:
            why = f"edges across it ({rungs:.0%}), not a smooth pipe"
        tc = (th[a] + th[b]) / 2
        near = [(vx + rr[r1 - 1] * np.cos(tc)) / s, (vy - rr[r1 - 1] * np.sin(tc)) / s]
        far = [(vx + rr[r0] * np.cos(tc)) / s, (vy - rr[r0] * np.sin(tc)) / s]
        tube = Tube("tube", near, far, float(half))
        if why is None and tube_geometry(tube, scene) is None:
            why = "doesn't fit under the ceiling"
        if half > max_half and why is None:
            why = f"too wide ({np.degrees(half):.1f} deg)"
        if debug is not None:
            debug.append(dict(theta=round(float(np.degrees(tc)), 2), half_deg=round(float(np.degrees(half)), 2),
                              edges=(round(float(sm[a]), 2), round(float(sm[b]), 2)), run=round(float(run_frac), 2),
                              fill=round(float(fill), 2), rungs=round(rungs, 2), why=why, tube=tube))
        if why is None:
            cands.append((float(min(sm[a], sm[b]) * run_frac), (a, b), tube))
    # Widened wedges can overlap: keep the best of each overlapping group.
    cands.sort(key=lambda x: -x[0])
    kept = []
    for sc_, (a, b), t in cands:
        if all(b < a2 or a > b2 for _, (a2, b2), _ in kept):
            kept.append((sc_, (a, b), t))
    kept.sort(key=lambda x: x[2].near[0])
    for k, (_, _, t) in enumerate(kept):
        t.name = f"tube{k + 1}"
    return [t for _, _, t in kept]


def _longest_run(mask: np.ndarray, gap: int = 6):
    """(start, end) of the longest run of True in a 1-D mask, bridging gaps
    of up to `gap` False entries; None if there is none."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return None
    best, start, prev = None, idx[0], idx[0]
    for i in list(idx[1:]) + [None]:
        if i is not None and i - prev <= gap + 1:
            prev = i
            continue
        if best is None or prev + 1 - start > best[1] - best[0]:
            best = (start, prev + 1)
        if i is not None:
            start = prev = i
    return best
