"""Automatic foreground cutouts: things standing on the floor in front of the
walls (people, chairs, lecterns) that are neither room-aligned boxes nor
surfaces of revolution. They become classic TIP billboards (Foreground),
standing where they meet the floor.

Where they are is decided by the room geometry; only tracing their outline
uses a learned model (Segment Anything, ViT-B), as a fallback for shapes
classical tracing can't separate: a robed figure in a painting shares its
colours with the walls and floor around it, so the empty-room background,
per-surface colour models and GrabCut all fail on it.

1. SAM proposes masks over the whole image (automatic mask generator).
2. The room geometry keeps the ones that stand in the room: their lowest
   point is on the floor in front of the back wall, a good part of them
   rises above the floor into a wall's image region (a flat rug or book on
   the floor stays flat), their height when stood up there is plausible,
   and their bottom isn't a line receding along a wall (a door).
3. A kept mask is often only part of the object (a robe, below a desk that
   hides the waist). SAM is prompted again with a point at its centre and a
   box reaching up above it, and the most stable of its answers that still
   covers the part is taken: the whole figure, without the cabinet behind it.

Returns Foregrounds (polygon + contact row) in full-res pixels.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .annotations import Foreground

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "sam_vit_b_01ec64.pth"


def sam_available(checkpoint: str | os.PathLike | None = None) -> bool:
    try:
        import segment_anything  # noqa: F401
    except ImportError:
        return False
    return Path(checkpoint or os.environ.get("STEPINSIDE_SAM", DEFAULT_CHECKPOINT)).exists()


def _load_sam(checkpoint):
    from segment_anything import sam_model_registry

    # MPS lacks float64, which SAM's mask post-processing uses; CPU is fine (~25 s).
    return sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to("cpu")


def _plane_map(image: np.ndarray, scene):
    """Per full-res pixel: (True where the room's floor is seen, plane index)."""
    import cv2

    from .auto_annotate import downscale
    from .revolve import room_background

    H, W = image.shape[:2]
    small, s = downscale(image, 800)
    _, _, pid = room_background(small, scene, s)
    pid = cv2.resize(pid.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(int)
    floor = np.array([p.name.endswith("_floor") for p in scene.planes] + [False])
    return floor[np.where(pid >= 0, pid, len(scene.planes))], pid


def _polygons(mask: np.ndarray, eps: float = 1.0, min_frac: float = 0.05):
    """Outlines of the mask's pieces. An object is often split by something
    in front of it (a desk across a figure's waist); every piece of at least
    min_frac of the largest one is kept."""
    import cv2

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    big = max(cv2.contourArea(c) for c in cs)
    out = []
    for c in sorted(cs, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_frac * big:
            continue
        c = cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(float)
        if len(c) >= 3:
            out.append(c.tolist())
    return out


def _bridge(mask: np.ndarray, gap: int) -> np.ndarray:
    """Join separate pieces of one object split by something in front of it.
    In each column where one piece lies directly above another, fill the
    gap between them if it is at most gap px. Unlike a closing, this leaves
    the outline's own dips (around a head, between arms) alone."""
    import cv2

    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    out = mask.copy()
    H, W = mask.shape
    for x in np.nonzero(mask.any(axis=0))[0]:
        col = lab[:, x]
        ys = np.nonzero(col)[0]
        # Runs of one label down the column; fill between runs of different pieces.
        breaks = np.nonzero(np.diff(ys) > 1)[0]
        starts = np.r_[ys[0], ys[breaks + 1]]
        ends = np.r_[ys[breaks], ys[-1]]
        for e, s0 in zip(ends[:-1], starts[1:]):
            if s0 - e - 1 <= gap and col[e] != col[s0]:
                out[e + 1:s0, x] = True
    return out


def find_cutouts(image: np.ndarray, scene, checkpoint=None, min_area: float = 0.003, max_area: float = 0.4,
                 min_above: float = 0.25, height_range=(0.4, 3.5), max_recede: float = 1.12,
                 solids=(), min_stand_ratio: float = 2.5, bridge: float = 0.12,
                 debug: list | None = None) -> tuple[list[Foreground], list[int]]:
    """Standing foreground objects as billboards (see the module docstring).

    solids: image-space silhouettes (boolean masks) of the solids already
    found. A mask mostly inside one is explained by it and skipped. But a
    solid that covers only part of an object (a dog's body without its
    head: a lathe can't hold a dog) is replaced by the cutout.

    An object whose pixels all lie on the floor can still be standing (a
    dog): if laid flat it would be a strip more than min_stand_ratio times
    longer in depth than it is wide, which a rug or a book is not.

    Returns (cutouts, indices of the solids they replace)."""
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor

    H, W = image.shape[:2]
    u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    vy = scene.principal[1]
    isfloor, pid = _plane_map(image, scene)
    # The back wall's bottom row: where the floor starts below the VP.
    col = isfloor[:, int(np.clip(scene.principal[0], 0, W - 1))]
    back_bottom = float(np.argmax(col)) if col.any() else vy
    cam_h = float(-scene.yf)
    f = float(scene.focal)
    import cv2

    # Distance (px) to the nearest floor/wall seam.
    seam = np.zeros((H, W), np.uint8)
    seam[1:] |= (isfloor[1:] != isfloor[:-1]).astype(np.uint8)
    seam[:, 1:] |= (isfloor[:, 1:] != isfloor[:, :-1]).astype(np.uint8)
    seam_dist = cv2.distanceTransform(1 - seam, cv2.DIST_L2, 3)
    seam_tol = 0.01 * H
    # ... and to any seam between room surfaces (wall/ceiling too).
    edge = np.zeros((H, W), np.uint8)
    edge[1:] |= (pid[1:] != pid[:-1]).astype(np.uint8)
    edge[:, 1:] |= (pid[:, 1:] != pid[:, :-1]).astype(np.uint8)
    room_dist = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 3)

    def on_room_seams(m):
        """Fraction of the mask's outline (away from the image border) that
        runs along seams between room surfaces: a piece of wall is bounded
        by them, an object standing in the room is not."""
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = np.concatenate([c[:, 0, :] for c in cs])
        inner = (pts[:, 0] > 2) & (pts[:, 0] < W - 3) & (pts[:, 1] > 2) & (pts[:, 1] < H - 3)
        pts = pts[inner]
        return float(np.mean(room_dist[pts[:, 1], pts[:, 0]] <= seam_tol)) if len(pts) else 1.0

    sam = _load_sam(checkpoint or os.environ.get("STEPINSIDE_SAM", DEFAULT_CHECKPOINT))
    masks = SamAutomaticMaskGenerator(sam, points_per_side=24, pred_iou_thresh=0.86,
                                      stability_score_thresh=0.9).generate(u8)

    def depth_ratio(m, ys, xs):
        """How far the mask's bottom recedes from one end to the other (z ratio)."""
        x0, x1 = xs.min(), xs.max()
        q = max(1, (x1 - x0) // 7)
        bot = lambda a, b: max(ys[(xs >= a) & (xs <= b)].max() - vy, 1.0)  # noqa: E731
        l, r = bot(x0, x0 + q), bot(x1 - q, x1)
        return max(l, r) / min(l, r)

    def hugs_seam(ys, xs):
        """Fraction of columns whose lowest pixel lies on a floor/wall seam:
        a strip of wall or a door ends along the seam across its width; a
        chair touches it only with its feet."""
        low = np.full(W, -1)
        np.maximum.at(low, xs, ys)
        cols = np.nonzero(low >= 0)[0]
        return float(np.mean(seam_dist[low[cols], cols] <= seam_tol))

    def stands_up(xs, yb, ytop):
        """Laid flat on the floor, would it be an implausibly long strip?"""
        if ytop <= vy + 1:
            return True  # reaches the horizon: it can't be flat on the floor
        z_near, z_far = f * cam_h / (yb - vy), f * cam_h / (ytop - vy)
        width = (xs.max() - xs.min() + 1) * z_near / f
        return (z_far - z_near) > min_stand_ratio * width

    def check(m):
        a = m.mean()
        ys, xs = np.nonzero(m)
        yb, ytop = int(ys.max()), int(ys.min())
        on_floor = isfloor[yb, xs[ys >= yb - 2]].mean()
        above = 1 - isfloor[m].mean()
        hm = cam_h * (yb - ytop) / max(yb - vy, 1)
        why = None
        if not min_area <= a <= max_area:
            why = f"area {a:.3f}"
        elif on_floor < 0.5 or yb < back_bottom + 0.02 * H:
            why = "bottom not on the floor in front of the back wall"
        elif yb >= H - 3:
            why = "cut off by the image bottom"
        elif above < min_above and not stands_up(xs, yb, ytop):
            why = f"flat on the floor ({above:.0%} above it)"
        elif not height_range[0] <= hm <= height_range[1]:
            why = f"height {hm:.1f} m"
        elif depth_ratio(m, ys, xs) > max_recede:
            why = "bottom recedes along a wall"
        elif hugs_seam(ys, xs) > 0.6:
            why = "bottom runs along a floor/wall seam (part of a wall)"
        elif on_room_seams(m) > 0.35:
            why = "outline follows the room's seams (a piece of a room surface)"
        elif any((m & sv).sum() > 0.8 * m.sum() for sv in solids):
            why = "a solid explains it"
        return why, a, yb, ytop, (xs.min(), ytop, xs.max(), yb)

    cands = []
    for k, rec in enumerate(masks):
        m = rec["segmentation"]
        why, a, yb, ytop, bbox = check(m)
        cands.append(dict(k=k, mask=m, area=a, contact=yb, top=ytop, bbox=bbox, why=why))
    if debug is not None:
        debug.extend(cands)

    keep = []
    for c in sorted((c for c in cands if c["why"] is None), key=lambda c: -c["area"]):
        if any((c["mask"] & d["mask"]).sum() > 0.5 * c["mask"].sum() for d in keep):
            continue
        keep.append(c)
    if not keep:
        return [], []

    # Grow each kept part to its whole object.
    pred = SamPredictor(sam)
    pred.set_image(u8)
    out = []
    for i, c in enumerate(keep):
        m = c["mask"]
        x0, ytop, x1, yb = c["bbox"]
        w = x1 - x0
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(dt.argmax(), dt.shape)  # the part's centre
        box = np.array([max(0, x0 - 0.15 * w), max(0, ytop - 1.2 * (yb - ytop)), min(W - 1, x1 + 0.15 * w),
                        min(H - 1, yb + 0.06 * H)])
        logits, _, _ = pred.predict(point_coords=np.array([[cx, cy]], float), point_labels=np.ones(1),
                                    box=box, multimask_output=True, return_logits=True)
        # SAM's answers range from the part to the part plus whatever is
        # behind it (a cabinet against the wall). Take the most stable answer
        # (least change between logit thresholds -1 and +1) that still covers
        # the part: leaks into a neighbouring object are the unstable ones.
        full, best = m, -1.0
        for lg in logits:
            mm = lg > 0
            stab = (lg > 1).sum() / max((lg > -1).sum(), 1)
            if (mm & m).sum() >= 0.85 * m.sum() and stab > best and check(mm | m)[0] is None:
                full, best = mm | m, stab
        # Whatever crosses the object in front of it (a desk board across a
        # figure's waist) splits the mask; it is in front anyway, so bridge
        # short vertical gaps and keep the figure in one piece.
        # Other objects SAM found on its own (the cabinet behind a figure)
        # extend well beyond the figure: remove their pixels from it.
        for rec in masks:
            o = rec["segmentation"]
            n_o = o.sum()
            if (o & m).sum() < 0.05 * n_o and (o & full).sum() < 0.5 * n_o:
                full &= ~o
        # Drop slivers (a cabinet's finial).
        r = max(1, int(0.003 * W))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        full = cv2.morphologyEx(full.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
        ys = np.nonzero(full)[0]
        if len(ys) == 0:
            continue
        full = _bridge(full, max(3, int(bridge * (ys.max() - ys.min()))))
        # Keep the pieces joined to the part that was found.
        n, lab = cv2.connectedComponents(full.astype(np.uint8), connectivity=8)
        joined = np.unique(lab[full & m])
        full = np.isin(lab, joined[joined > 0])
        if not full.any():
            continue
        ys = np.nonzero(full)[0]
        contact = float(ys.max())
        c["full"] = full
        # All pieces stand at the object's floor contact.
        for j, poly in enumerate(_polygons(full)):
            out.append(Foreground(f"cutout{i + 1}" + ("" if j == 0 else chr(ord("a") + j - 1)), poly, contact))
    replaced = sorted({j for c in keep if "full" in c for j, sv in enumerate(solids)
                       if (c["full"] & sv).sum() > 0.5 * sv.sum()})
    return out, replaced
