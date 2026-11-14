"""Offline ray-cast renderer for novel views of a TIP scene.

The interactive viewer is viewer/index.html (three.js). This renderer gives
the same picture from Python, for figures and for checking that the original
camera reproduces the input photo.

Box planes are hit-tested (from inside the box exactly one face is hit).
Foreground billboards are then composited over the box from far to near.
As in classic TIP they are drawn on top of the box, so a billboard slightly
outside a wall still shows.
"""

from __future__ import annotations

import numpy as np

from .geometry import Plane, Scene
from .textures import bilinear


def rotation(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """Camera-to-world rotation. Positive yaw turns right, positive pitch looks up."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    return Ry @ Rx


def _hit(plane: Plane, cam: np.ndarray, dirs: np.ndarray):
    """Ray/rectangle intersection. Returns (t, s, tt, valid)."""
    k = plane.normal_axis
    c = plane.origin[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (c - cam[k]) / dirs[..., k]
    p = cam + t[..., None] * dirs
    rel = p - plane.origin
    s = rel @ plane.u_axis / (plane.u_axis @ plane.u_axis)
    tt = rel @ plane.v_axis / (plane.v_axis @ plane.v_axis)
    eps = 1e-6
    valid = np.isfinite(t) & (t > 1e-6) & (s >= -eps) & (s <= 1 + eps) & (tt >= -eps) & (tt <= 1 + eps)
    return t, s, tt, valid


def _sample(tex: np.ndarray, s, tt):
    h, w = tex.shape[:2]
    vals, _ = bilinear(tex, s * w - 0.5, tt * h - 0.5)
    return vals


def render_view(
    scene: Scene,
    textures: dict,
    position=(0.0, 0.0, 0.0),
    yaw: float = 0.0,
    pitch: float = 0.0,
    scale: float = 1.0,
    show_foreground: bool = True,
    background=(0.0, 0.0, 0.0),
):
    """Render RGB (H, W, 3) and a hole mask (H, W) from a moved camera.

    The camera keeps the input camera's intrinsics. position is in scene units
    (meters, since the floor is camera_height below the input camera).
    """
    W, H = int(round(scene.width * scale)), int(round(scene.height * scale))
    f = scene.focal * scale
    vx, vy = scene.principal[0] * scale, scene.principal[1] * scale
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dirs = np.stack([(u - vx) / f, -(v - vy) / f, np.ones_like(u)], axis=-1)
    dirs = dirs @ rotation(yaw, pitch).T
    cam = np.asarray(position, dtype=float)

    best_t = np.full((H, W), np.inf)
    color = np.zeros((H, W, 3))
    alpha = np.zeros((H, W))
    for plane in scene.planes:
        if plane.kind != "box":
            continue
        t, s, tt, valid = _hit(plane, cam, dirs)
        closer = valid & (t < best_t)
        if not closer.any():
            continue
        vals = _sample(textures[plane.name], s[closer], tt[closer])
        best_t[closer] = t[closer]
        color[closer] = vals[:, :3]
        alpha[closer] = vals[:, 3]

    # Textures are premultiplied, so composite as premultiplied too.
    out = color + (1 - alpha)[..., None] * np.asarray(background)
    hole = alpha < 0.5

    if show_foreground:
        fgs = [p for p in scene.planes if p.kind == "foreground"]
        fgs.sort(key=lambda p: -np.linalg.norm(p.origin - cam))
        for plane in fgs:
            t, s, tt, valid = _hit(plane, cam, dirs)
            if not valid.any():
                continue
            vals = _sample(textures[plane.name], s[valid], tt[valid])
            a = vals[:, 3:4]
            out[valid] = vals[:, :3] + (1 - a) * out[valid]
            hole[valid] &= a[:, 0] < 0.5

    return np.clip(out, 0, 1), hole
