"""Turn the photograph into per-plane RGBA textures.

Each box texture is sampled by inverse mapping. For every texel we find its
3D point on the plane, project it into the photo and sample bilinearly. A
texel gets alpha = 0 (a hole) when its point falls outside the photo or
inside a foreground object, whose pixels belong to the billboard instead.
These holes are what a completion method has to fill.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .geometry import Plane, Scene, project


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample img (H, W, C) at float pixel coords. Returns (values, in_bounds)."""
    h, w = img.shape[:2]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= -0.5) & (x <= w - 0.5) & (y >= -0.5) & (y <= h - 0.5)
    xc = np.clip(np.nan_to_num(x), 0, w - 1)
    yc = np.clip(np.nan_to_num(y), 0, h - 1)
    x0 = np.floor(xc).astype(int)
    y0 = np.floor(yc).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    ax = (xc - x0)[..., None]
    ay = (yc - y0)[..., None]
    top = img[y0, x0] * (1 - ax) + img[y0, x1] * ax
    bot = img[y1, x0] * (1 - ax) + img[y1, x1] * ax
    return top * (1 - ay) + bot * ay, valid


def polygon_mask(polygons, width: int, height: int, dilate: int = 0) -> np.ndarray:
    """Boolean mask that is True inside any of the polygons."""
    m = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(m)
    for poly in polygons:
        d.polygon([tuple(p) for p in poly], fill=255)
    mask = np.asarray(m) > 0
    for _ in range(dilate):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return mask


def texture_resolution(plane: Plane, scene: Scene, max_res: int, n: int = 96) -> tuple[int, int]:
    """Choose (rows, cols) so one texel is about one photo pixel where the plane
    looks largest (its nearest visible part)."""
    s, t = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    u, v = project(plane.point(s, t), scene.focal, scene.principal)
    inside = (u >= 0) & (u < scene.width) & (v >= 0) & (v < scene.height)
    du = np.hypot(np.diff(u, axis=1), np.diff(v, axis=1))  # along s
    dv = np.hypot(np.diff(u, axis=0), np.diff(v, axis=0))  # along t
    in_s = inside[:, 1:] & inside[:, :-1]
    in_t = inside[1:, :] & inside[:-1, :]
    cols = (du[in_s].max() if in_s.any() else 1.0) * (n - 1)
    rows = (dv[in_t].max() if in_t.any() else 1.0) * (n - 1)
    clamp = lambda r: int(np.clip(round(r), 16, max_res))  # noqa: E731
    return clamp(rows), clamp(cols)


def box_texture(plane: Plane, scene: Scene, image: np.ndarray, fg_mask: np.ndarray, max_res: int) -> np.ndarray:
    rows, cols = texture_resolution(plane, scene, max_res)
    s = (np.arange(cols) + 0.5) / cols
    t = (np.arange(rows) + 0.5) / rows
    S, T = np.meshgrid(s, t)
    u, v = project(plane.point(S, T), scene.focal, scene.principal)
    rgb, valid = bilinear(image, u, v)
    ui = np.clip(np.round(np.nan_to_num(u)).astype(int), 0, scene.width - 1)
    vi = np.clip(np.round(np.nan_to_num(v)).astype(int), 0, scene.height - 1)
    valid &= ~fg_mask[vi, ui]
    alpha = valid.astype(np.float32)[..., None]
    return np.concatenate([rgb * alpha, alpha], axis=-1).astype(np.float32)


def foreground_texture(polygon, image: np.ndarray) -> np.ndarray:
    poly = np.asarray(polygon, dtype=float)
    u0, v0 = np.floor(poly.min(axis=0)).astype(int)
    u1, v1 = np.ceil(poly.max(axis=0)).astype(int)
    h, w = image.shape[:2]
    u0, v0 = max(u0, 0), max(v0, 0)
    u1, v1 = min(u1, w), min(v1, h)
    crop = image[v0:v1, u0:u1]
    mask = polygon_mask([poly - [u0, v0]], u1 - u0, v1 - v0).astype(np.float32)[..., None]
    return np.concatenate([crop * mask, mask], axis=-1).astype(np.float32)


def push_pull_fill(rgba: np.ndarray) -> np.ndarray:
    """Simple hole filling baseline: fill alpha==0 texels from a weighted
    image pyramid. The result is smooth, with no texture or line continuation."""
    rgb = rgba[..., :3] * rgba[..., 3:4]  # premultiplied
    a = rgba[..., 3:4].copy()
    if a.min() > 0.999:
        return rgba.copy()

    def down(x):
        h, w = x.shape[:2]
        x = x[: h - h % 2 or h, : w - w % 2 or w] if (h > 1 and w > 1) else x
        if x.shape[0] < 2 or x.shape[1] < 2:
            return x
        return 0.25 * (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2])

    pyr = [(rgb, a)]
    while min(pyr[-1][1].shape[:2]) > 1:
        c, al = pyr[-1]
        c2, a2 = down(c), down(al)
        if c2.shape == c.shape:
            break
        pyr.append((c2, a2))

    # Pull back down: at each level, fill low-alpha texels from the coarser level.
    filled_c, filled_a = pyr[-1]
    filled_c = np.where(filled_a > 1e-6, filled_c / np.maximum(filled_a, 1e-6), filled_c.mean((0, 1)))
    for c, al in reversed(pyr[:-1]):
        h, w = c.shape[:2]
        up = np.asarray(
            Image.fromarray(np.clip(filled_c * 255, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        own = np.where(al > 1e-6, c / np.maximum(al, 1e-6), 0)
        filled_c = own * al + up * (1 - al)
    out = np.concatenate([filled_c, np.ones_like(a)], axis=-1)
    return out.astype(np.float32)


def extract_textures(scene: Scene, image: np.ndarray, foreground_polys, max_res: int = 1536):
    """Return {plane_name: RGBA float texture in [0, 1]}."""
    fg_mask = polygon_mask(foreground_polys, scene.width, scene.height, dilate=3)
    textures = {}
    fg_iter = iter(foreground_polys)
    for plane in scene.planes:
        if plane.kind == "box":
            textures[plane.name] = box_texture(plane, scene, image, fg_mask, max_res)
        else:
            textures[plane.name] = foreground_texture(next(fg_iter), image)
    return textures


def load_image(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_pil(rgba: np.ndarray) -> Image.Image:
    arr = np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB")
