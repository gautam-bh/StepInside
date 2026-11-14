# StepInside

CS180 final project: turn a single interior photograph into a scene you can step into, moving a little sideways or forward and seeing objects shift against the walls behind them.

The starting point is **Tour Into the Picture** (TIP; Horry, Anjyo & Arai, SIGGRAPH 1997). The room is a textured box (back wall, floor, ceiling, left and right walls), and foreground objects are flat cutouts standing at their estimated depth. When the camera moves, it reveals regions the photo never saw. The project's research question is how to fill those: later we will add diffusion inpainting on the rectified wall and floor textures and compare it against inpainting the original photo directly.

**Status:** TIP baseline working on the hallway scene, with holes shown and a simple push-pull fill. Diffusion completion is not started.

| Annotation | Novel views (top: holes in magenta, bottom: push-pull fill) |
|---|---|
| <img src="outputs/hallway/annotation.jpg" width="240"> | <img src="outputs/hallway/contact_sheet.jpg" width="600"> |

Camera sweep with the simple fill: [`outputs/hallway/sweep.gif`](outputs/hallway/sweep.gif)

## Quick start

```bash
pip install -r requirements.txt          # numpy, pillow, matplotlib

# 1. Build the scene (geometry + per-plane textures) for the viewer
python scripts/build_scene.py data/annotations/hallway.json

# 2. Interactive viewer: serve the repo root, then open the page
python -m http.server 8000
#    -> http://localhost:8000/viewer/?scene=hallway

# 3. Offline renders for figures (novel views, contact sheet, GIF)
python scripts/render_views.py data/annotations/hallway.json

# Sanity tests
python tests/test_tip.py
```

**Viewer controls:** drag to move sideways and up/down · scroll or `W`/`S` to walk forward and back · `A`/`D` strafe · `Q`/`E` up/down · `Shift`+drag to look around · `F` toggles the simple fill · `G` toggles the foreground · `H` shows box edges · `Space` wiggles · `R` resets. Holes (no photo data) show in magenta.

## Adding a new scene

```bash
python scripts/annotate.py data/images/my_room.jpg          # click back wall, vanishing point, foreground outlines
python scripts/show_annotation.py data/annotations/my_room.json -o outputs/my_room/annotation.png   # check it
python scripts/build_scene.py data/annotations/my_room.json
```

You can also write the annotation JSON by hand (see [`data/annotations/hallway.json`](data/annotations/hallway.json)):

| field | meaning |
|---|---|
| `vanishing_point` | `[x, y]` pixel where the room's depth lines converge; must lie inside the back wall |
| `back_wall` | `[x_left, y_top, x_right, y_bottom]` rectangle of the far wall |
| `focal_35mm` | 35mm-equivalent focal length (iPhone main camera ≈ 26) |
| `camera_height` | eye height in meters; sets the scene's scale |
| `foreground[]` | `{name, polygon: [[x, y], ...], contact_y?}`; `contact_y` is the row where the object meets the floor (defaults to the polygon's lowest point) |

Good photos have one-point perspective (looking straight down the room), visible floor/wall/ceiling edges, sharp focus, and few reflective surfaces.

## How it works

1. **Camera.** Pinhole camera at the origin with the vanishing point as the principal point (the classic TIP assumption), so the room's depth axis is the optical axis. Focal length comes from `focal_35mm`.
2. **Box.** The back-wall rectangle is placed at depth `D`, chosen so the floor sits `camera_height` below the camera. Floor, ceiling and side walls run from the back wall toward the camera, out to where they leave the image ([`tip/geometry.py`](tip/geometry.py)).
3. **Textures.** For every texel on every plane, we find its 3D point, project it into the photo and sample bilinearly (inverse mapping, so there are no gaps). This removes the perspective distortion: each wall and floor becomes a flat, front-on image, which is where completion will happen. Texels that fall outside the photo or inside a foreground object get `alpha = 0`; these are the holes ([`tip/textures.py`](tip/textures.py)).
4. **Foreground.** Each outlined object becomes a billboard parallel to the image plane. Its depth comes from where its bottom meets the floor. Like classic TIP, billboards draw on top of the box.
5. **Rendering.** [`viewer/index.html`](viewer/index.html) uses three.js, with `setViewOffset` to reproduce the off-center principal point. [`tip/render.py`](tip/render.py) is a NumPy ray-caster that produces the same views for figures and tests. From the original camera it reproduces the input photo (checked in `tests/test_tip.py`).
6. **Simple fill baseline.** `push_pull_fill` fills holes from a weighted image pyramid. It gives smooth color with no texture or line continuation, the "simple filling" baseline from the proposal.

### Hallway scene numbers

`IMG_4484` from project 4, 1200×1600, f ≈ 1156 px. The box is 2.9 m wide, 3.2 m tall, and spans 2.0–7.0 m in depth. Rendered at half resolution, the share of hole pixels per view is:

| view | holes |
|---|---|
| original | 0.2% (thin ring from the dilated foreground mask) |
| 0.3 m left / right | 10.2% / 12.3% |
| 0.3 m up | 8.2% |
| 1.5 m forward | 3.4% |

## Layout

```
tip/            library: annotations, geometry, textures (+ push-pull fill), renderer, export
scripts/        annotate.py · show_annotation.py · build_scene.py · render_views.py
viewer/         three.js interactive viewer (loads scenes/<name>/scene.json)
data/images/    input photos
data/annotations/  per-scene annotation JSON
scenes/<name>/  built scene: scene.json, <plane>.png (RGBA with holes), <plane>_filled.jpg
outputs/<name>/ rendered figures
tests/          sanity tests
```

## Known limitations

- Assumes one-point perspective and a box-shaped room. Doorways, alcoves and the hallway beyond the back wall are flattened onto the back wall.
- Foreground objects are flat cutouts. Big sideways moves reveal that they're cards.
- The photo has no data close to the camera along the side walls, so even a small sideways step reveals large holes there (see the left/right views).
- The focal length is an assumption (26 mm equivalent), not calibrated.

## Next steps

- [ ] Diffusion inpainting on the rectified plane textures vs. on the original photo (the main experiment)
- [ ] Photograph about 6 scenes, plus held-out nearby views for evaluation
- [ ] Hole/seam/line-continuity comparison: unfilled vs. push-pull vs. diffusion
