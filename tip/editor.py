"""Interactive annotation editor (matplotlib), seeded by auto-annotation.

Controls (also shown in the window):
    drag a handle     move the vanishing point (red), back-wall corners
                      (magenta) or a box's corners
    b                 add a box behind the current last one: click its
                      top-left, then bottom-right corner
    f                 add a foreground object: click around its outline,
                      Enter to finish
    o                 add a room-aligned object (cabinet, desk): click its
                      front face's top-left, then bottom-right corner; drag
                      the orange back handle to set how deep it is
    Backspace         undo: last outline point / foreground / object / box
    a                 re-run automatic detection (VP, back wall, deeper boxes, objects)
    s                 save                      q   save and quit

The editing logic lives in AnnotationEditor and takes plain (x, y) events,
so tests can drive it without a window. attach() wires it to a matplotlib
figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .annotations import Annotation, Foreground, RoomObject, save_annotation

HANDLE_RADIUS = 18.0  # pixels in image coordinates


@dataclass
class EditorState:
    vanishing_point: list[float]
    back_wall: list[float]  # x1, y1, x2, y2
    segments: list[list[float]] = field(default_factory=list)
    foreground: list[dict] = field(default_factory=list)  # {name, polygon}
    objects: list[dict] = field(default_factory=list)  # {name, front, back_y}


class AnnotationEditor:
    def __init__(self, ann: Annotation, image_size: tuple[int, int], out_path=None, auto_fn=None):
        self.width, self.height = image_size
        self.ann = ann
        self.out_path = out_path
        self.auto_fn = auto_fn  # () -> (vp, back_wall[, segments[, objects]])
        self.state = EditorState(
            vanishing_point=list(ann.vanishing_point),
            back_wall=list(ann.back_wall),
            segments=[list(r) for r in ann.segments],
            foreground=[{"name": f.name, "polygon": [list(p) for p in f.polygon]} for f in ann.foreground],
            objects=[{"name": o.name, "front": list(o.front), "back_y": o.back_y} for o in ann.objects],
        )
        self.mode = "edit"  # edit | box | foreground | object
        self.pending: list[list[float]] = []
        self.drag = None  # handle id while dragging
        self.message = ""
        self.on_change = lambda: None

    # ------------------------------------------------------------------ handles
    def handles(self) -> dict:
        s = self.state
        h = {"vp": tuple(s.vanishing_point)}
        x1, y1, x2, y2 = s.back_wall
        h[("bw", 0)] = (x1, y1)
        h[("bw", 1)] = (x2, y2)
        for i, (a, b, c, d) in enumerate(s.segments):
            h[("seg", i, 0)] = (a, b)
            h[("seg", i, 1)] = (c, d)
        for i, o in enumerate(s.objects):
            x1, y1, x2, y2 = o["front"]
            h[("obj", i, 0)] = (x1, y1)
            h[("obj", i, 1)] = (x2, y2)
            h[("obj", i, 2)] = self.object_back_handle(o)
        return h

    def object_back_handle(self, o):
        """Where the object's back bottom edge would be, below its front's centre."""
        vx, vy = self.state.vanishing_point
        x1, _, x2, y2 = o["front"]
        k = (o["back_y"] - vy) / (y2 - vy) if y2 != vy else 1.0
        return (vx + ((x1 + x2) / 2 - vx) * k, o["back_y"])

    def handle_at(self, x, y, radius=HANDLE_RADIUS):
        best, best_d = None, radius
        for k, (hx, hy) in self.handles().items():
            d = np.hypot(hx - x, hy - y)
            if d <= best_d:
                best, best_d = k, d
        return best

    def _clip(self, x, y):
        return float(np.clip(x, 0, self.width - 1)), float(np.clip(y, 0, self.height - 1))

    def _move_handle(self, key, x, y):
        x, y = self._clip(x, y)
        s = self.state
        if key == "vp":
            s.vanishing_point = [x, y]
        elif key[0] == "bw":
            if key[1] == 0:
                s.back_wall[0], s.back_wall[1] = x, y
            else:
                s.back_wall[2], s.back_wall[3] = x, y
        elif key[0] == "seg":
            r = s.segments[key[1]]
            if key[2] == 0:
                r[0], r[1] = x, y
            else:
                r[2], r[3] = x, y
        elif key[0] == "obj":
            o = s.objects[key[1]]
            if key[2] == 0:
                o["front"][0], o["front"][1] = x, y
            elif key[2] == 1:
                o["front"][2], o["front"][3] = x, y
            else:
                vy = s.vanishing_point[1]
                o["back_y"] = float(np.clip(y, vy + 1, max(o["front"][3] - 1, vy + 1)))

    # ------------------------------------------------------------------ events
    def press(self, x, y):
        if x is None or y is None:
            return
        if self.mode == "edit":
            self.drag = self.handle_at(x, y)
        elif self.mode in ("box", "object"):
            self.pending.append(list(self._clip(x, y)))
            if len(self.pending) == 2:
                (ax, ay), (bx, by) = self.pending
                rect = [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]
                self.pending = []
                mode, self.mode = self.mode, "edit"
                if mode == "box":
                    self.message = "added box" if self._add_box(rect) else self.message
                else:
                    self._add_object(rect)
        elif self.mode == "foreground":
            self.pending.append(list(self._clip(x, y)))
        self.on_change()

    def motion(self, x, y):
        if self.drag is not None and x is not None and y is not None:
            self._move_handle(self.drag, x, y)
            self.on_change()

    def release(self, x=None, y=None):
        if self.drag is not None:
            self._normalise()
            self.drag = None
            self.on_change()

    def key(self, k: str):
        if k == "b":
            self.mode, self.pending = "box", []
            self.message = "box: click top-left, then bottom-right corner"
        elif k == "f":
            self.mode, self.pending = "foreground", []
            self.message = "foreground: click around the outline, Enter to finish"
        elif k == "o":
            self.mode, self.pending = "object", []
            self.message = "object: click its front face's top-left, then bottom-right corner"
        elif k == "enter":
            if self.mode == "foreground":
                if len(self.pending) >= 3:
                    name = f"object{len(self.state.foreground) + 1}"
                    self.state.foreground.append({"name": name, "polygon": self.pending})
                    self.message = f"added {name}"
                else:
                    self.message = "outline needs at least 3 points"
                self.pending = []
                self.mode = "edit"
        elif k == "escape":
            self.mode, self.pending = "edit", []
            self.message = ""
        elif k == "backspace":
            self.undo()
        elif k == "a" and self.auto_fn is not None:
            res = self.auto_fn()
            vp, bw = res[0], res[1]
            self.state.vanishing_point, self.state.back_wall = list(vp), list(bw)
            self.message = "auto-detected VP and back wall"
            if len(res) > 2:
                self.state.segments = [list(r) for r in res[2]]
                self.message = f"auto-detected VP, back wall and {len(res[2])} deeper boxes"
            if len(res) > 3:
                self.state.objects = [{"name": f"object{i + 1}", "front": list(f), "back_y": float(b)}
                                      for i, (f, b) in enumerate(res[3])]
                self.message += f", {len(res[3])} objects"
        elif k == "s":
            self.save()
        self.on_change()

    def undo(self):
        if self.pending:
            self.pending.pop()
        elif self.state.foreground:
            self.state.foreground.pop()
            self.message = "removed last foreground object"
        elif self.state.objects:
            self.state.objects.pop()
            self.message = "removed last room object"
        elif self.state.segments:
            self.state.segments.pop()
            self.message = "removed last box"

    # ------------------------------------------------------------------ model
    def _normalise(self):
        """Keep rectangles ordered (x1 < x2, y1 < y2) after dragging."""
        def fix(r):
            r[0], r[2] = sorted([r[0], r[2]])
            r[1], r[3] = sorted([r[1], r[3]])
        fix(self.state.back_wall)
        for r in self.state.segments:
            fix(r)
        for o in self.state.objects:
            fix(o["front"])

    def _add_box(self, rect) -> bool:
        vx, vy = self.state.vanishing_point
        prev = self.state.segments[-1] if self.state.segments else self.state.back_wall
        if not (rect[0] < vx < rect[2] and rect[1] < vy < rect[3]):
            self.message = "box rejected: it must contain the vanishing point"
            return False
        if not rect[3] < prev[3]:
            self.message = "box rejected: its bottom must be above the previous box's bottom (farther away)"
            return False
        self.state.segments.append([round(v, 1) for v in rect])
        return True

    def _add_object(self, rect) -> bool:
        vy = self.state.vanishing_point[1]
        if not rect[3] > vy + 1:
            self.message = "object rejected: its bottom must be below the vanishing point (on the floor)"
            return False
        # Start 30% deeper than its front; drag the back handle to adjust.
        back_y = vy + (rect[3] - vy) / 1.3
        name = f"object{len(self.state.objects) + 1}"
        self.state.objects.append({"name": name, "front": [round(v, 1) for v in rect], "back_y": round(back_y, 1)})
        self.message = f"added {name}: drag its orange back handle to set its depth"
        return True

    def problems(self) -> list[str]:
        """Reasons the current annotation can't be built (empty if fine)."""
        s = self.state
        vx, vy = s.vanishing_point
        out = []
        rects = [("back wall", s.back_wall)] + [(f"box {i + 1}", r) for i, r in enumerate(s.segments)]
        for name, (x1, y1, x2, y2) in rects:
            if not (x1 < vx < x2 and y1 < vy < y2):
                out.append(f"{name} must contain the vanishing point")
        for (n0, a), (n1, b) in zip(rects, rects[1:]):
            if not b[3] < a[3]:
                out.append(f"{n1} must be farther than {n0} (bottom edge higher)")
        for o in s.objects:
            if not vy < o["back_y"] < o["front"][3]:
                out.append(f"{o['name']} must stand on the floor, below the vanishing point")
        return out

    def to_annotation(self) -> Annotation:
        s = self.state
        r = lambda v: [round(float(x), 1) for x in v]  # noqa: E731
        return Annotation(
            image=self.ann.image,
            vanishing_point=tuple(r(s.vanishing_point)),
            back_wall=tuple(r(s.back_wall)),
            focal_35mm=self.ann.focal_35mm,
            camera_height=self.ann.camera_height,
            segments=[r(x) for x in s.segments],
            foreground=[Foreground(name=f["name"], polygon=[r(p) for p in f["polygon"]]) for f in s.foreground],
            objects=[RoomObject(name=o["name"], front=r(o["front"]), back_y=round(float(o["back_y"]), 1))
                     for o in s.objects],
        )

    def save(self) -> bool:
        probs = self.problems()
        if probs:
            self.message = "not saved: " + "; ".join(probs)
            return False
        if self.out_path is not None:
            save_annotation(self.to_annotation(), self.out_path)
            self.message = f"saved {self.out_path}"
        return True

    # ------------------------------------------------------------------ matplotlib
    def attach(self, fig, ax, image):
        """Draw into a matplotlib axes and connect mouse/keyboard events."""
        import matplotlib.patches as mpatches

        ax.imshow(image)
        ax.set_axis_off()
        self._artists = []
        colors = ["#ffc800", "#00c8ff", "#78ff78", "#ff7878", "#c878ff", "#ffffff"]

        def redraw():
            for a in self._artists:
                a.remove()
            self._artists = []
            s = self.state
            vx, vy = s.vanishing_point
            x1, y1, x2, y2 = s.back_wall
            add = self._artists.append
            for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                # Room-edge ray from the VP through each corner, to the border.
                dx, dy = cx - vx, cy - vy
                ts = [t for t in ([(-vx) / dx, (self.width - vx) / dx] if dx else [])
                      + ([(-vy) / dy, (self.height - vy) / dy] if dy else []) if t > 0]
                t = min(ts) if ts else 1
                add(ax.plot([cx, vx + t * dx], [cy, vy + t * dy], "-", color="cyan", lw=1.5)[0])
            add(ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="magenta", lw=2)))
            for i, (a, b, c, d) in enumerate(s.segments):
                add(ax.add_patch(mpatches.Rectangle((a, b), c - a, d - b, fill=False,
                                                    ec=colors[i % len(colors)], lw=1.5)))
            for o in s.objects:
                x1o, y1o, x2o, y2o = o["front"]
                k = (o["back_y"] - vy) / (y2o - vy) if y2o != vy else 1.0
                front = [(x1o, y1o), (x2o, y1o), (x2o, y2o), (x1o, y2o)]
                back = [(vx + (px - vx) * k, vy + (py - vy) * k) for px, py in front]
                for poly in (front, back):
                    pts = np.array(poly + [poly[0]])
                    add(ax.plot(pts[:, 0], pts[:, 1], "-", color="orange", lw=1.5)[0])
                for (px, py), (qx, qy) in zip(front, back):
                    add(ax.plot([px, qx], [py, qy], "-", color="orange", lw=1)[0])
                bx, by = self.object_back_handle(o)
                add(ax.plot([bx], [by], "D", color="orange", ms=7, mec="white")[0])
            for f in s.foreground:
                pts = np.array(f["polygon"] + [f["polygon"][0]])
                add(ax.plot(pts[:, 0], pts[:, 1], "-", color="lime", lw=2)[0])
            if self.pending:
                pts = np.array(self.pending)
                add(ax.plot(pts[:, 0], pts[:, 1], "o-", color="yellow", lw=1.5, ms=4)[0])
            hx, hy = zip(*[v for k, v in self.handles().items() if k != "vp" and not (k[0] == "obj" and k[2] == 2)])
            add(ax.plot(hx, hy, "s", color="magenta", ms=7, mec="white")[0])
            add(ax.plot([vx], [vy], "o", color="red", ms=10, mec="white")[0])
            probs = self.problems()
            status = self.message or ("⚠ " + probs[0] if probs else "")
            ax.set_title(
                f"[{self.mode}] drag handles · b box · o object · f foreground · ⌫ undo · a auto · s save · q quit\n{status}",
                fontsize=9,
            )
            fig.canvas.draw_idle()

        self.on_change = redraw
        fig.canvas.mpl_connect("button_press_event", lambda e: e.inaxes is ax and self.press(e.xdata, e.ydata))
        fig.canvas.mpl_connect("motion_notify_event", lambda e: self.motion(e.xdata, e.ydata))
        fig.canvas.mpl_connect("button_release_event", lambda e: self.release())

        def on_key(e):
            if e.key == "q":
                self.save()
                import matplotlib.pyplot as plt

                plt.close(fig)
            else:
                self.key(e.key)

        fig.canvas.mpl_connect("key_press_event", on_key)
        redraw()
