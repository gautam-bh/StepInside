"""Render the furnished synthetic test room (known geometry) as a demo image.

    python scripts/synthetic_demo.py
    -> data/images/furnished_room.png  (a cabinet, a low box, a poster and a door)

Then run it through the normal pipeline, e.g.
    python scripts/annotate.py data/images/furnished_room.png --auto-only
    python scripts/build_scene.py data/annotations/furnished_room.json
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_annotation_tools import FURNISHED, synthetic_room  # noqa: E402

(vp, bw), objs = FURNISHED[0]
img = synthetic_room(vp, bw, objects=objs, decor=True)
out = ROOT / "data" / "images" / "furnished_room.png"
Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out)
print("wrote", out)
