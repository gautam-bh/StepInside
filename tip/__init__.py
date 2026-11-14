"""Tour Into the Picture (TIP): single-image interior scenes as a textured box.

Pipeline:
    annotation JSON  ->  geometry.build_scene  ->  textures.extract_textures
                     ->  export.write_scene (three.js viewer)  /  render.render_view
"""

from .annotations import Annotation, load_annotation, save_annotation
from .geometry import Scene, Plane, build_scene, project

__all__ = [
    "Annotation",
    "load_annotation",
    "save_annotation",
    "Scene",
    "Plane",
    "build_scene",
    "project",
]
