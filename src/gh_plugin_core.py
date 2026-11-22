#! python 3
#requirements: numpy, opencv-python-headless, pillow, ezdxf, reportlab==4.2.5
"""Thin Grasshopper entry point; the reusable implementation ships as a library."""

import Grasshopper
import Rhino

from jiejing.plugin_runtime import run_jiejing


class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, window: Rhino.Geometry.Curve, recipe: str,
                  field_override: str, focus_x_shift: float,
                  focus_y_shift, composition_amount, borrowed_view_shape,
                  borrowed_view_curve: Rhino.Geometry.Curve,
                  image_path: str, image_mode: str, density: float,
                  line_weight, frame_weight, image_strength, polarity):
        return run_jiejing(
            window, recipe, field_override, focus_x_shift, focus_y_shift,
            composition_amount, borrowed_view_shape, borrowed_view_curve,
            image_path, image_mode, density, line_weight, frame_weight,
            image_strength, polarity)
