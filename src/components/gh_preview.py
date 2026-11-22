#! python 3
import Grasshopper
import Rhino
from jiejing.modular_runtime import preview_regions

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, Curves: list[Rhino.Geometry.Curve]):
        return preview_regions(Curves)
