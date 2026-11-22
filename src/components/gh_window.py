#! python 3
import Grasshopper
import Rhino
from jiejing.modular_runtime import validate_window

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, Window: Rhino.Geometry.Curve):
        return validate_window(Window)
