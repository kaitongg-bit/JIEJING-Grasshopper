#! python 3
import Grasshopper
import Rhino
from jiejing.modular_runtime import configure_component_ui, make_borrowed_view

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, Enabled: bool, Shape: int, Guide: Rhino.Geometry.Curve,
                  XShift: float, YShift: float, Amount: float):
        configure_component_ui(self, "borrowed")
        return make_borrowed_view(Enabled, Shape, Guide, XShift, YShift, Amount)
