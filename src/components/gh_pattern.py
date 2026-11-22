#! python 3
import Grasshopper
from jiejing.modular_runtime import configure_component_ui, make_pattern

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, Recipe: int, Density: float, LineWeight: float, FrameWeight: float):
        configure_component_ui(self, "pattern")
        return make_pattern(Recipe, Density, LineWeight, FrameWeight)
