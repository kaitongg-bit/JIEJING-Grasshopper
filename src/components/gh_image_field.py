#! python 3
import Grasshopper
from jiejing.modular_runtime import configure_component_ui, make_image_field

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, ImagePath: str, Mode: int, Strength: float, Polarity: int):
        configure_component_ui(self, "image")
        return make_image_field(ImagePath, Mode, Strength, Polarity)
