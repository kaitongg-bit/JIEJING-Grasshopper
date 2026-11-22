#! python 3
#requirements: ezdxf, reportlab==4.2.5
import Grasshopper
import Rhino
from jiejing.modular_runtime import export_curves

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, Curves: list[Rhino.Geometry.Curve], Folder: str,
                  Name: str, Trigger: bool):
        return export_curves(Curves, Folder, Name, Trigger)
