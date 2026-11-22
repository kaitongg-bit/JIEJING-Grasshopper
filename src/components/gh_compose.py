#! python 3
#requirements: numpy, opencv-python-headless, pillow, ezdxf, reportlab==4.2.5
import Grasshopper
import Rhino
from jiejing.modular_runtime import compose

class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, Window: Rhino.Geometry.Curve, Pattern: object,
                  BorrowedView: object, ImageField: object):
        return compose(Window, Pattern, BorrowedView, ImageField)
