#! python 3
"""Build the modular JIEJING Grasshopper source document."""

from pathlib import Path

import Grasshopper
import RhinoCodePluginGH
import System
import System.Drawing


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "JIEJING_Component.ghx"

RECIPE_CHOICES = [
    ("圆转四瓣 / Round Cell Morph", "round_cell_morph"),
    ("三角孔肖像 / Portrait Triangles", "portrait_triangles"),
    ("六瓣共享节点 / Six Petals", "six_petals"),
    ("鱼鳞层叠 / Fish Scale", "fish_scale_cascade"),
    ("柳条曲格 / Willow Curves", "willow_curves"),
    ("圆孔曲菱 / Tangent Round Cells", "tangent_round_cells"),
    ("对称阶梯回纹 / Stepped Fret", "stepped_fret"),
    ("莲瓣圆窗 / Lotus Roundel", "lotus_roundel"),
    ("等高流线 / Contour View", "contour_view"),
    ("交错长孔编织 / Herringbone", "herringbone_screen"),
    ("八角套方 / Octagon Square", "octagon_square"),
    ("五瓣梅花孔 / Plum Blossom", "plum_blossom_screen"),
    ("三层海水波 / Nested Waves", "nested_waves"),
    ("冰裂 / Ice Crack", "ice_crack"),
]
SHAPE_CHOICES = [
    ("椭圆 / Ellipse", "ellipse"), ("圆 / Circle", "circle"),
    ("六角 / Hexagon", "hexagon"), ("八角 / Octagon", "octagon"),
    ("切角矩形 / Clipped Rect", "clipped_rect"),
    ("海棠 / Begonia", "begonia"), ("梅花 / Plum", "plum"),
    ("如意 / Ruyi", "ruyi"), ("扇面 / Fan", "fan"),
    ("葫芦 / Gourd", "gourd"),
]


def signature(parameters):
    return '''#! python 3
import Grasshopper
import Rhino
class MyComponent(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, %s):
        return None
''' % parameters


SPECS = [
    {
        "id": "288b5e22-dccb-41ca-b5c4-65573f55e3b2",
        "name": "JIEJING Quick Generate", "nick": "Quick",
        "icon": "quick",
        "source": ROOT / "gh_plugin_core.py", "output": "Curves",
        "signature": "window: Rhino.Geometry.Curve, recipe: str, field_override: str, focus_x_shift: float, focus_y_shift: float, composition_amount: float, borrowed_view_shape: str, borrowed_view_curve: Rhino.Geometry.Curve, image_path: str, image_mode: str, density: float, line_weight: float, frame_weight: float, image_strength: float, polarity: str",
        "required": {"window"},
        "defaults": {"recipe": "round_cell_morph", "field_override": "recipe",
                     "focus_x_shift": 0.0, "focus_y_shift": 0.0,
                     "composition_amount": 1.0, "borrowed_view_shape": "ellipse",
                     "image_path": "", "image_mode": "portrait", "density": 1.0,
                     "line_weight": 1.0, "frame_weight": 1.0,
                     "image_strength": 1.0, "polarity": "material"},
        "choices": {"recipe": RECIPE_CHOICES,
                    "field_override": [("不借景 / Pattern", "recipe"),
                                       ("借景 / Borrowed View", "borrowed")],
                    "borrowed_view_shape": SHAPE_CHOICES,
                    "image_mode": [("轮廓 / Silhouette", "silhouette"),
                                   ("肖像细节 / Portrait", "portrait"),
                                   ("混合 / Hybrid", "hybrid")],
                    "polarity": [("图像控制材料 / Material", "material"),
                                 ("图像控制开口 / Opening", "opening")]},
        "description": "One-component beginner workflow. Generates fabrication curves and exports a preview.",
    },
    {
        "id": "f32d3510-9b23-4c50-a6c7-97501e6cb701",
        "name": "JIEJING Window", "nick": "Window",
        "icon": "window",
        "source": ROOT / "components" / "gh_window.py", "output": "Window",
        "signature": "Window: Rhino.Geometry.Curve", "required": {"Window"},
        "description": "Validate a closed planar window boundary in millimetres.",
    },
    {
        "id": "bb870c01-77ee-4979-b497-4bfde3cc486b",
        "name": "JIEJING Pattern", "nick": "Pattern",
        "icon": "pattern",
        "source": ROOT / "components" / "gh_pattern.py", "output": "Pattern",
        "signature": "Recipe: int, Density: float, LineWeight: float, FrameWeight: float",
        "required": set(),
        "defaults": {"Recipe": 0, "Density": 1.0,
                     "LineWeight": 1.0, "FrameWeight": 1.0},
        "choices": {"Recipe": [(label, index) for index, (label, _) in enumerate(RECIPE_CHOICES)]},
        "description": "Choose a curated lattice recipe and fabrication proportions.",
    },
    {
        "id": "57c17e75-77fa-46f9-a06e-c3a29c5a5eb8",
        "name": "JIEJING Borrowed View", "nick": "BorrowedView",
        "icon": "borrowed",
        "source": ROOT / "components" / "gh_borrowed_view.py", "output": "BorrowedView",
        "signature": "Enabled: bool, Shape: int, Guide: Rhino.Geometry.Curve, XShift: float, YShift: float, Amount: float",
        "required": set(),
        "defaults": {"Enabled": False, "Shape": 0, "XShift": 0.0,
                     "YShift": 0.0, "Amount": 1.0},
        "choices": {"Shape": [(label, index) for index, (label, _) in enumerate(SHAPE_CHOICES)]},
        "description": "Add an optional classical borrowed-view opening or a custom closed guide.",
    },
    {
        "id": "217767aa-0215-4d19-b627-5a03d5f6ed3b",
        "name": "JIEJING Image Field", "nick": "ImageField",
        "icon": "image",
        "source": ROOT / "components" / "gh_image_field.py", "output": "ImageField",
        "signature": "ImagePath: str, Mode: int, Strength: float, Polarity: int",
        "required": set(),
        "defaults": {"ImagePath": "", "Mode": 1, "Strength": 1.0,
                     "Polarity": 0},
        "choices": {"Mode": [("轮廓 / Silhouette", 0),
                              ("肖像细节 / Portrait", 1),
                              ("混合 / Hybrid", 2)],
                    "Polarity": [("图像控制材料 / Material", 0),
                                 ("图像控制开口 / Opening", 1)]},
        "description": "Optional non-AI image field for portraits, pets, silhouettes, and cartoons.",
    },
    {
        "id": "85e46f83-57ef-45a9-8afe-64fbf3ca924c",
        "name": "JIEJING Compose", "nick": "Compose",
        "icon": "compose",
        "source": ROOT / "components" / "gh_compose.py", "output": "Curves",
        "signature": "Window: Rhino.Geometry.Curve, Pattern: object, BorrowedView: object, ImageField: object",
        "required": {"Window", "Pattern"},
        "description": "Compose window, pattern, borrowed view, and image field into closed fabrication curves.",
    },
    {
        "id": "b7ab5ec8-cc98-47b1-9616-f04c5230c276",
        "name": "JIEJING Material Preview", "nick": "Preview",
        "icon": "preview",
        "source": ROOT / "components" / "gh_preview.py", "output": "Material",
        "signature": "Curves: list[Rhino.Geometry.Curve]", "required": {"Curves"},
        "list_inputs": {"Curves"},
        "description": "Create black material regions from closed cut contours for visual checking.",
    },
    {
        "id": "d69da97d-46b0-4608-9a40-dfbd6b77ea80",
        "name": "JIEJING Export", "nick": "Export",
        "icon": "export",
        "source": ROOT / "components" / "gh_export.py", "output": "Folder",
        "signature": "Curves: list[Rhino.Geometry.Curve], Folder: str, Name: str, Trigger: bool",
        "required": {"Curves"},
        "list_inputs": {"Curves"},
        "defaults": {"Folder": str(Path.home() / "Documents" / "JIEJING_exports"),
                     "Name": "jiejing_panel", "Trigger": False},
        "description": "Export closed curves as 1:1 millimetre DXF, SVG, and PDF files.",
    },
]


document = Grasshopper.Kernel.GH_Document()
for index, spec in enumerate(SPECS):
    source = spec["source"].read_text(encoding="utf-8")
    icon = System.Drawing.Bitmap(str(ROOT / "icons" / (spec["icon"] + "_light.png")))
    component = RhinoCodePluginGH.Components.Python3Component.Create(
        spec["name"], source, icon, False)
    component.NewInstanceGuid(System.Guid(spec["id"]))
    component.Name = spec["name"]
    component.NickName = spec["nick"]
    component.Description = spec["description"]

    if index == 0:
        # The legacy all-in-one source uses a separate parameter template to
        # avoid Rhino 8 staging its longer script with damaged indentation.
        parameter_template = RhinoCodePluginGH.Components.Python3Component.Create(
            spec["name"] + " Parameters", signature(spec["signature"]))
        parameter_template.SetParametersFromScript()
        parameter_template.CopyParameters()
        component.PasteParameters()
    else:
        # Thin modular sources can safely let RhinoCode bind their own
        # signatures. This keeps the compiled ScriptInstance and parameter
        # marshaling metadata in sync.
        component.SetParametersFromScript()

    # Parameter discovery resets the component artwork to the Python language
    # icon in Rhino 8.35. Apply the JIEJING icon only after the script and its
    # parameters have finished binding so the project publisher serializes it.
    component.SetIcon(icon)
    component.SetIconOverride(icon)

    for param in component.Params.Input:
        param.Optional = str(param.Name) not in spec.get("required", set())
        if str(param.Name) in spec.get("list_inputs", set()):
            param.Access = Grasshopper.Kernel.GH_ParamAccess.list
        if hasattr(param, "AddNamedValue"):
            for label, value in spec.get("choices", {}).get(str(param.Name), []):
                param.AddNamedValue(label, value)
        if str(param.Name) in spec.get("defaults", {}):
            value = spec["defaults"][str(param.Name)]
            if isinstance(value, bool):
                goo = Grasshopper.Kernel.Types.GH_Boolean(value)
            elif isinstance(value, (int, float)):
                goo = Grasshopper.Kernel.Types.GH_Number(float(value))
            else:
                goo = Grasshopper.Kernel.Types.GH_String(str(value))
            param.PersistentData.Append(goo)
    if len(component.Params.Output) > 1:
        output = component.Params.Output[1]
        output.Name = spec["output"]
        output.NickName = spec["output"]
        output.Description = spec["description"]

    component.CreateAttributes()
    component.Attributes.Pivot = System.Drawing.PointF(
        120 + (index % 4) * 260, 100 + (index // 4) * 220)
    document.AddObject(component, False)

io = Grasshopper.Kernel.GH_DocumentIO()
io.Document = document
saved = io.SaveQuiet(str(TARGET))
print("JIEJING MODULAR COMPONENTS", saved, len(SPECS), TARGET)
