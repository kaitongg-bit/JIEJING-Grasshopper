"""Small composable operations used by the modular Grasshopper components."""

import json
from pathlib import Path

import Grasshopper
import Rhino

from .plugin_runtime import PROFILE_NAMES, _number, run_jiejing
from .recipes import RECIPES

RUNTIME_CHOICES = {
    "pattern": {"Recipe": [
        "圆转四瓣 / Round Cell Morph", "三角孔肖像 / Portrait Triangles",
        "六瓣共享节点 / Six Petals", "鱼鳞层叠 / Fish Scale",
        "柳条曲格 / Willow Curves", "圆孔曲菱 / Tangent Round Cells",
        "对称阶梯回纹 / Stepped Fret", "莲瓣圆窗 / Lotus Roundel",
        "等高流线 / Contour View", "交错长孔 / Herringbone",
        "八角套方 / Octagon Square", "五瓣梅花孔 / Plum Blossom",
        "三层海水波 / Nested Waves", "冰裂 / Ice Crack"]},
    "borrowed": {"Shape": [
        "椭圆 / Ellipse", "圆 / Circle", "六角 / Hexagon", "八角 / Octagon",
        "切角矩形 / Clipped Rect", "海棠 / Begonia", "梅花 / Plum",
        "如意 / Ruyi", "扇面 / Fan", "葫芦 / Gourd"]},
    "image": {"Mode": ["轮廓 / Silhouette", "肖像细节 / Portrait", "混合 / Hybrid"],
              "Polarity": ["图像控制材料 / Material", "图像控制开口 / Opening"]},
}


def configure_component_ui(script_instance, kind):
    """Restore named integer menus that Rhino's script publisher strips."""
    component = getattr(script_instance, "Component", None)
    components = [component] if component is not None else []
    if not components and Grasshopper.Instances.ActiveCanvas is not None:
        document = Grasshopper.Instances.ActiveCanvas.Document
        component_names = {
            "pattern": "JIEJING Pattern",
            "borrowed": "JIEJING Borrowed View",
            "image": "JIEJING Image Field",
        }
        if document is not None:
            components = [item for item in document.Objects
                          if str(getattr(item, "Name", "")) == component_names.get(kind)]
    configured = False
    for target in components:
        for name, labels in RUNTIME_CHOICES.get(kind, {}).items():
            param = next((item for item in target.Params.Input if str(item.Name) == name), None)
            if param is None or not hasattr(param, "AddNamedValue"):
                continue
            if not bool(param.HasNamedValues):
                for index, label in enumerate(labels):
                    param.AddNamedValue(label, index)
                configured = True
    return configured


class PatternConfig:
    def __init__(self, recipe, density, line_weight, frame_weight):
        self.kind = "jiejing.pattern"
        self.recipe = recipe
        self.density = density
        self.line_weight = line_weight
        self.frame_weight = frame_weight

    def __str__(self):
        return "JIEJING Pattern: {}".format(self.recipe)


class BorrowedViewConfig:
    def __init__(self, enabled, shape, guide_curve, x_shift, y_shift, amount):
        self.kind = "jiejing.borrowed_view"
        self.enabled = enabled
        self.shape = shape
        self.guide_curve = guide_curve
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.amount = amount

    def __str__(self):
        return "JIEJING Borrowed View: {}".format("ON" if self.enabled else "OFF")


class ImageFieldConfig:
    def __init__(self, path, mode, strength, polarity):
        self.kind = "jiejing.image"
        self.path = path
        self.mode = mode
        self.strength = strength
        self.polarity = polarity

    def __str__(self):
        return "JIEJING Image Field: {}".format(Path(self.path).name if self.path else "OFF")


def _unwrap(value):
    return value.Value if hasattr(value, "Value") else value


def validate_window(window):
    window = _unwrap(window)
    if window is None:
        print("WINDOW | Connect one closed planar curve / 连接一条闭合平面窗框")
        return None
    if not window.IsClosed:
        print("WINDOW ERROR | Curve must be closed / 窗框必须闭合")
        return None
    box = window.GetBoundingBox(True)
    width = box.Max.X - box.Min.X
    height = box.Max.Y - box.Min.Y
    if box.Max.Z - box.Min.Z > .01 or min(width, height) < 100:
        print("WINDOW ERROR | World XY, minimum dimension 100 mm")
        return None
    print("WINDOW READY | {:.1f} x {:.1f} mm".format(width, height))
    return window


def make_pattern(recipe, density, line_weight, frame_weight):
    if isinstance(recipe, (int, float)):
        keys = list(RECIPES)
        index = int(recipe)
        key = keys[index] if 0 <= index < len(keys) else keys[0]
    else:
        key = str(recipe or "round_cell_morph").strip()
    if key not in RECIPES:
        print("PATTERN ERROR | Unknown recipe: {}".format(key))
        return None
    config = PatternConfig(
        key, _number(density, 1.0, .65, 1.45),
        _number(line_weight, 1.0, .65, 1.65),
        _number(frame_weight, 1.0, .65, 1.65))
    print("PATTERN READY | {}".format(RECIPES[key]["label"]))
    return config


def make_borrowed_view(enabled, shape, guide_curve, x_shift, y_shift, amount):
    enabled = bool(enabled)
    if isinstance(shape, (int, float)):
        index = int(shape)
        shape = PROFILE_NAMES[index] if 0 <= index < len(PROFILE_NAMES) else "ellipse"
    else:
        shape = str(shape or "ellipse").strip()
    if shape not in PROFILE_NAMES:
        shape = "ellipse"
    guide_curve = _unwrap(guide_curve)
    if guide_curve is not None and not guide_curve.IsClosed:
        print("VIEW ERROR | Custom guide must be closed / 自定义借景线必须闭合")
        return None
    config = BorrowedViewConfig(
        enabled, shape, guide_curve,
        _number(x_shift, 0.0, -.35, .35),
        _number(y_shift, 0.0, -.35, .35),
        _number(amount, 1.0, 0.0, 1.5))
    state = "ON" if enabled else "OFF"
    print("BORROWED VIEW {} | {}".format(state, "custom" if guide_curve else shape))
    return config


def make_image_field(image_path, mode, strength, polarity):
    image_text = str(image_path or "").strip()
    image = Path(image_text).expanduser() if image_text else None
    if image is not None and not image.is_file():
        print("IMAGE ERROR | File not found: {}".format(image))
        return None
    if isinstance(mode, (int, float)):
        modes = ("silhouette", "portrait", "hybrid")
        index = int(mode)
        mode = modes[index] if 0 <= index < len(modes) else "portrait"
    else:
        mode = str(mode or "portrait").strip()
    if mode not in ("silhouette", "portrait", "hybrid"):
        mode = "portrait"
    if isinstance(polarity, (int, float)):
        polarity = "opening" if int(polarity) == 1 else "material"
    else:
        polarity = str(polarity or "material").strip()
    if polarity not in ("material", "opening"):
        polarity = "material"
    config = ImageFieldConfig(
        str(image) if image else "", mode,
        _number(strength, 1.0, 0.0, 1.0), polarity)
    print("IMAGE FIELD | {}".format(image.name if image else "disabled"))
    return config


def compose(window, pattern, borrowed_view, image):
    window = validate_window(window)
    pattern = _unwrap(pattern)
    borrowed_view = _unwrap(borrowed_view)
    image = _unwrap(image)
    if window is None:
        return None
    if not isinstance(pattern, PatternConfig):
        print("COMPOSE ERROR | Connect JIEJING Pattern")
        return None
    if not isinstance(borrowed_view, BorrowedViewConfig):
        borrowed_view = make_borrowed_view(False, "ellipse", None, 0, 0, 1)
    if not isinstance(image, ImageFieldConfig):
        image = make_image_field("", "portrait", 1, "material")
    view_enabled = bool(borrowed_view.enabled)
    return run_jiejing(
        window, pattern.recipe, "borrowed" if view_enabled else "recipe",
        borrowed_view.x_shift, borrowed_view.y_shift, borrowed_view.amount,
        borrowed_view.shape, borrowed_view.guide_curve, image.path, image.mode,
        pattern.density, pattern.line_weight, pattern.frame_weight,
        image.strength, image.polarity)


def preview_regions(curves):
    curves = [_unwrap(curve) for curve in (curves or []) if curve is not None]
    if not curves:
        print("PREVIEW | Connect Compose curves")
        return None
    tolerance = Rhino.RhinoDoc.ActiveDoc.ModelAbsoluteTolerance if Rhino.RhinoDoc.ActiveDoc else .01
    regions = Rhino.Geometry.Brep.CreatePlanarBreps(curves, tolerance)
    result = list(regions) if regions else []
    print("PREVIEW READY | {} planar regions | BLACK=KEEP".format(len(result)))
    return result


def _curve_points(curve):
    curve = _unwrap(curve)
    success, polyline = curve.TryGetPolyline()
    if success:
        return [(point.X, point.Y) for point in polyline]
    parameters = curve.DivideByCount(96, True)
    return [(curve.PointAt(t).X, curve.PointAt(t).Y) for t in parameters]


def export_curves(curves, folder, name, trigger):
    if not bool(trigger):
        print("EXPORT | Set Trigger=True when the design is ready")
        return None
    curves = [_unwrap(curve) for curve in (curves or []) if curve is not None]
    if not curves:
        print("EXPORT ERROR | No curves")
        return None
    output = Path(str(folder or (Path.home() / "Documents" / "JIEJING_exports"))).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    stem = str(name or "jiejing_panel").strip().replace("/", "_")
    paths = [_curve_points(curve) for curve in curves]
    all_points = [point for path in paths for point in path]
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_x = max(point[0] for point in all_points)
    max_y = max(point[1] for point in all_points)
    width, height = max_x - min_x, max_y - min_y

    import ezdxf
    document = ezdxf.new("R2010", setup=True)
    document.layers.add("CUT", color=1)
    model = document.modelspace()
    for path in paths:
        model.add_lwpolyline(path, close=True, dxfattribs={"layer": "CUT"})
    dxf_path = output / (stem + ".dxf")
    document.saveas(str(dxf_path))

    svg_parts = []
    for path in paths:
        coords = " ".join("{:.3f},{:.3f}".format(x - min_x, max_y - y) for x, y in path)
        svg_parts.append("M " + coords + " Z")
    svg_path = output / (stem + ".svg")
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}mm" height="{1}mm" '
        'viewBox="0 0 {0} {1}"><path d="{2}" fill="black" fill-rule="evenodd"/></svg>'.format(
            width, height, " ".join(svg_parts)), encoding="utf-8")

    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    pdf_path = output / (stem + ".pdf")
    pdf = canvas.Canvas(str(pdf_path), pagesize=(width * mm, height * mm))
    pdf_path_data = pdf.beginPath()
    for points in paths:
        x, y = points[0]
        pdf_path_data.moveTo((x - min_x) * mm, (y - min_y) * mm)
        for x, y in points[1:]:
            pdf_path_data.lineTo((x - min_x) * mm, (y - min_y) * mm)
        pdf_path_data.close()
    pdf.drawPath(pdf_path_data, fill=1, stroke=0, fillMode=0)
    pdf.showPage()
    pdf.save()
    manifest = output / (stem + "_export.json")
    manifest.write_text(json.dumps({"dxf": str(dxf_path), "svg": str(svg_path),
                                    "pdf": str(pdf_path), "curve_count": len(curves)},
                                   indent=2), encoding="utf-8")
    print("EXPORT READY | DXF + SVG + PDF | {}".format(output))
    return str(output)
