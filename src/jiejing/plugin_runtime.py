"""Runtime shared by JIEJING Grasshopper components."""

import json
from pathlib import Path

import cv2
import Rhino

from . import RECIPES, build_generator_panel, dxf_from_mask, pdf_from_mask, svg_from_mask

PROFILE_NAMES = (
    "ellipse", "circle", "hexagon", "octagon", "clipped_rect",
    "begonia", "plum", "ruyi", "fan", "gourd",
)


def _number(value, default, low, high):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _curves_from_mask(mask, export_spec, origin):
    """Convert the exact CUT contour hierarchy into closed Rhino curves."""
    x0, y0 = export_spec["origin_px"]
    sx, sy = export_spec["scale_mm_per_px"]
    height_mm = export_spec["height_mm"]
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    curves = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        approx = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        points = [Rhino.Geometry.Point3d(
            origin.X + (float(px) - x0) * sx,
            origin.Y + height_mm - (float(py) - y0) * sy,
            origin.Z) for px, py in approx]
        points.append(points[0])
        curves.append(Rhino.Geometry.PolylineCurve(points))
    return curves


def run_jiejing(window, recipe, field_override, focus_x_shift, focus_y_shift,
                composition_amount, borrowed_view_shape, borrowed_view_curve,
                image_path, image_mode, density, line_weight, frame_weight,
                image_strength, polarity):
    if window is None:
        print("STEP 1 | Right-click Window and choose Set one Curve. | 请设置闭合窗框")
        return None
    if not window.IsClosed:
        print("WINDOW ERROR | 窗框必须是闭合曲线。")
        return None
    box = window.GetBoundingBox(True)
    width, height = box.Max.X - box.Min.X, box.Max.Y - box.Min.Y
    if box.Max.Z - box.Min.Z > .01 or min(width, height) < 100:
        print("WINDOW ERROR | 请使用世界XY平面内、宽高至少100 mm的窗框。")
        return None

    recipe_key = str(recipe or "round_cell_morph").strip()
    if recipe_key not in RECIPES:
        print("RECIPE ERROR | 未找到配方：", recipe_key)
        return None
    field_override = str(field_override or "recipe").strip()
    if field_override not in ("recipe", "borrowed"):
        field_override = "recipe"
    borrowed_view_shape = str(borrowed_view_shape or "ellipse").strip()
    if borrowed_view_shape not in PROFILE_NAMES:
        borrowed_view_shape = "ellipse"
    density = _number(density, 1.0, .65, 1.45)
    line_weight = _number(line_weight, 1.0, .65, 1.65)
    frame_weight = _number(frame_weight, 1.0, .65, 1.65)
    focus_x_shift = _number(focus_x_shift, 0.0, -.35, .35)
    focus_y_shift = _number(focus_y_shift, 0.0, -.35, .35)
    composition_amount = _number(composition_amount, 1.0, 0.0, 1.5)
    image_strength = _number(image_strength, 1.0, 0.0, 1.0)
    image_mode = str(image_mode or "portrait").strip()
    if image_mode not in ("silhouette", "portrait", "hybrid"):
        image_mode = "portrait"
    polarity = str(polarity or "material").strip()
    if polarity not in ("material", "opening"):
        polarity = "material"

    image_value = image_path.FullName if hasattr(image_path, "FullName") else image_path
    image_text = str(image_value or "").strip()
    image = Path(image_text).expanduser() if image_text and image_text not in ("None", "<null>") else None
    if image is not None and not image.is_file():
        print("IMAGE ERROR | 找不到图片：", image)
        return None

    domain = window.Domain
    sampled = [window.PointAt(domain.T0 + (domain.T1 - domain.T0) * i / 240.0)
               for i in range(240)]
    outline = [[max(0.0, min(width, point.X - box.Min.X)),
                max(0.0, min(height, point.Y - box.Min.Y))]
               for point in sampled]

    output_root = Path.home() / "Documents"
    if not output_root.is_dir():
        output_root = Path.home()
    out = output_root / "JIEJING_exports"
    out.mkdir(parents=True, exist_ok=True)
    borrowed_file = None
    if field_override == "borrowed" and borrowed_view_curve is not None:
        if not borrowed_view_curve.IsClosed:
            print("BORROWED VIEW ERROR | 借景轮廓必须闭合。")
            return None
        borrowed_domain = borrowed_view_curve.Domain
        borrowed_samples = [borrowed_view_curve.PointAt(
            borrowed_domain.T0 + (borrowed_domain.T1 - borrowed_domain.T0) * i / 160.0)
            for i in range(160)]
        borrowed_points = [[max(0.0, min(1.0, (point.X - box.Min.X) / width)),
                            max(0.0, min(1.0, (point.Y - box.Min.Y) / height))]
                           for point in borrowed_samples]
        borrowed_file = out / "borrowed_view.json"
        borrowed_file.write_text(json.dumps({"points": borrowed_points}), encoding="utf-8")

    recipe_data = RECIPES[recipe_key]
    attractor = (
        max(0.0, min(1.0, recipe_data["attractor_x"] + focus_x_shift)),
        max(0.0, min(1.0, recipe_data["attractor_y"] + focus_y_shift)),
    )
    field_strength = max(0.0, min(1.0,
        recipe_data["field_strength"] * composition_amount))
    field_mode = recipe_data["field_mode"] if field_override == "recipe" else "borrowed"
    pitch = max(55, min(260, round(recipe_data["pitch"] / density)))
    bar = max(8, min(80, round(recipe_data["bar"] * line_weight)))
    if image is not None and image_mode in ("portrait", "hybrid"):
        detail_factor = .38 if image_mode == "portrait" else .48
        pitch = max(28, min(55, round(recipe_data["pitch"] * detail_factor / density)))
        bar = max(5, min(18, round(recipe_data["bar"] * line_weight * .62)))
    frame = max(20, min(100, round(recipe_data["frame"] * frame_weight)))

    try:
        panel, report = build_generator_panel(
            recipe_data["generator"], pitch, bar, frame,
            window_shape="outline", window_width_mm=width,
            window_height_mm=height, window_outline_mm=outline,
            transform_mode=recipe_data["transform_mode"],
            attractor=attractor, field_strength=field_strength,
            field_mode=field_mode, borrowed_view_shape=borrowed_view_shape,
            borrowed_view_guide=borrowed_file,
            semantic_image=image, semantic_strength=image_strength,
            semantic_polarity=polarity, semantic_mode=image_mode)
    except Exception as error:
        print("GEOMETRY ERROR |", error)
        return None

    export_spec = report["export_spec"]
    (out / "panel.svg").write_text(svg_from_mask(panel, export_spec), encoding="utf-8")
    pdf_from_mask(panel, export_spec, out / "panel.pdf")
    dxf_from_mask(panel, export_spec, out / "panel.dxf")
    cv2.imwrite(str(out / "preview.png"), 255 - panel)
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    curves = _curves_from_mask(panel, export_spec, box.Min)
    image_note = " + {} image".format(image_mode) if image else ""
    view_note = " + borrowed view / 借景" if field_override == "borrowed" else ""
    print("READY | {}{}{} | {} closed curves | BLACK=KEEP / WHITE=CUT | {}".format(
        recipe_data["label"], view_note, image_note, len(curves), out))
    return curves
