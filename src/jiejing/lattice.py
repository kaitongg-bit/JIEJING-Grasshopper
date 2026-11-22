"""Turn a photographed subject into a connected, flat lattice panel.

This is a standalone geometry core. It writes SVG paths in millimetres, so the
same core can later be wrapped by a Grasshopper Python component or Rhino UI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from .motifs import PRESETS, load_custom_paths, render as render_motif
from .curated_presets import render_preset
from .parametric_generators import CONTROLLER_GENERATORS, GENERATORS, semantic_reveal


SIZE = 1200
PANEL_MM = 600.0
CENTER = SIZE // 2


def largest_filled_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError("No foreground found. Adjust --background-tolerance or use a transparent PNG.")
    index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    component = np.uint8(labels == index) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(component)
    cv2.drawContours(result, contours, -1, 255, -1)
    return result


def extract_subject(path: Path, tolerance: float) -> tuple[np.ndarray, np.ndarray, dict]:
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    if np.min(alpha) < 240:
        raw = np.uint8(alpha > 80) * 255
        extraction = "alpha"
    else:
        corners = np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3),
                                  rgb[-8:, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)])
        background = np.median(corners.astype(np.float32), axis=0)
        color_distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
        candidate_background = np.uint8(color_distance <= tolerance)
        # Only background reachable from an image edge is removed. Light areas
        # inside the subject remain part of its filled silhouette.
        flood = np.zeros((rgb.shape[0] + 2, rgb.shape[1] + 2), np.uint8)
        free = candidate_background.copy()
        cv2.floodFill(free, flood, (0, 0), 2)
        raw = np.uint8(free != 2) * 255
        extraction = "corner-background"
    silhouette = largest_filled_component(raw)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    details = np.uint8((grayscale < 95) & (silhouette > 0)) * 255
    return silhouette, details, {"extraction": extraction, "image_size": [int(rgb.shape[1]), int(rgb.shape[0])]}


def extract_internal_engraving(path: Path, silhouette: np.ndarray) -> np.ndarray:
    """Pick a few interior color boundaries for engraving on a solid symbol.

    The mask is kept in source-image coordinates so it can be placed with the
    silhouette's exact crop and scale. Dense photographs are left unengraved.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"))
    lightness = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
    lightness = cv2.GaussianBlur(lightness, (7, 7), 0)
    safe = cv2.erode(silhouette, np.ones((13, 13), np.uint8))
    for low, high in ((15, 45), (35, 85), (60, 130)):
        edges = cv2.bitwise_and(cv2.Canny(lightness, low, high), safe)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
        selected = np.zeros_like(edges)
        for index in range(1, count):
            if 30 <= stats[index, cv2.CC_STAT_AREA] <= 2000:
                selected[labels == index] = 255
        if cv2.countNonZero(selected) <= .045 * cv2.countNonZero(silhouette):
            return selected
    return np.zeros_like(silhouette)


def extract_major_light_inset(path: Path, silhouette: np.ndarray) -> np.ndarray:
    """Find one large, enclosed light region that can become a cutout.

    This is deliberately conservative: photographs without a clear interior
    region return an empty mask instead of arbitrary holes.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"))
    lightness = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
    ys, xs = np.where(silhouette > 0)
    if not len(xs):
        return np.zeros_like(silhouette)
    threshold = max(225, min(245, int(np.quantile(lightness[silhouette > 0], .74))))
    safe = cv2.erode(silhouette, np.ones((21, 21), np.uint8))
    candidate = np.uint8((lightness >= threshold) & (safe > 0)) * 255
    count, labels, stats, centers = cv2.connectedComponentsWithStats(candidate, 8)
    area = cv2.countNonZero(silhouette)
    lower_half = ys.min() + .50 * (ys.max() - ys.min())
    eligible = [i for i in range(1, count)
                if .05 * area <= stats[i, cv2.CC_STAT_AREA] <= .30 * area
                and centers[i, 1] > lower_half]
    if not eligible:
        return np.zeros_like(silhouette)
    largest = max(eligible, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    inset = np.uint8(labels == largest) * 255
    inset = cv2.morphologyEx(inset, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return cv2.bitwise_and(inset, safe)


def place_subject(silhouette: np.ndarray, details: np.ndarray, scale: float,
                  max_width: float = 720, max_height: float = 790,
                  center_px: tuple[int, int] = (CENTER, CENTER)) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(silhouette > 0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = silhouette[y0:y1, x0:x1]
    crop_details = details[y0:y1, x0:x1]
    h, w = crop.shape
    factor = min(max_width / w, max_height / h) * scale
    new_size = (max(1, round(w * factor)), max(1, round(h * factor)))
    small = cv2.resize(crop, new_size, interpolation=cv2.INTER_NEAREST)
    small_details = cv2.resize(crop_details, new_size, interpolation=cv2.INTER_NEAREST)
    x = center_px[0] - new_size[0] // 2
    y = center_px[1] - new_size[1] // 2
    placed = np.zeros((SIZE, SIZE), np.uint8)
    placed_details = np.zeros_like(placed)
    placed[y:y + new_size[1], x:x + new_size[0]] = small
    placed_details[y:y + new_size[1], x:x + new_size[0]] = small_details
    return placed, placed_details


def composition_placement(composition: str, spec: dict) -> tuple[float, float, tuple[int, int]]:
    """Subject size and position relative to the actual window bounding box."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    if composition == "curated":
        return width * .42, height * .60, (round(x0 + width * .5), round(y0 + height * .51))
    if composition == "curated_weave":
        return width * .32, height * .52, (round(x0 + width * .66), round(y0 + height * .53))
    if composition == "portrait":
        return min(720, width * .68), min(790, height * .78), (round(x0 + width * .5), round(y0 + height * .5))
    if composition == "aperture":
        return width * .52, height * .58, (round(x0 + width * .5), round(y0 + height * .5))
    if composition == "borrowed_view":
        return width * .37, height * .56, (round(x0 + width * .34), round(y0 + height * .58))
    if composition == "roundel":
        return width * .34, height * .48, (round(x0 + width * .5), round(y0 + height * .5))
    if composition == "diagonal":
        return width * .31, height * .44, (round(x0 + width * .27), round(y0 + height * .34))
    if composition == "diptych":
        return width * .38, height * .57, (round(x0 + width * .36), round(y0 + height * .52))
    if composition == "corners":
        return width * .40, height * .56, (round(x0 + width * .5), round(y0 + height * .5))
    if composition == "echo":
        return width * .20, height * .33, (round(x0 + width * .23), round(y0 + height * .60))
    if composition == "garden_veil":
        return width * .25, height * .40, (round(x0 + width * .25), round(y0 + height * .57))
    if composition == "framed_view":
        return width * .13, height * .17, (round(x0 + width * .55), round(y0 + height * .82))
    raise ValueError(f"Unknown composition: {composition}")


def place_composition_subject(silhouette: np.ndarray, details: np.ndarray, scale: float,
                              composition: str, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    max_width, max_height, center = composition_placement(composition, spec)
    if composition != "echo":
        return place_subject(silhouette, details, scale, max_width, max_height, center)
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    figures = np.zeros((SIZE, SIZE), np.uint8)
    marks = np.zeros_like(figures)
    for u, v in ((.23, .60), (.50, .39), (.77, .60)):
        figure, detail = place_subject(silhouette, details, scale, max_width, max_height,
                                       (round(x0 + width * u), round(y0 + height * v)))
        figures = cv2.bitwise_or(figures, figure)
        marks = cv2.bitwise_or(marks, detail)
    return figures, marks


def tracery_subject(subject: np.ndarray, details: np.ndarray, width: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Turn a filled figure into a cuttable outline plus optional engraved marks."""
    if not cv2.countNonZero(subject):
        return subject, np.zeros_like(subject)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    outline = cv2.subtract(subject, cv2.erode(subject, kernel))
    safe = cv2.erode(subject, np.ones((17, 17), np.uint8))
    marks = cv2.bitwise_and(details, safe)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(marks, 8)
    engrave = np.zeros_like(subject)
    for index in range(1, count):
        if 70 <= stats[index, cv2.CC_STAT_AREA] <= 25000:
            engrave[labels == index] = 255
    return outline, engrave


def garden_view_points(spec: dict) -> np.ndarray:
    """A restrained four-lobed opening, sized from the selected window."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    points = []
    for angle in np.linspace(0, 2 * np.pi, 361):
        radius = 1 + .20 * np.cos(4 * angle)
        points.append((round(x0 + width * (.70 + .29 * radius * np.cos(angle))),
                       round(y0 + height * (.50 + .37 * radius * np.sin(angle)))))
    return np.asarray(points, np.int32)


def symbol_from_json(path: Path, spec: dict, bar: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize user-authored cut and engraving strokes in window coordinates."""
    data = json.loads(path.read_text(encoding="utf-8"))
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    cut = np.zeros((SIZE, SIZE), np.uint8)
    engrave = np.zeros_like(cut)
    for key, target, thickness in (("cut_paths", cut, bar),
                                   ("engrave_paths", engrave, max(4, bar // 3))):
        for path_data in data.get(key, []):
            points = path_data.get("points")
            if not isinstance(points, list) or len(points) < 2:
                raise ValueError(f"Every {key} entry needs at least two points")
            if any(not isinstance(p, list) or len(p) != 2 or
                   not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in p)
                   for p in points):
                raise ValueError("Symbol points must be normalized [x,y] in [0,1]")
            xy = np.asarray([(x0 + width * p[0], y0 + height * p[1])
                             for p in points], np.float64)
            closed = bool(path_data.get("closed", False))
            if path_data.get("smooth", False) and len(xy) >= 3:
                segments = []
                for index in range(len(xy) if closed else len(xy) - 1):
                    p0 = xy[(index - 1) % len(xy)] if closed else xy[max(0, index - 1)]
                    p1 = xy[index]
                    p2 = xy[(index + 1) % len(xy)]
                    p3 = xy[(index + 2) % len(xy)] if closed else xy[min(len(xy) - 1, index + 2)]
                    for t in np.linspace(0, 1, 12, endpoint=False):
                        t2, t3 = t * t, t * t * t
                        segments.append(.5 * ((2 * p1) + (-p0 + p2) * t +
                                                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                                                (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
                if not closed:
                    segments.append(xy[-1])
                xy = np.asarray(segments)
            xy = np.rint(xy).astype(np.int32)
            cv2.polylines(target, [xy], closed,
                          255, thickness, cv2.LINE_AA)
    if not cv2.countNonZero(cut):
        raise ValueError("Symbol JSON needs at least one cut path")
    return cut, engrave


def borrowed_view_opening(inner: np.ndarray, spec: dict,
                          attractor: tuple[float, float] | None,
                          strength: float,
                          shape: str = "ellipse",
                          guide: Path | None = None) -> np.ndarray:
    """Return a real borrowed-view opening from a closed guide or built-in profile.

    Guide points are normalized to the selected window (0–1).  Without a guide,
    the attractor locates an elliptical opening and strength controls its extent.
    """
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    opening = np.zeros_like(inner)
    if guide is not None:
        data = json.loads(guide.read_text(encoding="utf-8"))
        points = data.get("points")
        if (not isinstance(points, list) or len(points) < 3 or
                any(not isinstance(point, list) or len(point) != 2 or
                    not all(isinstance(value, (int, float)) and 0 <= value <= 1
                            for value in point) for point in points)):
            raise ValueError("Borrowed-view guide needs at least three normalized 0–1 points")
        polygon = np.asarray([
            [round(x0 + width * point[0]), round(y0 + height * point[1])]
            for point in points], np.int32)
        cv2.fillPoly(opening, [polygon], 255, cv2.LINE_AA)
    else:
        ax, ay = attractor if attractor is not None else (.5, .5)
        amount = max(0.0, min(1.0, strength))
        radius_x = max(8, round(width * (.08 + .24 * amount)))
        radius_y = max(8, round(height * (.10 + .28 * amount)))
        center = (round(x0 + width * ax), round(y0 + height * ay))
        cx, cy = center
        if shape == "circle":
            radius = max(8, round(min(width, height) * (.10 + .28 * amount)))
            cv2.circle(opening, center, radius, 255, -1, cv2.LINE_AA)
        elif shape == "hexagon":
            points = np.asarray([
                (round(cx + radius_x * math.cos(math.pi / 6 + index * math.pi / 3)),
                 round(cy + radius_y * math.sin(math.pi / 6 + index * math.pi / 3)))
                for index in range(6)], np.int32)
            cv2.fillPoly(opening, [points], 255, cv2.LINE_AA)
        elif shape == "octagon":
            cut_x, cut_y = radius_x * .34, radius_y * .34
            points = np.asarray([
                (cx-radius_x+cut_x, cy-radius_y), (cx+radius_x-cut_x, cy-radius_y),
                (cx+radius_x, cy-radius_y+cut_y), (cx+radius_x, cy+radius_y-cut_y),
                (cx+radius_x-cut_x, cy+radius_y), (cx-radius_x+cut_x, cy+radius_y),
                (cx-radius_x, cy+radius_y-cut_y), (cx-radius_x, cy-radius_y+cut_y),
            ], np.int32)
            cv2.fillPoly(opening, [points], 255, cv2.LINE_AA)
        elif shape == "clipped_rect":
            half_x, half_y = radius_x * .64, radius_y * 1.06
            cut_x, cut_y = half_x * .28, half_y * .16
            points = np.asarray([
                (cx-half_x+cut_x, cy-half_y), (cx+half_x-cut_x, cy-half_y),
                (cx+half_x, cy-half_y+cut_y), (cx+half_x, cy+half_y-cut_y),
                (cx+half_x-cut_x, cy+half_y), (cx-half_x+cut_x, cy+half_y),
                (cx-half_x, cy+half_y-cut_y), (cx-half_x, cy-half_y+cut_y),
            ], np.int32)
            cv2.fillPoly(opening, [points], 255, cv2.LINE_AA)
        elif shape == "begonia":
            points = []
            for angle in np.linspace(0, 2 * math.pi, 240, endpoint=False):
                radius = 1.0 + .20 * math.cos(4 * angle)
                points.append((round(cx + radius_x * radius * math.cos(angle)),
                               round(cy + radius_y * radius * math.sin(angle))))
            cv2.fillPoly(opening, [np.asarray(points, np.int32)], 255, cv2.LINE_AA)
        elif shape == "plum":
            points = []
            for angle in np.linspace(0, 2 * math.pi, 300, endpoint=False):
                radius = 1.0 + .20 * math.cos(5 * angle - math.pi / 2)
                points.append((round(cx + radius_x * radius * math.cos(angle)),
                               round(cy + radius_y * radius * math.sin(angle))))
            cv2.fillPoly(opening, [np.asarray(points, np.int32)], 255, cv2.LINE_AA)
        elif shape == "ruyi":
            profile = [
                (0, -1.08), (.18, -.82), (.52, -.82), (.62, -.50),
                (.92, -.34), (.82, 0), (.96, .28), (.58, .42),
                (.46, .76), (0, 1.05), (-.46, .76), (-.58, .42),
                (-.96, .28), (-.82, 0), (-.92, -.34), (-.62, -.50),
                (-.52, -.82), (-.18, -.82),
            ]
            points = np.asarray([(round(cx + radius_x * x),
                                  round(cy + radius_y * y))
                                 for x, y in profile], np.int32)
            cv2.fillPoly(opening, [points], 255, cv2.LINE_AA)
        elif shape == "fan":
            arc = [(round(cx + radius_x * math.cos(angle)),
                    round(cy + radius_y * math.sin(angle)))
                   for angle in np.linspace(math.radians(200), math.radians(340), 120)]
            apex = (cx, round(cy + radius_y * .55))
            cv2.fillPoly(opening, [np.asarray(arc + [apex], np.int32)],
                         255, cv2.LINE_AA)
        elif shape == "gourd":
            cv2.ellipse(opening, (cx, round(cy - radius_y * .26)),
                        (round(radius_x * .60), round(radius_y * .52)),
                        0, 0, 360, 255, -1, cv2.LINE_AA)
            cv2.ellipse(opening, (cx, round(cy + radius_y * .25)),
                        (round(radius_x * .82), round(radius_y * .68)),
                        0, 0, 360, 255, -1, cv2.LINE_AA)
        else:
            cv2.ellipse(opening, center, (radius_x, radius_y),
                        0, 0, 360, 255, -1, cv2.LINE_AA)
    return cv2.bitwise_and(opening, inner)


def sprig_aperture_points(spec: dict) -> np.ndarray:
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    points = ((.22, .17), (.78, .17), (.91, .50), (.78, .83),
              (.22, .83), (.09, .50))
    return np.asarray([[round(x0 + width * u), round(y0 + height * v)]
                       for u, v in points], np.int32)


def composition_field(composition: str, inner: np.ndarray, spec: dict) -> np.ndarray:
    """Where the repeated motif may appear; white regions become intentional views."""
    if composition in ("portrait", "aperture"):
        return inner
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    region = np.zeros_like(inner)
    if composition == "borrowed_view":
        # The curved opening runs off the right edge, leaving an asymmetric view.
        cv2.ellipse(region, (round(x0 + width * .76), round(y0 + height * .50)),
                    (round(width * .43), round(height * .39)), 0, 0, 360, 255, -1)
        region = cv2.bitwise_not(region)
    elif composition == "framed_sprig":
        region[:] = 255
        cv2.fillPoly(region, [sprig_aperture_points(spec)], 0)
    elif composition == "framed_view":
        region[:] = 255
        cv2.rectangle(region, (round(x0 + width * .34), round(y0 + height * .22)),
                      (round(x0 + width * .72), round(y0 + height * .67)), 0, -1)
        cv2.ellipse(region, (round(x0 + width * .55), round(y0 + height * .82)),
                    (round(width * .095), round(height * .145)), 0, 0, 360, 0, -1)
    elif composition == "garden_veil":
        cv2.fillPoly(region, [garden_view_points(spec)], 255)
        region = cv2.bitwise_not(region)
        yy = np.arange(SIZE, dtype=np.float32)[:, None]
        xx = np.arange(SIZE, dtype=np.float32)[None, :]
        u = (xx - x0) / width
        v = (yy - y0) / height
        # A quiet central band keeps the symbol readable. Motifs occupy only
        # the upper and lower margins and terminate on deliberate curved rails.
        top = .36 + .045 * np.sin(np.pi * np.clip(u, 0, 1))
        bottom = .80 - .04 * np.sin(np.pi * np.clip(u, 0, 1))
        region = cv2.bitwise_and(region, np.uint8((v < top) | (v > bottom)) * 255)
    elif composition == "roundel":
        region[:] = 255
        cv2.ellipse(region, (round(x0 + width * .50), round(y0 + height * .50)),
                    (round(width * .34), round(height * .42)), 0, 0, 360, 0, -1)
    elif composition == "diagonal":
        yy, xx = np.ogrid[:SIZE, :SIZE]
        u = (xx - x0) / width
        v = (yy - y0) / height
        region[:] = np.uint8(v > .80 - .58 * u + .045 * np.sin(u * 2 * np.pi)) * 255
    elif composition == "diptych":
        cv2.rectangle(region, (round(x0 + width * .54), y0),
                      (x0 + width, y0 + height), 255, -1)
    elif composition == "corners":
        yy, xx = np.ogrid[:SIZE, :SIZE]
        u = (xx - x0) / width
        v = (yy - y0) / height
        distance = np.minimum(u, 1 - u) + np.minimum(v, 1 - v)
        region[:] = np.uint8(distance < .38) * 255
    elif composition == "echo":
        band = max(25, round(min(width, height) * .12))
        cv2.rectangle(region, (x0, y0), (x0 + width, y0 + height), 255, band * 2)
    else:
        raise ValueError(f"Unknown composition: {composition}")
    return cv2.bitwise_and(inner, region)


def composition_structure(composition: str, spec: dict, bar: int) -> np.ndarray:
    """Primary members that make each spatial grammar legible and connected."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    structure = np.zeros((SIZE, SIZE), np.uint8)
    point = lambda u, v: (round(x0 + width * u), round(y0 + height * v))
    if composition == "framed_sprig":
        cv2.polylines(structure, [sprig_aperture_points(spec)], True, 255,
                      max(8, round(bar * .85)), cv2.LINE_AA)
    elif composition == "borrowed_view":
        cv2.ellipse(structure, point(.76, .50),
                    (round(width * .43), round(height * .39)),
                    0, 0, 360, 255, max(8, round(bar * .8)))
    elif composition == "framed_view":
        cv2.rectangle(structure, point(.34, .22), point(.72, .67), 255,
                      max(12, round(bar * 1.5)), cv2.LINE_8)
        cv2.ellipse(structure, point(.55, .82),
                    (round(width * .095), round(height * .145)), 0, 0, 360,
                    255, max(8, round(bar * .85)), cv2.LINE_AA)
        cv2.line(structure, point(.55, .67), point(.55, .68), 255,
                 max(8, round(bar * .85)), cv2.LINE_AA)
    elif composition == "garden_veil":
        cv2.polylines(structure, [garden_view_points(spec)], True, 255,
                      max(8, round(bar * .8)), cv2.LINE_AA)
        for boundary in (lambda u: .36 + .045 * np.sin(np.pi * u),
                         lambda u: .80 - .04 * np.sin(np.pi * u)):
            rail = np.asarray([point(u, boundary(u)) for u in np.linspace(0, 1, 120)], np.int32)
            cv2.polylines(structure, [rail], False, 255, max(8, round(bar * .8)), cv2.LINE_AA)
        opening = np.zeros_like(structure)
        cv2.fillPoly(opening, [garden_view_points(spec)], 255)
        structure = cv2.bitwise_and(structure, cv2.bitwise_not(opening))
        cv2.polylines(structure, [garden_view_points(spec)], True, 255,
                      max(8, round(bar * .8)), cv2.LINE_AA)
    elif composition == "roundel":
        cv2.ellipse(structure, point(.5, .5), (round(width * .34), round(height * .42)),
                    0, 0, 360, 255, max(8, round(bar * .8)))
    elif composition == "diagonal":
        points = np.array([point(u, .80 - .58 * u + .045 * np.sin(u * 2 * np.pi))
                           for u in np.linspace(0, 1, 80)], np.int32)
        cv2.polylines(structure, [points], False, 255, max(8, round(bar * .8)), cv2.LINE_AA)
    elif composition == "diptych":
        cv2.line(structure, point(.53, 0), point(.53, 1), 255, max(10, bar), cv2.LINE_8)
    elif composition == "corners":
        cv2.line(structure, point(0, 0), point(1, 1), 255, max(8, round(bar * .75)), cv2.LINE_8)
        cv2.line(structure, point(1, 0), point(0, 1), 255, max(8, round(bar * .75)), cv2.LINE_8)
    elif composition == "echo":
        points = np.array([point(u, .48 + .13 * np.cos(4 * np.pi * (u - .23)))
                           for u in np.linspace(0, 1, 120)], np.int32)
        cv2.polylines(structure, [points], False, 255, max(8, round(bar * .75)), cv2.LINE_AA)
    return structure


def composition_ties(composition: str, subject: np.ndarray, spec: dict, bar: int) -> np.ndarray:
    """Place deliberate supports between an isolated figure and the main frame."""
    ties = np.zeros_like(subject)
    ys, xs = np.where(subject > 0)
    if not len(xs):
        return ties
    left, right, top, bottom = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    thickness = max(8, round(bar * .7))
    if composition == "roundel":
        middle = (top + bottom) // 2
        cv2.line(ties, (round(x0 + width * .16), middle), (left + 8, middle), 255, thickness)
        cv2.line(ties, (right - 8, middle), (round(x0 + width * .84), middle), 255, thickness)
        cv2.line(ties, ((left + right) // 2, bottom - 8),
                 ((left + right) // 2, round(y0 + height * .92)), 255, thickness)
    elif composition == "framed_view":
        cv2.line(ties, ((left + right) // 2, bottom - 5),
                 (round(x0 + width * .55), round(y0 + height * .965)),
                 255, thickness)
    elif composition == "diagonal":
        for fraction in (.38, .62):
            x = round(left + (right - left) * fraction)
            cv2.line(ties, (x, y0), (x, top + 10), 255, thickness)
    elif composition == "diptych":
        divider = round(x0 + width * .53)
        for fraction in (.38, .70):
            y = round(top + (bottom - top) * fraction)
            cv2.line(ties, (right - 8, y), (divider + 8, y), 255, thickness)
    return ties


def make_lattice(pattern: str, pitch: int, bar: int, layout: str = "grid", seed: int = 7,
                 custom_motif: Path | None = None, guide: Path | None = None,
                 clip_mask: np.ndarray | None = None,
                 transform_mode: str = "alternate",
                 attractor: tuple[float, float] | None = None,
                 field_strength: float = 0.0,
                 field_mode: str = "quiet") -> np.ndarray:
    if pattern in PRESETS or pattern == "custom":
        return render_motif(pattern, layout, pitch, bar, seed, custom_motif, guide, clip_mask,
                            transform_mode, attractor, field_strength, field_mode)
    if layout != "grid":
        raise ValueError("Legacy pattern studies support grid layout only")
    mask = np.zeros((SIZE, SIZE), np.uint8)
    if pattern == "diamond":
        for offset in range(-SIZE * 2, SIZE * 2, pitch):
            cv2.line(mask, (offset, 0), (offset + SIZE, SIZE), 255, bar, cv2.LINE_8)
            cv2.line(mask, (offset, 0), (offset - SIZE, SIZE), 255, bar, cv2.LINE_8)
    elif pattern == "vertical":
        for x in range(0, SIZE, pitch):
            cv2.line(mask, (x, 0), (x, SIZE), 255, bar, cv2.LINE_8)
        for y in range(0, SIZE, pitch * 2):
            cv2.line(mask, (0, y), (SIZE, y), 255, max(8, bar // 2), cv2.LINE_8)
    elif pattern == "radial":
        for angle in np.linspace(0, 2 * np.pi, max(16, round(2 * np.pi * 530 / pitch)), endpoint=False):
            end = (round(CENTER + 560 * np.cos(angle)), round(CENTER + 560 * np.sin(angle)))
            cv2.line(mask, (CENTER, CENTER), end, 255, bar, cv2.LINE_8)
        for radius in range(pitch, 550, pitch):
            cv2.circle(mask, (CENTER, CENTER), radius, 255, max(8, bar // 2), cv2.LINE_8)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    if clip_mask is None:
        clip_mask = np.zeros_like(mask)
        cv2.circle(clip_mask, (CENTER, CENTER), 530, 255, -1)
    return cv2.bitwise_and(mask, clip_mask)


def window_masks(shape: str, width_mm: float, height_mm: float, frame: int,
                 outline_points_mm: list[list[float]] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    if shape == "circle" and abs(width_mm - height_mm) > 1e-6:
        raise ValueError("Circular windows need equal width and height")
    if min(width_mm, height_mm) < 100 or max(width_mm, height_mm) > 3000:
        raise ValueError("Window dimensions must be between 100 and 3000 mm")
    outer = np.zeros((SIZE, SIZE), np.uint8)
    inner = np.zeros_like(outer)
    if shape == "circle":
        cv2.circle(outer, (CENTER, CENTER), 565, 255, -1)
        cv2.circle(inner, (CENTER, CENTER), 565 - frame, 255, -1)
        x0 = y0 = 35
        pixel_width = pixel_height = 1130
        motif_clip = np.zeros_like(outer)
        cv2.circle(motif_clip, (CENTER, CENTER), 530, 255, -1)
    elif shape in ("rectangle", "outline"):
        ratio = width_mm / height_mm
        if max(ratio, 1 / ratio) > 4:
            raise ValueError("This raster prototype supports window aspect ratios up to 4:1")
        pixel_width = 1130 if ratio >= 1 else round(1130 * ratio)
        pixel_height = 1130 if ratio <= 1 else round(1130 / ratio)
        if frame * 2 >= min(pixel_width, pixel_height) - 20:
            raise ValueError("Frame is too wide for the short side of this window")
        x0, y0 = CENTER - pixel_width // 2, CENTER - pixel_height // 2
        if shape == "rectangle":
            cv2.rectangle(outer, (x0, y0), (x0 + pixel_width, y0 + pixel_height), 255, -1)
            cv2.rectangle(inner, (x0 + frame, y0 + frame),
                          (x0 + pixel_width - frame, y0 + pixel_height - frame), 255, -1)
        else:
            if not outline_points_mm or len(outline_points_mm) < 3:
                raise ValueError("An outline window needs at least three points")
            points = np.asarray(outline_points_mm, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError("Outline points must be finite [x,y] coordinates")
            if np.any(points[:, 0] < -1e-6) or np.any(points[:, 0] > width_mm + 1e-6) or np.any(points[:, 1] < -1e-6) or np.any(points[:, 1] > height_mm + 1e-6):
                raise ValueError("Outline points must lie within the window bounding box")
            pixels = np.rint(np.column_stack((x0 + points[:, 0] * pixel_width / width_mm,
                                               y0 + pixel_height - points[:, 1] * pixel_height / height_mm))).astype(np.int32)
            cv2.fillPoly(outer, [pixels], 255, cv2.LINE_8)
            inner = cv2.erode(outer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                (frame * 2 + 1, frame * 2 + 1)))
            if cv2.countNonZero(inner) == 0:
                raise ValueError("Frame removes the entire interior of this window")
        motif_clip = outer
    else:
        raise ValueError(f"Unknown window shape: {shape}")
    spec = {"shape": shape, "width_mm": width_mm, "height_mm": height_mm,
            "origin_px": [x0, y0], "pixel_size": [pixel_width, pixel_height],
            "scale_mm_per_px": [width_mm / pixel_width, height_mm / pixel_height],
            "subject_max_px": [min(720, pixel_width * .68), min(790, pixel_height * .78)]}
    return outer, inner, {**spec, "motif_clip": motif_clip}


def resolve_islands(panel: np.ndarray, bridge_width: int = 10,
                    max_bridge_length: int = 55) -> tuple[np.ndarray, dict]:
    """Connect nearby islands; omit fragments needing conspicuous long bridges."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(panel, 8)
    if count <= 2:
        return panel, {"bridged_islands": 0, "discarded_specks": 0, "discarded_islands": 0}
    main_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main = np.uint8(labels == main_index) * 255
    output = main.copy()
    bridged = discarded = discarded_islands = 0
    for index in range(1, count):
        if index == main_index:
            continue
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 20:
            discarded += 1
            continue
        island = np.uint8(labels == index) * 255
        distances = cv2.distanceTransform(255 - main, cv2.DIST_L2, 5)
        ys, xs = np.where(island > 0)
        closest = int(np.argmin(distances[ys, xs]))
        px, py = int(xs[closest]), int(ys[closest])
        if distances[py, px] > max_bridge_length:
            discarded_islands += 1
            continue
        radius = int(np.ceil(distances[py, px])) + 4
        x0, x1 = max(0, px - radius), min(SIZE, px + radius + 1)
        y0, y1 = max(0, py - radius), min(SIZE, py + radius + 1)
        candidates_y, candidates_x = np.where(main[y0:y1, x0:x1] > 0)
        if not len(candidates_x):
            discarded += 1
            continue
        target = int(np.argmin((candidates_x + x0 - px) ** 2 + (candidates_y + y0 - py) ** 2))
        qx, qy = int(candidates_x[target] + x0), int(candidates_y[target] + y0)
        cv2.line(output, (px, py), (qx, qy), 255, bridge_width, cv2.LINE_8)
        output = cv2.bitwise_or(output, island)
        main = output
        bridged += 1
    return output, {"bridged_islands": bridged, "discarded_specks": discarded,
                    "discarded_islands": discarded_islands}


def embed_custom_knots(lattice: np.ndarray, motif_path: Path, inner: np.ndarray,
                       spec: dict, bar: int, size: int, seed: int,
                       placements=None, min_inside: float = .96) -> np.ndarray:
    """Replace a few field nodes with a user-drawn structural flower."""
    paths = load_custom_paths(motif_path)
    if not paths or len(paths[0]) < 5:
        raise ValueError("A flower knot needs one closed outline with at least five points")
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    # Asymmetric peripheral placement preserves the borrowed view. The factors
    # establish one primary flower and two quieter secondary knots.
    placements = placements or [(.18, .28, 1.00, -.16), (.19, .73, .82, .24),
                                (.83, .76, .92, -.10)]
    result = lattice.copy()
    base = lattice.copy()
    for u, v, factor, angle in placements:
        center = np.array([x0 + width * u, y0 + height * v], dtype=float)
        scale = float(size) * factor
        co, si = np.cos(angle), np.sin(angle)

        def transformed(points):
            return np.asarray([[round(center[0] + scale * (px * co - py * si)),
                                round(center[1] + scale * (px * si + py * co))]
                               for px, py in points], np.int32)

        outer = transformed(paths[0])
        footprint = np.zeros_like(result)
        cv2.fillPoly(footprint, [outer], 255)
        area = cv2.countNonZero(footprint)
        if area < 100 or cv2.countNonZero(cv2.bitwise_and(footprint, inner)) < area * min_inside:
            continue
        halo = cv2.dilate(footprint, np.ones((max(3, bar), max(3, bar)), np.uint8))
        if cv2.countNonZero(cv2.bitwise_and(halo, base)) < max(80, area * .025):
            continue
        # Existing bars terminate on the new flower boundary instead of
        # passing through it, so the motif replaces field geometry.
        result = cv2.bitwise_and(result, cv2.bitwise_not(footprint))
        knot = np.zeros_like(result)
        cv2.polylines(knot, [outer], True, 255, bar, cv2.LINE_AA)

        normalized = np.asarray(paths[0], dtype=float)
        centroid = normalized.mean(axis=0)
        relative = normalized - centroid
        polar = np.arctan2(relative[:, 1], relative[:, 0])
        radii = np.linalg.norm(relative, axis=1)
        valleys = []
        for target in np.linspace(-np.pi, np.pi, 5, endpoint=False):
            delta = np.abs(np.angle(np.exp(1j * (polar - target))))
            candidates = np.where(delta < np.pi / 5)[0]
            index = candidates[np.argmin(radii[candidates])] if len(candidates) else int(np.argmin(delta))
            valleys.append(tuple(outer[index]))

        centre_radius = max(round(bar * 1.35), round(scale * .07))
        centre_point = tuple(np.rint(center).astype(int))
        cv2.circle(knot, centre_point, centre_radius, 255,
                   max(7, round(bar * .70)), cv2.LINE_AA)
        for point in valleys:
            direction = np.asarray(point, dtype=float) - center
            length = max(1.0, float(np.linalg.norm(direction)))
            start = center + direction / length * centre_radius
            cv2.line(knot, tuple(np.rint(start).astype(int)), point, 255,
                     max(7, round(bar * .68)), cv2.LINE_AA)
        for extra in paths[1:3]:
            curve = transformed(extra)
            cv2.polylines(knot, [curve], True, 255, max(7, round(bar * .65)), cv2.LINE_AA)
            nearest = curve[np.argmin(np.linalg.norm(curve - center, axis=1))]
            cv2.line(knot, centre_point, tuple(nearest), 255,
                     max(7, round(bar * .62)), cv2.LINE_AA)
        result = cv2.bitwise_or(result, cv2.bitwise_and(knot, inner))
    return result


def embed_custom_aperture(lattice: np.ndarray, motif_path: Path, inner: np.ndarray,
                          spec: dict, bar: int, size: int) -> np.ndarray:
    """Use the user's outer contour as one large borrowed-view opening."""
    paths = load_custom_paths(motif_path)
    if not paths or len(paths[0]) < 5:
        raise ValueError("A drawn aperture needs one closed outline with at least five points")
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    center = np.array([x0 + width * .56, y0 + height * .50], dtype=float)
    scale = min(float(size) * 3.15, min(width, height) * .78)
    outer = np.asarray([[round(center[0] + scale * px), round(center[1] + scale * py)]
                        for px, py in paths[0]], np.int32)
    footprint = np.zeros_like(lattice)
    cv2.fillPoly(footprint, [outer], 255)
    footprint = cv2.bitwise_and(footprint, inner)
    result = cv2.bitwise_and(lattice, cv2.bitwise_not(footprint))
    rim = np.zeros_like(lattice)
    cv2.polylines(rim, [outer], True, 255, max(8, round(bar * .82)), cv2.LINE_AA)
    return cv2.bitwise_or(result, cv2.bitwise_and(rim, inner))


def build_panel(subject: np.ndarray, details: np.ndarray, pattern: str, pitch: int,
                bar: int, frame: int, cut_details: bool, layout: str = "grid",
                seed: int = 7, custom_motif: Path | None = None,
                guide: Path | None = None, window_shape: str = "circle",
                window_width_mm: float = 600, window_height_mm: float = 600,
                subject_clearance: int = 0, composition: str = "portrait",
                subject_clear_mask: np.ndarray | None = None,
                window_outline_mm: list[list[float]] | None = None,
                accent_motif: Path | None = None,
                accent_guide: Path | None = None,
                accent_pitch: int | None = None,
                accent_mode: str | None = None,
                transform_mode: str = "alternate",
                attractor: tuple[float, float] | None = None,
                field_strength: float = 0.0,
                field_mode: str = "quiet") -> tuple[np.ndarray, dict]:
    outer, inner, spec = window_masks(window_shape, window_width_mm, window_height_mm, frame,
                                      window_outline_mm)
    ring = cv2.subtract(outer, inner)
    if composition == "curated_weave" and pattern != "haitang":
        raise ValueError("The curated weave study currently supports the Haitang pattern only")
    if composition in ("curated", "curated_weave"):
        lattice = render_preset(pattern, inner, spec, bar, pitch, seed,
                                open_center=composition == "curated")
    else:
        lattice = make_lattice(pattern, pitch, bar, layout, seed, custom_motif, guide,
                               spec["motif_clip"], transform_mode, attractor,
                               field_strength, field_mode)
        lattice = cv2.bitwise_and(lattice, composition_field(composition, inner, spec))
        lattice = cv2.bitwise_or(lattice, cv2.bitwise_and(composition_structure(composition, spec, bar), inner))
    if accent_motif is not None:
        mode = accent_mode or ("sprig" if accent_guide is not None else "knot")
        if mode == "knot":
            lattice = embed_custom_knots(lattice, accent_motif, inner, spec, bar,
                                         accent_pitch or pitch, seed)
        elif mode == "border":
            lattice = embed_custom_knots(
                lattice, accent_motif, inner, spec, bar, accent_pitch or pitch, seed,
                placements=[(.08, .76, 1.55, .20), (.91, .23, 1.25, -.18)],
                min_inside=.52)
        elif mode == "aperture":
            lattice = embed_custom_aperture(lattice, accent_motif, inner, spec, bar,
                                             accent_pitch or pitch)
        elif mode == "sprig":
            if accent_guide is None:
                raise ValueError("Sprig accent needs a guide curve")
            accent = render_motif("custom", "sprig", accent_pitch or pitch, bar, seed,
                                  accent_motif, accent_guide, inner)
            lattice = cv2.bitwise_or(lattice, accent)
        else:
            raise ValueError("Unknown hand-drawn accent mode: {}".format(mode))
    subject = cv2.bitwise_and(subject, inner)
    details = cv2.bitwise_and(details, inner)
    clear_subject = cv2.bitwise_and(subject_clear_mask, inner) if subject_clear_mask is not None else subject
    effective_clearance = 0 if composition in ("corners", "echo") else subject_clearance
    if composition == "aperture" and cv2.countNonZero(subject):
        # The uploaded figure is the open view, outlined by a solid rim.
        outer_halo = cv2.dilate(subject, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * (effective_clearance + bar) + 1,) * 2))
        inner_halo = cv2.dilate(subject, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * max(2, effective_clearance) + 1,) * 2))
        aperture_rim = cv2.subtract(outer_halo, inner_halo)
        lattice = cv2.bitwise_and(lattice, cv2.bitwise_not(outer_halo))
        panel = cv2.bitwise_or(cv2.bitwise_or(ring, lattice), aperture_rim)
    else:
        if effective_clearance and cv2.countNonZero(clear_subject):
            radius = effective_clearance * 2 + 1
            halo = cv2.dilate(clear_subject, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)))
            lattice = cv2.bitwise_and(lattice, cv2.bitwise_not(halo))
        panel = cv2.bitwise_or(cv2.bitwise_or(ring, lattice), subject)
        if composition not in ("curated", "curated_weave"):
            panel = cv2.bitwise_or(panel, cv2.bitwise_and(composition_ties(composition, subject, spec, bar), inner))
    if cut_details and composition != "aperture":
        # Keep only sizeable dark marks fully inside the silhouette. This
        # preserves eyes and mouth on high-contrast images without punching
        # accidental holes through the subject's outer edge.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(details, 8)
        safe = cv2.erode(subject, np.ones((17, 17), np.uint8))
        holes = np.zeros_like(panel)
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            if 90 <= area <= 25000:
                component = np.uint8(labels == index) * 255
                component = cv2.bitwise_and(component, safe)
                if cv2.countNonZero(component) >= 90:
                    holes = cv2.bitwise_or(holes, component)
        panel = cv2.subtract(panel, holes)
    panel, island_report = resolve_islands(panel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(panel, 8)
    material_components = sorted((int(s[cv2.CC_STAT_AREA]) for s in stats[1:]), reverse=True)
    metrics = {
        "connected_material_components": count - 1,
        "component_areas_px": material_components[:10],
        "material_fraction": round(float(cv2.countNonZero(panel)) / float(cv2.countNonZero(outer)), 3),
        "requested_bar_width_mm": round(bar * min(spec["scale_mm_per_px"]), 2),
        "requested_frame_width_mm": round(frame * min(spec["scale_mm_per_px"]), 2),
        "window_shape": window_shape,
        "window_size_mm": [window_width_mm, window_height_mm],
        "composition": composition,
        "transform_mode": transform_mode,
        "attractor": list(attractor) if attractor is not None else None,
        "field_strength": round(field_strength, 3),
        "field_mode": field_mode,
        "subject_clearance_mm": round(effective_clearance * min(spec["scale_mm_per_px"]), 2),
        "export_spec": {k: v for k, v in spec.items() if k not in ("motif_clip", "subject_max_px")},
        **island_report,
    }
    return panel, metrics


def build_generator_panel(generator: str, pitch: int, bar: int, frame: int,
                          seed: int = 7, window_shape: str = "rectangle",
                          window_width_mm: float = 900, window_height_mm: float = 600,
                          window_outline_mm=None, accent_motif: Path | None = None,
                          accent_pitch: int | None = None,
                          accent_mode: str = "knot",
                          transform_mode: str = "alternate",
                          attractor: tuple[float, float] | None = None,
                          field_strength: float = 0.8,
                          field_mode: str = "quiet",
                          controller_image: Path | None = None,
                          controller_curve: Path | None = None,
                          borrowed_view_shape: str = "ellipse",
                          borrowed_view_guide: Path | None = None,
                          semantic_image: Path | None = None,
                          semantic_strength: float = 1.0,
                          semantic_polarity: str = "material",
                          semantic_mode: str = "silhouette") -> tuple[np.ndarray, dict]:
    """Build one whole-window generator without legacy motif/composition menus."""
    outer, inner, spec = window_masks(window_shape, window_width_mm, window_height_mm,
                                      frame, window_outline_mm)
    ring = cv2.subtract(outer, inner)
    generator_field_mode = "quiet" if field_mode == "borrowed" else field_mode
    if generator == "ice_field":
        bars = render_preset("binglie", inner, spec, bar, pitch, seed, open_center=False)
    elif generator in GENERATORS:
        kwargs = {"bar": bar, "pitch": pitch, "transform_mode": transform_mode,
                  "field_strength": field_strength, "field_mode": generator_field_mode}
        if attractor is not None:
            kwargs["attractor"] = attractor
        if controller_image is not None or controller_curve is not None:
            if generator not in CONTROLLER_GENERATORS:
                raise ValueError("This generator does not support image or curve controllers")
            if controller_image is not None:
                kwargs["controller_image"] = controller_image
            if controller_curve is not None:
                kwargs["controller_curve"] = controller_curve
        bars = GENERATORS[generator](inner, spec, **kwargs)
    else:
        raise ValueError("Unknown whole-window generator: {}".format(generator))
    if field_mode == "borrowed":
        opening = borrowed_view_opening(inner, spec, attractor, field_strength,
                                        borrowed_view_shape,
                                        borrowed_view_guide)
        bars = cv2.bitwise_and(bars, cv2.bitwise_not(opening))
        edge_width = max(5, round(bar * .72))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (edge_width * 2 + 1, edge_width * 2 + 1))
        inset = cv2.erode(opening, kernel, iterations=1)
        borrowed_frame = cv2.subtract(opening, inset)
        bars = cv2.bitwise_or(bars, borrowed_frame)
    if accent_motif is not None:
        if accent_mode == "knot":
            bars = embed_custom_knots(bars, accent_motif, inner, spec, bar,
                                      accent_pitch or pitch, seed)
        elif accent_mode == "border":
            bars = embed_custom_knots(
                bars, accent_motif, inner, spec, bar, accent_pitch or pitch, seed,
                placements=[(.08, .76, 1.55, .20), (.91, .23, 1.25, -.18)],
                min_inside=.52)
        elif accent_mode == "aperture":
            bars = embed_custom_aperture(bars, accent_motif, inner, spec, bar,
                                         accent_pitch or pitch)
        else:
            raise ValueError("Whole-window generators support knot, aperture, or border accents")
    if semantic_image is not None:
        bars = semantic_reveal(bars, inner, semantic_image, pitch, bar,
                               semantic_strength, semantic_polarity, semantic_mode)
    panel, island_report = resolve_islands(cv2.bitwise_or(ring, bars))
    count, _, stats, _ = cv2.connectedComponentsWithStats(panel, 8)
    components = sorted((int(s[cv2.CC_STAT_AREA]) for s in stats[1:]), reverse=True)
    metrics = {
        "generator": generator,
        "connected_material_components": count - 1,
        "component_areas_px": components[:10],
        "material_fraction": round(float(cv2.countNonZero(panel)) / float(cv2.countNonZero(outer)), 3),
        "requested_bar_width_mm": round(bar * min(spec["scale_mm_per_px"]), 2),
        "requested_frame_width_mm": round(frame * min(spec["scale_mm_per_px"]), 2),
        "window_shape": window_shape,
        "window_size_mm": [window_width_mm, window_height_mm],
        "transform_mode": transform_mode,
        "attractor": list(attractor) if attractor is not None else None,
        "field_strength": round(field_strength, 3),
        "field_mode": field_mode,
        "controller_image": str(controller_image) if controller_image else None,
        "controller_curve": str(controller_curve) if controller_curve else None,
        "borrowed_view_shape": borrowed_view_shape,
        "borrowed_view_guide": str(borrowed_view_guide) if borrowed_view_guide else None,
        "semantic_image": str(semantic_image) if semantic_image else None,
        "semantic_strength": round(semantic_strength, 3),
        "semantic_polarity": semantic_polarity,
        "semantic_mode": semantic_mode,
        "export_spec": {k:v for k,v in spec.items() if k not in ("motif_clip","subject_max_px")},
        **island_report,
    }
    return panel, metrics


def svg_from_mask(mask: np.ndarray, export_spec: dict | None = None,
                  engrave_mask: np.ndarray | None = None) -> str:
    if export_spec is None:
        export_spec = {"width_mm": PANEL_MM, "height_mm": PANEL_MM,
                       "origin_px": [0, 0], "scale_mm_per_px": [PANEL_MM / SIZE, PANEL_MM / SIZE]}
    x0, y0 = export_spec["origin_px"]
    sx, sy = export_spec["scale_mm_per_px"]
    width_mm, height_mm = export_spec["width_mm"], export_spec["height_mm"]
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        approx = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        # Rhino's SVG importer interprets path coordinates as model units.
        # Write the path itself in millimetres instead of relying on viewBox
        # scaling, which Rhino does not apply consistently.
        points = " ".join(f"{(x - x0) * sx:.2f},{(y - y0) * sy:.2f}" for x, y in approx)
        paths.append("M " + points + " Z")
    geometry = " ".join(paths)
    engraving = ""
    if engrave_mask is not None and cv2.countNonZero(engrave_mask):
        marks = []
        contours, _ = cv2.findContours(engrave_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 5:
                continue
            approx = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2)
            if len(approx) >= 3:
                points = " ".join(f"{(x - x0) * sx:.2f},{(y - y0) * sy:.2f}" for x, y in approx)
                marks.append("M " + points + " Z")
        engraving = f'<path d="{" ".join(marks)}" fill="#777" fill-rule="evenodd" id="ENGRAVE"/>\n'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" '
            f'height="{height_mm}mm" viewBox="0 0 {width_mm} {height_mm}">\n'
            f'<title>Picture Lattice flat panel</title>\n'
            f'<path d="{geometry}" fill="black" fill-rule="evenodd" id="CUT"/>\n'
            f'{engraving}</svg>\n')


def pdf_from_mask(mask: np.ndarray, export_spec: dict, output: Path,
                  engrave_mask: np.ndarray | None = None) -> None:
    """Write the same closed contour geometry to a full-size vector PDF."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    width_mm, height_mm = export_spec["width_mm"], export_spec["height_mm"]
    x0, y0 = export_spec["origin_px"]
    sx, sy = export_spec["scale_mm_per_px"]
    pdf = canvas.Canvas(str(output), pagesize=(width_mm * mm, height_mm * mm), pageCompression=1)
    pdf.setTitle("Picture Lattice flat panel - 1:1 vector geometry")
    path = pdf.beginPath()
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        approx = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        px, py = approx[0]
        path.moveTo((px - x0) * sx * mm, (height_mm - (py - y0) * sy) * mm)
        for px, py in approx[1:]:
            path.lineTo((px - x0) * sx * mm, (height_mm - (py - y0) * sy) * mm)
        path.close()
    pdf.setFillColorRGB(0, 0, 0)
    pdf.drawPath(path, fill=1, stroke=0, fillMode=0)
    if engrave_mask is not None and cv2.countNonZero(engrave_mask):
        marks = pdf.beginPath()
        contours, _ = cv2.findContours(engrave_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 5:
                continue
            approx = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            px, py = approx[0]
            marks.moveTo((px - x0) * sx * mm, (height_mm - (py - y0) * sy) * mm)
            for px, py in approx[1:]:
                marks.lineTo((px - x0) * sx * mm, (height_mm - (py - y0) * sy) * mm)
            marks.close()
        pdf.setFillColorRGB(.45, .45, .45)
        pdf.drawPath(marks, fill=1, stroke=0, fillMode=0)
    pdf.showPage()
    pdf.save()


def dxf_from_mask(mask: np.ndarray, export_spec: dict, output: Path,
                  engrave_mask: np.ndarray | None = None) -> None:
    """Write closed millimetre polylines for CAD review and laser handoff."""
    import ezdxf

    x0, y0 = export_spec["origin_px"]
    sx, sy = export_spec["scale_mm_per_px"]
    height_mm = export_spec["height_mm"]
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4  # millimetres
    doc.header["$MEASUREMENT"] = 1
    doc.layers.new("CUT", dxfattribs={"color": 1})
    if engrave_mask is not None and cv2.countNonZero(engrave_mask):
        doc.layers.new("ENGRAVE", dxfattribs={"color": 5})
    model = doc.modelspace()
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        approx = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        vertices = [((int(px) - x0) * sx, height_mm - (int(py) - y0) * sy)
                    for px, py in approx]
        model.add_lwpolyline(vertices, close=True, dxfattribs={"layer": "CUT"})
    if engrave_mask is not None and cv2.countNonZero(engrave_mask):
        contours, _ = cv2.findContours(engrave_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue
            approx = cv2.approxPolyDP(contour, 1.2, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            vertices = [((int(px) - x0) * sx, height_mm - (int(py) - y0) * sy)
                        for px, py in approx]
            model.add_lwpolyline(vertices, close=True, dxfattribs={"layer": "ENGRAVE"})
    doc.saveas(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", help="Optional subject image")
    parser.add_argument("--legacy-direct-image", action="store_true",
                        help="Research only: allow the rejected direct image-plus-preset experiment")
    parser.add_argument("--symbol-json", type=Path,
                        help="Normalized user-authored cut/engrave paths; replaces the image subject")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--generator", choices=["unit_field", "ice_field", "curve_weave",
                                                "radial_rosette", "growth_network",
                                                "meander_turns", "hex_plum", "coin_chain",
                                                "willow_leaves", "soft_strip_lattice",
                                                "stepped_medallion", "curved_cell_grid",
                                                "morph_cell_field", "hex_flower_field",
                                                "fish_scale_field", "phyllotaxis_star_screen",
                                                "contour_stream_field", "herringbone_screen",
                                                "octagon_square_grid", "plum_aperture_screen",
                                                "nested_wave_field",
                                                "lotus_rosette", "triangle_aperture_screen"],
                        help="Whole-window parametric engine; bypasses legacy pattern/composition/layout")
    parser.add_argument("--pattern", choices=[*PRESETS, "custom", "diamond", "vertical", "radial"], default="huiwen")
    parser.add_argument("--layout", choices=["grid", "radial", "curve", "vine", "sprig"], default="grid")
    parser.add_argument("--composition", choices=["curated", "curated_weave", "portrait", "aperture", "borrowed_view", "framed_view", "framed_sprig", "roundel", "diagonal", "diptych", "corners", "echo", "garden_veil"], default="portrait",
                        help="Subject and open-space arrangement, independent of motif and repeat layout")
    parser.add_argument("--custom-motif", type=Path, help="User-drawn polylines as normalized JSON")
    parser.add_argument("--accent-motif", type=Path,
                        help="User-drawn flower/shape woven along a guide over the traditional field")
    parser.add_argument("--accent-guide", type=Path,
                        help="Guide JSON for the hand-drawn accent")
    parser.add_argument("--accent-pitch", type=int,
                        help="Size for hand-drawn flower knots or spacing for legacy sprigs")
    parser.add_argument("--accent-mode", choices=["knot", "aperture", "border", "sprig"], default="knot",
                        help="Embed the drawing as structural flower knots; sprig is retained for old studies")
    parser.add_argument("--guide", type=Path, help="Normalized polyline JSON for curve layout")
    parser.add_argument("--seed", type=int, default=7, help="Variation seed for ice-crack pattern")
    parser.add_argument("--transform-mode",
                        choices=["alternate", "quarter_turn", "pinwheel", "nested", "field"],
                        default="alternate",
                        help="Repeated-unit symmetry grammar")
    parser.add_argument("--field-mode", choices=["quiet", "bloom", "vortex", "drift", "borrowed"],
                        default="quiet", help="How the attractor changes the lattice")
    parser.add_argument("--attractor-x", type=float, default=.68,
                        help="Attractor X position, normalized 0–1")
    parser.add_argument("--attractor-y", type=float, default=.48,
                        help="Attractor Y position, normalized 0–1")
    parser.add_argument("--field-strength", type=float, default=.8,
                        help="Attractor response strength, 0–1")
    parser.add_argument("--controller-image", type=Path,
                        help="Non-AI shape controller for compatible generators")
    parser.add_argument("--semantic-image", type=Path,
                        help="Non-AI image reveal for every whole-window generator")
    parser.add_argument("--semantic-strength", type=float, default=1.0,
                        help="Image reveal contrast, 0–1")
    parser.add_argument("--semantic-polarity", choices=["material", "opening"],
                        default="material",
                        help="Render the subject with heavier material or larger openings")
    parser.add_argument("--semantic-mode", choices=["silhouette", "portrait", "hybrid"],
                        default="silhouette",
                        help="Cutout silhouette, photographic tones, or cutout plus internal detail")
    parser.add_argument("--controller-curve", type=Path,
                        help="Normalized curve JSON for morph_cell_field; controls local cell shape")
    parser.add_argument("--borrowed-view-guide", type=Path,
                        help="Closed normalized curve JSON defining a true borrowed-view opening")
    parser.add_argument("--borrowed-view-shape",
                        choices=["ellipse", "circle", "hexagon", "octagon",
                                 "clipped_rect", "begonia", "plum", "ruyi",
                                 "fan", "gourd"],
                        default="ellipse", help="Preset outline for a borrowed-view opening")
    parser.add_argument("--pitch", type=int, default=110, help="Lattice spacing in working pixels")
    parser.add_argument("--bar", type=int, default=22, help="Nominal lattice bar width in working pixels")
    parser.add_argument("--frame", type=int, default=48, help="Frame width in working pixels")
    parser.add_argument("--window-shape", choices=["circle", "rectangle", "outline"], default="circle")
    parser.add_argument("--window-outline", type=Path,
                        help="JSON with points in local millimeters inside the selected window bounding box")
    parser.add_argument("--window-width-mm", type=float, default=600.0)
    parser.add_argument("--window-height-mm", type=float, default=600.0)
    parser.add_argument("--subject-scale", type=float, default=1.0)
    parser.add_argument("--subject-style", choices=["solid", "tracery"], default="solid",
                        help="Solid silhouette or cut outline with separate engraving marks")
    parser.add_argument("--subject-clearance", type=int, default=28,
                        help="White gap around an image subject, in working pixels")
    parser.add_argument("--background-tolerance", type=float, default=35.0)
    parser.add_argument("--no-details", action="store_true")
    args = parser.parse_args()
    if args.image is not None and not args.legacy_direct_image:
        parser.error("Positional subject images are an archived composition experiment. Use --semantic-image for local image reveal or --controller-image for a compatible shape controller.")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if args.symbol_json is not None and not args.symbol_json.is_file():
        parser.error(f"Symbol file not found: {args.symbol_json}")
    if args.controller_image is not None and not args.controller_image.is_file():
        parser.error(f"Controller image not found: {args.controller_image}")
    if args.semantic_image is not None and not args.semantic_image.is_file():
        parser.error(f"Semantic image not found: {args.semantic_image}")
    if args.controller_curve is not None and not args.controller_curve.is_file():
        parser.error(f"Controller curve not found: {args.controller_curve}")
    if args.borrowed_view_guide is not None and not args.borrowed_view_guide.is_file():
        parser.error(f"Borrowed-view guide not found: {args.borrowed_view_guide}")
    if args.controller_image is not None and args.controller_curve is not None:
        parser.error("Choose either --controller-image or --controller-curve")
    if (args.controller_image is not None or args.controller_curve is not None) and args.generator not in CONTROLLER_GENERATORS:
        parser.error("The selected generator does not support image or curve controllers")
    if args.image is not None and args.symbol_json is not None:
        parser.error("Choose either an image or --symbol-json as the subject")
    if args.generator and (args.image is not None or args.symbol_json is not None):
        parser.error("Whole-window generators are part of the no-image route")
    if args.pattern == "custom" and (args.custom_motif is None or not args.custom_motif.is_file()):
        parser.error("Custom pattern needs an existing --custom-motif JSON file")
    if args.pattern == "custom" and args.composition in ("curated", "curated_weave"):
        parser.error("Custom motifs need a field composition such as portrait or borrowed_view")
    if args.accent_motif and not args.accent_motif.is_file():
        parser.error("Hand-drawn accent needs an existing --accent-motif file")
    if args.accent_motif and args.accent_mode == "sprig" and (
            not args.accent_guide or not args.accent_guide.is_file()):
        parser.error("Sprig accent needs an existing --accent-guide file")
    if args.accent_motif and args.pattern == "custom":
        parser.error("Choose a named traditional pattern when using a hand-drawn accent")
    if args.accent_pitch is not None and not 55 <= args.accent_pitch <= 260:
        parser.error("Accent pitch must be 55–260")
    if args.composition == "curated_weave" and args.pattern != "haitang":
        parser.error("Curated weave currently supports --pattern haitang only")
    if args.layout in ("curve", "vine", "sprig") and (args.guide is None or not args.guide.is_file()):
        parser.error("Curve/vine/sprig layout needs an existing --guide JSON file")
    if args.subject_style == "tracery" and args.composition == "aperture":
        parser.error("Aperture already uses the subject as an opening; choose --subject-style solid")
    if args.window_shape == "outline" and (args.window_outline is None or not args.window_outline.is_file()):
        parser.error("Outline window needs --window-outline JSON")
    if not (24 <= args.pitch <= 260 and 5 <= args.bar <= 80 and 20 <= args.frame <= 100
            and 0.5 <= args.subject_scale <= 1.15 and 0 <= args.subject_clearance <= 80):
        parser.error("Use pitch 24–260, bar 5–80, frame 20–100, subject-scale 0.5–1.15, clearance 0–80")
    if not (0 <= args.attractor_x <= 1 and 0 <= args.attractor_y <= 1
            and 0 <= args.field_strength <= 1
            and 0 <= args.semantic_strength <= 1):
        parser.error("Use normalized attractor coordinates and field strength in the 0–1 range")
    args.out.mkdir(parents=True, exist_ok=True)
    outline_points = None
    if args.window_outline:
        try:
            outline_points = json.loads(args.window_outline.read_text(encoding="utf-8"))["points"]
        except (ValueError, KeyError, TypeError) as error:
            parser.error(f"Invalid window outline JSON: {error}")
    try:
        _, _, window_spec = window_masks(args.window_shape, args.window_width_mm,
                                         args.window_height_mm, args.frame, outline_points)
    except ValueError as error:
        parser.error(str(error))
    engrave = np.zeros((SIZE, SIZE), np.uint8)
    if args.symbol_json is not None:
        subject, engrave = symbol_from_json(args.symbol_json, window_spec, args.bar)
        subject_details = np.zeros_like(subject)
        info = {"extraction": "user-curves", "image_size": None}
    elif args.image is not None:
        silhouette, details, info = extract_subject(args.image, args.background_tolerance)
        subject, subject_details = place_composition_subject(silhouette, details,
                                                              args.subject_scale, args.composition, window_spec)
        if args.composition == "curated_weave" and args.subject_style == "solid" and not args.no_details:
            source_inset = extract_major_light_inset(args.image, silhouette)
            _, placed_inset = place_composition_subject(silhouette, source_inset,
                                                        args.subject_scale, args.composition, window_spec)
            subject_details = cv2.bitwise_or(subject_details, placed_inset)
            source_edges = extract_internal_engraving(args.image, silhouette)
            _, engrave = place_composition_subject(silhouette, source_edges,
                                                    args.subject_scale, args.composition, window_spec)
            engrave = cv2.dilate(engrave, np.ones((2, 2), np.uint8))
    else:
        subject = subject_details = np.zeros((SIZE, SIZE), np.uint8)
        info = {"extraction": "none", "image_size": None}
    clear_subject = None
    if args.subject_style == "tracery" and args.symbol_json is None:
        clear_subject = subject.copy()
        subject, engrave = tracery_subject(subject, subject_details)
        subject_details = np.zeros_like(subject)
    if args.generator:
        panel, metrics = build_generator_panel(
            args.generator, args.pitch, args.bar, args.frame, args.seed,
            args.window_shape, args.window_width_mm, args.window_height_mm,
            outline_points, args.accent_motif, args.accent_pitch, args.accent_mode,
            args.transform_mode, (args.attractor_x, args.attractor_y),
            args.field_strength, args.field_mode, args.controller_image,
            args.controller_curve, args.borrowed_view_shape, args.borrowed_view_guide,
            args.semantic_image,
            args.semantic_strength, args.semantic_polarity, args.semantic_mode)
    else:
        panel, metrics = build_panel(subject, subject_details, args.pattern, args.pitch,
                                     args.bar, args.frame, not args.no_details, args.layout,
                                     args.seed, args.custom_motif, args.guide,
                                     args.window_shape, args.window_width_mm, args.window_height_mm,
                                     args.subject_clearance, args.composition, clear_subject, outline_points,
                                     args.accent_motif, args.accent_guide, args.accent_pitch,
                                     args.accent_mode, args.transform_mode,
                                     (args.attractor_x, args.attractor_y),
                                     args.field_strength, args.field_mode)
        if args.semantic_image is not None:
            outer, inner, _ = window_masks(
                args.window_shape, args.window_width_mm, args.window_height_mm,
                args.frame, outline_points)
            ring = cv2.subtract(outer, inner)
            bars = semantic_reveal(cv2.bitwise_and(panel, inner), inner,
                                   args.semantic_image, args.pitch, args.bar,
                                   args.semantic_strength, args.semantic_polarity,
                                   args.semantic_mode)
            panel, semantic_islands = resolve_islands(cv2.bitwise_or(ring, bars))
            count, _, stats, _ = cv2.connectedComponentsWithStats(panel, 8)
            metrics.update(semantic_islands)
            metrics.update({
                "connected_material_components": count - 1,
                "component_areas_px": sorted(
                    (int(s[cv2.CC_STAT_AREA]) for s in stats[1:]), reverse=True)[:10],
                "material_fraction": round(
                    float(cv2.countNonZero(panel)) / float(cv2.countNonZero(outer)), 3),
                "semantic_image": str(args.semantic_image),
                "semantic_strength": round(args.semantic_strength, 3),
                "semantic_polarity": args.semantic_polarity,
                "semantic_mode": args.semantic_mode,
            })
    if args.composition == "curated_weave" and args.subject_style == "solid":
        engrave = cv2.bitwise_and(engrave, panel)
    Image.fromarray(subject).save(args.out / "subject_mask.png")
    preview = cv2.cvtColor(255 - panel, cv2.COLOR_GRAY2RGB)
    preview[engrave > 0] = (115, 115, 115)
    Image.fromarray(preview).save(args.out / "preview.png")
    svg = svg_from_mask(panel, metrics["export_spec"], engrave)
    (args.out / "panel.svg").write_text(svg, encoding="utf-8")
    pdf_from_mask(panel, metrics["export_spec"], args.out / "panel.pdf", engrave)
    dxf_from_mask(panel, metrics["export_spec"], args.out / "panel.dxf", engrave)
    # Rhino may reuse the previously imported SVG when several files share a
    # basename in different directories. Provide a distinct import filename.
    rhino_svg_name = f"{args.out.name}.svg"
    (args.out / rhino_svg_name).write_text(svg, encoding="utf-8")
    report = {**info, **metrics,
              "pattern": None if args.generator else args.pattern,
              "layout": None if args.generator else args.layout,
              "generator": args.generator,
              "subject_style": args.subject_style,
              "engrave_marks": int(cv2.connectedComponents(engrave, 8)[0] - 1),
              "pitch": args.pitch, "seed": args.seed,
              "custom_motif": str(args.custom_motif) if args.custom_motif else None,
              "guide": str(args.guide) if args.guide else None,
              "accent_motif": str(args.accent_motif) if args.accent_motif else None,
              "accent_guide": str(args.accent_guide) if args.accent_guide else None,
              "accent_pitch": args.accent_pitch,
              "accent_mode": args.accent_mode if args.accent_motif else None,
              "rhino_svg": rhino_svg_name,
              "source_image": str(args.image) if args.image else None,
              "symbol_json": str(args.symbol_json) if args.symbol_json else None,
              "window_outline": str(args.window_outline) if args.window_outline else None,
              "fabrication_status":
              "single_connected_piece" if metrics["connected_material_components"] == 1 else "review_disconnected_parts"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

