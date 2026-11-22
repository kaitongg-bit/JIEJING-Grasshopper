"""Deterministic non-AI whole-window generators for JIEJING studies.

Each function receives the usable inner mask and pixel export specification and
returns structural bars. The outer frame and fabrication checks are handled by
the existing lattice core.
"""
from __future__ import annotations

import math
import json

import cv2
import numpy as np


def _path(mask, points, width, closed=False):
    xy = np.rint(points).astype(np.int32)
    cv2.polylines(mask, [xy], closed, 255, width, cv2.LINE_AA)


def _flower(center, size, angle=0.0, count=5, samples=120):
    cx, cy = center
    points = []
    for t in np.linspace(0, 2 * math.pi, samples, endpoint=False):
        radius = size * (.34 + .075 * math.cos(count * t))
        a = t + angle
        points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return np.asarray(points)


def _field_warp(u, v, attractor, strength, mode):
    """Warp normalized coordinates with one broad, fabrication-safe field."""
    if attractor is None or strength <= 0:
        return u, v, 0.0, 0.0
    ax, ay = attractor
    dx, dy = u - ax, v - ay
    distance = max(1e-6, math.hypot(dx, dy))
    influence = min(1.0, strength) * math.exp(-((distance / .34) ** 2))
    rx, ry = dx / distance, dy / distance
    tx, ty = -ry, rx
    if mode == "quiet":
        amount = .11 * influence
        u, v = u + rx * amount, v + ry * amount
    elif mode == "bloom":
        # Larger units need more breathing room, so the field expands outward
        # instead of pulling every centre into a dense knot.
        amount = .060 * influence
        u, v = u + rx * amount, v + ry * amount
    elif mode == "vortex":
        amount = .10 * influence
        u, v = u + tx * amount, v + ty * amount
    elif mode == "drift":
        u, v = u + .09 * influence, v - .035 * influence
    return u, v, influence, math.atan2(dy, dx)


def _largest_center_component(mask):
    """Keep the most plausible centred foreground component."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(mask > 0), 8)
    if count <= 1:
        return np.uint8(mask > 0)
    height, width = mask.shape
    best_index = 1
    best_score = -1.0
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        cx, cy = centroids[index]
        distance = math.hypot((cx / width) - .5, (cy / height) - .5)
        edge_hits = int(x <= 1) + int(y <= 1) + int(x + w >= width - 1) + int(y + h >= height - 1)
        score = float(area) * max(.15, 1.15 - distance * 1.4) * (.35 if edge_hits >= 3 else 1.0)
        if score > best_score:
            best_index, best_score = index, score
    return np.uint8(labels == best_index)


def _tonal_field(image):
    """Preserve photographic light/dark structure for halftone control."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    low, high = np.percentile(clahe, (3, 97))
    normalized = np.clip((clahe.astype(np.float32) - low) / max(1.0, high - low), 0, 1)
    darkness = 1.0 - normalized
    local = np.abs(normalized - cv2.GaussianBlur(normalized, (0, 0),
                                                  max(3.0, min(image.shape[:2]) * .018)))
    if local.max() > 0:
        local /= local.max()
    edges = cv2.Canny(gray, 45, 135).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (0, 0), max(.8, min(image.shape[:2]) * .002))
    if edges.max() > 0:
        edges /= edges.max()
    field = np.clip(darkness * .72 + local * .17 + edges * .20, 0, 1)
    field = cv2.GaussianBlur(field, (0, 0), max(.7, min(image.shape[:2]) * .0015))
    low, high = np.percentile(field, (2, 98))
    return np.clip((field - low) / max(.001, high - low), 0, 1)


def _image_controller(path, width, height, mode="silhouette"):
    """Return a local, non-ML 0–1 image field.

    Transparent images use alpha. Simple backgrounds use corner colour distance.
    When border clutter suggests a casual photograph, local GrabCut supplies a
    centred-subject estimate. This is classical image processing, runs offline,
    and intentionally asks for a tighter crop when the scene is ambiguous.

    ``silhouette`` prioritizes the cutout, ``portrait`` uses the full cropped
    photograph's tones and edges, and ``hybrid`` keeps the cutout while restoring
    internal eyes, nose, mouth, and fur contrast.
    """
    if mode not in ("silhouette", "portrait", "hybrid"):
        raise ValueError("Image mode must be silhouette, portrait, or hybrid")
    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise ValueError("Image could not be read: {}".format(path))
    alpha = None
    if source.ndim == 2:
        image = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    elif source.shape[2] == 4:
        image = source[:, :, :3]
        alpha = source[:, :, 3].astype(np.float32) / 255.0
    else:
        image = source[:, :, :3]
    source_h, source_w = image.shape[:2]
    if mode == "portrait":
        # Cover-crop rather than letterbox: for a face photograph the features
        # should occupy the available window instead of becoming a small cutout.
        scale = max(width / source_w, height / source_h)
        cover_w = max(width, round(source_w * scale))
        cover_h = max(height, round(source_h * scale))
        cover = cv2.resize(image, (cover_w, cover_h), interpolation=cv2.INTER_AREA)
        left = (cover_w - width) // 2
        top = (cover_h - height) // 2
        return _tonal_field(cover[top:top + height, left:left + width])
    corner = max(2, round(min(source_h, source_w) * .08))
    samples = np.concatenate((
        image[:corner, :corner].reshape(-1, 3),
        image[:corner, -corner:].reshape(-1, 3),
        image[-corner:, :corner].reshape(-1, 3),
        image[-corner:, -corner:].reshape(-1, 3)), axis=0)
    background = np.median(samples, axis=0)
    scale = min(width / source_w, height / source_h) * .90
    target_w = max(1, round(source_w * scale))
    target_h = max(1, round(source_h * scale))
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    canvas = np.empty((height, width, 3), np.uint8)
    canvas[:] = np.rint(background).astype(np.uint8)
    left = (width - target_w) // 2
    top = (height - target_h) // 2
    canvas[top:top + target_h, left:left + target_w] = resized
    if alpha is not None and float(alpha.min()) < .94:
        alpha_resized = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)
        foreground = np.zeros((height, width), np.float32)
        foreground[top:top + target_h, left:left + target_w] = alpha_resized
        subject = np.clip(foreground, 0, 1)
    else:
        delta = np.linalg.norm(canvas.astype(np.float32) - background.astype(np.float32), axis=2)
        high = max(18.0, float(np.percentile(delta, 94)))
        colour_field = np.clip(delta / high, 0, 1)
        rough = np.uint8(colour_field > .16)
        border = max(4, round(min(width, height) * .035))
        border_mask = np.zeros_like(rough)
        border_mask[:border, :] = 1
        border_mask[-border:, :] = 1
        border_mask[:, :border] = 1
        border_mask[:, -border:] = 1
        border_clutter = float(np.count_nonzero(rough & border_mask)) / max(1, np.count_nonzero(border_mask))
        coverage = float(np.count_nonzero(rough)) / rough.size
        if border_clutter < .24 and coverage < .72:
            foreground = _largest_center_component(rough)
        else:
            grab_mask = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
            grab_mask[border_mask > 0] = cv2.GC_BGD
            central = np.zeros_like(rough)
            cv2.ellipse(central, (width // 2, height // 2),
                        (max(3, round(width * .31)), max(3, round(height * .38))),
                        0, 0, 360, 1, -1)
            grab_mask[central > 0] = cv2.GC_PR_FGD
            grab_mask[colour_field > .58] = cv2.GC_PR_FGD
            model_bg = np.zeros((1, 65), np.float64)
            model_fg = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(canvas, grab_mask, None, model_bg, model_fg, 5,
                            cv2.GC_INIT_WITH_MASK)
                grab = np.uint8((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD))
                foreground = _largest_center_component(grab)
            except cv2.error:
                foreground = _largest_center_component(rough)
        kernel = np.ones((5, 5), np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
        subject = cv2.GaussianBlur(foreground.astype(np.float32), (0, 0),
                                   max(1.5, min(width, height) * .004))
        if subject.max() > 0:
            subject /= subject.max()
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 55, 150).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (0, 0), max(2.0, min(width, height) * .010))
    if edges.max() > 0:
        edges /= edges.max()
    if mode == "hybrid":
        tone = _tonal_field(canvas)
        # A restrained baseline keeps the whole subject legible while dark
        # facial features receive enough range to survive the lattice sampling.
        field = np.clip(subject * (.24 + tone * .86) + edges * subject * .18, 0, 1)
    else:
        field = np.clip(subject * .86 + edges * subject * .30, 0, 1)
    field = cv2.GaussianBlur(field, (0, 0), max(2.0, min(width, height) * .012))
    if field.max() > 0:
        field /= field.max()
    return field


def semantic_reveal(mask, inner, path, pitch, bar, strength=1.0,
                    polarity="material", mode="silhouette"):
    """Reveal an ordinary image through local material/opening weight.

    This is deterministic image sampling, not semantic recognition.  The same
    signed-distance offset works after any whole-window generator: line-based
    lattices gain locally heavier strokes, while perforated screens gain locally
    smaller apertures.  The operation only adds material, so it cannot sever the
    source topology or create loose pieces.
    """
    if polarity not in ("material", "opening"):
        raise ValueError("Semantic polarity must be material or opening")
    if not 0 <= strength <= 1:
        raise ValueError("Semantic reveal strength must be between 0 and 1")
    if strength == 0:
        return cv2.bitwise_and(mask, inner)
    height, width = inner.shape
    field = _image_controller(path, width, height, mode)
    if polarity == "opening":
        field = 1.0 - field
    # Suppress weak photographic noise while keeping antialiased contours and
    # internal features.  The smoothstep makes a recognisable subject without
    # imposing a hard binary outline.
    field = np.clip((field - .08) / .82, 0, 1)
    field = field * field * (3.0 - 2.0 * field)

    solid = np.uint8(mask > 96)
    inside_distance = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform(1 - solid, cv2.DIST_L2, 5)
    signed_distance = inside_distance - outside_distance

    amplitude = min(max(5.0, bar * 1.55), max(7.0, pitch * .24)) * strength
    if mode in ("portrait", "hybrid"):
        # Photographic halftones need both larger and smaller openings. A small
        # negative range opens light fur while the larger positive range closes
        # around dark eyes and mouth. The bias is conservative enough to keep
        # the source lattice connected after fabrication cleanup.
        response = field * 1.18 - .18
    else:
        response = field
    local_offset = amplitude * response
    revealed = np.uint8((signed_distance + local_offset) >= 0) * 255
    return cv2.bitwise_and(revealed, inner)


def _curve_controller(path):
    """Load one or more normalized 0–1 controller polylines."""
    try:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError("Invalid controller curve JSON: {}".format(error))
    paths = data.get("paths")
    if paths is None and data.get("points") is not None:
        paths = [data["points"]]
    if not paths or any(len(item) < 2 for item in paths):
        raise ValueError("Controller curve needs at least one two-point path")
    arrays = [np.asarray(item, dtype=float) for item in paths]
    if any(array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all()
           or (array < 0).any() or (array > 1).any() for array in arrays):
        raise ValueError("Controller curve points must be normalized XY values")
    return arrays


def _distance_to_paths(point, paths):
    """Shortest normalized distance from a point to polyline segments."""
    point = np.asarray(point, dtype=float)
    best = float("inf")
    for path in paths:
        for a, b in zip(path, path[1:]):
            delta = b - a
            denom = float(delta @ delta)
            t = 0.0 if denom <= 1e-12 else min(1.0, max(0.0, float((point - a) @ delta) / denom))
            best = min(best, float(np.linalg.norm(point - (a + delta * t))))
    return best


def _controller_sample(u, v, strength, attractor, mode,
                       image_field=None, curve_paths=None):
    """Sample one controller source with a common 0–1 response."""
    if image_field is not None:
        height, width = image_field.shape
        x = min(width - 1, max(0, round(u * (width - 1))))
        y = min(height - 1, max(0, round(v * (height - 1))))
        influence = float(image_field[y, x])
        angle = 0.0
    elif curve_paths is not None:
        distance = _distance_to_paths((u, v), curve_paths)
        influence = math.exp(-((distance / .18) ** 2))
        angle = 0.0
    else:
        _, _, influence, angle = _field_warp(u, v, attractor, 1.0, mode)
    return min(1.0, max(0.0, influence * strength)), angle


def unit_field(inner, spec, bar=14, pitch=150, attractor=(.76, .48),
               field_strength=.85, field_mode="quiet", transform_mode="alternate"):
    """Interlocked four-lobed cells shaped by a multi-response field.

    The attractor changes position, scale and orientation together.  A quiet
    field opens a borrowed-view region; bloom, vortex and drift create other
    coherent responses without breaking the shared-node lattice.
    """
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    step_x = max(95, pitch)
    step_y = step_x * .73
    rows = []
    row = 0
    for y in np.arange(y0 + step_y * .45, y0 + height, step_y):
        centers = []
        offset = step_x * .5 if row % 2 else 0
        for col, x in enumerate(np.arange(x0 - step_x + offset, x0 + width + step_x, step_x)):
            u, v = (x - x0) / width, (y - y0) / height
            wu, wv, influence, radial = _field_warp(u, v, attractor,
                                                     field_strength, field_mode)
            px, py = x0 + width * wu, y0 + height * wv
            iy, ix = round(py), max(0, min(inner.shape[1] - 1, round(px)))
            if not (0 <= iy < inner.shape[0]) or inner[iy, ix] == 0:
                continue
            if field_mode == "quiet":
                scale = step_x * (.80 - .18 * influence)
            elif field_mode == "bloom":
                scale = step_x * (.76 + .20 * influence)
            else:
                scale = step_x * (.79 + .18 * influence)
            phase = ((row + col) % 2) * math.pi / 4 if transform_mode == "alternate" else 0
            angle = phase + (radial + math.pi / 2) * influence * (.45 if field_mode == "vortex" else .16)
            flower = _flower((px, py), scale, angle, count=4)
            _path(out, flower, bar, True)
            # A nested inner line gives the cell a traditional layered reading
            # near the field focus without adding an unrelated backing grid.
            if transform_mode == "nested" or (field_mode == "bloom" and influence > .58):
                _path(out, _flower((px, py), scale * .56, angle + math.pi / 4, count=4),
                      max(7, round(bar * .66)), True)
            centers.append((px, py, flower, scale))
        rows.append(centers)
        row += 1
    # Short tip-to-tip ties keep the changing cell sizes structurally honest.
    all_cells = [cell for row_cells in rows for cell in row_cells]
    for index, (x, y, flower, scale) in enumerate(all_cells):
        for j in range(index + 1, len(all_cells)):
            qx, qy, qflower, qscale = all_cells[j]
            distance = math.hypot(qx - x, qy - y)
            if distance > step_x * 1.24:
                continue
            delta = np.asarray([qx - x, qy - y], dtype=float)
            a = flower[np.argmax((flower - np.asarray([x, y])) @ delta)]
            b = qflower[np.argmax((qflower - np.asarray([qx, qy])) @ -delta)]
            if np.linalg.norm(a - b) < step_x * .62:
                cv2.line(out, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                         255, max(8, round(bar * .72)), cv2.LINE_AA)
    return cv2.bitwise_and(out, inner)


def curve_weave(inner, spec, bar=14, pitch=150, attractor=(.60, .50),
                field_strength=.95, field_mode="quiet", transform_mode="alternate"):
    """Continuous willow-strip families bend, rotate and open around a field."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    cx, cy = x0 + width * attractor[0], y0 + height * attractor[1]
    rx, ry = width * .23, height * .28
    samples = 280
    for row, base_v in enumerate(np.linspace(.12, .88, 7)):
        pts = []
        sign = -1 if base_v < .50 else 1
        for u in np.linspace(0, 1, samples):
            wave = .018 * math.sin(2 * math.pi * (u * 1.12 + row * .13))
            uu, vv, influence, radial = _field_warp(u, base_v + wave, attractor,
                                                     field_strength, field_mode)
            x = x0 + width * uu
            y = y0 + height * vv
            pts.append((x, y))
        _path(out, pts, bar)
    for col, base_u in enumerate((.10, .23, .37, .79, .91)):
        pts = []
        for v in np.linspace(0, 1, samples):
            wave = .014 * math.sin(2 * math.pi * (v * 1.08 + col * .17))
            uu, vv, influence, radial = _field_warp(base_u + wave, v, attractor,
                                                     field_strength, field_mode)
            if transform_mode == "pinwheel":
                uu += math.cos(radial + math.pi / 2) * influence * .025
                vv += math.sin(radial + math.pi / 2) * influence * .025
            x = x0 + width * uu
            y = y0 + height * vv
            pts.append((x, y))
        _path(out, pts, max(8, round(bar * .78)))
    # One quiet, four-lobed rim grows from the same curve vocabulary rather
    # than appearing as a generic ellipse pasted over the weave.
    rim = _flower((cx, cy), min(width, height) * .94, 0, count=4, samples=180)
    rim[:, 0] = cx + (rim[:, 0] - cx) * 1.28
    _path(out, rim, max(9, round(bar * .85)), True)
    return cv2.bitwise_and(out, inner)


def radial_rosette(inner, spec, bar=14, pitch=150, attractor=(.53, .50),
                   field_strength=.0, field_mode="quiet", transform_mode="alternate"):
    """Offset elliptical rings and twisted spokes form a rose-window system."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    cx, cy = x0 + width * .53, y0 + height * .50
    inner_rx, inner_ry = width * .14, height * .19
    outer_rx, outer_ry = width * .43, height * .43
    for factor in (.48, .72, 1.0):
        cv2.ellipse(out, (round(cx), round(cy)),
                    (round(outer_rx * factor), round(outer_ry * factor)),
                    0, 0, 360, 255, max(8, round(bar * (.72 if factor < 1 else .88))), cv2.LINE_AA)
    cv2.ellipse(out, (round(cx), round(cy)), (round(inner_rx), round(inner_ry)),
                0, 0, 360, 255, bar, cv2.LINE_AA)
    for index, angle in enumerate(np.linspace(0, 2 * math.pi, 14, endpoint=False)):
        pts = []
        for t in np.linspace(0, 1, 90):
            twist = (.13 if index % 2 == 0 else -.13) * math.sin(math.pi * t)
            a = angle + twist
            rx = inner_rx + (outer_rx - inner_rx) * t
            ry = inner_ry + (outer_ry - inner_ry) * t
            pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
        _path(out, pts, max(8, round(bar * .76)))
    # Four quiet structural ties anchor the outer ellipse to the frame.
    for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
        p = (cx + outer_rx * math.cos(a), cy + outer_ry * math.sin(a))
        q = ((x0 + width) if a == 0 else x0 if a == math.pi else cx,
             (y0 + height) if a == math.pi / 2 else y0 if a == 3 * math.pi / 2 else cy)
        cv2.line(out, tuple(np.rint(p).astype(int)), tuple(np.rint(q).astype(int)),
                 255, max(8, round(bar * .8)), cv2.LINE_AA)
    return cv2.bitwise_and(out, inner)


def _bezier(p0, p1, p2, p3, count=70):
    pts = []
    for t in np.linspace(0, 1, count):
        s = 1 - t
        pts.append(s ** 3 * p0 + 3 * s * s * t * p1 + 3 * s * t * t * p2 + t ** 3 * p3)
    return np.asarray(pts)


def _nearest_material(mask, point, radius=64):
    """Snap a designed branch node to nearby existing structure."""
    x, y = np.rint(point).astype(int)
    x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
    ys, xs = np.where(mask[y0:y1, x0:x1] > 0)
    if not len(xs):
        return np.asarray(point, dtype=float)
    candidates = np.column_stack((xs + x0, ys + y0)).astype(float)
    return candidates[np.argmin(np.linalg.norm(candidates - np.asarray(point), axis=1))]


def growth_network(inner, spec, bar=14, pitch=150, attractor=(.66, .43),
                   field_strength=.55, field_mode="drift", transform_mode="alternate"):
    """A composed flowering sprig with one trunk, three branches, and clear voids."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    def q(u, v):
        return np.asarray([x0 + width * u, y0 + height * v], dtype=float)

    # Main branch has a clear rising gesture and connects frame to frame.
    trunk = _bezier(q(0, .82), q(.23, .76), q(.70, .36), q(1, .25), 180)
    _path(out, trunk, max(10, round(bar * 1.05)))
    branches = [
        (_bezier(q(.31, .67), q(.25, .54), q(.22, .34), q(.25, .20), 90), .25, .20),
        (_bezier(q(.55, .49), q(.58, .35), q(.65, .24), q(.69, .15), 90), .69, .15),
        (_bezier(q(.73, .36), q(.79, .34), q(.84, .39), q(.88, .43), 90), .88, .43),
    ]
    for curve, _, _ in branches:
        curve[0] = _nearest_material(out, curve[0], 54)
        _path(out, curve, max(8, round(bar * .82)))

    # Three modest blossoms terminate the side branches.  Their centres are
    # solid structural nodes, so the branch-to-flower relationship is legible.
    flower_size = min(width, height) * .125
    for index, (_, u, v) in enumerate(branches):
        point = q(u, v)
        flower = _flower(point, flower_size * (1.0 if index != 1 else 1.08),
                         -.30 + index * .22, count=5, samples=140)
        _path(out, flower, max(8, round(bar * .78)), True)
        cv2.circle(out, tuple(np.rint(point).astype(int)),
                   max(7, round(bar * .80)), 255, -1, cv2.LINE_AA)

    # Almond leaves alternate along the trunk and branches.  Each starts on
    # existing material and opens toward the available negative space.
    leaf_sites = [
        (.39, .61, -.95, 1), (.47, .55, .55, -1),
        (.61, .45, -.85, 1), (.79, .35, .55, -1),
        (.27, .48, -.15, -1), (.61, .30, .20, 1),
    ]
    leaf_length = min(width, height) * .085
    for u, v, angle, side in leaf_sites:
        base = _nearest_material(out, q(u, v), 60)
        direction = np.asarray([math.cos(angle), math.sin(angle)])
        normal = np.asarray([-direction[1], direction[0]]) * side
        tip = base + direction * leaf_length
        upper = _bezier(base, base + direction*leaf_length*.34 + normal*leaf_length*.24,
                        tip - direction*leaf_length*.25 + normal*leaf_length*.16,
                        tip, 36)
        lower = _bezier(tip, tip - direction*leaf_length*.25 - normal*leaf_length*.16,
                        base + direction*leaf_length*.34 - normal*leaf_length*.18,
                        base, 36)
        _path(out, np.vstack((upper, lower[1:])), max(7, round(bar * .62)), True)
    return cv2.bitwise_and(out, inner)


def meander_turns(inner, spec, bar=14, pitch=150, attractor=(.68, .46),
                  field_strength=.75, field_mode="quiet", transform_mode="quarter_turn"):
    """One continuous orthogonal fret path, folded row by row.

    The previous version drew separate horizontal bands and added arbitrary
    vertical stitches.  Those stitches created broken-looking T junctions.
    This version is a single serpentine path: every turn belongs to the same
    construction and row changes happen only at the frame edge.
    """
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    step = max(92.0, pitch * .76)
    row_gap = max(78.0, pitch * .62)
    rows = list(np.arange(y0 + row_gap * .52, y0 + height - row_gap * .30,
                          row_gap))
    points = []
    for row_index, base_y in enumerate(rows):
        left_to_right = row_index % 2 == 0
        edge_start = x0 if left_to_right else x0 + width
        edge_end = x0 + width if left_to_right else x0
        if not points:
            points.append((edge_start, base_y))
        else:
            points.append((edge_start, base_y))
        travelled = 0.0
        cell_index = 0
        while travelled < width - 1:
            local_step = min(step, width - travelled)
            sign = -1 if (row_index + cell_index) % 2 else 1
            amplitude = row_gap * (.20 + .10 * min(1.0, field_strength))
            distances = (0.18, 0.18, 0.50, 0.50, 0.82, 0.82, 1.0)
            offsets = (0.0, sign * amplitude, sign * amplitude,
                       -sign * amplitude, -sign * amplitude, 0.0, 0.0)
            for distance, offset in zip(distances, offsets):
                run = travelled + local_step * distance
                x = x0 + run if left_to_right else x0 + width - run
                points.append((x, base_y + offset))
            travelled += local_step
            cell_index += 1
        points.append((edge_end, base_y))
    _path(out, points, bar)
    return cv2.bitwise_and(out, inner)


def hex_plum(inner, spec, bar=14, pitch=150, attractor=(.68, .45),
             field_strength=.8, field_mode="bloom", transform_mode="alternate"):
    """Shared-edge hexagons whose selected cells become six-angle-through-plum nodes."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    radius = max(62, pitch * .60)
    step_x = radius * 1.5
    step_y = radius * math.sqrt(3)
    col = 0
    for x in np.arange(x0 - radius, x0 + width + radius, step_x):
        y_offset = step_y / 2 if col % 2 else 0
        for row, y in enumerate(np.arange(y0 - radius + y_offset,
                                          y0 + height + radius, step_y)):
            outer = np.asarray([(x + radius * math.cos(k * math.pi / 3),
                                 y + radius * math.sin(k * math.pi / 3))
                                for k in range(6)])
            _path(out, outer, bar, True)
            influence, radial = 0.0, 0.0
            if attractor is not None:
                dx=(x-(x0+width*attractor[0]))/width
                dy=(y-(y0+height*attractor[1]))/height
                influence=min(1.0,field_strength)*math.exp(-((math.hypot(dx,dy)/.34)**2))
                radial=math.atan2(dy,dx)
            use_plum = influence > .30 or (row + col * 2) % 5 == 0
            if use_plum:
                flower_size = radius * (1.00 + .26 * influence)
                flower = _flower((x, y), flower_size,
                                 radial * .18 + (row % 2) * math.pi / 6, count=5)
                _path(out, flower, max(8, round(bar * .78)), True)
                for angle in np.linspace(0, 2 * math.pi, 5, endpoint=False):
                    a=(x+flower_size*.22*math.cos(angle),y+flower_size*.22*math.sin(angle))
                    b=(x+radius*.86*math.cos(angle),y+radius*.86*math.sin(angle))
                    cv2.line(out,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),
                             255,max(7,round(bar*.64)),cv2.LINE_AA)
            else:
                inner_hex = np.asarray([(x + radius * .36 * math.cos(k * math.pi / 3),
                                         y + radius * .36 * math.sin(k * math.pi / 3))
                                        for k in range(6)])
                _path(out, inner_hex, max(7, round(bar * .68)), True)
                for k in (0, 2, 4):
                    cv2.line(out, tuple(np.rint(inner_hex[k]).astype(int)),
                             tuple(np.rint(outer[k]).astype(int)), 255,
                             max(7, round(bar * .64)), cv2.LINE_AA)
        col += 1
    return cv2.bitwise_and(out, inner)


def coin_chain(inner, spec, bar=14, pitch=150, attractor=(.67, .46),
               field_strength=.82, field_mode="bloom", transform_mode="nested"):
    """Interlocking money-ring chains with nested, field-scaled openings."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    base_radius = max(56, pitch * .56)
    step_x = base_radius * 1.92
    step_y = base_radius * 1.92
    for row, y in enumerate(np.arange(y0 - base_radius, y0 + height + base_radius, step_y)):
        offset = step_x * .5 if row % 2 else 0
        for col, x in enumerate(np.arange(x0 - base_radius + offset,
                                          x0 + width + base_radius, step_x)):
            u, v = (x - x0) / width, (y - y0) / height
            _, _, influence, radial = _field_warp(u, v, attractor,
                                                   field_strength, field_mode)
            radius = base_radius * (1 + (.34 if field_mode == "bloom" else -.18) * influence)
            cv2.circle(out, (round(x), round(y)), round(radius), 255, bar, cv2.LINE_AA)
            cv2.circle(out, (round(x), round(y)), round(radius * .48), 255,
                       max(7, round(bar * .68)), cv2.LINE_AA)
            phase = (row + col) % 2 * math.pi / 4 + radial * influence * .22
            for a in np.linspace(0, 2 * math.pi, 4, endpoint=False) + phase:
                p=(round(x+radius*.48*math.cos(a)),round(y+radius*.48*math.sin(a)))
                q=(round(x+radius*.98*math.cos(a)),round(y+radius*.98*math.sin(a)))
                cv2.line(out,p,q,255,max(7,round(bar*.62)),cv2.LINE_AA)
    return cv2.bitwise_and(out, inner)


def willow_leaves(inner, spec, bar=14, pitch=150, attractor=(.66, .48),
                  field_strength=.9, field_mode="vortex", transform_mode="field"):
    """Long willow-leaf openings share endpoints along several flowing bands."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    count = max(5, round(width / max(105, pitch * .82)))
    for band, base_v in enumerate((.16, .36, .58, .79)):
        nodes=[]
        for i,u in enumerate(np.linspace(0,1,count+1)):
            v=base_v+.035*math.sin(2*math.pi*(u+band*.17))
            wu,wv,influence,radial=_field_warp(u,v,attractor,field_strength,field_mode)
            nodes.append(np.asarray([x0+width*wu,y0+height*wv]))
        for index,(a,b) in enumerate(zip(nodes,nodes[1:])):
            delta=b-a; length=np.linalg.norm(delta)
            if length<2: continue
            tangent=delta/length; normal=np.asarray([-tangent[1],tangent[0]])
            mid=(a+b)/2
            u=(mid[0]-x0)/width; v=(mid[1]-y0)/height
            _,_,influence,radial=_field_warp(u,v,attractor,field_strength,field_mode)
            leaf_height=length*(.18+.14*influence)
            sign=-1 if (band+index)%2 else 1
            upper=[]; lower=[]
            for t in np.linspace(0,1,42):
                centre=a+delta*t
                bulge=math.sin(math.pi*t)
                upper.append(centre+normal*leaf_height*bulge*sign)
                lower.append(centre-normal*leaf_height*bulge*(.62+.18*influence)*sign)
            _path(out,upper,bar); _path(out,lower,max(8,round(bar*.82)))
        # bands already reach the side frame; sparse vertical leaf-stem ties
        # connect neighboring rows and make the whole field read as one weave.
        if band<3:
            for idx in (1,count-1):
                if 0<=idx<len(nodes):
                    next_v=(.36,.58,.79)[band]
                    u=idx/count
                    wu,wv,_,_=_field_warp(u,next_v,attractor,field_strength,field_mode)
                    cv2.line(out,tuple(np.rint(nodes[idx]).astype(int)),
                             (round(x0+width*wu),round(y0+height*wv)),255,
                             max(7,round(bar*.62)),cv2.LINE_AA)
    return cv2.bitwise_and(out, inner)


def soft_strip_lattice(inner, spec, bar=14, pitch=150, attractor=(.67, .46),
                       field_strength=.78, field_mode="quiet", transform_mode="alternate"):
    """Shared-node bowed strips; no closed circle is used.

    The basic opening is a leaf or a softly distorted quadrilateral.  Every
    strip is a cubic segment between shared construction nodes.  Alternating
    bow directions make the eye follow continuous S-curves across the window,
    instead of reading a field of stamped rings.
    """
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    columns = max(5, round(width / max(130, pitch * 1.24)))
    rows = max(4, round(height / max(110, pitch * .96)))

    # A slightly staggered construction net.  Border nodes deliberately land
    # on the frame, so the designed network is connected before repair.
    nodes = []
    for row in range(rows + 1):
        row_nodes = []
        v = row / rows
        for column in range(columns + 1):
            u = column / columns
            # Small coherent phase shifts keep the lattice hand-bent without
            # destroying its shared-node topology.
            if 0 < column < columns:
                u += .020 * math.sin(row * 1.37 + column * .54)
            if 0 < row < rows:
                v_local = v + .025 * math.sin(column * 1.11 + row * .61)
            else:
                v_local = v
            wu, wv, influence, radial = _field_warp(
                u, v_local, attractor, field_strength, field_mode)
            # Keep frame anchors fixed while allowing interior nodes to drift.
            if column in (0, columns):
                wu = u
            if row in (0, rows):
                wv = v
            row_nodes.append(np.asarray([x0 + width * wu, y0 + height * wv]))
        nodes.append(row_nodes)

    # Bowed horizontal members form long S-curves.  The sign changes at every
    # shared node, so neighboring segments meet as one flowing strip.
    for row in range(rows + 1):
        for column in range(columns):
            a, b = nodes[row][column], nodes[row][column + 1]
            delta = b - a
            length = max(1.0, np.linalg.norm(delta))
            normal = np.asarray([-delta[1], delta[0]]) / length
            mid = (a + b) * .5
            u, v = (mid[0] - x0) / width, (mid[1] - y0) / height
            _, _, influence, radial = _field_warp(
                u, v, attractor, field_strength, field_mode)
            sign = -1 if (row + column) % 2 else 1
            if transform_mode == "pinwheel":
                sign *= -1 if math.sin(radial * 2) < 0 else 1
            bow = min(length * (.24 + .08 * influence), height / rows * .32)
            c1 = a + delta * .32 + normal * bow * sign
            c2 = a + delta * .68 + normal * bow * sign
            _path(out, _bezier(a, c1, c2, b, 42), bar)

    # The vertical members are more restrained.  Their opposite bow creates
    # curved four-sided and occasional leaf-like voids, not circle lenses.
    for column in range(columns + 1):
        for row in range(rows):
            a, b = nodes[row][column], nodes[row + 1][column]
            delta = b - a
            length = max(1.0, np.linalg.norm(delta))
            normal = np.asarray([-delta[1], delta[0]]) / length
            mid = (a + b) * .5
            u, v = (mid[0] - x0) / width, (mid[1] - y0) / height
            _, _, influence, radial = _field_warp(
                u, v, attractor, field_strength, field_mode)
            sign = 1 if (row + column) % 2 else -1
            bow = min(length * (.16 + .06 * influence), width / columns * .22)
            c1 = a + delta * .34 + normal * bow * sign
            c2 = a + delta * .66 + normal * bow * sign
            _path(out, _bezier(a, c1, c2, b, 36), max(8, round(bar * .86)))
    return cv2.bitwise_and(out, inner)


def stepped_medallion(inner, spec, bar=14, pitch=150, attractor=(.50, .50),
                      field_strength=.5, field_mode="quiet", transform_mode="quarter_turn"):
    """Rebuild the reference window as an 8 by 11 orthogonal edge graph.

    The occupied horizontal and vertical edges were read from the supplied
    timber-window photograph.  Drawing the graph, rather than mirroring a few
    ornamental outlines, preserves its continuous stepped topology, repeated
    square returns, full centre rail, and vertical centre spine.
    """
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    xs = np.linspace(x0, x0 + width, 9)
    ys = np.linspace(y0, y0 + height, 12)
    horizontal = {
        1: "11111111",
        2: "00100100",
        3: "10111101",
        4: "01011010",
        5: "11011011",
        6: "11111111",
        7: "11011011",
        8: "01011010",
        9: "10111101",
        10: "00100100",
    }
    vertical = {
        1: "11101111011",
        2: "11010110101",
        3: "00110110110",
        4: "11111111111",
        5: "00110110110",
        6: "11010110101",
        7: "11101111011",
    }
    stroke = max(8, round(bar * .82))
    for row, edges in horizontal.items():
        for column, occupied in enumerate(edges):
            if occupied == "1":
                cv2.line(out, (round(xs[column]), round(ys[row])),
                         (round(xs[column + 1]), round(ys[row])),
                         255, stroke, cv2.LINE_8)
    for column, edges in vertical.items():
        for row, occupied in enumerate(edges):
            if occupied == "1":
                cv2.line(out, (round(xs[column]), round(ys[row])),
                         (round(xs[column]), round(ys[row + 1])),
                         255, stroke, cv2.LINE_8)
    return cv2.bitwise_and(out, inner)


def curved_cell_grid(inner, spec, bar=14, pitch=150, attractor=(.66, .46),
                     field_strength=.50, field_mode="quiet", transform_mode="field"):
    """Tangent round cells leave small concave diamonds between four cells.

    This is a tessellation, not overlapping rings: adjacent cells meet at one
    cardinal node and never cross to create a lens.  Each nominal circle is
    drawn as four cubic arc segments so the attractor can turn it into a mild
    oval while preserving the same node graph.
    """
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    step = max(150.0, pitch * 1.34)
    columns = max(5, int(round(width / step)) + 1)
    rows = max(5, int(round(height / step)) + 1)
    step_x = width / max(1, columns - 1)
    step_y = height / max(1, rows - 1)
    # Tangency in the tighter direction guarantees no lens-shaped overlap.
    base_rx = step_x * .50
    base_ry = step_y * .50
    kappa = .5522847498

    for row in range(rows):
        cy0 = y0 + row * step_y
        for column in range(columns):
            cx0 = x0 + column * step_x
            u, v = (cx0 - x0) / width, (cy0 - y0) / height
            _, _, influence, radial = _field_warp(
                u, v, attractor, field_strength, field_mode)
            # Modulate curvature but keep cardinal points on the shared grid.
            oval = .10 * influence
            if transform_mode == "alternate":
                oval *= -1 if (row + column) % 2 else 1
            rx = base_rx
            ry = base_ry
            kx = kappa * (1 + oval)
            ky = kappa * (1 - oval)
            centre = np.asarray([cx0, cy0], dtype=float)
            top = centre + np.asarray([0.0, -ry])
            right = centre + np.asarray([rx, 0.0])
            bottom = centre + np.asarray([0.0, ry])
            left = centre + np.asarray([-rx, 0.0])
            segments = (
                (top, top + np.asarray([kx * rx, 0]),
                 right + np.asarray([0, -ky * ry]), right),
                (right, right + np.asarray([0, ky * ry]),
                 bottom + np.asarray([kx * rx, 0]), bottom),
                (bottom, bottom + np.asarray([-kx * rx, 0]),
                 left + np.asarray([0, ky * ry]), left),
                (left, left + np.asarray([0, -ky * ry]),
                 top + np.asarray([-kx * rx, 0]), top),
            )
            for segment in segments:
                _path(out, _bezier(*segment, count=26), bar)

    # Explicit tangent nodes make raster and DXF topology unambiguous.  They
    # thicken only the shared points; the concave diamond apertures stay open.
    node_radius = max(2, round(bar * .34))
    for row in range(rows):
        y = round(y0 + row * step_y)
        for column in range(columns - 1):
            x = round(x0 + (column + .5) * step_x)
            cv2.circle(out, (x, y), node_radius, 255, -1, cv2.LINE_AA)
    for row in range(rows - 1):
        y = round(y0 + (row + .5) * step_y)
        for column in range(columns):
            x = round(x0 + column * step_x)
            cv2.circle(out, (x, y), node_radius, 255, -1, cv2.LINE_AA)
    return cv2.bitwise_and(out, inner)


def morph_cell_field(inner, spec, bar=14, pitch=150, attractor=(.58, .48),
                     field_strength=.82, field_mode="bloom", transform_mode="field",
                     controller_image=None, controller_curve=None):
    """Topology-preserving circle-to-four-lobed cell morph.

    Every cell keeps the same four cardinal contact nodes, so a point field or
    uploaded image can reshape the apertures without tearing the material
    network apart.  This is the first generator built as element + arrangement
    + controller + response rather than as one frozen preset.
    """
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    # Unlike the larger reference-cell generators, an image controller needs
    # enough samples to express a silhouette and internal contrast.  Pitch is
    # therefore allowed down to a denser, still fabricable field.
    step = max(72.0, pitch * 1.12)
    columns = max(5, int(round(width / step)) + 1)
    rows = max(4, int(round(height / step)) + 1)
    step_x = width / max(1, columns - 1)
    step_y = height / max(1, rows - 1)
    radius_x = step_x * .50
    radius_y = step_y * .50
    if controller_image and controller_curve:
        raise ValueError("Choose either an image or curve controller, not both")
    image_field = _image_controller(controller_image, width, height) if controller_image else None
    curve_paths = _curve_controller(controller_curve) if controller_curve else None

    for row in range(rows):
        cy = y0 + row * step_y
        for column in range(columns):
            cx = x0 + column * step_x
            u, v = (cx - x0) / width, (cy - y0) / height
            if image_field is not None:
                sample_x = min(width - 1, max(0, round(cx - x0)))
                sample_y = min(height - 1, max(0, round(cy - y0)))
                influence = float(image_field[sample_y, sample_x])
            elif curve_paths is not None:
                distance = _distance_to_paths((u, v), curve_paths)
                influence = math.exp(-((distance / .18) ** 2))
            else:
                _, _, influence, _ = _field_warp(
                    u, v, attractor, field_strength, field_mode)
            influence = min(1.0, max(0.0, influence * field_strength))
            # r(theta) is normalized by (1+a), so theta=0/90/180/270
            # always lands on the shared contact nodes.  Diagonals move inward
            # continuously, turning a circle into a restrained four-lobed cell.
            morph = .015 + .30 * influence
            points = []
            for theta in np.linspace(0, 2 * math.pi, 128, endpoint=False):
                radial = (1 + morph * math.cos(4 * theta)) / (1 + morph)
                points.append((cx + radius_x * math.cos(theta) * radial,
                               cy + radius_y * math.sin(theta) * radial))
            local_bar = max(8, round(bar * (.82 + .50 * influence)))
            _path(out, points, local_bar, True)

    # Reinforce only the invariant contact nodes.  These are structural facts,
    # not auto-repair bridges, and remain stable for every controller image.
    node_radius = max(2, round(bar * .36))
    for row in range(rows):
        y = round(y0 + row * step_y)
        for column in range(columns - 1):
            x = round(x0 + (column + .5) * step_x)
            cv2.circle(out, (x, y), node_radius, 255, -1, cv2.LINE_AA)
    for row in range(rows - 1):
        y = round(y0 + (row + .5) * step_y)
        for column in range(columns):
            x = round(x0 + column * step_x)
            cv2.circle(out, (x, y), node_radius, 255, -1, cv2.LINE_AA)
    return cv2.bitwise_and(out, inner)


def hex_flower_field(inner, spec, bar=14, pitch=125, attractor=(.62, .46),
                     field_strength=.78, field_mode="bloom", transform_mode="field",
                     controller_image=None, controller_curve=None):
    """Six-neighbour shared-node cells morph from round to six-lobed flowers."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    if controller_image and controller_curve:
        raise ValueError("Choose either an image or curve controller, not both")
    image_field = _image_controller(controller_image, width, height) if controller_image else None
    curve_paths = _curve_controller(controller_curve) if controller_curve else None
    step = max(76.0, pitch * 1.04)
    dy = step * math.sqrt(3) / 2
    row = -1
    for cy in np.arange(y0 - dy, y0 + height + dy * 1.5, dy):
        offset = step * .5 if row % 2 else 0.0
        for cx in np.arange(x0 - step + offset, x0 + width + step, step):
            u, v = (cx - x0) / width, (cy - y0) / height
            influence, _ = _controller_sample(
                u, v, field_strength, attractor, field_mode, image_field, curve_paths)
            morph = .01 + .22 * influence
            points = []
            for theta in np.linspace(0, 2 * math.pi, 144, endpoint=False):
                radial = (1 + morph * math.cos(6 * theta)) / (1 + morph)
                radius = step * .50 * radial
                points.append((cx + radius * math.cos(theta),
                               cy + radius * math.sin(theta)))
            local_bar = max(8, round(bar * (.84 + .38 * influence)))
            _path(out, points, local_bar, True)
        row += 1
    return cv2.bitwise_and(out, inner)


def fish_scale_field(inner, spec, bar=14, pitch=130, attractor=(.62, .40),
                     field_strength=.72, field_mode="drift", transform_mode="alternate",
                     controller_image=None, controller_curve=None):
    """Staggered scallop arcs share endpoints and apices across adjacent rows."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    if controller_image and controller_curve:
        raise ValueError("Choose either an image or curve controller, not both")
    image_field = _image_controller(controller_image, width, height) if controller_image else None
    curve_paths = _curve_controller(controller_curve) if controller_curve else None
    rx = max(44.0, pitch * .52)
    dx = rx * 2
    dy = max(42.0, pitch * .62)
    row = -1
    for baseline in np.arange(y0 - dy, y0 + height + dy, dy):
        offset = rx if row % 2 else 0.0
        for cx in np.arange(x0 - dx + offset, x0 + width + dx, dx):
            u, v = (cx - x0) / width, (baseline - y0) / height
            influence, _ = _controller_sample(
                u, v, field_strength, attractor, field_mode, image_field, curve_paths)
            power = .82 + 1.55 * influence
            points = []
            for theta in np.linspace(math.pi, 0, 80):
                points.append((cx + rx * math.cos(theta),
                               baseline + dy * (math.sin(theta) ** power)))
            _path(out, points, max(8, round(bar * (.86 + .30 * influence))))
        row += 1
    return cv2.bitwise_and(out, inner)


def phyllotaxis_star_screen(inner, spec, bar=14, pitch=115, attractor=(.50, .50),
                            field_strength=.72, field_mode="bloom", transform_mode="field",
                            controller_image=None, controller_curve=None):
    """A perforated modern screen of four-point stars on a golden-angle spiral."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    material = inner.copy()
    if controller_image and controller_curve:
        raise ValueError("Choose either an image or curve controller, not both")
    image_field = _image_controller(controller_image, width, height) if controller_image else None
    curve_paths = _curve_controller(controller_curve) if controller_curve else None
    cx, cy = x0 + width * .5, y0 + height * .5
    max_radius = min(width, height) * .465
    count = max(90, min(360, round(math.pi * max_radius * max_radius / max(52, pitch * .64) ** 2)))
    golden = math.pi * (3 - math.sqrt(5))
    spacing = max_radius / math.sqrt(count)
    for index in range(1, count + 1):
        radius = max_radius * math.sqrt(index / count)
        angle = index * golden
        px, py = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
        ix, iy = round(px), round(py)
        if not (0 <= iy < inner.shape[0] and 0 <= ix < inner.shape[1] and inner[iy, ix]):
            continue
        u, v = (px - x0) / width, (py - y0) / height
        influence, _ = _controller_sample(
            u, v, field_strength, attractor, field_mode, image_field, curve_paths)
        radial_growth = .58 + .42 * (radius / max_radius)
        # Keep neighboring apertures separated by a real material neck.  The
        # golden-angle nearest-neighbour distance is roughly 1.7 * spacing.
        outer = spacing * (.62 + .15 * influence) * radial_growth
        points = []
        shape_morph = .56 * influence
        for k in range(32):
            a = angle * .08 + k * 2 * math.pi / 32
            r = outer * (1 + shape_morph * math.cos(4 * a)) / (1 + shape_morph)
            points.append((px + r * math.cos(a), py + r * math.sin(a)))
        cv2.fillPoly(material, [np.rint(points).astype(np.int32)], 0, cv2.LINE_AA)
    return cv2.bitwise_and(material, inner)


def contour_stream_field(inner, spec, bar=14, pitch=105, attractor=(.60, .48),
                         field_strength=.76, field_mode="quiet", transform_mode="field",
                         controller_image=None, controller_curve=None):
    """Nested topographic loops with real two-dimensional direction changes."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    if controller_image and controller_curve:
        raise ValueError("Choose either an image or curve controller, not both")
    ax, ay = attractor if attractor is not None else (.5, .5)
    cx = x0 + width * (.5 + (ax - .5) * .32)
    cy = y0 + height * (.5 + (ay - .5) * .32)
    count = max(8, min(11, round(min(width, height) / max(58, pitch * .62))))
    loops = []
    for index, scale in enumerate(np.linspace(.12, .46, count)):
        rx = width * scale
        ry = height * (.11 + (scale - .12) * .88)
        points = []
        for theta in np.linspace(0, 2 * math.pi, 260, endpoint=False):
            modulation = (1 + .055 * math.sin(3 * theta + index * .46)
                          + .025 * math.cos(5 * theta - index * .31))
            shear = .055 * math.sin(theta) * math.cos(theta + index * .12)
            points.append((cx + rx * modulation * math.cos(theta) + width * shear,
                           cy + ry * modulation * math.sin(theta)))
        loop = np.asarray(points, dtype=float)
        loops.append(loop)
        _path(out, loop, max(8, round(bar * (.72 + index / count * .18))), True)

    # Join adjacent contours with short, deliberately staggered bridges.  The
    # connection point walks around the hill instead of becoming one arbitrary
    # diagonal seam.  The outermost loop receives one short frame tie.
    bridge_width = max(7, round(bar * .64))
    for index in range(len(loops) - 1):
        angle = (index * 2.18 + .45) % (2 * math.pi)
        sample = round(angle / (2 * math.pi) * len(loops[index])) % len(loops[index])
        cv2.line(out, tuple(np.rint(loops[index][sample]).astype(int)),
                 tuple(np.rint(loops[index + 1][sample]).astype(int)),
                 255, bridge_width, cv2.LINE_AA)
    outer = loops[-1]
    outer_point = outer[int(np.argmax(outer[:, 0]))]
    cv2.line(out, tuple(np.rint(outer_point).astype(int)),
             (x0 + width, round(outer_point[1])), 255,
             bridge_width, cv2.LINE_AA)
    return cv2.bitwise_and(out, inner)


def herringbone_screen(inner, spec, bar=14, pitch=120, attractor=(.50, .50),
                       field_strength=.40, field_mode="quiet", transform_mode="alternate",
                       controller_image=None, controller_curve=None):
    """Alternating rounded slots form a modern herringbone perforated screen."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    material = inner.copy()
    step = max(64.0, pitch * .66)
    row_step = step * .70
    length = step * .72
    slot = max(10, round(step * .22))
    row = -1
    for cy in np.arange(y0 - row_step, y0 + height + row_step, row_step):
        offset = step * .5 if row % 2 else 0.0
        for column, cx in enumerate(np.arange(x0 - step + offset,
                                               x0 + width + step, step)):
            angle = math.pi / 4 if (row + column) % 2 else -math.pi / 4
            direction = np.asarray([math.cos(angle), math.sin(angle)])
            a = np.asarray([cx, cy]) - direction * length * .5
            b = np.asarray([cx, cy]) + direction * length * .5
            cv2.line(material, tuple(np.rint(a).astype(int)),
                     tuple(np.rint(b).astype(int)), 0, slot, cv2.LINE_AA)
            radius = max(2, slot // 2)
            cv2.circle(material, tuple(np.rint(a).astype(int)), radius, 0, -1, cv2.LINE_AA)
            cv2.circle(material, tuple(np.rint(b).astype(int)), radius, 0, -1, cv2.LINE_AA)
        row += 1
    return cv2.bitwise_and(material, inner)


def octagon_square_grid(inner, spec, bar=14, pitch=132, attractor=(.62, .46),
                        field_strength=.45, field_mode="quiet", transform_mode="alternate",
                        controller_image=None, controller_curve=None):
    """Tangent chamfered cells create the classic octagon-and-square reading."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    if controller_image and controller_curve:
        raise ValueError("Choose either an image or curve controller, not both")
    image_field = _image_controller(controller_image, width, height) if controller_image else None
    curve_paths = _curve_controller(controller_curve) if controller_curve else None
    step = max(86.0, pitch * .92)
    half = step * .5
    for cy in np.arange(y0, y0 + height + step, step):
        for cx in np.arange(x0, x0 + width + step, step):
            u, v = (cx - x0) / width, (cy - y0) / height
            influence, _ = _controller_sample(
                u, v, field_strength, attractor, field_mode, image_field, curve_paths)
            chamfer = half * (.34 + .10 * influence)
            points = [(cx - chamfer, cy - half), (cx + chamfer, cy - half),
                      (cx + half, cy - chamfer), (cx + half, cy + chamfer),
                      (cx + chamfer, cy + half), (cx - chamfer, cy + half),
                      (cx - half, cy + chamfer), (cx - half, cy - chamfer)]
            _path(out, points, max(8, round(bar * (.88 + .24 * influence))), True)
    return cv2.bitwise_and(out, inner)


def plum_aperture_screen(inner, spec, bar=14, pitch=112, attractor=(.60, .46),
                         field_strength=.68, field_mode="bloom", transform_mode="alternate",
                         controller_image=None, controller_curve=None):
    """Five-petal plum apertures repeat on a staggered field in a solid screen."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    material = inner.copy()
    if controller_image and controller_curve:
        raise ValueError("Choose either an image or curve controller, not both")
    image_field = _image_controller(controller_image, width, height) if controller_image else None
    curve_paths = _curve_controller(controller_curve) if controller_curve else None
    step = max(72.0, pitch * .92)
    dy = step * .86
    row = -1
    for cy in np.arange(y0 - dy, y0 + height + dy, dy):
        offset = step * .5 if row % 2 else 0.0
        for column, cx in enumerate(np.arange(x0 - step + offset,
                                               x0 + width + step, step)):
            u, v = (cx - x0) / width, (cy - y0) / height
            influence, _ = _controller_sample(
                u, v, field_strength, attractor, field_mode, image_field, curve_paths)
            radius = step * (.245 + .055 * influence)
            lobe = .13 + .20 * influence
            rotation = ((row + column) % 2) * math.pi / 5 - math.pi / 2
            points = []
            for theta in np.linspace(0, 2 * math.pi, 100, endpoint=False):
                radial = radius * (1 + lobe * math.cos(5 * (theta - rotation))) / (1 + lobe)
                points.append((cx + radial * math.cos(theta), cy + radial * math.sin(theta)))
            cv2.fillPoly(material, [np.rint(points).astype(np.int32)], 0, cv2.LINE_AA)
        row += 1
    return cv2.bitwise_and(material, inner)


def nested_wave_field(inner, spec, bar=14, pitch=126, attractor=(.58, .42),
                      field_strength=.62, field_mode="drift", transform_mode="alternate",
                      controller_image=None, controller_curve=None):
    """Three nested scallop arcs share endpoints, forming a seigaiha-like field."""
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = np.zeros_like(inner)
    if controller_image and controller_curve:
        raise ValueError("Choose either an image or curve controller, not both")
    image_field = _image_controller(controller_image, width, height) if controller_image else None
    curve_paths = _curve_controller(controller_curve) if controller_curve else None
    rx = max(46.0, pitch * .54)
    dx = rx * 2
    dy = max(48.0, pitch * .68)
    row = -1
    for baseline in np.arange(y0 - dy, y0 + height + dy, dy):
        offset = rx if row % 2 else 0.0
        for cx in np.arange(x0 - dx + offset, x0 + width + dx, dx):
            u, v = (cx - x0) / width, (baseline - y0) / height
            influence, _ = _controller_sample(
                u, v, field_strength, attractor, field_mode, image_field, curve_paths)
            power = .88 + .85 * influence
            for level, factor in enumerate((1.0, .68, .38)):
                points = []
                for theta in np.linspace(math.pi, 0, 72):
                    points.append((cx + rx * math.cos(theta),
                                   baseline + dy * factor * (math.sin(theta) ** power)))
                _path(out, points, max(7, round(bar * (1.0 - level * .16))))
        row += 1
    return cv2.bitwise_and(out, inner)


def lotus_rosette(inner, spec, bar=14, pitch=150, attractor=(.50, .50),
                  field_strength=.45, field_mode="bloom", transform_mode="alternate"):
    """Concentric lotus-petal rings with a beaded outer band and frame ties."""
    x0,y0=spec["origin_px"]; width,height=spec["pixel_size"]
    out=np.zeros_like(inner)
    cx=x0+width*(.50+((attractor[0]-.50)*.18 if attractor else 0))
    cy=y0+height*(.50+((attractor[1]-.50)*.14 if attractor else 0))
    base=min(width,height); r0=base*.09; r1=base*.24; r2=base*.40
    cv2.circle(out,(round(cx),round(cy)),round(r0),255,bar,cv2.LINE_AA)
    cv2.circle(out,(round(cx),round(cy)),round(r1),255,max(8,round(bar*.82)),cv2.LINE_AA)
    cv2.circle(out,(round(cx),round(cy)),round(r2),255,bar,cv2.LINE_AA)
    def petal(angle,inner_r,outer_r,half_width):
        radial=np.asarray([math.cos(angle),math.sin(angle)])
        normal=np.asarray([-radial[1],radial[0]])
        a=np.asarray([cx,cy])+radial*inner_r
        b=np.asarray([cx,cy])+radial*outer_r
        upper=_bezier(a,a+normal*half_width+radial*(outer_r-inner_r)*.35,
                      b+normal*half_width*.18-radial*(outer_r-inner_r)*.18,b,34)
        lower=_bezier(b,b-normal*half_width*.18-radial*(outer_r-inner_r)*.18,
                      a-normal*half_width+radial*(outer_r-inner_r)*.35,a,34)
        return np.vstack((upper,lower))
    for i,a in enumerate(np.linspace(0,2*math.pi,8,endpoint=False)):
        _path(out,petal(a,r0,r1,base*.105),bar,True)
    for i,a in enumerate(np.linspace(0,2*math.pi,12,endpoint=False)+math.pi/12):
        _path(out,petal(a,r1,r2,base*.075),max(8,round(bar*.82)),True)
        node=(round(cx+r2*math.cos(a)),round(cy+r2*math.sin(a)))
        cv2.circle(out,node,max(7,round(base*.026)),255,max(7,round(bar*.65)),cv2.LINE_AA)
    # Cardinal ties anchor the rosette to the selected window frame.
    for a in (0,math.pi/2,math.pi,3*math.pi/2):
        start=(round(cx+r2*math.cos(a)),round(cy+r2*math.sin(a)))
        end=((x0+width) if a==0 else x0 if a==math.pi else round(cx),
             (y0+height) if a==math.pi/2 else y0 if a==3*math.pi/2 else round(cy))
        cv2.line(out,start,end,255,max(8,round(bar*.78)),cv2.LINE_AA)
    return cv2.bitwise_and(out,inner)


def triangle_aperture_screen(inner, spec, bar=7, pitch=34, attractor=(.50, .50),
                             field_strength=.8, field_mode="quiet",
                             transform_mode="alternate"):
    """A connected sheet of triangular apertures for photographic halftones.

    The apertures start uniformly sized; ``semantic_reveal`` subsequently
    expands or contracts them from the photographic tone field. This mirrors
    architectural perforated-image screens while keeping one continuous sheet.
    """
    x0, y0 = spec["origin_px"]
    width, height = spec["pixel_size"]
    out = inner.copy()
    step = max(22.0, float(pitch))
    row_step = step * .78
    radius = step * .49
    row = 0
    for cy in np.arange(y0 - row_step, y0 + height + row_step, row_step):
        offset = step * .5 if row % 2 else 0.0
        col = 0
        for cx in np.arange(x0 - step + offset, x0 + width + step, step):
            angle = -math.pi / 2 if (row + col) % 2 == 0 else math.pi / 2
            points = np.asarray([
                (cx + radius * math.cos(angle + i * 2 * math.pi / 3),
                 cy + radius * math.sin(angle + i * 2 * math.pi / 3))
                for i in range(3)], dtype=np.int32)
            cv2.fillPoly(out, [points], 0, cv2.LINE_AA)
            col += 1
        row += 1
    return cv2.bitwise_and(out, inner)


GENERATORS = {
    "unit_field": unit_field,
    "curve_weave": curve_weave,
    "radial_rosette": radial_rosette,
    "growth_network": growth_network,
    "meander_turns": meander_turns,
    "hex_plum": hex_plum,
    "coin_chain": coin_chain,
    "willow_leaves": willow_leaves,
    "soft_strip_lattice": soft_strip_lattice,
    "stepped_medallion": stepped_medallion,
    "curved_cell_grid": curved_cell_grid,
    "morph_cell_field": morph_cell_field,
    "hex_flower_field": hex_flower_field,
    "fish_scale_field": fish_scale_field,
    "phyllotaxis_star_screen": phyllotaxis_star_screen,
    "contour_stream_field": contour_stream_field,
    "herringbone_screen": herringbone_screen,
    "octagon_square_grid": octagon_square_grid,
    "plum_aperture_screen": plum_aperture_screen,
    "nested_wave_field": nested_wave_field,
    "lotus_rosette": lotus_rosette,
    "triangle_aperture_screen": triangle_aperture_screen,
}

CONTROLLER_GENERATORS = {
    "morph_cell_field",
    "hex_flower_field",
    "fish_scale_field",
    "phyllotaxis_star_screen",
    "contour_stream_field",
    "octagon_square_grid",
    "plum_aperture_screen",
    "nested_wave_field",
}

