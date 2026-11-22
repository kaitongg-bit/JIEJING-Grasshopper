"""Parametric motif library and reusable placement rules for flat window screens.

These are contemporary geometric constructions based on named traditional
window-pattern families, not surveys or exact reproductions of historic joinery.
Coordinates for tile paths are normalized around each tile's centre.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


SIZE = 1200
CENTER = SIZE // 2
PRESETS = ("huiwen", "binglie", "haitang", "guibei", "fangsheng", "yulin")


def _circle_path(radius: float, lobes: int = 0, modulation: float = 0.0,
                 start: float = 0.0, end: float = 2 * math.pi, n: int = 120) -> list[tuple[float, float]]:
    return [((radius * (1 + modulation * math.cos(lobes * t))) * math.cos(t),
             (radius * (1 + modulation * math.cos(lobes * t))) * math.sin(t))
            for t in np.linspace(start, end, n)]


def tile_paths(name: str) -> list[list[tuple[float, float]]]:
    if name == "huiwen":  # 回纹: orthogonal recurring turns
        return [[(-.46, -.46), (.46, -.46), (.46, .46), (-.46, .46),
                 (-.46, -.24), (.22, -.24), (.22, .22), (-.22, .22),
                 (-.22, -.02), (.03, -.02)]]
    if name == "haitang":  # 海棠: four softly indented lobes
        return [_circle_path(.36, 4, .23), _circle_path(.22, 4, .16)]
    if name == "guibei":  # 龟背: hexagonal cell and inset hexagon
        return [[(.43 * math.cos(math.pi / 6 + k * math.pi / 3),
                  .43 * math.sin(math.pi / 6 + k * math.pi / 3)) for k in range(7)],
                [(.23 * math.cos(math.pi / 6 + k * math.pi / 3),
                  .23 * math.sin(math.pi / 6 + k * math.pi / 3)) for k in range(7)]]
    if name == "fangsheng":  # 方胜: two overlapping diamonds
        return [[(-.40, 0), (-.12, -.36), (.16, 0), (-.12, .36), (-.40, 0)],
                [(-.16, 0), (.12, -.36), (.40, 0), (.12, .36), (-.16, 0)]]
    if name == "yulin":  # 鱼鳞: staggered repeated arcs
        return [_circle_path(.38, start=math.pi, end=2 * math.pi, n=50),
                [(x, y + .35) for x, y in _circle_path(.32, start=math.pi, end=2 * math.pi, n=50)]]
    raise ValueError(f"Unknown tile motif: {name}")


def load_custom_paths(path: Path) -> list[list[tuple[float, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    paths = data.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("Custom motif JSON needs a nonempty 'paths' list")
    out = []
    for polyline in paths:
        if not isinstance(polyline, list) or len(polyline) < 2:
            raise ValueError("Each custom path needs at least two points")
        points = []
        for point in polyline:
            if not isinstance(point, list) or len(point) != 2 or not all(isinstance(v, (int, float)) for v in point):
                raise ValueError("Each custom point must be [x,y]")
            if any(abs(v) > .5 for v in point):
                raise ValueError("Custom coordinates must be in [-0.5,0.5]")
            points.append((float(point[0]), float(point[1])))
        out.append(points)
    return out


def _stroke(mask: np.ndarray, points: list[tuple[float, float]], center: tuple[float, float],
            size: float, angle: float, bar: int) -> None:
    co, si = math.cos(angle), math.sin(angle)
    xy = np.array([[round(center[0] + size * (x * co - y * si)),
                    round(center[1] + size * (x * si + y * co))] for x, y in points], np.int32)
    cv2.polylines(mask, [xy], False, 255, bar, cv2.LINE_8)


def _stamp(mask: np.ndarray, paths: list[list[tuple[float, float]]],
           center: tuple[float, float], size: float, angle: float, bar: int) -> None:
    for points in paths:
        _stroke(mask, points, center, size, angle, bar)


def _ice_crack(pitch: int, bar: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    subdiv = cv2.Subdiv2D((0, 0, SIZE, SIZE))
    count = max(35, min(180, round(SIZE * SIZE / (pitch * pitch * 2.4))))
    points = rng.uniform(15, SIZE - 15, (count, 2))
    for x, y in points:
        subdiv.insert((float(x), float(y)))
    facets, _ = subdiv.getVoronoiFacetList([])
    for facet in facets:
        polygon = np.rint(facet).astype(np.int32)
        cv2.polylines(mask, [polygon], True, 255, bar, cv2.LINE_8)
    return mask


def _tortoise_shell(pitch: int, bar: int) -> np.ndarray:
    """Shared-edge hexagon cells with connected inset geometry."""
    mask = np.zeros((SIZE, SIZE), np.uint8)
    radius = pitch * .52
    step_x = radius * 1.5
    step_y = radius * math.sqrt(3)
    column = 0
    for x in np.arange(-radius, SIZE + radius, step_x):
        y_offset = step_y / 2 if column % 2 else 0
        for y in np.arange(-radius + y_offset, SIZE + radius, step_y):
            outer = np.array([[round(x + radius * math.cos(k * math.pi / 3)),
                               round(y + radius * math.sin(k * math.pi / 3))]
                              for k in range(6)], np.int32)
            inner = np.array([[round(x + radius * .3 * math.cos(k * math.pi / 3)),
                               round(y + radius * .3 * math.sin(k * math.pi / 3))]
                              for k in range(6)], np.int32)
            cv2.polylines(mask, [outer, inner], True, 255, bar, cv2.LINE_8)
            for k in (0, 2, 4):
                cv2.line(mask, tuple(inner[k]), tuple(outer[k]), 255, bar, cv2.LINE_8)
        column += 1
    return mask


def _flower_lattice(pitch: int, bar: int) -> np.ndarray:
    """Linked four-lobed blossoms without the default square backing grid."""
    mask = np.zeros((SIZE, SIZE), np.uint8)
    outer = _circle_path(.40, 4, .17)
    inner = _circle_path(.19, 4, .12)
    centers = range(pitch // 2, SIZE + pitch, pitch)
    for x in centers:
        for y in centers:
            _stroke(mask, outer, (x, y), pitch, 0, bar)
            _stroke(mask, inner, (x, y), pitch, 0, max(7, round(bar * .7)))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(mask, (round(x + dx * pitch * .21), round(y + dy * pitch * .21)),
                         (round(x + dx * pitch * .48), round(y + dy * pitch * .48)),
                         255, max(7, round(bar * .7)), cv2.LINE_8)
    return mask


def _read_guide(path: Path) -> list[list[tuple[float, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    paths = data.get("paths") or [data.get("points")]
    if not isinstance(paths, list) or not 1 <= len(paths) <= 3:
        raise ValueError("Guide JSON needs 1–3 paths")
    result = []
    for points in paths:
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("Each guide needs at least two points")
        guide = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in point):
                raise ValueError("Guide points must be normalized [x,y] in [0,1]")
            guide.append((point[0] * SIZE, point[1] * SIZE))
        result.append(guide)
    return result


def _sample_polyline(points: list[tuple[float, float]], spacing: float,
                     distances=None) -> list[tuple[float, float, float]]:
    xy = np.asarray(points, dtype=float)
    segment = np.diff(xy, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total < spacing and distances is None:
        raise ValueError("Guide curve is shorter than one repeat spacing")
    samples = []
    for distance in (distances if distances is not None else np.arange(spacing / 2, total, spacing)):
        index = min(len(lengths) - 1, int(np.searchsorted(cumulative, distance, side="right") - 1))
        fraction = (distance - cumulative[index]) / max(lengths[index], 1e-6)
        position = xy[index] + fraction * segment[index]
        angle = math.atan2(segment[index, 1], segment[index, 0])
        samples.append((position[0], position[1], angle))
    return samples


def _field_value(x: float, y: float, attractor: tuple[float, float] | None,
                 strength: float = 0.0) -> tuple[float, float, float]:
    """Return smooth influence, radial angle and normalized distance.

    The attractor is expressed in normalized window coordinates.  The field is
    deliberately broad: traditional lattice rhythm should bend gradually, not
    produce a digital-looking local bump.
    """
    if attractor is None or strength <= 0:
        return 0.0, 0.0, 1.0
    ax, ay = attractor[0] * SIZE, attractor[1] * SIZE
    dx, dy = x - ax, y - ay
    distance = math.hypot(dx, dy) / SIZE
    influence = min(1.0, strength) * math.exp(-((distance / .34) ** 2))
    return influence, math.atan2(dy, dx), distance


def _transform_stamp(mask: np.ndarray, paths: list[list[tuple[float, float]]],
                     center: tuple[float, float], size: float, angle: float,
                     bar: int, index: int, transform_mode: str,
                     attractor: tuple[float, float] | None,
                     field_strength: float, field_mode: str) -> None:
    """Place one motif with coherent symmetry and field response.

    Translation is only the starting operation.  Alternating reflection,
    quarter-turn rhythm, nested scale and a smooth attractor-driven tangent
    field create the repeated transformations found in richer lattice systems.
    """
    influence, radial, _ = _field_value(center[0], center[1], attractor, field_strength)
    local_size = size
    local_angle = angle
    if transform_mode == "alternate":
        local_angle += (index % 2) * math.pi
    elif transform_mode == "quarter_turn":
        local_angle += (index % 4) * math.pi / 2
    elif transform_mode == "pinwheel":
        local_angle += radial + math.pi / 2
    elif transform_mode == "nested":
        _stamp(mask, paths, center, size * .58, angle + math.pi / 4, max(6, round(bar * .72)))
    elif transform_mode == "field":
        local_angle += influence * (radial + math.pi / 2 - local_angle)
    if field_mode == "bloom":
        local_size *= 1 + .42 * influence
    elif field_mode == "quiet":
        local_size *= 1 - .28 * influence
    elif field_mode == "vortex":
        local_angle += influence * 1.15
        local_size *= 1 + .16 * influence
    elif field_mode == "drift":
        local_angle += influence * .42
        local_size *= 1 + .18 * influence
    _stamp(mask, paths, center, local_size, local_angle, bar)
    # User drawings and several traditional units contain an outer contour plus
    # one or more inner contours.  Join each inner contour to the outer member
    # at the shortest local gap so it becomes intentional lattice, not an
    # isolated island repaired later by an arbitrary bridge.
    if len(paths) > 1:
        outer = np.asarray(paths[0], dtype=float)
        co, si = math.cos(local_angle), math.sin(local_angle)
        def world(point):
            x, y = point
            return (round(center[0] + local_size * (x * co - y * si)),
                    round(center[1] + local_size * (x * si + y * co)))
        for inner_path in paths[1:]:
            inner = np.asarray(inner_path, dtype=float)
            delta = outer[:, None, :] - inner[None, :, :]
            oi, ii = np.unravel_index(np.argmin(np.sum(delta * delta, axis=2)),
                                      (len(outer), len(inner)))
            cv2.line(mask, world(outer[oi]), world(inner[ii]), 255,
                     max(5, round(bar * .62)), cv2.LINE_AA)


def _bezier_branch(start: tuple[float, float], end: tuple[float, float],
                   tangent: float, side: float, length: float,
                   count: int = 36) -> np.ndarray:
    """A tangential side branch that grows out of a guide without a kink."""
    p0 = np.asarray(start, dtype=float)
    p3 = np.asarray(end, dtype=float)
    along = np.asarray([math.cos(tangent), math.sin(tangent)])
    across = np.asarray([-along[1], along[0]]) * side
    p1 = p0 + along * length * .24 + across * length * .10
    p2 = p3 - across * length * .30 - along * length * .10
    points = []
    for t in np.linspace(0, 1, count):
        s = 1 - t
        points.append(s ** 3 * p0 + 3 * s * s * t * p1 + 3 * s * t * t * p2 + t ** 3 * p3)
    return np.rint(points).astype(np.int32)


def _draw_leaf(mask: np.ndarray, stem: tuple[float, float], angle: float,
               length: float, side: float, width: int) -> None:
    """Draw one closed willow-like leaf attached to the branch."""
    p = np.asarray(stem, dtype=float)
    tangent = np.asarray([math.cos(angle), math.sin(angle)])
    normal = np.asarray([-tangent[1], tangent[0]]) * side
    points = []
    for t in np.linspace(0, 1, 28):
        points.append(p + tangent * length * t + normal * length * .18 * math.sin(math.pi * t))
    for t in np.linspace(1, 0, 28):
        points.append(p + tangent * length * t - normal * length * .07 * math.sin(math.pi * t))
    cv2.polylines(mask, [np.rint(points).astype(np.int32)], True, 255, width, cv2.LINE_AA)


def render(name: str, layout: str, pitch: int, bar: int, seed: int = 7,
           custom_motif: Path | None = None, guide: Path | None = None,
           clip_mask: np.ndarray | None = None,
           transform_mode: str = "alternate",
           attractor: tuple[float, float] | None = None,
           field_strength: float = 0.0,
           field_mode: str = "quiet") -> np.ndarray:
    if name == "binglie":
        if layout != "grid":
            raise ValueError("Ice-crack pattern uses grid/field layout only")
        mask = _ice_crack(pitch, bar, seed)
    elif name == "haitang" and layout == "grid":
        mask = _flower_lattice(pitch, bar)
    elif name == "guibei" and layout == "grid":
        mask = _tortoise_shell(pitch, bar)
    else:
        paths = load_custom_paths(custom_motif) if name == "custom" and custom_motif else tile_paths(name)
        mask = np.zeros((SIZE, SIZE), np.uint8)
        if layout == "grid":
            # A shared construction grid joins neighboring cells and lets a
            # custom drawn module participate in a single panel.
            for x in range(0, SIZE + pitch, pitch):
                cv2.line(mask, (x, 0), (x, SIZE), 255, max(7, bar // 2), cv2.LINE_8)
            for y in range(0, SIZE + pitch, pitch):
                cv2.line(mask, (0, y), (SIZE, y), 255, max(7, bar // 2), cv2.LINE_8)
            index = 0
            for row, y in enumerate(range(pitch // 2, SIZE, pitch)):
                for col, x in enumerate(range(pitch // 2, SIZE, pitch)):
                    angle = 0.0
                    if transform_mode == "alternate" and (row + col) % 2:
                        angle = math.pi
                    _transform_stamp(mask, paths, (x, y), pitch, angle, bar,
                                     index, transform_mode, attractor,
                                     field_strength, field_mode)
                    index += 1
        elif layout == "radial":
            index = 0
            for ring, radius in enumerate(range(pitch, 550, pitch)):
                cv2.circle(mask, (CENTER, CENTER), radius, 255, max(7, bar // 2), cv2.LINE_8)
                count = max(8, round(2 * math.pi * radius / pitch))
                phase = (ring % 2) * math.pi / max(count, 1)
                for angle in np.linspace(0, 2 * math.pi, count, endpoint=False) + phase:
                    x, y = CENTER + radius * math.cos(angle), CENTER + radius * math.sin(angle)
                    size = pitch * (.70 + .10 * math.sin((ring + 1) * math.pi / 3))
                    _transform_stamp(mask, paths, (x, y), size, angle + math.pi / 2,
                                     bar, index, transform_mode, attractor,
                                     field_strength, field_mode)
                    cv2.line(mask, (round(x), round(y)),
                             (round(x + pitch * .38 * math.cos(angle)), round(y + pitch * .38 * math.sin(angle))),
                             255, max(7, bar // 2), cv2.LINE_8)
                    index += 1
            for angle in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                cv2.line(mask, (CENTER, CENTER),
                         (round(CENTER + 550 * math.cos(angle)), round(CENTER + 550 * math.sin(angle))),
                         255, max(7, bar // 2), cv2.LINE_8)
        elif layout in ("curve", "vine", "sprig"):
            if guide is None:
                raise ValueError("Curve and vine layouts need --guide JSON")
            for guide_points in _read_guide(guide):
                guide_xy = np.rint(guide_points).astype(np.int32)
                spine_width = max(7, round(bar * (.72 if layout == "curve" else .82)))
                if layout != "curve":
                    cv2.polylines(mask, [guide_xy], False, 255, spine_width, cv2.LINE_AA)
                length = float(np.linalg.norm(np.diff(np.asarray(guide_points), axis=0), axis=1).sum())
                if layout == "sprig":
                    distances = [length * fraction for fraction in (.24, .51, .79)]
                elif layout == "vine":
                    count = max(4, round(length / (pitch * .95)))
                    fractions = np.linspace(.10, .90, count)
                    distances = [length * (f + .025 * math.sin(i * 1.7))
                                 for i, f in enumerate(fractions)]
                else:
                    count = max(5, round(length / (pitch * .72)))
                    fractions = np.linspace(.08, .92, count)
                    distances = [length * (f + .018 * math.sin(i * 1.4))
                                 for i, f in enumerate(fractions)]
                samples = _sample_polyline(guide_points, pitch, distances)
                chain_nodes = []
                for index, (x, y, angle) in enumerate(samples):
                    dx, dy = math.cos(angle + math.pi / 2), math.sin(angle + math.pi / 2)
                    influence, radial, _ = _field_value(x, y, attractor, field_strength)
                    if layout == "vine":
                        side = (-1 if index % 2 else 1) * (1 if index % 5 != 3 else -1)
                        branch_length = pitch * (.48 + .16 * (.5 + .5 * math.sin(index * 1.45)))
                        branch_length *= 1 + (.22 if field_mode == "bloom" else -.12) * influence
                        cx, cy = x + side * dx * branch_length, y + side * dy * branch_length
                        branch = _bezier_branch((x, y), (cx, cy), angle, side,
                                                branch_length, 42)
                        cv2.polylines(mask, [branch], False, 255,
                                      max(6, round(bar * .66)), cv2.LINE_AA)
                        scale = pitch * (.56 + .18 * (.5 + .5 * math.sin(index * 1.17 + .4)))
                        _transform_stamp(mask, paths, (cx, cy), scale,
                                         angle + side * .28, bar, index,
                                         transform_mode, attractor, field_strength, field_mode)
                        # A leaf close to the branch root establishes hierarchy
                        # between stem, secondary branch and terminal flower.
                        leaf_start = tuple(branch[len(branch) // 3].astype(float))
                        _draw_leaf(mask, leaf_start, angle + side * .58,
                                   pitch * .24, side, max(5, round(bar * .52)))
                    elif layout == "sprig":
                        # A folded sprig is intentionally unequal: one main
                        # flower, one secondary flower, and one small terminal.
                        side_sequence = (1, -1, 1)
                        scale_sequence = (1.00, .72, .54)
                        angle_sequence = (.62, -.78, .48)
                        side = side_sequence[index % 3]
                        branch_length = pitch * (.64, .52, .44)[index % 3]
                        cx = x + side * dx * branch_length + math.cos(angle) * pitch * .10
                        cy = y + side * dy * branch_length + math.sin(angle) * pitch * .10
                        branch = _bezier_branch((x, y), (cx, cy), angle, side,
                                                branch_length, 48)
                        cv2.polylines(mask, [branch], False, 255,
                                      max(6, round(bar * (.72 if index == 0 else .58))), cv2.LINE_AA)
                        if index == 0:
                            # The user's drawing is the one principal bloom;
                            # secondary tips become leaves and buds instead of
                            # mechanical smaller copies of the same symbol.
                            _transform_stamp(mask, paths, (cx, cy), pitch * scale_sequence[index % 3],
                                             angle + angle_sequence[index % 3], bar, index,
                                             transform_mode, attractor, field_strength, field_mode)
                        else:
                            _draw_leaf(mask, (cx, cy), angle + side * .52,
                                       pitch * (.34 if index == 1 else .26), side,
                                       max(6, round(bar * .60)))
                            if index == 1:
                                _draw_leaf(mask, (cx, cy), angle - side * .28,
                                           pitch * .24, -side,
                                           max(5, round(bar * .48)))
                        for fraction, leaf_side in ((.28, -side), (.56, side)):
                            leaf_point = tuple(branch[round((len(branch) - 1) * fraction)].astype(float))
                            _draw_leaf(mask, leaf_point, angle + leaf_side * .42,
                                       pitch * (.24 if index == 0 else .19), leaf_side,
                                       max(5, round(bar * .50)))
                    else:
                        # Repeated units sit on a continuous rail, but change
                        # scale and phase gradually instead of behaving like
                        # identical beads on a string.
                        scale = pitch * (.84 + .10 * math.sin(math.pi * (index + 1) / (len(samples) + 1)))
                        if field_mode == "bloom":
                            scale *= 1 + .32 * influence
                        elif field_mode == "quiet":
                            scale *= 1 - .20 * influence
                        _transform_stamp(mask, paths, (x, y), scale,
                                         angle + .12 * math.sin(index * 1.3),
                                         bar, index, transform_mode, attractor,
                                         field_strength, field_mode)
                        chain_nodes.append((np.asarray([x, y], dtype=float), scale, angle))
                if layout == "curve" and chain_nodes:
                    # Join neighboring outlines at their tangent-facing edges.
                    # This produces a continuous floral band, not beads on a
                    # line running through every motif centre.
                    joins = [(np.asarray(guide_points[0], dtype=float), 0.0,
                              chain_nodes[0][0], chain_nodes[0][1] * .18)]
                    for left, right in zip(chain_nodes, chain_nodes[1:]):
                        joins.append((left[0], left[1] * .18, right[0], right[1] * .18))
                    joins.append((chain_nodes[-1][0], chain_nodes[-1][1] * .18,
                                  np.asarray(guide_points[-1], dtype=float), 0.0))
                    for start, radius_start, end, radius_end in joins:
                        delta = end - start
                        distance = np.linalg.norm(delta)
                        if distance < 1:
                            continue
                        direction = delta / distance
                        a = start + direction * radius_start
                        b = end - direction * radius_end
                        cv2.line(mask, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                                 255, max(6, round(bar * .58)), cv2.LINE_AA)
        else:
            raise ValueError(f"Unknown layout: {layout}")
    if clip_mask is None:
        clip_mask = np.zeros_like(mask)
        cv2.circle(clip_mask, (CENTER, CENTER), 530, 255, -1)
    return cv2.bitwise_and(mask, clip_mask)

