"""Curated, fabrication-safe starting recipes for the Rhino dialog."""

RECIPES = {
    "round_cell_morph": {
        "label": "Round Cell Morph | 圆转四瓣",
        "generator": "morph_cell_field", "pitch": 115, "bar": 12, "frame": 38,
        "transform_mode": "field", "field_mode": "bloom",
        "attractor_x": .60, "attractor_y": .48, "field_strength": .82,
        "note": "A quiet focus morphs tangent round cells into four-lobed openings; borrowed view is optional."
    },
    "portrait_triangles": {
        "label": "Portrait Triangle Halftone | 三角孔肖像半调",
        "generator": "triangle_aperture_screen", "pitch": 28, "bar": 5, "frame": 38,
        "transform_mode": "alternate", "field_mode": "quiet",
        "attractor_x": .50, "attractor_y": .50, "field_strength": .80,
        "image_use": "semantic_reveal", "semantic_strength": 1.0,
        "semantic_polarity": "material", "semantic_mode": "portrait",
        "note": "Fine triangular apertures preserve eyes, nose, mouth, and photographic tone."
    },
    "six_petals": {
        "label": "Six Petals | 六瓣共享节点",
        "generator": "hex_flower_field", "pitch": 118, "bar": 12, "frame": 38,
        "transform_mode": "field", "field_mode": "bloom",
        "attractor_x": .63, "attractor_y": .45, "field_strength": .82,
        "note": "A hexagonal contact graph changes from round to six-lobed cells."
    },
    "fish_scale_cascade": {
        "label": "Fish Scale Cascade | 鱼鳞层叠",
        "generator": "fish_scale_field", "pitch": 125, "bar": 13, "frame": 38,
        "transform_mode": "alternate", "field_mode": "drift",
        "attractor_x": .62, "attractor_y": .42, "field_strength": .78,
        "note": "Shared scallop endpoints create a continuous directional scale field."
    },
    "willow_curves": {
        "label": "Willow Curves | 柳条曲格",
        "generator": "soft_strip_lattice", "pitch": 150, "bar": 13, "frame": 38,
        "transform_mode": "alternate", "field_mode": "drift",
        "attractor_x": .58, "attractor_y": .48, "field_strength": .68,
        "note": "Continuous bowed strips form irregular curved openings."
    },
    "tangent_round_cells": {
        "label": "Round and Curved Diamond | 圆孔曲菱",
        "generator": "curved_cell_grid", "pitch": 150, "bar": 14, "frame": 38,
        "transform_mode": "alternate", "field_mode": "quiet",
        "attractor_x": .62, "attractor_y": .45, "field_strength": .24,
        "note": "Tangent round cells leave concave diamonds without overlapping rings."
    },
    "stepped_fret": {
        "label": "Symmetric Stepped Fret | 对称阶梯回纹",
        "generator": "stepped_medallion", "pitch": 150, "bar": 14, "frame": 38,
        "transform_mode": "quarter_turn", "field_mode": "quiet",
        "attractor_x": .50, "attractor_y": .50, "field_strength": .45,
        "note": "A central cross, mirrored stepped returns, side squares, and nested centre follow a traditional whole-window hierarchy."
    },
    "lotus_roundel": {
        "label": "Lotus Roundel | 莲瓣圆窗",
        "generator": "lotus_rosette", "pitch": 150, "bar": 14, "frame": 38,
        "transform_mode": "alternate", "field_mode": "bloom",
        "attractor_x": .50, "attractor_y": .50, "field_strength": .38,
        "note": "Two lotus-petal rings for circular or near-square windows."
    },
    "contour_view": {
        "label": "Contour View | 等高流线",
        "generator": "contour_stream_field", "pitch": 95, "bar": 13, "frame": 38,
        "transform_mode": "field", "field_mode": "quiet",
        "attractor_x": .62, "attractor_y": .48, "field_strength": .82,
        "note": "Nested two-dimensional contour loops bend around a focus and connect back to the frame."
    },
    "herringbone_screen": {
        "label": "Herringbone Screen | 交错长孔编织",
        "generator": "herringbone_screen", "pitch": 115, "bar": 13, "frame": 38,
        "transform_mode": "alternate", "field_mode": "quiet",
        "attractor_x": .50, "attractor_y": .50, "field_strength": .40,
        "note": "A modern perforated sheet with alternating rounded slots."
    },
    "octagon_square": {
        "label": "Octagon and Square | 八角套方",
        "generator": "octagon_square_grid", "pitch": 132, "bar": 13, "frame": 38,
        "transform_mode": "alternate", "field_mode": "quiet",
        "attractor_x": .62, "attractor_y": .46, "field_strength": .45,
        "note": "Tangent chamfered cells create large octagons and small square gaps."
    },
    "plum_blossom_screen": {
        "label": "Plum Blossom Screen | 五瓣梅花孔",
        "generator": "plum_aperture_screen", "pitch": 112, "bar": 12, "frame": 38,
        "transform_mode": "alternate", "field_mode": "bloom",
        "attractor_x": .60, "attractor_y": .46, "field_strength": .68,
        "note": "A modern solid screen perforated with staggered five-petal plum openings."
    },
    "nested_waves": {
        "label": "Nested Water Waves | 三层海水波",
        "generator": "nested_wave_field", "pitch": 126, "bar": 12, "frame": 38,
        "transform_mode": "alternate", "field_mode": "drift",
        "attractor_x": .58, "attractor_y": .42, "field_strength": .62,
        "note": "Three arcs share every scale endpoint, producing a layered water rhythm."
    },
    "ice_crack": {
        "label": "Ice Crack | 冰裂",
        "generator": "ice_field", "pitch": 165, "bar": 13, "frame": 38,
        "transform_mode": "alternate", "field_mode": "quiet",
        "attractor_x": .67, "attractor_y": .46, "field_strength": .60,
        "note": "A full irregular subdivision field; borrowed-view openings are added only when requested."
    },
}

RECIPE_NAMES = list(RECIPES)

