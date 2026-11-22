#! python 3
"""Create installable JIEJING preset Value List user objects."""

from pathlib import Path

import Grasshopper
import System
import System.Drawing


ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "user_objects"
INSTALL_DIR = Path(str(Grasshopper.Folders.DefaultUserObjectFolder))

SELECTORS = [
    {
        "file": "JIEJING_Recipe.ghuser",
        "guid": "7ca564d5-73e1-44c4-a884-c4b34a7ddf01",
        "name": "JIEJING Recipe Selector",
        "nick": "Recipe",
        "description": "Choose one of the curated JIEJING lattice recipes.",
        "icon": "pattern_light.png",
        "items": [
            "圆转四瓣 / Round Cell Morph", "三角孔肖像 / Portrait Triangles",
            "六瓣共享节点 / Six Petals", "鱼鳞层叠 / Fish Scale",
            "柳条曲格 / Willow Curves", "圆孔曲菱 / Tangent Round Cells",
            "对称阶梯回纹 / Stepped Fret", "莲瓣圆窗 / Lotus Roundel",
            "等高流线 / Contour View", "交错长孔编织 / Herringbone",
            "八角套方 / Octagon Square", "五瓣梅花孔 / Plum Blossom",
            "三层海水波 / Nested Waves", "冰裂 / Ice Crack",
        ],
    },
    {
        "file": "JIEJING_ViewShape.ghuser",
        "guid": "b30a2975-fc73-408b-bd7c-dc37796347b1",
        "name": "JIEJING View Shape Selector",
        "nick": "ViewShape",
        "description": "Choose a classical borrowed-view opening shape.",
        "icon": "borrowed_light.png",
        "items": [
            "椭圆 / Ellipse", "圆 / Circle", "六角 / Hexagon",
            "八角 / Octagon", "切角矩形 / Clipped Rect", "海棠 / Begonia",
            "梅花 / Plum", "如意 / Ruyi", "扇面 / Fan", "葫芦 / Gourd",
        ],
    },
    {
        "file": "JIEJING_ImageMode.ghuser",
        "guid": "2ed91dbb-2b6b-4403-8d83-7d74eb456260",
        "name": "JIEJING Image Mode Selector",
        "nick": "ImageMode",
        "description": "Choose how image information controls the lattice field.",
        "icon": "image_light.png",
        "items": ["轮廓 / Silhouette", "肖像细节 / Portrait", "混合 / Hybrid"],
    },
    {
        "file": "JIEJING_Polarity.ghuser",
        "guid": "407b9c6e-7a51-4a7a-9974-69878de2c9d0",
        "name": "JIEJING Polarity Selector",
        "nick": "Polarity",
        "description": "Choose whether the image controls retained material or openings.",
        "icon": "preview_light.png",
        "items": ["图像控制材料 / Material", "图像控制开口 / Opening"],
    },
]


def build_selector(spec, target_dir):
    value_list = Grasshopper.Kernel.Special.GH_ValueList()
    value_list.Name = spec["name"]
    value_list.NickName = spec["nick"]
    value_list.Description = spec["description"]
    value_list.Category = "JIEJING"
    value_list.SubCategory = "Selectors"
    value_list.ListMode = Grasshopper.Kernel.Special.GH_ValueListMode.DropDown
    value_list.ListItems.Clear()
    for index, label in enumerate(spec["items"]):
        item = Grasshopper.Kernel.Special.GH_ValueListItem(label, str(index))
        item.Selected = index == 0
        value_list.ListItems.Add(item)

    user = Grasshopper.Kernel.GH_UserObject()
    # Guid identifies this reusable preset. BaseGuid must identify the
    # underlying Grasshopper object type; using the preset Guid here makes
    # Grasshopper report "base type is missing" when the toolbar item is used.
    user.Guid = System.Guid(spec["guid"])
    user.BaseGuid = value_list.ComponentGuid
    user.Description = Grasshopper.Kernel.GH_InstanceDescription(
        spec["name"], spec["nick"], spec["description"], "JIEJING", "Selectors")
    user.Exposure = Grasshopper.Kernel.GH_Exposure.primary
    user.Icon = System.Drawing.Bitmap(str(ROOT / "icons" / spec["icon"]))
    user.SetDataFromObject(value_list)
    target_dir.mkdir(parents=True, exist_ok=True)
    user.Path = str(target_dir / spec["file"])
    if not user.SaveToFile():
        raise RuntimeError("Could not save " + user.Path)
    return user.Path


created = []
for selector in SELECTORS:
    created.append(build_selector(selector, PACKAGE_DIR))
    created.append(build_selector(selector, INSTALL_DIR))

print("JIEJING USER OBJECTS", len(created), "files")
for path in created:
    print(path)
