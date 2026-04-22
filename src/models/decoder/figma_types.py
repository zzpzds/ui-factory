"""
Figma 节点类型枚举及属性定义。
"""

from enum import Enum
from typing import Literal

# Figma 基础节点类型
NodeType = Literal[
    "DOCUMENT",
    "CANVAS",
    "FRAME",
    "GROUP",
    "TEXT",
    "RECTANGLE",
    "ELLIPSE",
    "LINE",
    "VECTOR",
    "BOOLEAN_OPERATION",
    "INSTANCE",
    "COMPONENT",
]

# 所有可生成的设计节点类型
DESIGN_NODE_TYPES = [
    "FRAME",
    "GROUP",
    "TEXT",
    "RECTANGLE",
    "ELLIPSE",
    "LINE",
    "VECTOR",
    "BOOLEAN_OPERATION",
    "INSTANCE",
]

NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(DESIGN_NODE_TYPES)}
IDX_TO_NODE_TYPE = {i: t for t, i in NODE_TYPE_TO_IDX.items()}
NUM_NODE_TYPES = len(DESIGN_NODE_TYPES)


class LayoutMode(Enum):
    """CSS 布局模式对应 Figma 布局模式"""

    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"
    ABSOLUTE = "ABSOLUTE"


class BlendMode(Enum):
    """混合模式"""

    PASS_THROUGH = "PASS_THROUGH"
    NORMAL = "NORMAL"


class FontWeight(Enum):
    """字体粗细"""

    THIN = 100
    EXTRA_LIGHT = 200
    LIGHT = 300
    REGULAR = 400
    MEDIUM = 500
    SEMI_BOLD = 600
    BOLD = 700
    EXTRA_BOLD = 800
    BLACK = 900


# HTML 标签到 Figma 节点类型的映射
TAG_TO_NODE_TYPE = {
    "a": "INSTANCE",
    "button": "INSTANCE",
    "div": "FRAME",
    "header": "FRAME",
    "footer": "FRAME",
    "nav": "FRAME",
    "section": "FRAME",
    "article": "FRAME",
    "main": "FRAME",
    "aside": "FRAME",
    "p": "TEXT",
    "h1": "TEXT",
    "h2": "TEXT",
    "h3": "TEXT",
    "h4": "TEXT",
    "h5": "TEXT",
    "h6": "TEXT",
    "span": "TEXT",
    "strong": "TEXT",
    "em": "TEXT",
    "img": "RECTANGLE",
    "video": "RECTANGLE",
    "canvas": "RECTANGLE",
    "ul": "FRAME",
    "ol": "FRAME",
    "li": "FRAME",
    "table": "FRAME",
    "tr": "FRAME",
    "td": "FRAME",
    "th": "FRAME",
    "form": "FRAME",
    "input": "RECTANGLE",
    "textarea": "RECTANGLE",
    "select": "RECTANGLE",
}


def html_tag_to_node_type(tag: str) -> str:
    """将 HTML 标签映射为 Figma 节点类型"""
    return TAG_TO_NODE_TYPE.get(tag.lower(), "FRAME")
