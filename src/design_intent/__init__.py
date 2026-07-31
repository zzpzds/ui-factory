"""设计意图恢复的数据契约、弱监督、评测与导出工具。"""

from .schema import (
    BBox,
    Canvas,
    DesignElement,
    DesignGroup,
    DesignIntentIR,
    LayoutConstraint,
    PageGraph,
    PageNode,
    StyleToken,
    TreeEdge,
)
from .grouping import (
    descendant_element_ids,
    direct_children,
    normalize_group_source_elements,
    validate_group_structure,
)
from .validation import validate_intent_ir, validate_page_graph
from .weak_supervision import build_weak_intent

__all__ = [
    "BBox",
    "Canvas",
    "DesignElement",
    "DesignGroup",
    "DesignIntentIR",
    "LayoutConstraint",
    "PageGraph",
    "PageNode",
    "StyleToken",
    "TreeEdge",
    "build_weak_intent",
    "descendant_element_ids",
    "direct_children",
    "normalize_group_source_elements",
    "validate_intent_ir",
    "validate_group_structure",
    "validate_page_graph",
]
