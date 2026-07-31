"""Design Intent IR 的可重渲染 HTML 导出。"""

from __future__ import annotations

from html import escape
import json
from typing import Any

from .schema import DesignIntentIR


def _rgba(value: Any, fallback: str = "transparent") -> str:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return fallback
    red, green, blue = [round(float(channel) * 255) for channel in value[:3]]
    alpha = max(0.0, min(1.0, float(value[3])))
    return f"rgba({red},{green},{blue},{alpha:.4f})"


def _token_css_value(token, override: dict[str, Any] | None) -> str:
    value = override if override is not None else token.value
    if token.kind == "COLOR":
        return _rgba(value.get("rgba"), "rgba(0,0,0,1)")
    if token.kind == "TEXT":
        return f"{float(value.get('font_size', 16))}px"
    if token.kind == "RADIUS":
        return f"{float(value.get('radius', 0))}px"
    if token.kind == "SPACING":
        return f"{float(value.get('spacing', 0))}px"
    return "initial"


def _entity_css(
    entity,
    token_by_id: dict[str, Any],
) -> str:
    box = entity.bbox
    style = entity.style
    rules = {
        "position": "absolute",
        "left": f"{box.x}px",
        "top": f"{box.y}px",
        "width": f"{box.width}px",
        "height": f"{box.height}px",
        "box-sizing": "border-box",
        "background": _rgba(style.get("background_color")),
        "color": _rgba(style.get("text_color"), "rgba(0,0,0,1)"),
        "border-color": _rgba(style.get("border_color")),
        "border-style": "solid",
        "border-width": f"{float(style.get('border_width', 0))}px",
        "border-radius": f"{float(style.get('border_radius', 0))}px",
        "opacity": str(max(0.0, min(1.0, float(style.get("opacity", 1))))),
        "font-size": f"{float(style.get('font_size', 16))}px",
        "font-weight": str(int(style.get("font_weight", 400))),
        "text-align": str(style.get("text_align", "start")),
        "overflow": "hidden",
        "white-space": "pre-wrap",
        "margin": "0",
        "padding": "0",
    }
    for token_id in getattr(entity, "style_token_refs", []):
        token = token_by_id.get(token_id)
        if token is None:
            continue
        variable = f"var(--{token_id})"
        if token.kind == "COLOR":
            property_name = token.value.get("property", "foreground")
            css_property = {
                "background": "background",
                "foreground": "color",
                "border": "border-color",
            }.get(property_name, "color")
            rules[css_property] = variable
        elif token.kind == "TEXT":
            rules["font-size"] = variable
        elif token.kind == "RADIUS":
            rules["border-radius"] = variable
    return ";".join(f"{key}:{value}" for key, value in rules.items())


def export_editable_html(
    ir: DesignIntentIR,
    token_overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    """导出固定画布 HTML；token 使用 CSS variable，便于编辑行为测试。"""
    token_overrides = token_overrides or {}
    token_by_id = {token.id: token for token in ir.style_tokens}
    variables = ";".join(
        f"--{token.id}:{_token_css_value(token, token_overrides.get(token.id))}"
        for token in ir.style_tokens
    )

    # 大组先绘制，原子元素最后绘制，避免容器背景覆盖内容。
    groups = sorted(ir.groups, key=lambda group: group.bbox.area, reverse=True)
    body = []
    for group in groups:
        body.append(
            f'<div data-intent-id="{escape(group.id)}" '
            f'style="{_entity_css(group, token_by_id)}"></div>'
        )
    for element in ir.elements:
        css = _entity_css(element, token_by_id)
        source = element.style.get("image_src")
        if element.type == "IMAGE" and source:
            body.append(
                f'<img data-intent-id="{escape(element.id)}" '
                f'src="{escape(str(source), quote=True)}" style="{css};object-fit:cover">'
            )
        else:
            body.append(
                f'<div data-intent-id="{escape(element.id)}" '
                f'style="{css}">{escape(element.text)}</div>'
            )

    metadata = json.dumps(ir.provenance, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{{variables}}}
html,body{{margin:0;width:{ir.canvas.width}px;height:{ir.canvas.height}px;overflow:hidden}}
body{{position:relative;background:white;font-family:Arial,sans-serif}}
</style>
</head>
<body>
{''.join(body)}
<script type="application/json" id="design-intent-provenance">{metadata}</script>
</body>
</html>"""
