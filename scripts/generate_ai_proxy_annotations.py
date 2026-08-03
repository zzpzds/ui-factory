"""基于盲态视觉审阅结果生成 Pilot 的独立 AI 代理标注。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation.store import AnnotationStore
from src.design_intent.grouping import normalize_group_source_elements
from src.design_intent.schema import (
    BBox,
    DesignElement,
    DesignGroup,
    DesignIntentIR,
    LayoutConstraint,
    PageGraph,
    StyleToken,
    TreeEdge,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/thesis/12_ai_proxy_annotation_protocol.md"


def E(
    element_id: str,
    nodes: int | list[int],
    element_type: str,
    name: str,
    *,
    text: str | None = None,
    bbox: list[float] | None = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "id": element_id,
        "nodes": [nodes] if isinstance(nodes, int) else nodes,
        "type": element_type,
        "name": name,
        "text": text,
        "bbox": bbox,
        "confidence": confidence,
    }


def G(
    group_id: str,
    role: str,
    name: str,
    children: list[str],
    mode: str,
    *,
    source_node_id: int | None = None,
    gap: float = 0.0,
    padding: list[float] | None = None,
    primary: str = "START",
    cross: str = "START",
    horizontal_resize: str = "FIXED",
    vertical_resize: str = "HUG",
    confidence: float = 0.85,
) -> dict[str, Any]:
    return {
        "id": group_id,
        "role": role,
        "name": name,
        "children": children,
        "mode": mode,
        "source_node_id": source_node_id,
        "gap": gap,
        "padding": padding or [0, 0, 0, 0],
        "primary": primary,
        "cross": cross,
        "horizontal_resize": horizontal_resize,
        "vertical_resize": vertical_resize,
        "confidence": confidence,
    }


def T(
    token_id: str,
    kind: str,
    name: str,
    members: list[str],
) -> dict[str, Any]:
    return {"id": token_id, "kind": kind, "name": name, "members": members}


SPECS: dict[str, dict[str, Any]] = {
    "0001": {
        "elements": [
            E("e_logo", 1, "TEXT", "站点标识"),
            E("e_about", 3, "TEXT", "关于链接"),
            E("e_heading", 5, "TEXT", "欢迎标题"),
            E("e_intro", 6, "TEXT", "介绍文本"),
            E("e_primary", 8, "BUTTON_VISUAL", "主要按钮"),
            E("e_secondary", 9, "BUTTON_VISUAL", "次要按钮"),
            E("e_copyright", 11, "TEXT", "版权信息"),
        ],
        "groups": [
            G("g_nav", "NAV", "顶部导航", ["e_logo", "e_about"], "HORIZONTAL", source_node_id=0, gap=16, padding=[10, 10, 10, 10], primary="SPACE_BETWEEN", cross="CENTER", horizontal_resize="STRETCH"),
            G("g_actions", "CONTAINER", "操作按钮", ["e_primary", "e_secondary"], "HORIZONTAL", source_node_id=7, gap=10, cross="CENTER"),
            G("g_intro", "SECTION", "欢迎内容", ["e_heading", "e_intro", "g_actions"], "VERTICAL", source_node_id=4, gap=16, padding=[20, 20, 20, 20], horizontal_resize="STRETCH"),
        ],
        "tokens": [],
    },
    "0020": {
        "elements": [
            E("e_contact", 10, "TEXT", "联系入口"),
            E("e_social", 15, "ICON", "社交图标", confidence=0.75),
            E("e_nav_home", 20, "TEXT", "首页导航"),
            E("e_nav_about", 22, "TEXT", "品牌介绍导航"),
            E("e_nav_products", 24, "TEXT", "产品导航"),
            E("e_nav_magazine", 26, "TEXT", "杂志导航"),
            E("e_hero_image", 32, "IMAGE", "文章封面背景"),
            E("e_hero_title", 36, "TEXT", "文章标题"),
            E("e_breadcrumb", 49, "TEXT", "首页面包屑"),
            E("e_sponsor_heading", [41, 42], "TEXT", "赞助商标题", text="PATROCINADO POR"),
            E("e_sponsor_image", 43, "IMAGE", "赞助商图片"),
            E("e_sponsor_title", 44, "TEXT", "赞助商名称"),
            E("e_sponsor_body", 46, "TEXT", "赞助商说明"),
            E("e_back", 54, "TEXT", "返回链接"),
            E("e_author", 58, "TEXT", "作者"),
            E("e_date", 60, "TEXT", "发布日期"),
            E("e_intro", 63, "TEXT", "文章导语"),
            E("e_ingredient", 66, "TEXT", "食材条目"),
        ],
        "groups": [
            G("g_top_nav", "NAV", "顶部辅助导航", ["e_contact", "e_social"], "HORIZONTAL", source_node_id=3, gap=16, primary="SPACE_BETWEEN", cross="CENTER", horizontal_resize="STRETCH"),
            G("g_primary_nav", "NAV", "主导航", ["e_nav_home", "e_nav_about", "e_nav_products", "e_nav_magazine"], "HORIZONTAL", source_node_id=16, gap=24, cross="CENTER", horizontal_resize="STRETCH"),
            G("g_header", "SECTION", "页面页眉", ["g_top_nav", "g_primary_nav"], "VERTICAL", source_node_id=2, gap=20, horizontal_resize="STRETCH"),
            G("g_hero", "SECTION", "文章主视觉", ["e_hero_image", "e_hero_title", "e_breadcrumb"], "FREE", source_node_id=30, confidence=0.92),
            G("g_sponsor", "CARD", "赞助商卡片", ["e_sponsor_heading", "e_sponsor_image", "e_sponsor_title", "e_sponsor_body"], "VERTICAL", source_node_id=37, gap=16, padding=[20, 20, 20, 20], cross="CENTER"),
            G("g_meta", "CONTAINER", "作者信息", ["e_author", "e_date"], "HORIZONTAL", source_node_id=56, gap=12, cross="CENTER"),
            G("g_article", "SECTION", "文章正文", ["e_back", "g_meta", "e_intro", "e_ingredient"], "VERTICAL", gap=20, horizontal_resize="STRETCH"),
            G("g_main", "SECTION", "页面主体", ["g_hero", "g_article", "g_sponsor"], "FREE", source_node_id=28, horizontal_resize="STRETCH"),
        ],
        "tokens": [
            T("token_nav_text", "TEXT", "主导航文字", ["e_nav_home", "e_nav_about", "e_nav_products", "e_nav_magazine"]),
            T("token_sponsor_color", "COLOR", "赞助紫色", ["e_sponsor_heading", "e_sponsor_title", "e_sponsor_body"]),
        ],
    },
    "0319": {
        "elements": [
            E("e_nav_home", 6, "TEXT", "首页导航"),
            E("e_nav_about", 8, "TEXT", "关于导航"),
            E("e_nav_roster", 10, "TEXT", "成员导航"),
            E("e_nav_member", 12, "TEXT", "加入会员导航"),
            E("e_nav_manufacturing", 14, "TEXT", "制造业导航"),
            E("e_nav_programs", 16, "TEXT", "州项目导航"),
            E("e_hero_background", 26, "SHAPE", "主视觉背景"),
            E("e_title", 30, "TEXT", "页面标题"),
            E("e_summary", [38, 39, 40], "TEXT", "夏威夷概述"),
            E("e_section_title", 45, "TEXT", "农业部门标题"),
            E("e_section_body", [46, 47], "TEXT", "农业部门正文"),
        ],
        "groups": [
            G("g_nav", "NAV", "主导航", ["e_nav_home", "e_nav_about", "e_nav_roster", "e_nav_member", "e_nav_manufacturing", "e_nav_programs"], "HORIZONTAL", source_node_id=3, gap=22, cross="CENTER", horizontal_resize="STRETCH"),
            G("g_hero", "SECTION", "标题主视觉", ["e_hero_background", "e_title"], "FREE", source_node_id=26),
            G("g_body", "SECTION", "正文内容", ["e_summary", "e_section_title", "e_section_body"], "VERTICAL", gap=24, padding=[48, 128, 32, 128], horizontal_resize="STRETCH"),
            G("g_page", "SECTION", "页面内容", ["g_hero", "g_body"], "VERTICAL", gap=0, horizontal_resize="STRETCH"),
        ],
        "tokens": [
            T("token_nav_text", "TEXT", "导航文字", ["e_nav_home", "e_nav_about", "e_nav_roster", "e_nav_member", "e_nav_manufacturing", "e_nav_programs"]),
            T("token_body_text", "TEXT", "正文文字", ["e_summary", "e_section_body"]),
        ],
    },
    "0419": {
        "elements": [
            E("e_site_name", 8, "TEXT", "站点名称"),
            E("e_site_domain", 9, "TEXT", "站点域名"),
            E("e_title", 15, "TEXT", "文章标题"),
            E("e_posted", 17, "TEXT", "发布提示"),
            E("e_date", [19, 20], "TEXT", "发布日期", text="06/12/2022"),
            E("e_by", 21, "TEXT", "作者提示"),
            E("e_author", 23, "TEXT", "作者"),
            E("e_p1", 31, "TEXT", "正文段落 1"),
            E("e_p2", 32, "TEXT", "正文段落 2"),
            E("e_p3", 33, "TEXT", "正文段落 3"),
            E("e_p4", 34, "TEXT", "正文段落 4"),
            E("e_p5", 35, "TEXT", "正文段落 5"),
            E("e_p6", 36, "TEXT", "正文段落 6"),
            E("e_p7", 37, "TEXT", "正文段落 7", confidence=0.78),
        ],
        "groups": [
            G("g_brand", "SECTION", "站点品牌", ["e_site_name", "e_site_domain"], "VERTICAL", source_node_id=7, gap=0, padding=[0, 16, 0, 16], horizontal_resize="STRETCH"),
            G("g_meta", "CONTAINER", "文章元信息", ["e_posted", "e_date", "e_by", "e_author"], "HORIZONTAL", source_node_id=16, gap=10, cross="CENTER"),
            G("g_hero", "SECTION", "文章页头", ["e_title", "g_meta"], "VERTICAL", gap=28, padding=[96, 15, 96, 15], horizontal_resize="STRETCH"),
            G("g_body", "SECTION", "文章正文", ["e_p1", "e_p2", "e_p3", "e_p4", "e_p5", "e_p6", "e_p7"], "VERTICAL", source_node_id=29, gap=16, horizontal_resize="STRETCH"),
            G("g_page", "SECTION", "文章页面", ["g_hero", "g_body"], "VERTICAL", gap=80, horizontal_resize="STRETCH"),
        ],
        "tokens": [
            T("token_body_text", "TEXT", "文章正文文字", ["e_p1", "e_p2", "e_p3", "e_p4", "e_p5", "e_p6", "e_p7"]),
        ],
    },
    "0601": {
        "elements": [
            E("e_logo", 4, "IMAGE", "赛事站点标识"),
            *[E(f"e_nav_{node}", node, "TEXT", name) for node, name in [(11, "信息导航"), (13, "报名导航"), (15, "赛程导航"), (18, "选手导航"), (20, "轮次导航"), (22, "分组导航")]],
            E("e_title", 23, "TEXT", "分组页标题"),
            E("e_t1_h0", [28, 29], "TEXT", "半决赛状态"), E("e_t1_h1", 30, "TEXT", "半决赛列 1"), E("e_t1_h2", 32, "TEXT", "半决赛列 2"), E("e_t1_h3", 34, "TEXT", "半决赛列 3"), E("e_t1_h4", 36, "TEXT", "半决赛积分列"),
            E("e_t1_r1_rank", 39, "TEXT", "半决赛第一名"), E("e_t1_r1_name", 41, "TEXT", "半决赛第一名选手"), E("e_t1_r1_points", 45, "TEXT", "半决赛第一名积分"),
            E("e_t1_r2_rank", 47, "TEXT", "半决赛第二名"), E("e_t1_r2_name", 49, "TEXT", "半决赛第二名选手"), E("e_t1_r2_points", 53, "TEXT", "半决赛第二名积分"),
            E("e_t1_r3_rank", 55, "TEXT", "半决赛第三名"), E("e_t1_r3_name", 57, "TEXT", "半决赛第三名选手"), E("e_t1_r3_points", 61, "TEXT", "半决赛第三名积分"),
            E("e_t2_h0", [68, 69], "TEXT", "四分之一决赛状态"), E("e_t2_h1", 70, "TEXT", "四分之一决赛列 1"), E("e_t2_h2", 72, "TEXT", "四分之一决赛列 2"), E("e_t2_h3", 74, "TEXT", "四分之一决赛列 3"), E("e_t2_h4", 76, "TEXT", "四分之一决赛积分列"),
            E("e_t2_r1_rank", 79, "TEXT", "四分之一决赛第一名"), E("e_t2_r1_name", 81, "TEXT", "四分之一决赛第一名选手"), E("e_t2_r1_points", 85, "TEXT", "四分之一决赛第一名积分"),
            E("e_t2_r2_rank", 87, "TEXT", "四分之一决赛第二名"), E("e_t2_r2_name", 89, "TEXT", "四分之一决赛第二名选手"), E("e_t2_r2_points", 93, "TEXT", "四分之一决赛第二名积分"),
            E("e_t2_r3_rank", 95, "TEXT", "四分之一决赛第三名"), E("e_t2_r3_name", 97, "TEXT", "四分之一决赛第三名选手"), E("e_t2_r3_points", 101, "TEXT", "四分之一决赛第三名积分"),
            E("e_history_title", 104, "TEXT", "赛事历史标题"),
            E("e_t3_h0", 109, "TEXT", "历史日期列"), E("e_t3_h1", 110, "TEXT", "历史赛事列"), E("e_t3_h2", 111, "TEXT", "历史成绩列"),
            E("e_t3_r1_date", [113, 114], "TEXT", "历史日期 1"), E("e_t3_r1_name", 116, "TEXT", "历史赛事 1"), E("e_t3_r1_result", 118, "TEXT", "历史成绩 1"),
            E("e_t3_r2_date", [121, 122], "TEXT", "历史日期 2"), E("e_t3_r2_name", 124, "TEXT", "历史赛事 2"), E("e_t3_r2_result", 126, "TEXT", "历史成绩 2"),
            E("e_t3_r3_date", [129, 130], "TEXT", "历史日期 3"), E("e_t3_r3_name", 132, "TEXT", "历史赛事 3"), E("e_t3_r3_result", 134, "TEXT", "历史成绩 3"),
            E("e_t3_r4_date", [137, 138], "TEXT", "历史日期 4"), E("e_t3_r4_name", 140, "TEXT", "历史赛事 4"), E("e_t3_r4_result", 142, "TEXT", "历史成绩 4"), E("e_t3_r4_award", 143, "TEXT", "历史奖项 4"),
            E("e_t3_r5_date", [145, 146], "TEXT", "历史日期 5"), E("e_t3_r5_name", 148, "TEXT", "历史赛事 5"), E("e_t3_r5_result", 150, "TEXT", "历史成绩 5", confidence=0.8),
        ],
        "groups": [
            G("g_nav", "NAV", "站点导航", ["e_nav_11", "e_nav_13", "e_nav_15", "e_nav_18", "e_nav_20", "e_nav_22"], "GRID", gap=0, cross="CENTER"),
            G("g_header", "SECTION", "站点页头", ["e_logo", "g_nav"], "VERTICAL", gap=80, cross="CENTER", horizontal_resize="STRETCH"),
            G("g_t1_header", "TABLE_ROW", "半决赛表头", ["e_t1_h0", "e_t1_h1", "e_t1_h2", "e_t1_h3", "e_t1_h4"], "HORIZONTAL", source_node_id=27, gap=2, horizontal_resize="STRETCH"),
            G("g_t1_r1", "TABLE_ROW", "半决赛第一行", ["e_t1_r1_rank", "e_t1_r1_name", "e_t1_r1_points"], "HORIZONTAL", source_node_id=38, gap=2, horizontal_resize="STRETCH"),
            G("g_t1_r2", "TABLE_ROW", "半决赛第二行", ["e_t1_r2_rank", "e_t1_r2_name", "e_t1_r2_points"], "HORIZONTAL", source_node_id=46, gap=2, horizontal_resize="STRETCH"),
            G("g_t1_r3", "TABLE_ROW", "半决赛第三行", ["e_t1_r3_rank", "e_t1_r3_name", "e_t1_r3_points"], "HORIZONTAL", source_node_id=54, gap=2, horizontal_resize="STRETCH"),
            G("g_table_1", "TABLE", "半决赛积分表", ["g_t1_header", "g_t1_r1", "g_t1_r2", "g_t1_r3"], "VERTICAL", source_node_id=25, gap=2, horizontal_resize="STRETCH"),
            G("g_t2_header", "TABLE_ROW", "四分之一决赛表头", ["e_t2_h0", "e_t2_h1", "e_t2_h2", "e_t2_h3", "e_t2_h4"], "HORIZONTAL", source_node_id=67, gap=2, horizontal_resize="STRETCH"),
            G("g_t2_r1", "TABLE_ROW", "四分之一决赛第一行", ["e_t2_r1_rank", "e_t2_r1_name", "e_t2_r1_points"], "HORIZONTAL", source_node_id=78, gap=2, horizontal_resize="STRETCH"),
            G("g_t2_r2", "TABLE_ROW", "四分之一决赛第二行", ["e_t2_r2_rank", "e_t2_r2_name", "e_t2_r2_points"], "HORIZONTAL", source_node_id=86, gap=2, horizontal_resize="STRETCH"),
            G("g_t2_r3", "TABLE_ROW", "四分之一决赛第三行", ["e_t2_r3_rank", "e_t2_r3_name", "e_t2_r3_points"], "HORIZONTAL", source_node_id=94, gap=2, horizontal_resize="STRETCH"),
            G("g_table_2", "TABLE", "四分之一决赛积分表", ["g_t2_header", "g_t2_r1", "g_t2_r2", "g_t2_r3"], "VERTICAL", source_node_id=65, gap=2, horizontal_resize="STRETCH"),
            G("g_t3_header", "TABLE_ROW", "历史表头", ["e_t3_h0", "e_t3_h1", "e_t3_h2"], "HORIZONTAL", source_node_id=108, gap=2, horizontal_resize="STRETCH"),
            G("g_t3_r1", "TABLE_ROW", "历史第一行", ["e_t3_r1_date", "e_t3_r1_name", "e_t3_r1_result"], "HORIZONTAL", source_node_id=112, gap=2, horizontal_resize="STRETCH"),
            G("g_t3_r2", "TABLE_ROW", "历史第二行", ["e_t3_r2_date", "e_t3_r2_name", "e_t3_r2_result"], "HORIZONTAL", source_node_id=120, gap=2, horizontal_resize="STRETCH"),
            G("g_t3_r3", "TABLE_ROW", "历史第三行", ["e_t3_r3_date", "e_t3_r3_name", "e_t3_r3_result"], "HORIZONTAL", source_node_id=128, gap=2, horizontal_resize="STRETCH"),
            G("g_t3_r4", "TABLE_ROW", "历史第四行", ["e_t3_r4_date", "e_t3_r4_name", "e_t3_r4_result", "e_t3_r4_award"], "HORIZONTAL", source_node_id=136, gap=2, horizontal_resize="STRETCH"),
            G("g_t3_r5", "TABLE_ROW", "历史第五行", ["e_t3_r5_date", "e_t3_r5_name", "e_t3_r5_result"], "HORIZONTAL", source_node_id=144, gap=2, horizontal_resize="STRETCH"),
            G("g_table_3", "TABLE", "赛事历史表", ["g_t3_header", "g_t3_r1", "g_t3_r2", "g_t3_r3", "g_t3_r4", "g_t3_r5"], "VERTICAL", source_node_id=106, gap=2, horizontal_resize="STRETCH"),
            G("g_content", "SECTION", "赛事数据内容", ["e_title", "g_table_1", "g_table_2", "e_history_title", "g_table_3"], "VERTICAL", gap=16, cross="CENTER", horizontal_resize="STRETCH"),
        ],
        "tokens": [
            T("token_nav_text", "TEXT", "站点导航文字", ["e_nav_11", "e_nav_13", "e_nav_15", "e_nav_18", "e_nav_20", "e_nav_22"]),
            T("token_table_text", "TEXT", "表格正文文字", ["e_t1_r1_rank", "e_t1_r1_name", "e_t1_r1_points", "e_t1_r2_rank", "e_t1_r2_name", "e_t1_r2_points", "e_t1_r3_rank", "e_t1_r3_name", "e_t1_r3_points", "e_t2_r1_rank", "e_t2_r1_name", "e_t2_r1_points", "e_t2_r2_rank", "e_t2_r2_name", "e_t2_r2_points", "e_t2_r3_rank", "e_t2_r3_name", "e_t2_r3_points"]),
        ],
    },
    "0860": {
        "elements": [
            E("e_brand", [2, 3], "TEXT", "通胀主题标识", text="★ Citrus fruits price inflation since 2022"),
            *[E(f"e_nav_{node}", node, "TEXT", name) for node, name in [(7, "美国导航"), (9, "加拿大导航"), (11, "英国导航"), (13, "澳大利亚导航"), (15, "欧洲导航"), (17, "更多导航")]],
            E("e_headline", [22, 23, 24, 25], "TEXT", "价格换算标题", text="Citrus fruits priced at $20 in 2022 → $19.66 in 2023"),
            E("e_calc_title", 30, "TEXT", "计算器标题"),
            E("e_cost_label", 33, "TEXT", "金额标签"), E("e_currency", 36, "TEXT", "货币符号"), E("e_cost_input", 37, "INPUT", "金额输入框"),
            E("e_start_label", 39, "TEXT", "开始年份标签"), E("e_start_input", 41, "INPUT", "开始年份输入框"),
            E("e_end_label", 43, "TEXT", "结束年份标签"), E("e_end_input", 45, "INPUT", "结束年份输入框"),
            E("e_calculate", 48, "BUTTON_VISUAL", "计算按钮"),
            E("e_related_title", 50, "TEXT", "相关查询提示"), E("e_related_link", 53, "TEXT", "相关查询链接"),
            E("e_article_title", 59, "TEXT", "结果标题"),
            E("e_article_p1", [60, 61, 62], "TEXT", "结果说明 1"),
            E("e_article_p2", [63, 64, 65, 66, 67, 68], "TEXT", "结果说明 2"),
            E("e_article_p3", [69, 70, 71, 72, 73], "TEXT", "结果说明 3"),
            E("e_chart_title", [75, 76], "TEXT", "历史图表标题"),
            E("e_chart_subtitle", 77, "TEXT", "图表来源"),
            E("e_chart_body", [79, 80, 81, 82], "TEXT", "图表摘要"),
        ],
        "groups": [
            G("g_nav_links", "NAV", "地区导航", ["e_nav_7", "e_nav_9", "e_nav_11", "e_nav_13", "e_nav_15", "e_nav_17"], "HORIZONTAL", source_node_id=5, gap=0, cross="CENTER"),
            G("g_nav", "NAV", "顶部导航", ["e_brand", "g_nav_links"], "HORIZONTAL", source_node_id=0, primary="SPACE_BETWEEN", cross="CENTER", horizontal_resize="STRETCH"),
            G("g_cost_field", "CONTAINER", "金额字段", ["e_cost_label", "e_currency", "e_cost_input"], "FREE", source_node_id=32),
            G("g_start_field", "CONTAINER", "开始年份字段", ["e_start_label", "e_start_input"], "VERTICAL", source_node_id=38, gap=6, horizontal_resize="STRETCH"),
            G("g_end_field", "CONTAINER", "结束年份字段", ["e_end_label", "e_end_input"], "VERTICAL", source_node_id=42, gap=6, horizontal_resize="STRETCH"),
            G("g_form", "FORM", "通胀计算表单", ["g_cost_field", "g_start_field", "g_end_field", "e_calculate"], "VERTICAL", source_node_id=31, gap=18, horizontal_resize="STRETCH"),
            G("g_related", "LIST", "相关价格查询", ["e_related_title", "e_related_link"], "VERTICAL", source_node_id=49, gap=12),
            G("g_calculator", "SECTION", "计算器面板", ["e_calc_title", "g_form", "g_related"], "VERTICAL", source_node_id=29, gap=20, horizontal_resize="STRETCH"),
            G("g_chart", "CARD", "历史价格摘要", ["e_chart_title", "e_chart_subtitle", "e_chart_body"], "VERTICAL", source_node_id=74, gap=22, padding=[44, 26, 20, 26], cross="CENTER", horizontal_resize="STRETCH"),
            G("g_article", "SECTION", "通胀结果说明", ["e_article_title", "e_article_p1", "e_article_p2", "e_article_p3", "g_chart"], "VERTICAL", source_node_id=58, gap=20, horizontal_resize="STRETCH"),
            G("g_main", "SECTION", "计算器与结果", ["g_calculator", "g_article"], "HORIZONTAL", gap=54, horizontal_resize="STRETCH"),
            G("g_page", "SECTION", "通胀页面", ["e_headline", "g_main"], "VERTICAL", gap=24, horizontal_resize="STRETCH"),
        ],
        "tokens": [
            T("token_nav_text", "TEXT", "地区导航文字", ["e_nav_7", "e_nav_9", "e_nav_11", "e_nav_13", "e_nav_15", "e_nav_17"]),
            T("token_field_label", "TEXT", "表单标签", ["e_cost_label", "e_start_label", "e_end_label"]),
            T("token_body_text", "TEXT", "说明正文", ["e_article_p1", "e_article_p2", "e_article_p3", "e_chart_body"]),
        ],
    },
    "0052": {
        "elements": [
            E("e_logo", 13, "IMAGE", "南洋视界标识"),
            *[E(f"e_nav_{node}", node, "TEXT", name) for node, name in [(20, "新加坡导航"), (22, "中港台导航"), (24, "国际导航"), (26, "财经导航"), (28, "IT 导航"), (30, "科学导航"), (32, "健康导航"), (34, "观点导航"), (36, "文化导航"), (38, "关于导航"), (40, "广告导航")]],
            E("e_breadcrumb", [55, 57, 59, 60], "TEXT", "文章面包屑"),
            E("e_title", 62, "TEXT", "文章标题"),
            E("e_meta", [66, 67, 68, 69], "TEXT", "文章日期与分类"),
            *[E(f"e_share_{node}", node, "BUTTON_VISUAL", name) for node, name in [(74, "WhatsApp 分享"), (76, "Facebook 分享"), (78, "Twitter 分享"), (80, "Google+ 分享"), (82, "Pinterest 分享"), (84, "LinkedIn 分享")]],
            E("e_main_image", 89, "IMAGE", "文章主图"),
            E("e_body", 86, "TEXT", "文章正文", bbox=[120, 775, 690, 25], confidence=0.72),
            E("e_sidebar_title", 94, "TEXT", "热门新闻标题"),
            E("e_news_1_image", 101, "IMAGE", "热门新闻 1 图片"), E("e_news_1_text", 103, "TEXT", "热门新闻 1 标题"),
            E("e_news_2_image", 108, "IMAGE", "热门新闻 2 图片"), E("e_news_2_text", 110, "TEXT", "热门新闻 2 标题"),
            E("e_news_3_image", 115, "IMAGE", "热门新闻 3 图片"), E("e_news_3_text", 117, "TEXT", "热门新闻 3 标题"),
            E("e_news_4_image", 122, "IMAGE", "热门新闻 4 图片"), E("e_news_4_text", 124, "TEXT", "热门新闻 4 标题"),
            E("e_news_5_text", 128, "TEXT", "热门新闻 5 标题"), E("e_news_6_text", 132, "TEXT", "热门新闻 6 标题"), E("e_news_7_text", 136, "TEXT", "热门新闻 7 标题"), E("e_news_8_text", 140, "TEXT", "热门新闻 8 标题"),
        ],
        "groups": [
            G("g_nav", "NAV", "频道导航", ["e_nav_20", "e_nav_22", "e_nav_24", "e_nav_26", "e_nav_28", "e_nav_30", "e_nav_32", "e_nav_34", "e_nav_36", "e_nav_38", "e_nav_40"], "HORIZONTAL", source_node_id=18, gap=0, cross="CENTER", horizontal_resize="STRETCH"),
            G("g_header", "SECTION", "站点页头", ["e_logo", "g_nav"], "VERTICAL", gap=0, horizontal_resize="STRETCH"),
            G("g_article_header", "SECTION", "文章信息", ["e_breadcrumb", "e_title", "e_meta"], "VERTICAL", source_node_id=52, gap=8, horizontal_resize="STRETCH"),
            G("g_share", "CONTAINER", "社交分享", ["e_share_74", "e_share_76", "e_share_78", "e_share_80", "e_share_82", "e_share_84"], "HORIZONTAL", gap=4, cross="CENTER"),
            G("g_article", "SECTION", "文章正文栏", ["g_article_header", "g_share", "e_main_image", "e_body"], "VERTICAL", gap=20, horizontal_resize="STRETCH"),
            G("g_news_1", "LIST_ITEM", "热门新闻 1", ["e_news_1_image", "e_news_1_text"], "HORIZONTAL", source_node_id=97, gap=10, cross="CENTER", horizontal_resize="STRETCH"),
            G("g_news_2", "LIST_ITEM", "热门新闻 2", ["e_news_2_image", "e_news_2_text"], "HORIZONTAL", source_node_id=104, gap=10, cross="CENTER", horizontal_resize="STRETCH"),
            G("g_news_3", "LIST_ITEM", "热门新闻 3", ["e_news_3_image", "e_news_3_text"], "HORIZONTAL", source_node_id=111, gap=10, cross="CENTER", horizontal_resize="STRETCH"),
            G("g_news_4", "LIST_ITEM", "热门新闻 4", ["e_news_4_image", "e_news_4_text"], "HORIZONTAL", source_node_id=118, gap=10, cross="CENTER", horizontal_resize="STRETCH"),
            G("g_news_list", "LIST", "热门新闻列表", ["g_news_1", "g_news_2", "g_news_3", "g_news_4", "e_news_5_text", "e_news_6_text", "e_news_7_text", "e_news_8_text"], "VERTICAL", source_node_id=96, gap=10, horizontal_resize="STRETCH"),
            G("g_sidebar", "SECTION", "热门新闻侧栏", ["e_sidebar_title", "g_news_list"], "VERTICAL", gap=10, horizontal_resize="STRETCH"),
            G("g_main", "SECTION", "文章与侧栏", ["g_article", "g_sidebar"], "HORIZONTAL", gap=32, horizontal_resize="STRETCH"),
        ],
        "tokens": [
            T("token_nav_text", "TEXT", "频道导航文字", ["e_nav_20", "e_nav_22", "e_nav_24", "e_nav_26", "e_nav_28", "e_nav_30", "e_nav_32", "e_nav_34", "e_nav_36", "e_nav_38", "e_nav_40"]),
            T("token_sidebar_text", "TEXT", "热门新闻标题文字", ["e_news_1_text", "e_news_2_text", "e_news_3_text", "e_news_4_text", "e_news_5_text", "e_news_6_text", "e_news_7_text", "e_news_8_text"]),
        ],
    },
    "1094": {
        "elements": [
            E("e_top_blog", 6, "TEXT", "博客入口"), E("e_top_podcast", 8, "TEXT", "播客入口"),
            *[E(f"e_nav_{node}", node, "TEXT", name) for node, name in [(14, "首页导航"), (16, "开始导航"), (18, "营销创意导航"), (20, "演讲导航"), (22, "电话导航"), (24, "联系导航")]],
            E("e_title", 37, "TEXT", "文章标题"), E("e_image", 39, "IMAGE", "文章插图"),
            E("e_p1", 40, "TEXT", "文章导语"), E("e_p2", 41, "TEXT", "文章正文"), E("e_more", 43, "BUTTON_VISUAL", "阅读全文链接"),
            E("e_services_title", 47, "TEXT", "服务标题"), E("e_service_image", 50, "IMAGE", "咨询服务图"),
        ],
        "groups": [
            G("g_top_nav", "NAV", "顶部内容入口", ["e_top_blog", "e_top_podcast"], "HORIZONTAL", gap=16, cross="CENTER"),
            G("g_primary_nav", "NAV", "主导航", ["e_nav_14", "e_nav_16", "e_nav_18", "e_nav_20", "e_nav_22", "e_nav_24"], "HORIZONTAL", source_node_id=11, gap=22, cross="CENTER"),
            G("g_header", "SECTION", "站点页头", ["g_top_nav", "g_primary_nav"], "VERTICAL", source_node_id=9, gap=24, horizontal_resize="STRETCH"),
            G("g_article", "CARD", "新闻稿文章摘要", ["e_title", "e_image", "e_p1", "e_p2", "e_more"], "VERTICAL", source_node_id=35, gap=14, horizontal_resize="STRETCH"),
            G("g_services", "SECTION", "服务介绍", ["e_services_title", "e_service_image"], "VERTICAL", gap=12, cross="CENTER", horizontal_resize="STRETCH"),
            G("g_main", "SECTION", "页面内容", ["g_article", "g_services"], "VERTICAL", gap=56, horizontal_resize="STRETCH"),
        ],
        "tokens": [
            T("token_nav_text", "TEXT", "主导航文字", ["e_nav_14", "e_nav_16", "e_nav_18", "e_nav_20", "e_nav_22", "e_nav_24"]),
            T("token_body_text", "TEXT", "文章摘要文字", ["e_p1", "e_p2"]),
        ],
    },
    "1429": {
        "elements": [
            E("e_top_home", 13, "TEXT", "顶部首页"), E("e_top_books", 16, "TEXT", "顶部图书"), E("e_top_contact", 19, "TEXT", "顶部联系"), E("e_top_portfolio", 22, "TEXT", "顶部作品集"),
            E("e_site_title", 28, "TEXT", "作者姓名"), E("e_site_subtitle", 29, "TEXT", "作者简介"), E("e_posts", 33, "TEXT", "文章入口"), E("e_comments", 32, "TEXT", "评论入口"), E("e_search", 39, "INPUT", "站内搜索"),
            E("e_banner", 41, "IMAGE", "植物横幅", confidence=0.82),
            E("e_post1_title", 53, "TEXT", "文章 1 标题"), E("e_post1_body", 55, "TEXT", "文章 1 摘要"), E("e_post1_more", 56, "BUTTON_VISUAL", "文章 1 继续阅读"), E("e_post1_meta", [57, 58, 59, 60], "TEXT", "文章 1 元信息"),
            E("e_post2_title", 64, "TEXT", "文章 2 标题"), E("e_post2_body", 66, "TEXT", "文章 2 摘要"), E("e_post2_more", 67, "BUTTON_VISUAL", "文章 2 继续阅读"), E("e_post2_meta", [68, 69, 70, 71, 72, 73], "TEXT", "文章 2 元信息"),
            E("e_pager_home", 77, "TEXT", "分页首页"), E("e_pager_newer", 79, "TEXT", "较新文章分页"),
            E("e_menu_title", 83, "TEXT", "侧栏菜单标题"),
            *[E(f"e_menu_{node}", node, "TEXT", f"侧栏菜单 {node}") for node in [87, 89, 92, 94, 96, 99, 101, 103, 105, 107, 109, 112, 114, 116, 118]],
            E("e_recent_title", 121, "TEXT", "近期文章标题"), E("e_recent_item", 124, "TEXT", "近期文章条目", confidence=0.78),
        ],
        "groups": [
            G("g_top_nav", "NAV", "顶部导航", ["e_top_portfolio", "e_top_contact", "e_top_books", "e_top_home"], "HORIZONTAL", source_node_id=10, gap=12, cross="CENTER"),
            G("g_brand", "SECTION", "作者品牌", ["e_site_title", "e_site_subtitle"], "VERTICAL", gap=0),
            G("g_utility", "NAV", "内容入口", ["e_posts", "e_comments"], "HORIZONTAL", gap=12, cross="CENTER"),
            G("g_header", "SECTION", "博客页头", ["g_top_nav", "g_brand", "g_utility", "e_search", "e_banner"], "FREE", source_node_id=7, horizontal_resize="STRETCH"),
            G("g_post1", "CARD", "文章摘要 1", ["e_post1_title", "e_post1_body", "e_post1_more", "e_post1_meta"], "VERTICAL", gap=12, horizontal_resize="STRETCH"),
            G("g_post2", "CARD", "文章摘要 2", ["e_post2_title", "e_post2_body", "e_post2_more", "e_post2_meta"], "VERTICAL", gap=12, horizontal_resize="STRETCH"),
            G("g_pager", "NAV", "文章分页", ["e_pager_home", "e_pager_newer"], "HORIZONTAL", gap=32, primary="SPACE_BETWEEN", horizontal_resize="STRETCH"),
            G("g_posts", "LIST", "文章列表", ["g_post1", "g_post2", "g_pager"], "VERTICAL", gap=28, horizontal_resize="STRETCH"),
            G("g_menu", "LIST", "侧栏菜单", ["e_menu_title", *[f"e_menu_{node}" for node in [87, 89, 92, 94, 96, 99, 101, 103, 105, 107, 109, 112, 114, 116, 118]]], "VERTICAL", gap=2, horizontal_resize="STRETCH"),
            G("g_recent", "LIST", "近期文章", ["e_recent_title", "e_recent_item"], "VERTICAL", gap=6, horizontal_resize="STRETCH"),
            G("g_sidebar", "SECTION", "博客侧栏", ["g_menu", "g_recent"], "VERTICAL", gap=28, horizontal_resize="STRETCH"),
            G("g_main", "SECTION", "博客双栏内容", ["g_posts", "g_sidebar"], "HORIZONTAL", gap=28, horizontal_resize="STRETCH"),
        ],
        "tokens": [
            T("token_top_nav", "TEXT", "顶部导航文字", ["e_top_home", "e_top_books", "e_top_contact", "e_top_portfolio"]),
            T("token_post_body", "TEXT", "文章摘要文字", ["e_post1_body", "e_post2_body"]),
            T("token_sidebar_text", "TEXT", "侧栏菜单文字", [*[f"e_menu_{node}" for node in [87, 89, 92, 94, 96, 99, 101, 103, 105, 107, 109, 112, 114, 116, 118]]]),
        ],
    },
    "1474": {
        "elements": [
            *[E(f"e_nav_{node}", node, "TEXT", f"顶部导航 {node}") for node in [13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]],
            E("e_logo", 41, "IMAGE", "Agility Lana 标识"), E("e_site_title", 44, "TEXT", "站点名称"), E("e_site_subtitle", 45, "TEXT", "站点说明"), E("e_posts", 49, "TEXT", "文章入口"), E("e_comments", 48, "TEXT", "评论入口"), E("e_search", 55, "INPUT", "站内搜索"), E("e_dog_banner", 57, "IMAGE", "犬类横幅", confidence=0.82),
            E("e_categories_title", 63, "TEXT", "分类标题"),
            E("e_category_1", [65, 66], "TEXT", "Agility 分类"), E("e_category_2", [67, 68], "TEXT", "常规分类"), E("e_category_3", [69, 70], "TEXT", "服从分类"), E("e_category_4", [71, 72], "TEXT", "日程分类"), E("e_category_5", [73, 74], "TEXT", "幼犬分类"),
            E("e_archives_title", 77, "TEXT", "归档标题"),
            *[E(f"e_archive_{node}", node, "TEXT", f"归档月份 {node}") for node in [80, 82, 84, 86, 88, 90, 92, 94, 96, 98]],
            E("e_prev", 102, "TEXT", "上一篇"), E("e_next", 104, "TEXT", "下一篇"), E("e_title", 107, "TEXT", "文章标题"), E("e_p1", 109, "TEXT", "文章导语"), E("e_p2", [110, 111], "TEXT", "周六赛程"), E("e_p3", [112, 113], "TEXT", "周日赛程"), E("e_p4", [114, 115], "TEXT", "报名信息"), E("e_p5", 116, "TEXT", "截止日期"), E("e_p6", 117, "TEXT", "结束语"),
            E("e_recent_title", 123, "TEXT", "近期文章标题"),
            *[E(f"e_recent_{node}", node, "TEXT", f"近期文章 {node}") for node in [126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148]],
        ],
        "groups": [
            G("g_top_nav", "NAV", "顶部导航", [*[f"e_nav_{node}" for node in [13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]]], "GRID", source_node_id=11, gap=0, horizontal_resize="FIXED"),
            G("g_brand", "SECTION", "站点品牌", ["e_logo", "e_site_title", "e_site_subtitle"], "HORIZONTAL", gap=134, cross="CENTER"),
            G("g_utility", "NAV", "文章入口", ["e_posts", "e_comments"], "HORIZONTAL", gap=12, cross="CENTER"),
            G("g_header", "SECTION", "博客页头", ["g_top_nav", "g_brand", "g_utility", "e_search", "e_dog_banner"], "FREE", source_node_id=8, horizontal_resize="STRETCH"),
            G("g_categories", "LIST", "文章分类", ["e_categories_title", "e_category_1", "e_category_2", "e_category_3", "e_category_4", "e_category_5"], "VERTICAL", gap=2, horizontal_resize="STRETCH"),
            G("g_archives", "LIST", "月份归档", ["e_archives_title", *[f"e_archive_{node}" for node in [80, 82, 84, 86, 88, 90, 92, 94, 96, 98]]], "VERTICAL", gap=2, horizontal_resize="STRETCH"),
            G("g_left_sidebar", "SECTION", "分类与归档侧栏", ["g_categories", "g_archives"], "VERTICAL", gap=14, horizontal_resize="STRETCH"),
            G("g_post_nav", "NAV", "上一篇与下一篇", ["e_prev", "e_next"], "HORIZONTAL", gap=32, primary="SPACE_BETWEEN", horizontal_resize="STRETCH"),
            G("g_article", "SECTION", "文章正文", ["g_post_nav", "e_title", "e_p1", "e_p2", "e_p3", "e_p4", "e_p5", "e_p6"], "VERTICAL", gap=14, horizontal_resize="STRETCH"),
            G("g_recent", "LIST", "近期文章列表", ["e_recent_title", *[f"e_recent_{node}" for node in [126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148]]], "VERTICAL", gap=2, horizontal_resize="STRETCH"),
            G("g_main", "SECTION", "博客三栏内容", ["g_left_sidebar", "g_article", "g_recent"], "HORIZONTAL", gap=17, horizontal_resize="STRETCH"),
        ],
        "tokens": [
            T("token_top_nav", "TEXT", "顶部导航文字", [*[f"e_nav_{node}" for node in [13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]]]),
            T("token_sidebar_text", "TEXT", "侧栏链接文字", ["e_category_1", "e_category_2", "e_category_3", "e_category_4", "e_category_5", *[f"e_archive_{node}" for node in [80, 82, 84, 86, 88, 90, 92, 94, 96, 98]], *[f"e_recent_{node}" for node in [126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148]]]),
            T("token_body_text", "TEXT", "文章正文文字", ["e_p1", "e_p2", "e_p3", "e_p4", "e_p5", "e_p6"]),
        ],
    },
}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group(0)) if match else default


def _style(node: Any) -> dict[str, Any]:
    style = node.computed_style
    return {
        "background_color": style.get("backgroundColor"),
        "text_color": style.get("color"),
        "border_color": style.get("borderColor"),
        "border_radius": _number(style.get("borderRadius")),
        "border_width": _number(style.get("borderWidth")),
        "opacity": _number(style.get("opacity"), 1.0),
        "font_size": _number(style.get("fontSize"), 16.0),
        "font_weight": int(_number(style.get("fontWeight"), 400.0)),
        "text_align": str(style.get("textAlign", "start")),
        "image_src": node.attributes.get("src"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_value(kind: str, member: DesignElement) -> dict[str, Any]:
    if kind == "TEXT":
        return {
            "font_size": member.style.get("font_size"),
            "font_weight": member.style.get("font_weight"),
        }
    if kind == "COLOR":
        return {
            "property": "foreground",
            "rgba": member.style.get("text_color"),
        }
    if kind == "RADIUS":
        return {"radius": member.style.get("border_radius")}
    return {"spacing": 0.0}


def build_annotation(
    sample_id: str,
    graph_path: Path,
    screenshot_path: Path,
) -> DesignIntentIR:
    graph = PageGraph.load(graph_path)
    spec = SPECS[sample_id]
    nodes = {node.id: node for node in graph.nodes}

    elements: list[DesignElement] = []
    element_by_id: dict[str, DesignElement] = {}
    for item in spec["elements"]:
        source_nodes = [nodes[node_id] for node_id in item["nodes"]]
        bbox = (
            BBox(*item["bbox"])
            if item["bbox"] is not None
            else BBox.union([node.bbox for node in source_nodes])
        )
        text = item["text"]
        if text is None and item["type"] in {"TEXT", "BUTTON_VISUAL"}:
            text = " ".join(
                node.text.strip() for node in source_nodes if node.text.strip()
            )
        element = DesignElement(
            id=item["id"],
            source_node_ids=item["nodes"],
            type=item["type"],
            bbox=bbox,
            name=item["name"],
            text=text or "",
            style=_style(source_nodes[0]),
            confidence=item["confidence"],
        )
        elements.append(element)
        element_by_id[element.id] = element

    groups: list[DesignGroup] = []
    group_by_id: dict[str, DesignGroup] = {}
    layouts: list[LayoutConstraint] = []
    tree: list[TreeEdge] = []
    child_ids: set[str] = set()
    entity_bbox: dict[str, BBox] = {item.id: item.bbox for item in elements}
    for item in spec["groups"]:
        missing = set(item["children"]) - set(entity_bbox)
        if missing:
            raise ValueError(f"{sample_id}/{item['id']} 子实体尚未定义：{sorted(missing)}")
        source_node_id = item["source_node_id"]
        source_bbox = nodes[source_node_id].bbox if source_node_id is not None else None
        bbox = (
            source_bbox
            if source_bbox is not None and source_bbox.area > 0
            else BBox.union([entity_bbox[child] for child in item["children"]])
        )
        group = DesignGroup(
            id=item["id"],
            source_element_ids=[],
            role=item["role"],
            bbox=bbox,
            name=item["name"],
            style=_style(nodes[source_node_id]) if source_node_id is not None else {},
            source_node_id=source_node_id,
            confidence=item["confidence"],
        )
        groups.append(group)
        group_by_id[group.id] = group
        entity_bbox[group.id] = group.bbox
        layouts.append(
            LayoutConstraint(
                target_id=group.id,
                mode=item["mode"],
                gap=item["gap"],
                padding=item["padding"],
                primary_align=item["primary"],
                cross_align=item["cross"],
                horizontal_resize=item["horizontal_resize"],
                vertical_resize=item["vertical_resize"],
                confidence=item["confidence"],
            )
        )
        for order, child_id in enumerate(item["children"]):
            if child_id in child_ids:
                raise ValueError(f"{sample_id}/{child_id} 被多个 group 直接引用")
            child_ids.add(child_id)
            tree.append(
                TreeEdge(
                    parent_id=group.id,
                    child_id=child_id,
                    order=order,
                    confidence=item["confidence"],
                )
            )

    all_ids = set(entity_bbox)
    for order, entity_id in enumerate(
        sorted(
            all_ids - child_ids,
            key=lambda value: (
                round(entity_bbox[value].y, 3),
                round(entity_bbox[value].x, 3),
                value,
            ),
        )
    ):
        tree.append(
            TreeEdge(
                parent_id="page_root",
                child_id=entity_id,
                order=order,
                confidence=0.9,
            )
        )

    tokens: list[StyleToken] = []
    for item in spec["tokens"]:
        member_ids = list(dict.fromkeys(item["members"]))
        first_element = next(
            (element_by_id[member] for member in member_ids if member in element_by_id),
            None,
        )
        if first_element is None:
            raise ValueError(f"{sample_id}/{item['id']} 没有原子成员")
        tokens.append(
            StyleToken(
                id=item["id"],
                kind=item["kind"],
                value=_token_value(item["kind"], first_element),
                member_ids=member_ids,
                name=item["name"],
                confidence=0.78,
            )
        )

    ir = DesignIntentIR(
        schema_version="1.0",
        canvas=graph.canvas,
        elements=elements,
        groups=groups,
        layouts=layouts,
        tree=tree,
        style_tokens=tokens,
        provenance={
            "source_sample_id": sample_id,
            "label_source": "ai_proxy_annotation",
            "annotator": "annotator_ai",
            "annotator_kind": "ai_proxy",
            "model_provider": "OpenAI",
            "model_family": "Codex GPT-5 family",
            "annotation_protocol": PROTOCOL,
            "annotation_protocol_version": "1.0",
            "status": "complete",
            "weak_labels_viewed": False,
            "human_labels_viewed": False,
            "visual_input_viewed": True,
            "page_graph_viewed": True,
            "screenshot_sha256": _sha256(screenshot_path),
            "page_graph_sha256": _sha256(graph_path),
            "generated_at": "2026-08-03T00:00:00+08:00",
            "limitations": [
                "AI 代理标注不能作为第二位真人设计师标注。",
                "人机一致性不能表述为双人标注者一致性。",
            ],
        },
    )
    normalize_group_source_elements(ir)
    token_members = {token.id: set(token.member_ids) for token in tokens}
    for element in ir.elements:
        element.style_token_refs = sorted(
            token_id for token_id, members in token_members.items()
            if element.id in members
        )
    errors = AnnotationStore._submission_errors(ir, graph)
    if errors:
        raise ValueError(f"{sample_id} 校验失败：\n- " + "\n- ".join(errors))
    return ir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_dir",
        default="data/annotations/intent_pilot_v1",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    assignment = json.loads(
        (package_dir / "assignment.json").read_text(encoding="utf-8")
    )
    output_dir = package_dir / "annotator_ai"
    if not args.check:
        output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for item in assignment["samples"]:
        sample_id = item["sample_id"]
        if sample_id not in SPECS:
            raise ValueError(f"缺少样本 {sample_id} 的盲审规范")
        graph_path = REPO_ROOT / item["page_graph"]
        screenshot_path = REPO_ROOT / item["screenshot"]
        ir = build_annotation(sample_id, graph_path, screenshot_path)
        if not args.check:
            output_path = output_dir / f"{sample_id}.json"
            ir.dump(output_path)
        summary = {
            "sample_id": sample_id,
            "elements": len(ir.elements),
            "groups": len(ir.groups),
            "tokens": len(ir.style_tokens),
            "status": ir.provenance["status"],
        }
        if not args.check:
            summary["annotation_sha256"] = _sha256(output_path)
        summaries.append(summary)
    if not args.check:
        manifest = {
            "schema_version": "1.0",
            "track": "ai_proxy_annotation",
            "annotator": "annotator_ai",
            "protocol": PROTOCOL,
            "blind_annotation_completed_at": "2026-08-03T00:00:00+08:00",
            "human_labels_viewed_before_freeze": False,
            "weak_labels_viewed_before_freeze": False,
            "samples": summaries,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
