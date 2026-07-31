"""
用 Playwright 渲染 data/processed/ 下的 HTML，生成：
- screenshot.png：1280×800 截图（原始分辨率，保留 CSS/图像/字体）
- page_graph.json：统一的 Page Implementation Graph（主格式）
- nodes.pt：节点渲染框 Tensor [N, 4]（224×224 坐标系）
- node_texts.json：节点文本列表（与 nodes.pt 顺序一致，含 inner text）
- parents.pt：每个节点的父索引 LongTensor [N]（-1 表示根）
- styles.json：每节点的 computed style 字典列表（与上述同序）

旧的四个节点文件在迁移期继续生成，用于复现已有实验。

运行：
    python scripts/render_pages.py [--data_dir data/processed]
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from src.design_intent.page_graph import page_graph_from_browser_payload

VIEWPORT_W = 1280
VIEWPORT_H = 800
TARGET_SIZE = 224

SKIP_TAGS = {"html", "head", "body", "script", "style", "meta", "link", "noscript"}

JS_GET_NODES = """
() => {
    const skipTags = new Set(['html','head','body','script','style','meta','link','noscript']);
    const escapeAttr = (s) => s.replace(/"/g, '&quot;');

    // 颜色字符串解析为 [r,g,b,a]，归一化到 0-1。失败返回 null。
    const parseColor = (s) => {
        if (!s || s === 'transparent' || s === 'none') return null;
        const m = s.match(/rgba?\\(([^)]+)\\)/);
        if (!m) return null;
        const parts = m[1].split(',').map(x => parseFloat(x.trim()));
        if (parts.length < 3) return null;
        return [parts[0]/255, parts[1]/255, parts[2]/255, parts.length > 3 ? parts[3] : 1.0];
    };
    const parsePx = (s) => {
        if (!s) return 0;
        const m = s.match(/(-?[0-9.]+)/);
        return m ? parseFloat(m[1]) : 0;
    };

    // 第一遍：过滤并序列化
    const kept = [];
    const elemToIdx = new Map();
    for (const el of document.querySelectorAll('*')) {
        if (skipTags.has(el.tagName.toLowerCase())) continue;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') continue;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) continue;
        if (
            rect.right <= 0 || rect.bottom <= 0 ||
            rect.left >= window.innerWidth || rect.top >= window.innerHeight
        ) continue;
        const clippedRect = {
            left: Math.max(0, rect.left),
            top: Math.max(0, rect.top),
            right: Math.min(window.innerWidth, rect.right),
            bottom: Math.min(window.innerHeight, rect.bottom),
        };

        // 序列化（与 parse_html_nodes 保持一致）
        let text = '<' + el.tagName.toLowerCase();
        for (const attr of el.attributes) {
            text += ' ' + attr.name + '="' + escapeAttr(attr.value) + '"';
        }
        // 只取直接文本子节点，不递归后代
        let inner = '';
        for (const child of el.childNodes) {
            if (child.nodeType === 3) inner += child.textContent;
        }
        inner = inner.trim();
        if (inner.length > 100) inner = inner.substring(0, 100);
        text += '>' + inner + '</' + el.tagName.toLowerCase() + '>';

        const attributes = {};
        for (const attr of el.attributes) attributes[attr.name] = attr.value;

        // 计算样式既是模型输入证据，也是弱标签规则的证据来源。
        const cs = {
            backgroundColor: parseColor(style.backgroundColor),
            color: parseColor(style.color),
            borderColor: parseColor(style.borderTopColor),
            borderRadius: parsePx(style.borderTopLeftRadius),
            borderWidth: parsePx(style.borderTopWidth),
            opacity: parseFloat(style.opacity) || 1.0,
            fontSize: parsePx(style.fontSize),
            fontWeight: parseInt(style.fontWeight) || 400,
            textAlign: style.textAlign || 'start',
            display: style.display || 'block',
            position: style.position || 'static',
            flexDirection: style.flexDirection || 'row',
            justifyContent: style.justifyContent || 'normal',
            alignItems: style.alignItems || 'normal',
            gap: parsePx(style.gap),
            rowGap: parsePx(style.rowGap),
            columnGap: parsePx(style.columnGap),
            paddingTop: parsePx(style.paddingTop),
            paddingRight: parsePx(style.paddingRight),
            paddingBottom: parsePx(style.paddingBottom),
            paddingLeft: parsePx(style.paddingLeft),
            marginTop: parsePx(style.marginTop),
            marginRight: parsePx(style.marginRight),
            marginBottom: parsePx(style.marginBottom),
            marginLeft: parsePx(style.marginLeft),
            lineHeight: parsePx(style.lineHeight),
            letterSpacing: parsePx(style.letterSpacing),
            overflowX: style.overflowX || 'visible',
            overflowY: style.overflowY || 'visible',
            objectFit: style.objectFit || 'fill',
            gridTemplateColumns: style.gridTemplateColumns || 'none',
        };

        elemToIdx.set(el, kept.length);
        kept.push({
            el, tag: el.tagName.toLowerCase(), text, inner, attributes,
            rect: clippedRect, cs
        });
    }

    // 第二遍：解析最近保留祖先作为父节点
    return kept.map(({ el, tag, text, inner, attributes, rect, cs }) => {
        let p = el.parentElement;
        while (p && !elemToIdx.has(p)) p = p.parentElement;
        return {
            tag: tag,
            text: text,
            direct_text: inner,
            attributes: attributes,
            parent_idx: p ? elemToIdx.get(p) : -1,
            x1: rect.left,
            y1: rect.top,
            x2: rect.right,
            y2: rect.bottom,
            style: cs,
        };
    });
}
"""


def scale_box(x1, y1, x2, y2):
    """将 1280×800 坐标映射到 224×224"""
    sx = TARGET_SIZE / VIEWPORT_W
    sy = TARGET_SIZE / VIEWPORT_H
    return [
        max(0.0, min(TARGET_SIZE, x1 * sx)),
        max(0.0, min(TARGET_SIZE, y1 * sy)),
        max(0.0, min(TARGET_SIZE, x2 * sx)),
        max(0.0, min(TARGET_SIZE, y2 * sy)),
    ]


def render_one(
    page, sample_dir: str, force: bool = False
) -> tuple[bool, str | None]:
    html_path = os.path.join(sample_dir, "page.html")
    screenshot_path = os.path.join(sample_dir, "screenshot.png")
    nodes_path = os.path.join(sample_dir, "nodes.pt")
    texts_path = os.path.join(sample_dir, "node_texts.json")
    parents_path = os.path.join(sample_dir, "parents.pt")
    styles_path = os.path.join(sample_dir, "styles.json")
    page_graph_path = os.path.join(sample_dir, "page_graph.json")

    if not os.path.exists(html_path):
        return False, "missing_page_html"
    # 已处理过则跳过（包含 styles.json 才算完整，旧样本会被重渲）
    expected = (
        nodes_path,
        texts_path,
        parents_path,
        screenshot_path,
        styles_path,
        page_graph_path,
    )
    if not force and all(os.path.exists(p) for p in expected):
        try:
            with open(page_graph_path, encoding="utf-8") as f:
                metadata = json.load(f).get("metadata", {})
            if (
                metadata.get("viewport_filter") == "intersects_and_clipped_v2"
                and metadata.get("resource_policy")
                == "block_media_and_font_v1"
            ):
                return True, None
        except (OSError, ValueError, TypeError):
            pass

    try:
        html_url = Path(html_path).as_uri()
        # DOM 可用后最多再等 5 秒外部资源；资源超时不应丢弃整个页面。
        page.goto(html_url, timeout=10000, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("load", timeout=5000)
        except PlaywrightTimeoutError:
            pass

        # 截图
        page.screenshot(path=screenshot_path, clip={"x": 0, "y": 0, "width": VIEWPORT_W, "height": VIEWPORT_H})

        # 提取节点框 + 父索引 + 样式
        nodes = page.evaluate(JS_GET_NODES)
        if not nodes:
            return False, "empty_page_graph"

        node_texts = [n["text"] for n in nodes]
        boxes = [scale_box(n["x1"], n["y1"], n["x2"], n["y2"]) for n in nodes]
        parents = [n["parent_idx"] for n in nodes]
        styles = [n["style"] for n in nodes]

        torch.save(torch.tensor(boxes, dtype=torch.float32), nodes_path)
        torch.save(torch.tensor(parents, dtype=torch.long), parents_path)
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(node_texts, f, ensure_ascii=False)
        with open(styles_path, "w", encoding="utf-8") as f:
            json.dump(styles, f, ensure_ascii=False)

        graph_payload = [
            {
                "parent_id": n["parent_idx"],
                "tag": n["tag"],
                "text": n["direct_text"],
                "attributes": n["attributes"],
                "bbox": {
                    "x": n["x1"],
                    "y": n["y1"],
                    "width": max(0.0, n["x2"] - n["x1"]),
                    "height": max(0.0, n["y2"] - n["y1"]),
                },
                "computed_style": n["style"],
            }
            for n in nodes
        ]
        graph = page_graph_from_browser_payload(
            sample_id=os.path.basename(sample_dir),
            payload=graph_payload,
            canvas_width=VIEWPORT_W,
            canvas_height=VIEWPORT_H,
            metadata={
                "source_format": "playwright_page_graph_v1",
                "coordinate_space": "viewport_pixels",
                "target_vit_size": TARGET_SIZE,
                "viewport_filter": "intersects_and_clipped_v2",
                "resource_policy": "block_media_and_font_v1",
            },
        )
        graph.dump(page_graph_path)

        return True, None

    except Exception as e:
        print(f"\n  跳过 {os.path.basename(sample_dir)}: {e}")
        return False, f"{type(e).__name__}: {e}"


def main(
    data_dir: str,
    force: bool = False,
    sleep_between: float = 0.2,
    manifest_path: str | None = None,
    limit: int = 0,
    report_path: str | None = None,
):
    sample_dirs = sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    if manifest_path:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        selected_ids = set(manifest["sample_ids"])
        sample_dirs = [
            path for path in sample_dirs
            if os.path.basename(path) in selected_ids
        ]
        missing = selected_ids - {
            os.path.basename(path) for path in sample_dirs
        }
        if missing:
            raise FileNotFoundError(
                f"manifest 中 {len(missing)} 个样本目录不存在："
                f"{sorted(missing)[:5]}"
            )
    if limit > 0:
        sample_dirs = sample_dirs[:limit]
    print(f"共 {len(sample_dirs)} 个样本，开始渲染...")

    success, failed = 0, 0
    failures: list[dict[str, str]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})
        page = context.new_page()
        page.set_default_timeout(10000)
        # 每个 page 只注册一次路由，避免批量运行时处理器不断累积。
        page.route("**/*", lambda route: (
            route.abort()
            if route.request.resource_type in {"media", "font"}
            else route.continue_()
        ))

        for sample_dir in tqdm(sample_dirs, desc="渲染"):
            ok, error = render_one(page, sample_dir, force=force)
            if ok:
                success += 1
            else:
                failed += 1
                failures.append(
                    {
                        "sample_id": os.path.basename(sample_dir),
                        "error": error or "unknown",
                    }
                )
            # 留资源冗余：每条样本之间稍作 sleep
            if sleep_between > 0:
                time.sleep(sleep_between)

        browser.close()

    report = {
        "data_dir": data_dir,
        "manifest": manifest_path,
        "requested": len(sample_dirs),
        "success": success,
        "failed": failed,
        "failures": failures,
    }
    if report_path:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--force", action="store_true", help="强制重渲已有样本")
    parser.add_argument("--sleep", type=float, default=0.2, help="样本间 sleep 秒数")
    parser.add_argument("--manifest", default=None, help="包含 sample_ids 的 Pilot manifest")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    main(
        os.path.abspath(args.data_dir),
        force=args.force,
        sleep_between=args.sleep,
        manifest_path=args.manifest,
        limit=args.limit,
        report_path=args.report,
    )
