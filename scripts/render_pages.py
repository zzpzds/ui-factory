"""
用 Playwright 渲染 data/processed/ 下的 HTML，生成：
- screenshot.png：1280×800 截图（原始分辨率，保留 CSS/图像/字体）
- nodes.pt：节点渲染框 Tensor [N, 4]（224×224 坐标系）
- node_texts.json：节点文本列表（与 nodes.pt 顺序一致，含 inner text）
- parents.pt：每个节点的父索引 LongTensor [N]（-1 表示根）

运行：
    python scripts/render_pages.py [--data_dir data/processed]
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from playwright.sync_api import sync_playwright
from src.data.preprocessing import parse_html_nodes

VIEWPORT_W = 1280
VIEWPORT_H = 800
TARGET_SIZE = 224

SKIP_TAGS = {"html", "head", "body", "script", "style", "meta", "link", "noscript"}

JS_GET_NODES = """
() => {
    const skipTags = new Set(['html','head','body','script','style','meta','link','noscript']);
    const escapeAttr = (s) => s.replace(/"/g, '&quot;');

    // 第一遍：过滤并序列化
    const kept = [];
    const elemToIdx = new Map();
    for (const el of document.querySelectorAll('*')) {
        if (skipTags.has(el.tagName.toLowerCase())) continue;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') continue;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) continue;

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

        elemToIdx.set(el, kept.length);
        kept.push({ el, text, rect });
    }

    // 第二遍：解析最近保留祖先作为父节点
    return kept.map(({ el, text, rect }) => {
        let p = el.parentElement;
        while (p && !elemToIdx.has(p)) p = p.parentElement;
        return {
            text: text,
            parent_idx: p ? elemToIdx.get(p) : -1,
            x1: rect.left,
            y1: rect.top,
            x2: rect.right,
            y2: rect.bottom,
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


def render_one(page, sample_dir: str) -> bool:
    html_path = os.path.join(sample_dir, "page.html")
    screenshot_path = os.path.join(sample_dir, "screenshot.png")
    nodes_path = os.path.join(sample_dir, "nodes.pt")
    texts_path = os.path.join(sample_dir, "node_texts.json")
    parents_path = os.path.join(sample_dir, "parents.pt")

    if not os.path.exists(html_path):
        return False
    # 已处理过则跳过（包含 parents.pt 才算完整，旧样本会被重渲）
    if all(os.path.exists(p) for p in (nodes_path, texts_path, parents_path, screenshot_path)):
        return True

    try:
        # 只屏蔽流式 media（视频/音频），保留 CSS / 图像 / 字体，让截图接近真实视觉
        page.route("**/*", lambda route: (
            route.abort() if route.request.resource_type == "media"
            else route.continue_()
        ))

        html_url = Path(html_path).as_uri()
        # 等到 load 而不是 domcontentloaded，确保样式和图像完成应用
        page.goto(html_url, timeout=15000, wait_until="load")

        # 截图
        page.screenshot(path=screenshot_path, clip={"x": 0, "y": 0, "width": VIEWPORT_W, "height": VIEWPORT_H})

        # 提取节点框 + 父索引
        nodes = page.evaluate(JS_GET_NODES)
        if not nodes:
            return False

        node_texts = [n["text"] for n in nodes]
        boxes = [scale_box(n["x1"], n["y1"], n["x2"], n["y2"]) for n in nodes]
        parents = [n["parent_idx"] for n in nodes]

        torch.save(torch.tensor(boxes, dtype=torch.float32), nodes_path)
        torch.save(torch.tensor(parents, dtype=torch.long), parents_path)
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(node_texts, f, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"\n  跳过 {os.path.basename(sample_dir)}: {e}")
        return False


def main(data_dir: str):
    sample_dirs = sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    print(f"共 {len(sample_dirs)} 个样本，开始渲染...")

    success, failed = 0, 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})
        page = context.new_page()

        for sample_dir in tqdm(sample_dirs, desc="渲染"):
            if render_one(page, sample_dir):
                success += 1
            else:
                failed += 1

        browser.close()

    print(f"\n完成：成功 {success}，失败/跳过 {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    args = parser.parse_args()
    main(os.path.abspath(args.data_dir))
