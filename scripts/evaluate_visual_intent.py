"""把 Intent IR 重渲染为截图并计算视觉约束指标。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from playwright.sync_api import sync_playwright

from src.design_intent.html_export import export_editable_html
from src.design_intent.schema import DesignIntentIR
from src.design_intent.visual_metrics import compute_visual_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--intent_name", default="weak_intent.json")
    parser.add_argument("--output_dir", default="outputs/intent-visual")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    ir = DesignIntentIR.load(sample_dir / args.intent_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "render.html"
    screenshot_path = output_dir / "render.png"
    html_path.write_text(export_editable_html(ir), encoding="utf-8")

    width = int(round(ir.canvas.width))
    height = int(round(ir.canvas.height))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        page.screenshot(
            path=str(screenshot_path),
            clip={"x": 0, "y": 0, "width": width, "height": height},
        )
        browser.close()

    predicted = Image.open(screenshot_path)
    reference = Image.open(sample_dir / "screenshot.png")
    if reference.size != predicted.size:
        reference = reference.resize(predicted.size)
    metrics = compute_visual_metrics(predicted, reference)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
