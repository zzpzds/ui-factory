from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

from scripts.serve_intent_annotation import AnnotationHandler
from src.annotation import AnnotationStore


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "data/annotations/intent_gold_v1"


def test_reference_ui_distinguishes_ai_silver_and_human_gold():
    AnnotationHandler.store = AnnotationStore(REPO_ROOT, PACKAGE_DIR)
    AnnotationHandler.adjudication_store = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), AnnotationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            console_errors = []
            page.on(
                "console",
                lambda message: console_errors.append(message.text)
                if message.type == "error" else None,
            )
            page.goto(url)
            assert page.request.get(f"{url}/favicon.ico").ok
            page.select_option("#sampleSelect", "0044")
            page.wait_for_function(
                "document.querySelector('#assistanceBadge').textContent.includes('AI')"
            )

            assert page.locator("#assistanceBadge").text_content() == "AI 多阶段银标"
            assert page.locator("#assistanceTitle").text_content() == "AI 多视角参考标注"
            assert not page.locator(".review-confirmation").is_visible()
            assert not page.locator("#saveButton").is_visible()
            assert not page.locator("#submitButton").is_visible()
            assert "✓ AI 0044" in page.locator(
                '#sampleSelect option[value="0044"]'
            ).text_content()
            page.locator('[data-tab="entities"]').click()
            page.locator("#entityList .list-row").first.click(timeout=2000)
            assert page.locator("#inspectorTitle").text_content() != "未选择实体"
            assert page.locator("#entityInspector input:not(:disabled)").count() == 0
            assert page.locator("#entityInspector select:not(:disabled)").count() == 0
            assert page.locator("#entityInspector textarea:not(:disabled)").count() == 0
            assert not page.locator("#deleteEntity").is_visible()

            page.select_option("#sampleSelect", "0001")
            page.wait_for_function(
                "document.querySelector('#assistanceBadge').textContent.includes('Gold')"
            )
            assert page.locator("#assistanceBadge").text_content() == "人工复核 Gold"
            assert page.locator("#assistanceTitle").text_content() == "单人 AI 辅助定稿"
            assert "✓ 人 0001" in page.locator(
                '#sampleSelect option[value="0001"]'
            ).text_content()
            page.locator('[data-tab="tokens"]').click()
            assert page.locator("#tokenReviewStatus").text_content() == "未采集"
            assert page.locator(".token-result-label").text_content() == (
                "Token 标签不可用"
            )
            assert "未采集 Token 人工标签" in page.locator(
                "#tokenList"
            ).text_content()
            assert console_errors == []

            page.set_viewport_size({"width": 390, "height": 844})
            page.reload()
            page.select_option("#sampleSelect", "0044")
            page.wait_for_function(
                "document.querySelector('#assistanceBadge').textContent.includes('AI')"
            )
            mobile_layout = page.evaluate("""() => {
                const root = document.documentElement;
                const right = document.querySelector('.right-panel').getBoundingClientRect();
                return {
                    scrollHeight: root.scrollHeight,
                    requiredHeight: Math.ceil(right.bottom + window.scrollY),
                    horizontalOverflow: root.scrollWidth > root.clientWidth,
                };
            }""")
            assert not mobile_layout["horizontalOverflow"]
            assert mobile_layout["scrollHeight"] >= mobile_layout["requiredHeight"]
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
