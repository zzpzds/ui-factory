"""启动本地 Design Intent 人工标注工作台。"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import sys
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.annotation import AdjudicationStore, AnnotationStore


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = REPO_ROOT / "annotation_app"


class AnnotationHandler(BaseHTTPRequestHandler):
    store: AnnotationStore
    adjudication_store: AdjudicationStore | None

    def _require_adjudication(self) -> AdjudicationStore:
        if self.adjudication_store is None:
            raise ValueError("当前标注包尚未启用人机差异仲裁。")
        return self.adjudication_store

    def _json(self, payload, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self._json({"error": "文件不存在"}, 404)
            return
        content = path.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            content_type
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream",
        )
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    @staticmethod
    def _parts(path: str) -> list[str]:
        return [unquote(part) for part in path.split("/") if part]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parts = self._parts(parsed.path)
        try:
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                return
            if parsed.path == "/api/assignment":
                payload = self.store.assignment_payload()
                payload["adjudication"] = (
                    self.adjudication_store.summary()
                    if self.adjudication_store is not None
                    else None
                )
                self._json(payload)
                return
            if parsed.path == "/api/adjudication":
                self._json(self._require_adjudication().summary())
                return
            if len(parts) == 3 and parts[:2] == ["api", "adjudication"]:
                self._json(self._require_adjudication().payload(parts[2]))
                return
            if len(parts) == 4 and parts[:2] == ["api", "sample"]:
                sample_id, resource = parts[2], parts[3]
                if resource == "graph":
                    self._json(self.store.graph(sample_id).to_dict())
                    return
                if resource == "screenshot":
                    self._file(
                        self.store.screenshot_path(sample_id), "image/png"
                    )
                    return
            if len(parts) == 4 and parts[:2] == ["api", "annotation"]:
                annotator, sample_id = parts[2], parts[3]
                self._json(
                    self.store.annotation(annotator, sample_id).to_dict()
                )
                return

            static_path = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
            target = (STATIC_DIR / static_path).resolve()
            if not target.is_relative_to(STATIC_DIR.resolve()):
                self._json({"error": "非法路径"}, 400)
                return
            self._file(target)
        except (KeyError, ValueError) as error:
            self._json({"error": str(error)}, 400)
        except FileNotFoundError as error:
            self._json({"error": str(error)}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        parts = self._parts(parsed.path)
        if len(parts) == 3 and parts[:2] == ["api", "adjudication"]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 20 * 1024 * 1024:
                    raise ValueError("请求体尺寸非法。")
                payload = json.loads(self.rfile.read(length))
                submit = parse_qs(parsed.query).get("submit", ["0"])[0] == "1"
                result = self._require_adjudication().save(
                    parts[2],
                    payload.get("draft", {}),
                    review_payload=payload.get("review"),
                    submit=submit,
                )
                self._json(
                    result,
                    200 if not submit or result["reviewed"] else 422,
                )
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, 400)
            return
        if len(parts) != 4 or parts[:2] != ["api", "annotation"]:
            self._json({"error": "未知接口"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 10 * 1024 * 1024:
                raise ValueError("请求体尺寸非法。")
            payload = json.loads(self.rfile.read(length))
            submit = parse_qs(parsed.query).get("submit", ["0"])[0] == "1"
            result = self.store.save(
                parts[2], parts[3], payload, submit=submit
            )
            self._json(result, 200 if not submit or result["submitted"] else 422)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)

    def log_message(self, format: str, *args) -> None:
        print(f"[annotation] {self.address_string()} {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_dir", default="data/annotations/intent_pilot_v1"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    AnnotationHandler.store = AnnotationStore(
        REPO_ROOT, REPO_ROOT / args.package_dir
    )
    package_dir = REPO_ROOT / args.package_dir
    has_adjudication = all(
        (package_dir / directory).is_dir()
        for directory in ("annotator_ai", "adjudication", "gold_drafts")
    )
    AnnotationHandler.adjudication_store = (
        AdjudicationStore(REPO_ROOT, package_dir)
        if has_adjudication
        else None
    )
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    print(f"Intent Annotator: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
