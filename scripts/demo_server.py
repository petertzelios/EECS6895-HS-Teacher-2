from __future__ import annotations

import argparse
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from llm_utils import create_llm_client
from pipeline_core import run_multi_agent
from retrieval_agent import (
    RetrievalAgent,
    default_index_dir,
    default_student_support_index_dir,
)


INDEX_DIR = default_index_dir()
STUDENT_SUPPORT_INDEX_DIR = default_student_support_index_dir()
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_MODEL = os.getenv("DEMO_DEFAULT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
FETCH_K = int(os.getenv("FETCH_K", "50"))
FINAL_K = int(os.getenv("FINAL_K", "6"))
BASE_INDEX_NAMES = [
    "curriculum_overview",
    "exam_questions",
    "exam_scoring",
]
STUDENT_SUPPORT_INDEX_NAMES = [
    "college_info",
    "time_management",
]
INDEX_NAMES = BASE_INDEX_NAMES + STUDENT_SUPPORT_INDEX_NAMES
INDEX_SOURCES = {
    name: STUDENT_SUPPORT_INDEX_DIR for name in STUDENT_SUPPORT_INDEX_NAMES
}
SUPPORTED_FORCED_AGENTS = {None, "regents_agent", "curriculum_agent", "college_support_agent"}

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
FRONTEND_CANDIDATES = [
    REPO_ROOT / "web" / "index.html",
    THIS_DIR / "index.html",
    REPO_ROOT / "index.html",
]
FRONTEND_PATH = next((path for path in FRONTEND_CANDIDATES if path.exists()), FRONTEND_CANDIDATES[0])


def openrouter_headers() -> Optional[Dict[str, str]]:
    app_title = os.getenv("OPENROUTER_APP_TITLE", "HS Teacher Retrieval Demo")
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    headers: Dict[str, str] = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-Title"] = app_title
    return headers or None


def client_for_model(model_name: str):
    model_name = (model_name or "").strip()
    lower = model_name.lower()

    if lower.startswith("deepseek"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY for DeepSeek model requests.")
        return create_llm_client(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    headers = openrouter_headers() if base_url and "openrouter.ai" in base_url else None
    if api_key:
        return create_llm_client(api_key=api_key, base_url=base_url, default_headers=headers)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return create_llm_client(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            default_headers=openrouter_headers(),
        )

    raise RuntimeError("Missing OPENAI_API_KEY / OPENROUTER_API_KEY for OpenAI-compatible model requests.")


def sanitize_evidence(evidence):
    items = []
    for row in evidence:
        items.append(
            {
                "doc_id": row.get("doc_id"),
                "candidate_id": row.get("candidate_id"),
                "retrieved_from": row.get("retrieved_from"),
                "source_file": row.get("source_file"),
                "page": row.get("page"),
                "score": row.get("score"),
                "snippet": (" ".join((row.get("text") or "").split()))[:260],
            }
        )
    return items


retrieval_agent = RetrievalAgent(
    client=None,
    index_dir=INDEX_DIR,
    embed_model_name=EMBED_MODEL_NAME,
    fetch_k=FETCH_K,
    final_k=FINAL_K,
    index_names=INDEX_NAMES,
    index_sources=INDEX_SOURCES,
)
request_lock = threading.Lock()


def run_request(
    query: str,
    model: str,
    use_rag: bool,
    forced_agent: Optional[str],
) -> Dict[str, Any]:
    client = client_for_model(model)

    with request_lock:
        retrieval_agent.client = client
        if retrieval_agent.pageindex_refiner is not None:
            retrieval_agent.pageindex_refiner.client = client

        result = run_multi_agent(
            user_query=query,
            client=client,
            retrieval_agent=retrieval_agent,
            model=model,
            routing_mode="manual" if forced_agent else "llm",
            forced_agent=forced_agent,
            fetch_k_per_query=FETCH_K,
            final_k=FINAL_K,
            use_rag=use_rag,
        )

    return {
        "query": result["query"],
        "agent": result["agent"],
        "route": result["route"],
        "use_rag": result["use_rag"],
        "answer": result["answer"],
        "retrieval_plan": result["retrieval_plan"],
        "rewrites": result["rewrites"],
        "top_evidence": sanitize_evidence(result["top_evidence"]),
    }


def frontend_html() -> bytes:
    html = FRONTEND_PATH.read_text(encoding="utf-8")
    config = {
        "defaultModel": DEFAULT_MODEL,
        "fetchK": FETCH_K,
        "finalK": FINAL_K,
        "gpuEnabled": retrieval_agent.embed_device == "cuda",
        "reasoningBackends": sorted(retrieval_agent.experimental_backends),
        "loadedIndexes": sorted(retrieval_agent.indexes.keys()),
        "missingIndexes": retrieval_agent.missing_indexes,
    }
    injected = html.replace("__DEMO_CONFIG__", json.dumps(config, ensure_ascii=False))
    return injected.encode("utf-8")


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "HSTeacherDemo/0.1"

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_html(frontend_html())
            return
        if self.path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "default_model": DEFAULT_MODEL,
                    "gpu_enabled": retrieval_agent.embed_device == "cuda",
                    "index_dir": INDEX_DIR,
                    "student_support_index_dir": STUDENT_SUPPORT_INDEX_DIR,
                    "loaded_indexes": sorted(retrieval_agent.indexes.keys()),
                    "missing_indexes": retrieval_agent.missing_indexes,
                    "frontend_path": str(FRONTEND_PATH),
                }
            )
            return
        self._send_json({"error": f"Unknown path: {self.path}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._send_json({"error": f"Unknown path: {self.path}"}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON body."}, status=HTTPStatus.BAD_REQUEST)
            return

        query = (payload.get("query") or "").strip()
        model = (payload.get("model") or DEFAULT_MODEL).strip()
        forced_agent = payload.get("forced_agent") or None
        use_rag = bool(payload.get("use_rag", True))

        if not query:
            self._send_json({"error": "Query is required."}, status=HTTPStatus.BAD_REQUEST)
            return
        if forced_agent not in SUPPORTED_FORCED_AGENTS:
            self._send_json({"error": "Unsupported forced_agent value."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            result = run_request(
                query=query,
                model=model,
                use_rag=use_rag,
                forced_agent=forced_agent,
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(result)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the terminal-style multi-agent demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Serving demo UI at http://{args.host}:{args.port}")
    print(f"Using frontend: {FRONTEND_PATH}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
