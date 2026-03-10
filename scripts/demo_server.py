from __future__ import annotations

import argparse
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    config = {
        "defaultModel": DEFAULT_MODEL,
        "fetchK": FETCH_K,
        "finalK": FINAL_K,
        "gpuEnabled": retrieval_agent.embed_device == "cuda",
        "reasoningBackends": sorted(retrieval_agent.experimental_backends),
        "loadedIndexes": sorted(retrieval_agent.indexes.keys()),
        "missingIndexes": retrieval_agent.missing_indexes,
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HS Teacher Retrieval Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121933;
      --panel-2: #182142;
      --text: #edf2ff;
      --muted: #aab7e8;
      --accent: #7aa2ff;
      --accent-2: #9a7cff;
      --border: #2a3566;
      --success: #27c281;
      --danger: #ff6b6b;
      --warning: #ffb454;
      --code-bg: #0f1530;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #0a0f1f 0%, #0d1330 100%);
      color: var(--text);
    }}

    .container {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}

    .header {{
      margin-bottom: 20px;
    }}

    .title {{
      font-size: 28px;
      font-weight: 800;
      letter-spacing: 0.2px;
      margin: 0 0 8px;
    }}

    .subtitle {{
      color: var(--muted);
      margin: 0;
      line-height: 1.5;
    }}

    .grid {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
      align-items: start;
    }}

    .panel {{
      background: rgba(18, 25, 51, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.28);
      overflow: hidden;
    }}

    .panel-header {{
      padding: 16px 18px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.02);
      font-weight: 700;
    }}

    .panel-body {{
      padding: 16px 18px 18px;
    }}

    label {{
      display: block;
      font-size: 13px;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    textarea, input, select {{
      width: 100%;
      border: 1px solid var(--border);
      background: #0d1430;
      color: var(--text);
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      outline: none;
    }}

    textarea {{
      min-height: 160px;
      resize: vertical;
      line-height: 1.5;
    }}

    .row {{
      display: grid;
      gap: 14px;
      margin-bottom: 14px;
    }}

    .checkbox-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 14px 0 18px;
      color: var(--text);
    }}

    .checkbox-row input {{
      width: auto;
      transform: scale(1.1);
    }}

    .btn {{
      width: 100%;
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      font: inherit;
      font-weight: 800;
      color: white;
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      box-shadow: 0 10px 20px rgba(122, 162, 255, 0.25);
    }}

    .btn:disabled {{
      opacity: 0.65;
      cursor: not-allowed;
    }}

    .meta {{
      margin-top: 16px;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.6;
    }}

    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.03);
      margin: 4px 6px 0 0;
      color: var(--text);
      font-size: 12px;
    }}

    .answer-wrap {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}

    .status {{
      font-size: 14px;
      color: var(--muted);
    }}

    .answer-card {{
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
    }}

    .answer-card-header {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .answer-card-body {{
      padding: 18px;
    }}

    .assistant-answer {{
      line-height: 1.72;
      font-size: 16px;
      color: var(--text);
      white-space: normal;
      overflow-wrap: anywhere;
    }}

    .assistant-answer h1,
    .assistant-answer h2,
    .assistant-answer h3,
    .assistant-answer h4 {{
      margin-top: 1.1em;
      margin-bottom: 0.5em;
    }}

    .assistant-answer p {{
      margin: 0.6em 0;
    }}

    .assistant-answer ul,
    .assistant-answer ol {{
      padding-left: 1.4rem;
      margin: 0.6em 0;
    }}

    .assistant-answer li {{
      margin: 0.35em 0;
    }}

    .assistant-answer strong {{
      color: #ffffff;
    }}

    .assistant-answer code {{
      background: var(--code-bg);
      border: 1px solid var(--border);
      padding: 0.15em 0.4em;
      border-radius: 6px;
      font-size: 0.95em;
    }}

    .assistant-answer pre {{
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      overflow: auto;
    }}

    .assistant-answer blockquote {{
      margin: 0.8em 0;
      padding: 0.2em 1em;
      border-left: 4px solid var(--accent);
      color: #dbe4ff;
      background: rgba(255,255,255,0.02);
      border-radius: 0 10px 10px 0;
    }}

    .assistant-answer table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1em 0;
    }}

    .assistant-answer th,
    .assistant-answer td {{
      border: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
    }}

    .assistant-answer .katex-display {{
      overflow-x: auto;
      overflow-y: hidden;
      padding: 0.65rem 0.1rem;
      margin: 0.6rem 0;
    }}

    .details {{
      background: rgba(255,255,255,0.02);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }}

    .details summary {{
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 700;
    }}

    .details-content {{
      padding: 0 16px 16px;
    }}

    pre.json {{
      margin: 0;
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      white-space: pre-wrap;
      word-break: break-word;
      color: #dbe4ff;
      line-height: 1.55;
      font-size: 13px;
    }}

    .error {{
      color: #ffd0d0;
      background: rgba(255, 107, 107, 0.12);
      border: 1px solid rgba(255, 107, 107, 0.35);
      padding: 12px 14px;
      border-radius: 12px;
    }}

    .loading {{
      color: #dfe7ff;
      background: rgba(122, 162, 255, 0.12);
      border: 1px solid rgba(122, 162, 255, 0.35);
      padding: 12px 14px;
      border-radius: 12px;
    }}

    @media (max-width: 900px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}

      .container {{
        padding: 16px;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1 class="title">HS Teacher Retrieval Demo</h1>
      <p class="subtitle">
        Ask curriculum, Regents, or college-support questions. Markdown and LaTeX equations are rendered in the answer panel.
      </p>
    </div>

    <div class="grid">
      <section class="panel">
        <div class="panel-header">Controls</div>
        <div class="panel-body">
          <div class="row">
            <div>
              <label for="query">Query</label>
              <textarea id="query" placeholder="Example: Give me a Regents-style practice chemistry balancing equation question."></textarea>
            </div>
          </div>

          <div class="row">
            <div>
              <label for="model">Model</label>
              <input id="model" />
            </div>
            <div>
              <label for="forced_agent">Force Agent</label>
              <select id="forced_agent">
                <option value="">Auto route</option>
                <option value="regents_agent">regents_agent</option>
                <option value="curriculum_agent">curriculum_agent</option>
                <option value="college_support_agent">college_support_agent</option>
              </select>
            </div>
          </div>

          <div class="checkbox-row">
            <input type="checkbox" id="use_rag" checked>
            <label for="use_rag" style="margin:0; text-transform:none; letter-spacing:normal; font-size:14px; font-weight:600; color:var(--text);">
              Use retrieval (RAG)
            </label>
          </div>

          <button id="sendBtn" class="btn">Run</button>

          <div class="meta" id="meta"></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">Answer</div>
        <div class="panel-body">
          <div class="answer-wrap">
            <div class="status" id="status">Ready.</div>
            <div id="messageArea"></div>

            <div class="answer-card">
              <div class="answer-card-header">
                <div>
                  <strong id="agentLine">Agent: —</strong>
                </div>
                <div id="routeLine" class="status">Route confidence: —</div>
              </div>
              <div class="answer-card-body">
                <div id="answer"></div>
              </div>
            </div>

            <details class="details">
              <summary>Debug details</summary>
              <div class="details-content">
                <h4>Top Evidence</h4>
                <pre id="evidence" class="json"></pre>
                <h4>Retrieval Plan</h4>
                <pre id="retrieval_plan" class="json"></pre>
                <h4>Rewrites</h4>
                <pre id="rewrites" class="json"></pre>
              </div>
            </details>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const DEMO_CONFIG = {json.dumps(config, ensure_ascii=False)};

    const queryEl = document.getElementById("query");
    const modelEl = document.getElementById("model");
    const forcedAgentEl = document.getElementById("forced_agent");
    const useRagEl = document.getElementById("use_rag");
    const sendBtn = document.getElementById("sendBtn");

    const statusEl = document.getElementById("status");
    const messageAreaEl = document.getElementById("messageArea");
    const answerEl = document.getElementById("answer");
    const agentLineEl = document.getElementById("agentLine");
    const routeLineEl = document.getElementById("routeLine");
    const evidenceEl = document.getElementById("evidence");
    const retrievalPlanEl = document.getElementById("retrieval_plan");
    const rewritesEl = document.getElementById("rewrites");
    const metaEl = document.getElementById("meta");

    modelEl.value = DEMO_CONFIG.defaultModel || "gpt-4o-mini";

    metaEl.innerHTML = `
      <div><strong>Loaded indexes:</strong></div>
      ${(DEMO_CONFIG.loadedIndexes || []).map(x => `<span class="pill">${{escapeHtml(x)}}</span>`).join("") || "<span class='pill'>None</span>"}
      <div style="margin-top:10px;"><strong>Reasoning backends:</strong></div>
      ${(DEMO_CONFIG.reasoningBackends || []).map(x => `<span class="pill">${{escapeHtml(x)}}</span>`).join("") || "<span class='pill'>None</span>"}
    `;

    function escapeHtml(text) {{
      return String(text ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }}

    function showMessage(kind, text) {{
      const cls = kind === "error" ? "error" : "loading";
      messageAreaEl.innerHTML = `<div class="${{cls}}">${{escapeHtml(text)}}</div>`;
    }}

    function clearMessage() {{
      messageAreaEl.innerHTML = "";
    }}

    function renderAssistantAnswer(text) {{
      const safeText = text || "";
      const html = marked.parse(safeText);
      answerEl.innerHTML = `<div class="assistant-answer">${{html}}</div>`;

      if (window.renderMathInElement) {{
        renderMathInElement(answerEl, {{
          delimiters: [
            {{ left: "$$", right: "$$", display: true }},
            {{ left: "\\\\[", right: "\\\\]", display: true }},
            {{ left: "$", right: "$", display: false }},
            {{ left: "\\\\(", right: "\\\\)", display: false }}
          ],
          throwOnError: false
        }});
      }}
    }}

    async function sendQuery() {{
      const query = queryEl.value.trim();
      const model = modelEl.value.trim();
      const forced_agent = forcedAgentEl.value || null;
      const use_rag = !!useRagEl.checked;

      if (!query) {{
        showMessage("error", "Please enter a query.");
        return;
      }}

      sendBtn.disabled = true;
      statusEl.textContent = "Running...";
      showMessage("loading", "Generating response...");
      agentLineEl.textContent = "Agent: —";
      routeLineEl.textContent = "Route confidence: —";
      answerEl.innerHTML = "";
      evidenceEl.textContent = "";
      retrievalPlanEl.textContent = "";
      rewritesEl.textContent = "";

      try {{
        const resp = await fetch("/api/chat", {{
          method: "POST",
          headers: {{
            "Content-Type": "application/json"
          }},
          body: JSON.stringify({{
            query,
            model,
            forced_agent,
            use_rag
          }})
        }});

        const data = await resp.json();

        if (!resp.ok) {{
          throw new Error(data.error || `HTTP ${{resp.status}}`);
        }}

        clearMessage();
        statusEl.textContent = "Done.";
        agentLineEl.textContent = `Agent: ${{data.agent || "—"}}`;

        const conf = data.route && typeof data.route.confidence !== "undefined"
          ? data.route.confidence
          : "—";
        const rationale = data.route && data.route.rationale ? data.route.rationale : "";
        routeLineEl.textContent = `Route confidence: ${{conf}}${{rationale ? " | " + rationale : ""}}`;

        renderAssistantAnswer(data.answer || "");

        evidenceEl.textContent = JSON.stringify(data.top_evidence || [], null, 2);
        retrievalPlanEl.textContent = JSON.stringify(data.retrieval_plan || {{}}, null, 2);
        rewritesEl.textContent = JSON.stringify(data.rewrites || [], null, 2);
      }} catch (err) {{
        statusEl.textContent = "Failed.";
        showMessage("error", err.message || "Unknown error");
      }} finally {{
        sendBtn.disabled = false;
      }}
    }}

    sendBtn.addEventListener("click", sendQuery);

    queryEl.addEventListener("keydown", (event) => {{
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {{
        sendQuery();
      }}
    }});

    renderAssistantAnswer("Ask a question on the left.\\\\n\\\\nFor best equation rendering, the model should return LaTeX like `\\\\(x^2 + 1\\\\)` or `\\\\[\\\\text{C}_3\\\\text{H}_8 + 5\\\\text{O}_2 \\\\rightarrow 3\\\\text{CO}_2 + 4\\\\text{H}_2\\\\text{O}\\\\]`.");
  </script>
</body>
</html>
"""
    return html.encode("utf-8")


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
    print(f"Serving demo on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\nShutting down...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()