"""
GPT-powered multi-agent high-school teacher assistant.

This script is adapted from the GPT portion of:
- hs_teacher_multi_agent_notebook_v5-3.ipynb

It assumes FAISS indexes have already been built using the same embedding model
(sentence-transformers/all-MiniLM-L6-v2) and saved as:
    <index_name>.faiss
    <index_name>.meta.jsonl

Environment variables:
- OPENAI_API_KEY or OPENROUTER_API_KEY: required for LLM calls
- INDEX_DIR: optional override for the index directory
- OPENAI_MODEL: optional override for the model name
"""

from __future__ import annotations

import argparse
import json
import os

from llm_utils import build_llm_client
from pipeline_core import run_multi_agent
from retrieval_agent import RetrievalAgent, default_index_dir


# =========================
# Config
# =========================

INDEX_DIR = default_index_dir()
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

FETCH_K = int(os.getenv("FETCH_K", "50"))
FINAL_K = int(os.getenv("FINAL_K", "6"))

INDEX_NAMES = [
    "curriculum_overview",
    "exam_questions",
    "exam_scoring",
]


# =========================
# Clients / models
# =========================

client = build_llm_client()
retrieval_agent = RetrievalAgent(
    client=client,
    index_dir=INDEX_DIR,
    embed_model_name=EMBED_MODEL_NAME,
    fetch_k=FETCH_K,
    final_k=FINAL_K,
    index_names=INDEX_NAMES,
)


def ask_multi_agent(
    user_query: str,
    routing_mode: str = "llm",
    forced_agent: str | None = None,
    fetch_k_per_query: int = FETCH_K,
    final_k: int = FINAL_K,
    model: str = OPENAI_MODEL,
    show_debug: bool = False,
):
    result = run_multi_agent(
        user_query=user_query,
        client=client,
        retrieval_agent=retrieval_agent,
        model=model,
        routing_mode=routing_mode,
        forced_agent=forced_agent,
        fetch_k_per_query=fetch_k_per_query,
        final_k=final_k,
        use_rag=True,
    )

    print("=" * 110)
    print("USER QUERY:", user_query)
    print("ROUTED TO:", result["agent"])
    print("RATIONALE:", result["route"].get("rationale"))
    print("CONFIDENCE:", result["route"].get("confidence"))
    print("=" * 110)
    print()
    print(result["answer"])

    if show_debug:
        print("\n--- Retrieval Plan ---")
        print(json.dumps(result["retrieval_plan"], ensure_ascii=False, indent=2))

        print("\n--- Rewrites ---")
        for i, rq in enumerate(result["rewrites"], start=1):
            print(f"{i}. {rq}")

        print("\n--- Top Evidence ---")
        for e in result["top_evidence"]:
            snippet = (e.get("text", "")[:220] or "").replace("\n", " ")
            print(
                f"- {e['doc_id']} | idx={e.get('retrieved_from')} | "
                f"page={e.get('page')} | score={e.get('score', 0):.3f}"
            )
            print("  ", snippet, "...")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GPT multi-agent HS teacher assistant.")
    parser.add_argument("query", nargs="?", help="User query to answer.")
    parser.add_argument("--routing-mode", default="llm", choices=["llm", "manual"])
    parser.add_argument(
        "--forced-agent",
        default=None,
        choices=[None, "regents_agent", "curriculum_agent", "study_skills_agent"],
    )
    parser.add_argument("--show-debug", action="store_true")
    parser.add_argument("--model", default=OPENAI_MODEL)
    args = parser.parse_args()

    if not args.query:
        parser.error("Please pass a query string.")

    ask_multi_agent(
        user_query=args.query,
        routing_mode=args.routing_mode,
        forced_agent=args.forced_agent,
        model=args.model,
        show_debug=args.show_debug,
    )


if __name__ == "__main__":
    main()
