"""
GPT-powered multi-agent high-school teacher assistant.

This script is adapted from the GPT portion of:
- hs_teacher_multi_agent_notebook_v5-3.ipynb

It assumes FAISS indexes have already been built using the same embedding model
(sentence-transformers/all-MiniLM-L6-v2) and saved as:
    <index_name>.faiss
    <index_name>.meta.jsonl

Environment variables:
- OPENAI_API_KEY: required for OpenAI calls
- INDEX_DIR: optional override for the index directory
- OPENAI_MODEL: optional override for the model name
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# =========================
# Config
# =========================

INDEX_DIR = os.getenv("INDEX_DIR", "./indexes_hs_teacher_clean")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

FETCH_K = int(os.getenv("FETCH_K", "50"))
FINAL_K = int(os.getenv("FINAL_K", "6"))

INDEX_NAMES = [
    "standards_core",
    "curriculum_overview",
    "exam_questions",
    "exam_scoring",
]


# =========================
# Clients / models
# =========================

client = OpenAI()
st_model = SentenceTransformer(EMBED_MODEL_NAME)


# =========================
# Index loading
# =========================

def load_index(index_dir: str, name: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index_path = os.path.join(index_dir, f"{name}.faiss")
    meta_path = os.path.join(index_dir, f"{name}.meta.jsonl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index file: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    index = faiss.read_index(index_path)
    meta: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return index, meta


INDEXES: Dict[str, Tuple[faiss.Index, List[Dict[str, Any]]]] = {
    name: load_index(INDEX_DIR, name) for name in INDEX_NAMES
}


# =========================
# Retrieval helpers
# =========================

def embed_384(texts: List[str]) -> np.ndarray:
    vecs = st_model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def infer_subject_from_query(query: str) -> Optional[str]:
    q = query.lower()
    mapping = {
        "algebra i": "Algebra 1",
        "algebra 1": "Algebra 1",
        "algebra ii": "Algebra 2",
        "algebra 2": "Algebra 2",
        "geometry": "Geometry",
        "ela": "High School English Language Arts",
        "english language arts": "High School English Language Arts",
        "english": "High School English Language Arts",
        "chemistry": "Chemistry",
        "physics": "Physics",
        "living environment": "Living Environment",
        "biology": "Life Science: Biology",
        "global history": "Global History and Geography II",
    }
    for k, v in mapping.items():
        if k in q:
            return v
    return None


def infer_admin_from_query(query: str) -> Optional[str]:
    q = query.lower()
    patterns = [
        r"january\s+20\d{2}",
        r"august\s+20\d{2}",
        r"june\s+20\d{2}",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(0).title()
    return None


QUESTIONISH_PATTERNS = [
    r"\bPart I\b",
    r"\bQuestion\b",
    r"\([0-9]+\)",
    r"\b[A-D]\)",
    r"(?m)^\s*[0-9]{1,2}[\.)]\s",
]


def looks_like_question_text(text: str) -> bool:
    t = (text or "").replace("\n", " ")
    return any(re.search(p, t) for p in QUESTIONISH_PATTERNS)


def is_front_matter_exam(row: Dict[str, Any]) -> bool:
    src = os.path.basename(row.get("source", "")).lower()
    page = row.get("page")
    if "exam" in src and isinstance(page, int) and page <= 1:
        return not looks_like_question_text(row.get("text", ""))
    return False


def module_hint(query: str) -> Optional[str]:
    m = re.search(r"\bmodule\s*(\d+)\b", query.lower())
    return m.group(1) if m else None


def retrieve_faiss(
    query: str,
    index_name: str,
    fetch_k: int = FETCH_K,
    final_k: int = FINAL_K,
) -> List[Dict[str, Any]]:
    index, meta = INDEXES[index_name]
    if not meta or index.ntotal == 0:
        return []

    qv = embed_384([query])
    local_fetch_k = min(fetch_k, len(meta))
    if local_fetch_k <= 0:
        return []

    scores, ids = index.search(qv, local_fetch_k)

    subj = infer_subject_from_query(query)
    admin = infer_admin_from_query(query)
    mod = module_hint(query)

    cands: List[Dict[str, Any]] = []
    for sc, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        row = dict(meta[idx])
        row["score"] = float(sc)
        cands.append(row)

    def boosted_score(r: Dict[str, Any]) -> float:
        score = r["score"]
        if subj and r.get("subject") == subj:
            score += 0.15
        if admin and r.get("admin") == admin:
            score += 0.10
        if index_name == "curriculum_overview" and mod:
            src = (r.get("source") or "").lower().replace("\\", "/")
            base = os.path.basename(src)
            if f"-m{mod}-" in base or f"/module {mod}/" in src:
                score += 0.25
        if index_name == "exam_questions" and r.get("doc_type") == "exam":
            score += 0.10
        return score

    cands.sort(key=boosted_score, reverse=True)

    if index_name == "exam_questions":
        cands = [r for r in cands if not is_front_matter_exam(r)]
        cands.sort(
            key=lambda r: (looks_like_question_text(r.get("text", "")), boosted_score(r)),
            reverse=True,
        )

    out: List[Dict[str, Any]] = []
    seen = set()
    for r in cands:
        if r["doc_id"] in seen:
            continue
        seen.add(r["doc_id"])
        out.append(r)
        if len(out) >= final_k:
            break
    return out


# =========================
# LLM retrieval helpers
# =========================

def rewrite_queries(
    user_query: str,
    agent_name: str,
    index_names: List[str],
    model: str = OPENAI_MODEL,
) -> List[str]:
    prompt = f"""
Rewrite the user request into 4 short search queries for PDF retrieval.

Agent: {agent_name}
Candidate indexes: {", ".join(index_names)}

Rules:
- Return exactly 4 lines
- No numbering
- Each line <= 14 words
- Preserve subject / grade / exam / module details when present
- Prefer retrieval-oriented wording

User request: {user_query}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "low"},
    )
    lines = [x.strip() for x in resp.output_text.splitlines() if x.strip()]
    if len(lines) >= 4:
        return lines[:4]
    return [user_query] * 4


def rerank_with_gpt(
    user_query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = FINAL_K,
    model: str = OPENAI_MODEL,
) -> List[Dict[str, Any]]:
    packed = []
    for c in candidates[:60]:
        packed.append(
            {
                "doc_id": c.get("doc_id"),
                "pdf": os.path.basename(c.get("source", "")),
                "page": c.get("page"),
                "subject": c.get("subject"),
                "admin": c.get("admin"),
                "snippet": (c.get("text", "")[:280] or "").replace("\n", " "),
            }
        )

    prompt = f"""
You are selecting the best evidence chunks for answering a teacher-assistant query.

Return ONLY a JSON array of doc_id strings from best to worst.
Choose at most {top_n} ids.

User query:
{user_query}

Candidates:
{json.dumps(packed, ensure_ascii=False)}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "low"},
    )

    try:
        chosen_ids = json.loads(resp.output_text.strip())
        chosen_ids = [x for x in chosen_ids if isinstance(x, str)]
    except Exception:
        chosen_ids = []

    by_id = {c["doc_id"]: c for c in candidates}
    reranked = [by_id[d] for d in chosen_ids if d in by_id]
    return reranked[:top_n] if reranked else candidates[:top_n]


# =========================
# Agent specifications
# =========================

AGENT_SPECS = {
    "regents_agent": {
        "description": (
            "Handles Regents-style practice questions, exam questions, scoring guides, "
            "rubrics, and answer explanations."
        ),
        "indexes": ["exam_questions", "exam_scoring"],
    },
    "curriculum_agent": {
        "description": (
            "Handles NYS curriculum teaching support, standards alignment, module guidance, "
            "lesson framing, and teaching explanations."
        ),
        "indexes": ["standards_core", "curriculum_overview"],
    },
    "study_skills_agent": {
        "description": (
            "Handles time management, study habits, and college readiness. "
            "Not implemented in this script."
        ),
        "indexes": [],
    },
}


def keyword_router_fallback(query: str) -> str:
    q = query.lower()

    regents_terms = [
        "regents",
        "practice question",
        "practice problems",
        "multiple choice",
        "constructed response",
        "rubric",
        "rating guide",
        "full credit",
        "scoring",
        "model response",
        "exam",
    ]
    curriculum_terms = [
        "standard",
        "standards",
        "curriculum",
        "module",
        "lesson",
        "teach",
        "explain",
        "how do i teach",
        "nys",
        "next generation",
        "ela",
        "algebra",
        "geometry",
    ]
    study_terms = [
        "time management",
        "study plan",
        "organization",
        "planner",
        "college essay",
        "college application",
        "study skills",
    ]

    if any(t in q for t in regents_terms):
        return "regents_agent"
    if any(t in q for t in study_terms):
        return "study_skills_agent"
    if any(t in q for t in curriculum_terms):
        return "curriculum_agent"
    return "curriculum_agent"


def route_with_llm(query: str, model: str = OPENAI_MODEL) -> Dict[str, Any]:
    prompt = f"""
Route the user query to exactly one agent.

Available agents:
- regents_agent: Regents-style practice questions, exam questions, scoring guides, rubrics, answer explanations
- curriculum_agent: NYS curriculum teaching, standards, modules, lessons, concept explanations
- study_skills_agent: time management, study skills, college readiness (not implemented)

Return ONLY valid JSON with keys:
- agent
- rationale
- confidence

User query:
{query}
""".strip()

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "low"},
        )
        data = json.loads(resp.output_text.strip())
        if data.get("agent") not in AGENT_SPECS:
            raise ValueError("Bad agent")
        return data
    except Exception:
        fallback = keyword_router_fallback(query)
        return {
            "agent": fallback,
            "rationale": "Fallback keyword router used because LLM router output was unavailable.",
            "confidence": 0.4,
        }


# =========================
# Agent retrieval pipeline
# =========================

def gather_agent_evidence(
    user_query: str,
    agent_name: str,
    fetch_k_per_query: int = FETCH_K,
    final_k: int = FINAL_K,
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    spec = AGENT_SPECS[agent_name]
    index_names = [
        name
        for name in spec["indexes"]
        if name in INDEXES and INDEXES[name][0].ntotal > 0 and len(INDEXES[name][1]) > 0
    ]

    if not index_names:
        return {
            "rewrites": [],
            "candidate_pool": [],
            "top_evidence": [],
        }

    rewrites = rewrite_queries(user_query, agent_name, index_names, model=model)

    pool = []
    seen = set()

    per_index_final = max(final_k, 4)
    for rq in rewrites:
        for idx_name in index_names:
            hits = retrieve_faiss(
                rq,
                index_name=idx_name,
                fetch_k=fetch_k_per_query,
                final_k=per_index_final,
            )
            for h in hits:
                if h["doc_id"] not in seen:
                    seen.add(h["doc_id"])
                    h["retrieved_from"] = idx_name
                    pool.append(h)

    pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    pool = pool[: min(len(pool), max(25, final_k * 5))]

    top = rerank_with_gpt(user_query, pool, top_n=final_k, model=model) if pool else []

    return {
        "rewrites": rewrites,
        "candidate_pool": pool,
        "top_evidence": top,
    }


# =========================
# Answer generation
# =========================

def format_context(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "NO EVIDENCE FOUND."
    chunks = []
    for i, e in enumerate(evidence, start=1):
        chunks.append(
            f"""CHUNK {i}
DOC_ID: {e["doc_id"]}
INDEX: {e.get("retrieved_from")}
PDF: {os.path.basename(e.get("source", ""))}
PAGE: {e.get("page")}
SUBJECT: {e.get("subject")}
ADMIN: {e.get("admin")}
TEXT:
{e.get("text", "")}"""
        )
    return "\n\n".join(chunks)


def answer_with_regents_agent(
    user_query: str,
    evidence: List[Dict[str, Any]],
    model: str = OPENAI_MODEL,
) -> str:
    context = format_context(evidence)
    prompt = f"""
You are the Regents Agent for a high school teacher assistant.

Your job:
- Answer Regents-related questions
- Create Regents-style practice only when the user asks for practice
- Use ONLY the evidence when making factual claims about real NYS exams, scoring guides, or released materials
- If the evidence is insufficient, say so clearly
- Never claim a generated question is an official released Regents question unless the evidence shows that

Output style:
- Clear teacher-friendly answer
- Use short sections or bullets when helpful
- End with a brief "Sources used" line listing DOC_IDs

User request:
{user_query}

Evidence:
{context}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "low"},
    )
    return resp.output_text.strip()


def answer_with_curriculum_agent(
    user_query: str,
    evidence: List[Dict[str, Any]],
    model: str = OPENAI_MODEL,
) -> str:
    context = format_context(evidence)
    prompt = f"""
You are the Curriculum Agent for a high school teacher assistant.

Your job:
- Help teach using NYS math / ELA curriculum and standards
- Use ONLY the evidence for factual claims about curriculum structure, modules, standards, or official materials
- If the user asks for a teaching explanation, you may explain in your own words, but keep it aligned to the evidence
- If evidence is weak or incomplete, say what you can and what is missing

Output style:
- Practical and teacher-friendly
- Prioritize teaching steps, alignment, lesson framing, and student-facing explanation when relevant
- End with a brief "Sources used" line listing DOC_IDs

User request:
{user_query}

Evidence:
{context}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "low"},
    )
    return resp.output_text.strip()


def answer_with_unimplemented_agent(user_query: str) -> str:
    return (
        "This query routed to the study-skills / time-management agent, "
        "but that agent is not implemented in this script yet."
    )


# =========================
# Orchestration
# =========================

def ask_multi_agent(
    user_query: str,
    routing_mode: str = "llm",
    forced_agent: Optional[str] = None,
    fetch_k_per_query: int = FETCH_K,
    final_k: int = FINAL_K,
    model: str = OPENAI_MODEL,
    show_debug: bool = False,
) -> Dict[str, Any]:
    if forced_agent is not None:
        chosen_agent = forced_agent
        route_info = {
            "agent": forced_agent,
            "rationale": "Agent was manually forced by the caller.",
            "confidence": 1.0,
        }
    elif routing_mode == "manual":
        raise ValueError("For manual mode, pass forced_agent explicitly.")
    else:
        route_info = route_with_llm(user_query, model=model)
        chosen_agent = route_info["agent"]

    retrieval = gather_agent_evidence(
        user_query=user_query,
        agent_name=chosen_agent,
        fetch_k_per_query=fetch_k_per_query,
        final_k=final_k,
        model=model,
    )

    if chosen_agent == "regents_agent":
        answer = answer_with_regents_agent(user_query, retrieval["top_evidence"], model=model)
    elif chosen_agent == "curriculum_agent":
        answer = answer_with_curriculum_agent(user_query, retrieval["top_evidence"], model=model)
    else:
        answer = answer_with_unimplemented_agent(user_query)

    result = {
        "query": user_query,
        "route": route_info,
        "agent": chosen_agent,
        "rewrites": retrieval["rewrites"],
        "top_evidence": retrieval["top_evidence"],
        "answer": answer,
    }

    print("=" * 110)
    print("USER QUERY:", user_query)
    print("ROUTED TO:", chosen_agent)
    print("RATIONALE:", route_info.get("rationale"))
    print("CONFIDENCE:", route_info.get("confidence"))
    print("=" * 110)
    print()
    print(answer)

    if show_debug:
        print("\n--- Rewrites ---")
        for i, rq in enumerate(retrieval["rewrites"], start=1):
            print(f"{i}. {rq}")

        print("\n--- Top Evidence ---")
        for e in retrieval["top_evidence"]:
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
    parser.add_argument("--forced-agent", default=None, choices=[None, "regents_agent", "curriculum_agent", "study_skills_agent"])
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
