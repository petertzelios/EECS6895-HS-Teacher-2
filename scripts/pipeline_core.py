from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from llm_utils import complete_text


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
            "Handles NYS curriculum teaching support, module guidance, "
            "lesson framing, and teaching explanations."
        ),
        "indexes": ["curriculum_overview"],
    },
    "study_skills_agent": {
        "description": (
            "Handles time management, study habits, and college readiness. "
            "Not implemented in this project."
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

    if any(term in q for term in regents_terms):
        return "regents_agent"
    if any(term in q for term in study_terms):
        return "study_skills_agent"
    if any(term in q for term in curriculum_terms):
        return "curriculum_agent"
    return "curriculum_agent"


def route_with_llm(query: str, client, model: str) -> Dict[str, Any]:
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
        data = json.loads(
            complete_text(client, prompt, model=model, reasoning_effort="low", temperature=0.0)
        )
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
    client,
    model: str,
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

    return complete_text(client, prompt, model=model, reasoning_effort="low", temperature=0.0)


def answer_with_curriculum_agent(
    user_query: str,
    evidence: List[Dict[str, Any]],
    client,
    model: str,
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

    return complete_text(client, prompt, model=model, reasoning_effort="low", temperature=0.0)


def answer_without_rag(
    user_query: str,
    agent_name: str,
    client,
    model: str,
) -> str:
    if agent_name == "regents_agent":
        instructions = """
You are answering a Regents-related question WITHOUT retrieval or source documents.

Rules:
- Use your own prior knowledge only
- If you are unsure, say so clearly
- Do not pretend you saw official documents
- If you mention a likely official source, label it explicitly as a guess
- End with one line: Likely source guess: ...
""".strip()
    elif agent_name == "curriculum_agent":
        instructions = """
You are answering a curriculum-related question WITHOUT retrieval or source documents.

Rules:
- Use your own prior knowledge only
- If you are unsure, say so clearly
- Do not pretend you saw official curriculum files
- If you mention a likely official source, label it explicitly as a guess
- End with one line: Likely source guess: ...
""".strip()
    else:
        return (
            "This query routed to the study-skills / time-management agent, "
            "but that agent is not implemented in this project.\n"
            "Likely source guess: none"
        )

    prompt = f"""
{instructions}

User request:
{user_query}
""".strip()
    return complete_text(client, prompt, model=model, reasoning_effort="low", temperature=0.0)


def answer_for_agent(
    user_query: str,
    agent_name: str,
    evidence: List[Dict[str, Any]],
    client,
    model: str,
    use_rag: bool,
) -> str:
    if not use_rag:
        return answer_without_rag(user_query, agent_name, client, model)
    if agent_name == "regents_agent":
        return answer_with_regents_agent(user_query, evidence, client, model)
    if agent_name == "curriculum_agent":
        return answer_with_curriculum_agent(user_query, evidence, client, model)
    return (
        "This query routed to the study-skills / time-management agent, "
        "but that agent is not implemented in this project."
    )


def run_multi_agent(
    user_query: str,
    client,
    retrieval_agent,
    model: str,
    routing_mode: str = "llm",
    forced_agent: Optional[str] = None,
    fetch_k_per_query: int = 50,
    final_k: int = 6,
    use_rag: bool = True,
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
        route_info = route_with_llm(user_query, client=client, model=model)
        chosen_agent = route_info["agent"]

    if use_rag and retrieval_agent is not None:
        retrieval = retrieval_agent.run(
            user_query=user_query,
            agent_name=chosen_agent,
            index_names=AGENT_SPECS[chosen_agent]["indexes"],
            model=model,
            fetch_k_per_query=fetch_k_per_query,
            final_k=final_k,
        )
    else:
        retrieval = {
            "retrieval_plan": {
                "indexes": [],
                "subject_aliases": [],
                "exam_admin": None,
                "module": None,
                "grade": None,
            },
            "rewrites": [],
            "candidate_pool": [],
            "top_evidence": [],
        }

    answer = answer_for_agent(
        user_query=user_query,
        agent_name=chosen_agent,
        evidence=retrieval["top_evidence"],
        client=client,
        model=model,
        use_rag=use_rag,
    )

    return {
        "query": user_query,
        "route": route_info,
        "agent": chosen_agent,
        "use_rag": use_rag,
        "retrieval_plan": retrieval["retrieval_plan"],
        "rewrites": retrieval["rewrites"],
        "top_evidence": retrieval["top_evidence"],
        "answer": answer,
    }
