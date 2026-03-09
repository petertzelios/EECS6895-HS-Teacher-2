from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from llm_utils import complete_text, create_llm_client
from pipeline_core import run_multi_agent
from retrieval_agent import RetrievalAgent, default_index_dir


EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = default_index_dir()
FETCH_K = int(os.getenv("FETCH_K", "50"))
FINAL_K = int(os.getenv("FINAL_K", "6"))

OPENAI_MODEL = os.getenv("EVAL_OPENAI_MODEL", "gpt-4o-mini")
DEEPSEEK_MODEL = os.getenv("EVAL_DEEPSEEK_MODEL", "deepseek-chat")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
BENCHMARK_PATH = Path(os.getenv("EVAL_BENCHMARK_PATH", "eval/benchmark_cases.json"))


@dataclass
class BenchmarkCase:
    case_id: str
    agent: str
    query: str
    reference_answer: str
    expected_keyword_groups: List[List[str]]
    expected_source_patterns: List[str]
    expected_source_hints: List[str]
    expected_indexes: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=data["id"],
            agent=data["agent"],
            query=data["query"],
            reference_answer=data["reference_answer"],
            expected_keyword_groups=data.get("expected_keyword_groups", []),
            expected_source_patterns=data.get("expected_source_patterns", []),
            expected_source_hints=data.get("expected_source_hints", []),
            expected_indexes=data.get("expected_indexes", []),
        )


@dataclass
class SystemConfig:
    system_id: str
    provider: str
    model: str
    use_rag: bool
    api_key: str
    base_url: Optional[str] = None
    default_headers: Optional[Dict[str, str]] = None


def load_benchmark(path: Path) -> List[BenchmarkCase]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [BenchmarkCase.from_dict(item) for item in data]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keyword_coverage(answer: str, keyword_groups: List[List[str]]) -> float:
    if not keyword_groups:
        return 0.0
    haystack = normalize_text(answer)
    hits = 0
    for group in keyword_groups:
        if any(normalize_text(term) in haystack for term in group):
            hits += 1
    return hits / len(keyword_groups)


def answer_source_guess_score(answer: str, source_hints: List[str], source_patterns: List[str]) -> float:
    hints = list(source_hints) + list(source_patterns)
    if not hints:
        return 0.0
    haystack = normalize_text(answer)
    hits = 0
    for hint in hints:
        if normalize_text(hint) in haystack:
            hits += 1
    return min(1.0, hits / max(1, len(hints)))


def retrieval_alignment_score(result: Dict[str, Any], case: BenchmarkCase) -> Optional[float]:
    if not result.get("use_rag"):
        return None

    evidence = result.get("top_evidence", [])
    if not evidence:
        return 0.0

    source_hits = []
    for pattern in case.expected_source_patterns:
        normalized = normalize_text(pattern)
        matched = False
        for item in evidence:
            candidates = [
                item.get("source_file", ""),
                os.path.basename(item.get("source", "")),
                item.get("doc_id", ""),
            ]
            if any(normalized in normalize_text(candidate) for candidate in candidates):
                matched = True
                break
        source_hits.append(1.0 if matched else 0.0)

    index_hits = []
    for index_name in case.expected_indexes:
        matched = any(item.get("retrieved_from") == index_name for item in evidence)
        index_hits.append(1.0 if matched else 0.0)

    parts = []
    if source_hits:
        parts.append(mean(source_hits))
    if index_hits:
        parts.append(mean(index_hits))
    return mean(parts) if parts else 0.0


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in text")
    return json.loads(match.group(0))


def judge_answer(
    case: BenchmarkCase,
    result: Dict[str, Any],
    judge_client,
    judge_model: str,
    keyword_score: float,
    retrieval_score: Optional[float],
) -> Dict[str, Any]:
    def heuristic_judge(reason: str) -> Dict[str, Any]:
        heuristic_correctness = round(keyword_score * 5, 2)
        if result.get("use_rag"):
            heuristic_grounding = round((retrieval_score or 0.0) * 5, 2)
        else:
            answer_text = normalize_text(result.get("answer", ""))
            heuristic_grounding = 3.0 if ("guess" in answer_text or "unsure" in answer_text) else 1.5
        heuristic_source = round(answer_source_guess_score(
            result.get("answer", ""),
            case.expected_source_hints,
            case.expected_source_patterns,
        ) * 5, 2)
        return {
            "correctness": heuristic_correctness,
            "grounding": heuristic_grounding,
            "source_accuracy": heuristic_source,
            "rationale": reason,
        }

    if judge_client is None:
        return heuristic_judge(
            "Fallback heuristic judge used because no OpenAI judge client was available."
        )

    evidence = result.get("top_evidence", [])[:4]
    evidence_summary = []
    for item in evidence:
        evidence_summary.append(
            {
                "doc_id": item.get("doc_id"),
                "index": item.get("retrieved_from"),
                "source_file": item.get("source_file") or os.path.basename(item.get("source", "")),
                "page": item.get("page"),
                "snippet": (item.get("text", "")[:240] or "").replace("\n", " "),
            }
        )

    prompt = f"""
You are grading a benchmark answer for a high-school teacher assistant.

Return ONLY valid JSON with keys:
- correctness
- grounding
- source_accuracy
- rationale

Rubric for each numeric score:
- 0 = completely wrong
- 1 = mostly wrong
- 2 = weak / partially correct
- 3 = mixed
- 4 = strong
- 5 = excellent

How to score:
- correctness: factual accuracy against the reference answer
- grounding: for RAG runs, whether the answer stays aligned to the retrieved evidence; for no-RAG runs, whether the answer is appropriately cautious instead of inventing evidence
- source_accuracy: whether cited or guessed sources match the expected source type/file hints

Benchmark case:
{json.dumps({
    "id": case.case_id,
    "query": case.query,
    "reference_answer": case.reference_answer,
    "expected_keyword_groups": case.expected_keyword_groups,
    "expected_source_patterns": case.expected_source_patterns,
    "expected_source_hints": case.expected_source_hints,
    "expected_indexes": case.expected_indexes,
}, ensure_ascii=False)}

System output:
{json.dumps({
    "system": result.get("system_id"),
    "provider": result.get("provider"),
    "model": result.get("model"),
    "use_rag": result.get("use_rag"),
    "answer": result.get("answer"),
    "top_evidence": evidence_summary,
    "keyword_coverage": keyword_score,
    "retrieval_alignment": retrieval_score,
}, ensure_ascii=False)}
""".strip()

    try:
        judged = extract_json_object(
            complete_text(
                judge_client,
                prompt,
                model=judge_model,
                reasoning_effort="low",
                temperature=0.0,
            )
        )
        return {
            "correctness": float(judged.get("correctness", 0)),
            "grounding": float(judged.get("grounding", 0)),
            "source_accuracy": float(judged.get("source_accuracy", 0)),
            "rationale": str(judged.get("rationale", "")),
        }
    except Exception as exc:
        return heuristic_judge(
            f"Fallback heuristic judge used because judge model call failed: {exc}"
        )


def overall_score_100(keyword_score: float, judge_scores: Dict[str, Any]) -> float:
    keyword_5 = keyword_score * 5.0
    overall_5 = (
        0.55 * judge_scores["correctness"]
        + 0.20 * judge_scores["grounding"]
        + 0.15 * judge_scores["source_accuracy"]
        + 0.10 * keyword_5
    )
    return round((overall_5 / 5.0) * 100.0, 1)


def openai_like_headers(base_url: Optional[str]) -> Optional[Dict[str, str]]:
    if not base_url or "openrouter.ai" not in base_url:
        return None
    headers = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    app_title = os.getenv("OPENROUTER_APP_TITLE", "HS Teacher Eval")
    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-Title"] = app_title
    return headers or None


def build_systems(openai_api_key: Optional[str], deepseek_api_key: Optional[str]) -> List[SystemConfig]:
    systems = []
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_headers = openai_like_headers(openai_base_url)
    if openai_api_key:
        systems.append(
            SystemConfig(
                system_id="openai_rag",
                provider="openai",
                model=OPENAI_MODEL,
                use_rag=True,
                api_key=openai_api_key,
                base_url=openai_base_url,
                default_headers=openai_headers,
            )
        )
        systems.append(
            SystemConfig(
                system_id="openai_no_rag",
                provider="openai",
                model=OPENAI_MODEL,
                use_rag=False,
                api_key=openai_api_key,
                base_url=openai_base_url,
                default_headers=openai_headers,
            )
        )
    if deepseek_api_key:
        systems.append(
            SystemConfig(
                system_id="deepseek_rag",
                provider="deepseek",
                model=DEEPSEEK_MODEL,
                use_rag=True,
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
        )
        systems.append(
            SystemConfig(
                system_id="deepseek_no_rag",
                provider="deepseek",
                model=DEEPSEEK_MODEL,
                use_rag=False,
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
        )
    return systems


def run_case(system: SystemConfig, case: BenchmarkCase, retrieval_agent, client) -> Dict[str, Any]:
    result = run_multi_agent(
        user_query=case.query,
        client=client,
        retrieval_agent=retrieval_agent if system.use_rag else None,
        model=system.model,
        routing_mode="manual",
        forced_agent=case.agent,
        fetch_k_per_query=FETCH_K,
        final_k=FINAL_K,
        use_rag=system.use_rag,
    )
    result["system_id"] = system.system_id
    result["provider"] = system.provider
    result["model"] = system.model
    result["case_id"] = case.case_id
    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_system: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        by_system.setdefault(item["system_id"], []).append(item)

    summary = {}
    for system_id, rows in by_system.items():
        summary[system_id] = {
            "avg_overall_100": round(mean(row["scores"]["overall_100"] for row in rows), 2),
            "avg_keyword_coverage": round(mean(row["scores"]["keyword_coverage"] for row in rows), 3),
            "avg_correctness": round(mean(row["scores"]["judge"]["correctness"] for row in rows), 3),
            "avg_grounding": round(mean(row["scores"]["judge"]["grounding"] for row in rows), 3),
            "avg_source_accuracy": round(mean(row["scores"]["judge"]["source_accuracy"] for row in rows), 3),
            "avg_retrieval_alignment": round(
                mean(
                    row["scores"]["retrieval_alignment"]
                    for row in rows
                    if row["scores"]["retrieval_alignment"] is not None
                ),
                3,
            ) if any(row["scores"]["retrieval_alignment"] is not None for row in rows) else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenAI/DeepSeek with and without RAG.")
    parser.add_argument("--benchmark", default=str(BENCHMARK_PATH))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional path to save detailed JSON results.")
    args = parser.parse_args()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not openai_api_key and not deepseek_api_key:
        raise RuntimeError("Set OPENAI_API_KEY and/or DEEPSEEK_API_KEY before running evaluation.")

    benchmark = load_benchmark(Path(args.benchmark))
    if args.max_cases is not None:
        benchmark = benchmark[: args.max_cases]

    systems = build_systems(openai_api_key=openai_api_key, deepseek_api_key=deepseek_api_key)
    if not systems:
        raise RuntimeError("No runnable systems were configured.")

    judge_client = create_llm_client(openai_api_key) if openai_api_key else None

    provider_clients = {}
    retrieval_agents = {}
    for system in systems:
        if system.provider not in provider_clients:
            provider_clients[system.provider] = create_llm_client(
                api_key=system.api_key,
                base_url=system.base_url,
                default_headers=system.default_headers,
            )
        if system.provider not in retrieval_agents:
            retrieval_agents[system.provider] = RetrievalAgent(
                client=provider_clients[system.provider],
                index_dir=INDEX_DIR,
                embed_model_name=EMBED_MODEL_NAME,
                fetch_k=FETCH_K,
                final_k=FINAL_K,
            )

    detailed_results = []
    for system in systems:
        client = provider_clients[system.provider]
        retrieval_agent = retrieval_agents[system.provider]
        print(f"\n=== Running {system.system_id} ({system.model}) ===")
        for case in benchmark:
            print(f"- {case.case_id}: {case.query}")
            try:
                result = run_case(system, case, retrieval_agent, client)
                keyword_score = keyword_coverage(result["answer"], case.expected_keyword_groups)
                retrieval_score = retrieval_alignment_score(result, case)
                judge_scores = judge_answer(
                    case=case,
                    result=result,
                    judge_client=judge_client,
                    judge_model=JUDGE_MODEL,
                    keyword_score=keyword_score,
                    retrieval_score=retrieval_score,
                )
                score_blob = {
                    "keyword_coverage": round(keyword_score, 3),
                    "answer_source_guess": round(
                        answer_source_guess_score(
                            result["answer"],
                            case.expected_source_hints,
                            case.expected_source_patterns,
                        ),
                        3,
                    ),
                    "retrieval_alignment": round(retrieval_score, 3) if retrieval_score is not None else None,
                    "judge": judge_scores,
                    "overall_100": overall_score_100(keyword_score, judge_scores),
                }
                detailed_results.append({
                    "system_id": system.system_id,
                    "provider": system.provider,
                    "model": system.model,
                    "case_id": case.case_id,
                    "query": case.query,
                    "use_rag": system.use_rag,
                    "retrieval_plan": result["retrieval_plan"],
                    "rewrites": result["rewrites"],
                    "top_evidence": [
                        {
                            "doc_id": item.get("doc_id"),
                            "retrieved_from": item.get("retrieved_from"),
                            "source_file": item.get("source_file") or os.path.basename(item.get("source", "")),
                            "page": item.get("page"),
                            "score": item.get("score"),
                        }
                        for item in result["top_evidence"]
                    ],
                    "answer": result["answer"],
                    "scores": score_blob,
                })
                print(f"  overall={score_blob['overall_100']} keyword={score_blob['keyword_coverage']} retrieval={score_blob['retrieval_alignment']}")
            except Exception as exc:
                detailed_results.append({
                    "system_id": system.system_id,
                    "provider": system.provider,
                    "model": system.model,
                    "case_id": case.case_id,
                    "query": case.query,
                    "use_rag": system.use_rag,
                    "error": str(exc),
                    "scores": {
                        "keyword_coverage": 0.0,
                        "answer_source_guess": 0.0,
                        "retrieval_alignment": None,
                        "judge": {
                            "correctness": 0.0,
                            "grounding": 0.0,
                            "source_accuracy": 0.0,
                            "rationale": f"Run failed: {exc}",
                        },
                        "overall_100": 0.0,
                    },
                })
                print(f"  failed: {exc}")

    summary = summarize(detailed_results)
    print("\n=== Summary ===")
    for system_id, row in summary.items():
        print(
            f"{system_id}: overall={row['avg_overall_100']} "
            f"correctness={row['avg_correctness']} grounding={row['avg_grounding']} "
            f"source={row['avg_source_accuracy']} retrieval={row['avg_retrieval_alignment']}"
        )

    output_path = Path(args.output) if args.output else Path(
        "eval/results"
    ) / f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": [case.case_id for case in benchmark],
        "summary": summary,
        "results": detailed_results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {output_path}")


if __name__ == "__main__":
    main()
