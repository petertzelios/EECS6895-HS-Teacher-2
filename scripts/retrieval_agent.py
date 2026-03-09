from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from llm_utils import complete_text
from reasoning_backends import (
    OpenVikingRefiner,
    PageIndexRefiner,
    PdfSourceResolver,
    default_data_root,
)


DEFAULT_INDEX_NAMES = [
    "curriculum_overview",
    "exam_questions",
    "exam_scoring",
]


def infer_embed_device() -> str:
    explicit = os.getenv("EMBED_DEVICE")
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def first_existing_path(candidates: Sequence[str], fallback: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    return fallback


def default_index_dir() -> str:
    env_value = os.getenv("INDEX_DIR")
    if env_value:
        return env_value
    return first_existing_path(
        [
            "./indexes_hs_teacher_clean",
            "../indexes_hs_teacher_clean",
            "/mnt/f/SchoolWorks/BigData/MultiAgent/indexes_hs_teacher_clean",
        ],
        "./indexes_hs_teacher_clean",
    )


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


@dataclass
class QueryProfile:
    subject_aliases: Tuple[str, ...]
    exam_admin: Optional[str]
    module: Optional[str]
    grade: Optional[str]


class RetrievalAgent:
    def __init__(
        self,
        client: Optional[OpenAI],
        index_dir: str,
        embed_model_name: str,
        fetch_k: int,
        final_k: int,
        index_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.client = client
        self.index_dir = index_dir
        self.fetch_k = fetch_k
        self.final_k = final_k
        self.index_names = list(index_names or DEFAULT_INDEX_NAMES)
        self.embed_device = infer_embed_device()
        self.st_model = SentenceTransformer(embed_model_name, device=self.embed_device)
        self.indexes: Dict[str, Tuple[faiss.Index, List[Dict[str, Any]]]] = {
            name: load_index(index_dir, name) for name in self.index_names
        }
        self._rows_by_source_page: Optional[Dict[Tuple[str, int], List[Dict[str, Any]]]] = None
        self.source_resolver = PdfSourceResolver(default_data_root())
        backend_list = os.getenv("RETRIEVAL_EXPERIMENTAL_BACKENDS", "pageindex,openviking")
        self.experimental_backends = {
            item.strip().lower() for item in backend_list.split(",") if item.strip()
        }
        cache_root = os.getenv("RETRIEVAL_CACHE_DIR", ".cache/retrieval")
        self.pageindex_refiner = (
            PageIndexRefiner(
                client=self.client,
                source_resolver=self.source_resolver,
                cache_dir=os.path.join(cache_root, "pageindex"),
            )
            if "pageindex" in self.experimental_backends
            else None
        )
        self.openviking_refiner = (
            OpenVikingRefiner(
                source_resolver=self.source_resolver,
                cache_dir=os.path.join(cache_root, "openviking"),
            )
            if "openviking" in self.experimental_backends
            else None
        )

    def embed_384(self, texts: List[str]) -> np.ndarray:
        vecs = self.st_model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def available_indexes(self) -> List[str]:
        return list(self.indexes.keys())

    def infer_subject_aliases(self, query: str) -> Tuple[str, ...]:
        q = query.lower()
        mapping = {
            "algebra i": ("Algebra 1", "Algebra I"),
            "algebra 1": ("Algebra 1", "Algebra I"),
            "algebra ii": ("Algebra 2", "Algebra II"),
            "algebra 2": ("Algebra 2", "Algebra II"),
            "geometry": ("Geometry",),
            "ela": ("High School English Language Arts", "ELA"),
            "english language arts": ("High School English Language Arts", "ELA"),
            "english": ("High School English Language Arts", "ELA"),
            "chemistry": ("Chemistry",),
            "physics": ("Physics",),
            "living environment": ("Living Environment",),
            "biology": ("Life Science: Biology",),
            "life science": ("Life Science: Biology",),
            "global history": ("Global History and Geography II",),
            "us history": ("US History and Government",),
            "precalculus": ("Precalculus",),
        }
        for key, values in mapping.items():
            if key in q:
                return values
        return ()

    def infer_exam_admin(self, query: str) -> Optional[str]:
        q = query.lower()
        direct = re.search(r"\b([168]20\d{2})\b", q)
        if direct:
            return direct.group(1)

        month_map = {
            "january": "1",
            "june": "6",
            "august": "8",
        }
        for month, prefix in month_map.items():
            match = re.search(rf"\b{month}\s+(20\d{{2}})\b", q)
            if match:
                return f"{prefix}{match.group(1)}"
        return None

    def module_hint(self, query: str) -> Optional[str]:
        match = re.search(r"\bmodule\s*(\d+)\b", query.lower())
        return match.group(1) if match else None

    def wants_overview(self, query: str) -> bool:
        q = query.lower()
        return any(
            phrase in q
            for phrase in [
                "what is included",
                "what is in",
                "overview",
                "summarize",
                "summary",
            ]
        )

    def wants_curriculum_structure(self, query: str) -> bool:
        q = query.lower()
        return self.wants_overview(query) or any(
            phrase in q
            for phrase in [
                "essential question",
                "introduction",
                "title",
                "what is module",
                "what texts",
                "which texts",
                "how many lessons",
                "number of lessons",
                "unit 1 texts",
                "unit 2 texts",
            ]
        )

    def wants_exam_structure(self, query: str) -> bool:
        q = query.lower()
        return any(
            phrase in q
            for phrase in [
                "how many questions",
                "total questions",
                "how many total questions",
                "four parts",
                "part i",
                "part ii",
                "exam structure",
                "allowed tools",
                "permitted tools",
                "what tools",
                "tools are allowed",
                "what materials",
                "allowed materials",
                "permitted materials",
                "allowed to use",
                "graphing calculator",
                "straightedge",
                "ruler",
                "compass",
                "tools",
            ]
        )

    def build_query_profile(self, query: str) -> QueryProfile:
        return QueryProfile(
            subject_aliases=self.infer_subject_aliases(query),
            exam_admin=self.infer_exam_admin(query),
            module=self.module_hint(query),
            grade=self.infer_grade(query),
        )

    def infer_grade(self, query: str) -> Optional[str]:
        q = query.lower()
        explicit = re.search(r"\bgrade\s*(\d{1,2})\b", q)
        if explicit:
            return f"Grade {explicit.group(1)}"
        ordinal = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s+grade\b", q)
        if ordinal:
            return f"Grade {ordinal.group(1)}"
        return None

    def normalize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        out["admin"] = out.get("admin") or out.get("exam_admin")
        out["_source_identity"] = self.source_identity(out)
        out["candidate_id"] = self.candidate_id(out)
        return out

    def source_identity(self, row: Dict[str, Any]) -> str:
        source = (row.get("source") or "").replace("\\", "/").strip()
        if source:
            return os.path.normpath(source).replace("\\", "/")

        resolved = self.source_resolver.resolve(row)
        if resolved is not None:
            return str(resolved.resolve()).replace("\\", "/")

        source_file = row.get("source_file") or os.path.basename(row.get("source", ""))
        return source_file or "unknown-source"

    def candidate_id(self, row: Dict[str, Any]) -> str:
        source_identity = row.get("_source_identity") or self.source_identity(row)
        doc_id = row.get("doc_id") or f"page-{row.get('page', 'na')}"
        return f"{source_identity}::{doc_id}"

    def page_key(self, row: Dict[str, Any]) -> Tuple[str, int]:
        return (
            row.get("_source_identity") or self.source_identity(row),
            int(row.get("page", 0)),
        )

    def normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    def query_tokens(self, query: str) -> List[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "for",
            "how",
            "in",
            "is",
            "of",
            "on",
            "or",
            "the",
            "to",
            "what",
            "which",
        }
        return [
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) > 1 and token not in stopwords
        ]

    def row_subject_values(self, row: Dict[str, Any]) -> Tuple[str, ...]:
        values = []
        for key in ("subject", "course", "grade"):
            value = row.get(key)
            if value and value not in values:
                values.append(value)
        return tuple(values)

    def rows_for_source_page(self, source_identity: str, page: int) -> List[Dict[str, Any]]:
        if self._rows_by_source_page is None:
            cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
            for index_name, (_, meta_rows) in self.indexes.items():
                for raw_row in meta_rows:
                    normalized = self.normalize_row(raw_row)
                    key = (normalized.get("_source_identity"), normalized.get("page"))
                    if key[0] is None or key[1] is None:
                        continue
                    row = dict(normalized)
                    row["retrieved_from"] = index_name
                    cache.setdefault(key, []).append(row)
            self._rows_by_source_page = cache
        return list(self._rows_by_source_page.get((source_identity, page), []))

    def looks_like_question_text(self, text: str) -> bool:
        patterns = [
            r"\bPart I\b",
            r"\bQuestion\b",
            r"\([0-9]+\)",
            r"\b[A-D]\)",
            r"(?m)^\s*[0-9]{1,2}[\.)]\s",
        ]
        normalized = (text or "").replace("\n", " ")
        return any(re.search(pattern, normalized) for pattern in patterns)

    def is_front_matter_exam(self, row: Dict[str, Any]) -> bool:
        src = os.path.basename(row.get("source", "")).lower()
        page = row.get("page")
        if "exam" in src and isinstance(page, int) and page <= 1:
            return not self.looks_like_question_text(row.get("text", ""))
        return False

    def heuristic_rewrites(
        self,
        user_query: str,
        agent_name: str,
        profile: QueryProfile,
    ) -> List[str]:
        primary_subject = profile.subject_aliases[0] if profile.subject_aliases else ""
        module_text = f"module {profile.module}" if profile.module else ""
        exam_admin = profile.exam_admin or ""
        grade_text = profile.grade or ""

        if agent_name == "regents_agent":
            seeds = [
                user_query,
                " ".join(x for x in [primary_subject, exam_admin, "regents exam question"] if x),
                " ".join(x for x in [primary_subject, exam_admin, "rating guide scoring"] if x),
                " ".join(x for x in [primary_subject, "released regents practice question"] if x),
            ]
            if self.wants_exam_structure(user_query):
                seeds.insert(
                    1,
                    " ".join(
                        x for x in [primary_subject, exam_admin, "exam booklet four parts total questions"] if x
                    ),
                )
        else:
            seeds = [
                user_query,
                " ".join(x for x in [grade_text, primary_subject, module_text, "curriculum overview"] if x),
                " ".join(x for x in [grade_text, primary_subject, "lesson assessment overview"] if x),
                " ".join(x for x in [grade_text, primary_subject, "teacher materials unit overview"] if x),
            ]
            if self.wants_curriculum_structure(user_query):
                seeds.insert(
                    1,
                    " ".join(
                        x
                        for x in [grade_text, primary_subject, module_text, "module overview introduction"]
                        if x
                    ),
                )
                seeds.insert(
                    2,
                    " ".join(
                        x
                        for x in [grade_text, primary_subject, module_text, "essential question texts lessons"]
                        if x
                    ),
                )

        out: List[str] = []
        seen = set()
        for seed in seeds:
            normalized = " ".join(seed.split())
            if normalized and normalized not in seen:
                out.append(normalized)
                seen.add(normalized)
        while len(out) < 4:
            out.append(user_query)
        return out[:4]

    def rewrite_queries(
        self,
        user_query: str,
        agent_name: str,
        index_names: List[str],
        model: str,
    ) -> List[str]:
        profile = self.build_query_profile(user_query)
        fallback = self.heuristic_rewrites(user_query, agent_name, profile)
        if self.client is None:
            return fallback

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

        try:
            text = complete_text(
                self.client,
                prompt,
                model=model,
                reasoning_effort="low",
                temperature=0.0,
            )
            lines = [x.strip(" -0123456789.") for x in text.splitlines() if x.strip()]
            if len(lines) >= 4:
                return lines[:4]
        except Exception:
            pass
        return fallback

    def retrieve_faiss(
        self,
        query: str,
        index_name: str,
        fetch_k: Optional[int] = None,
        final_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        index, meta = self.indexes[index_name]
        if not meta or index.ntotal == 0:
            return []

        local_fetch_k = min(fetch_k or self.fetch_k, len(meta))
        if local_fetch_k <= 0:
            return []

        profile = self.build_query_profile(query)
        scores, ids = index.search(self.embed_384([query]), local_fetch_k)
        normalized_query = self.normalize_text(query)
        query_tokens = self.query_tokens(query)

        cands: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0 or idx >= len(meta):
                continue
            row = self.normalize_row(meta[idx])
            row["score"] = float(score)
            cands.append(row)

        def boosted_score(row: Dict[str, Any]) -> float:
            score = row["score"]
            text = self.normalize_text(row.get("text", ""))
            source_file = self.normalize_text(row.get("source_file") or os.path.basename(row.get("source", "")))
            token_hits = sum(1 for token in query_tokens if token in text or token in source_file)
            score += min(token_hits, 6) * 0.03
            if profile.subject_aliases:
                row_values = self.row_subject_values(row)
                if any(value in profile.subject_aliases for value in row_values):
                    score += 0.15
            if profile.exam_admin and row.get("admin") == profile.exam_admin:
                score += 0.10
            if profile.grade and row.get("grade") == profile.grade:
                score += 0.25
            if index_name == "curriculum_overview" and profile.module:
                src = (row.get("source") or "").lower().replace("\\", "/")
                base = os.path.basename(src)
                if f"-m{profile.module}-" in base or f"/module {profile.module}/" in src:
                    score += 0.25
                if self.wants_overview(query) and "overview" in base:
                    score += 0.15
            if index_name == "curriculum_overview" and self.wants_curriculum_structure(query):
                if "module-overview" in source_file or "unit-overview" in source_file:
                    score += 0.28
                if any(term in source_file for term in ["rubric", "checklist", "assessment", "lesson"]):
                    score -= 0.10
                if isinstance(row.get("page"), int) and row.get("page") <= 1:
                    score += 0.18
                if "essential question" in normalized_query and "essential question" in text:
                    score += 0.45
                if "title" in normalized_query and row.get("page") == 0:
                    score += 0.22
                if any(term in normalized_query for term in ["how many lessons", "number of lessons"]) and (
                    "number of lessons" in text or "lessons 1" in text
                ):
                    score += 0.22
                if any(term in normalized_query for term in ["what texts", "which texts", "unit 1 texts", "unit 2 texts"]) and "texts" in text:
                    score += 0.18
                if "introduction" in normalized_query and "introduction" in text:
                    score += 0.18
            if index_name == "exam_questions" and row.get("doc_type") in {"exam", "exam_questions"}:
                score += 0.10
                if self.wants_exam_structure(query):
                    if any(
                        phrase in text
                        for phrase in [
                            "total of 35 questions",
                            "four parts",
                            "graphing calculator",
                            "straightedge",
                            "ruler",
                        ]
                    ):
                        score += 0.25
            if index_name == "exam_scoring" and row.get("doc_type") == "exam_scoring":
                score += 0.05
            return score

        for row in cands:
            row["raw_score"] = row["score"]
            row["score"] = boosted_score(row)

        cands.sort(key=boosted_score, reverse=True)
        if index_name == "exam_questions":
            if not self.wants_exam_structure(query):
                cands = [row for row in cands if not self.is_front_matter_exam(row)]
                cands.sort(
                    key=lambda row: (self.looks_like_question_text(row.get("text", "")), boosted_score(row)),
                    reverse=True,
                )

        out: List[Dict[str, Any]] = []
        seen = set()
        target_k = final_k or self.final_k
        for row in cands:
            if row["candidate_id"] in seen:
                continue
            seen.add(row["candidate_id"])
            out.append(row)
            if len(out) >= target_k:
                break
        return out

    def rerank_with_llm(
        self,
        user_query: str,
        candidates: List[Dict[str, Any]],
        top_n: int,
        model: str,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if self.client is None:
            return candidates[:top_n]

        packed = []
        for candidate in candidates[:60]:
            packed.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "doc_id": candidate.get("doc_id"),
                    "pdf": os.path.basename(candidate.get("source", "")),
                    "page": candidate.get("page"),
                    "subject": candidate.get("subject"),
                    "course": candidate.get("course"),
                    "admin": candidate.get("admin"),
                    "snippet": (candidate.get("text", "")[:280] or "").replace("\n", " "),
                }
            )

        prompt = f"""
You are selecting the best evidence chunks for answering a teacher-assistant query.

Return ONLY a JSON array of candidate_id strings from best to worst.
Choose at most {top_n} ids.

User query:
{user_query}

Candidates:
{json.dumps(packed, ensure_ascii=False)}
""".strip()

        try:
            text = complete_text(
                self.client,
                prompt,
                model=model,
                reasoning_effort="low",
                temperature=0.0,
            )
            chosen_ids = json.loads(text.strip())
            chosen_ids = [candidate_id for candidate_id in chosen_ids if isinstance(candidate_id, str)]
        except Exception:
            chosen_ids = []

        llm_rank = {candidate_id: idx for idx, candidate_id in enumerate(chosen_ids)}

        def blended_rank(candidate: Dict[str, Any]) -> Tuple[float, float]:
            heuristic = float(candidate.get("score", 0.0))
            if candidate.get("candidate_id") in llm_rank:
                llm_bonus = float(top_n - llm_rank[candidate["candidate_id"]]) * 0.08
            else:
                llm_bonus = 0.0
            return (heuristic + llm_bonus, heuristic)

        reranked = sorted(candidates, key=blended_rank, reverse=True)
        return reranked[:top_n]

    def should_use_reasoning_refinement(self, query: str) -> bool:
        return self.wants_curriculum_structure(query) or self.wants_exam_structure(query)

    def apply_reasoning_refinement(
        self,
        user_query: str,
        candidates: List[Dict[str, Any]],
        model: str,
    ) -> List[Dict[str, Any]]:
        if not candidates or not self.should_use_reasoning_refinement(user_query):
            return candidates

        boosts: Dict[Tuple[str, int], float] = {}
        focus_candidates = candidates[:12]

        if self.pageindex_refiner is not None and self.pageindex_refiner.enabled:
            for key, value in self.pageindex_refiner.collect_page_boosts(
                user_query,
                focus_candidates,
                model=model,
                max_docs=1,
            ).items():
                boosts[key] = max(boosts.get(key, 0.0), value)

        if self.openviking_refiner is not None and self.openviking_refiner.enabled:
            for key, value in self.openviking_refiner.collect_page_boosts(
                user_query,
                focus_candidates,
                max_docs=1,
            ).items():
                boosts[key] = max(boosts.get(key, 0.0), value)

        if not boosts:
            return candidates

        refined: List[Dict[str, Any]] = []
        existing_candidate_ids = set()
        max_score = max(float(candidate.get("score", 0.0)) for candidate in candidates)
        for candidate in candidates:
            row = dict(candidate)
            existing_candidate_ids.add(row.get("candidate_id"))
            key = self.page_key(row)
            if key in boosts:
                row["score"] = float(row.get("score", 0.0)) + boosts[key]
                row["reasoning_boost"] = boosts[key]
            refined.append(row)

        for (source_identity, page), boost in boosts.items():
            for row in self.rows_for_source_page(source_identity, page):
                if row.get("candidate_id") in existing_candidate_ids:
                    continue
                injected = dict(row)
                injected["score"] = max_score + boost
                injected["reasoning_boost"] = boost
                refined.append(injected)
                existing_candidate_ids.add(injected.get("candidate_id"))
        refined.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        return refined

    def run(
        self,
        user_query: str,
        agent_name: str,
        index_names: Sequence[str],
        model: str,
        fetch_k_per_query: Optional[int] = None,
        final_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        enabled_indexes = [
            name
            for name in index_names
            if name in self.indexes and self.indexes[name][0].ntotal > 0 and len(self.indexes[name][1]) > 0
        ]
        profile = self.build_query_profile(user_query)

        if not enabled_indexes:
            return {
                "retrieval_plan": {
                    "indexes": [],
                    "subject_aliases": list(profile.subject_aliases),
                    "exam_admin": profile.exam_admin,
                    "module": profile.module,
                    "grade": profile.grade,
                    "embedding_device": self.embed_device,
                },
                "rewrites": [],
                "candidate_pool": [],
                "top_evidence": [],
            }

        rewrites = self.rewrite_queries(user_query, agent_name, enabled_indexes, model=model)

        pool = []
        seen = set()
        per_index_final = max(final_k or self.final_k, 4)

        for rewrite in rewrites:
            for index_name in enabled_indexes:
                hits = self.retrieve_faiss(
                    rewrite,
                    index_name=index_name,
                    fetch_k=fetch_k_per_query or self.fetch_k,
                    final_k=per_index_final,
                )
                for hit in hits:
                    if hit["candidate_id"] in seen:
                        continue
                    seen.add(hit["candidate_id"])
                    hit["retrieved_from"] = index_name
                    pool.append(hit)

        pool.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        pool = self.apply_reasoning_refinement(user_query, pool, model=model)
        pool = pool[: min(len(pool), max(25, (final_k or self.final_k) * 5))]
        top = self.rerank_with_llm(user_query, pool, top_n=final_k or self.final_k, model=model)

        return {
            "retrieval_plan": {
                "indexes": enabled_indexes,
                "subject_aliases": list(profile.subject_aliases),
                "exam_admin": profile.exam_admin,
                "module": profile.module,
                "grade": profile.grade,
                "embedding_device": self.embed_device,
                "reasoning_backends": [
                    name
                    for name, enabled in [
                        ("pageindex", bool(self.pageindex_refiner and self.pageindex_refiner.enabled)),
                        ("openviking", bool(self.openviking_refiner and self.openviking_refiner.enabled)),
                    ]
                    if enabled
                ],
            },
            "rewrites": rewrites,
            "candidate_pool": pool,
            "top_evidence": top,
        }
