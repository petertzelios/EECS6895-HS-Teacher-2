from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_utils import complete_text


def first_existing_path(candidates: Sequence[str], fallback: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    return fallback


def default_data_root() -> str:
    env_value = os.getenv("ROOT_DIR")
    if env_value:
        return env_value
    return first_existing_path(
        [
            "./midterm_data_clean",
            "../midterm_data_clean",
            "/mnt/f/SchoolWorks/BigData/MultiAgent/midterm_data_clean",
        ],
        "./midterm_data_clean",
    )


def available_api_key() -> Optional[str]:
    return (
        os.getenv("CHATGPT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )


class PdfSourceResolver:
    def __init__(self, data_root: str) -> None:
        self.data_root = Path(data_root).resolve()
        self._basename_index: Dict[str, List[Path]] = {}

    def _build_basename_index(self) -> None:
        if self._basename_index or not self.data_root.exists():
            return
        for path in self.data_root.rglob("*.pdf"):
            self._basename_index.setdefault(path.name.lower(), []).append(path)

    def resolve(self, row: Dict[str, Any]) -> Optional[Path]:
        source = (row.get("source") or "").replace("\\", "/")
        marker = "midterm_data_clean/"
        if marker in source:
            rel = source.split(marker, 1)[1]
            candidate = self.data_root / rel
            if candidate.exists():
                return candidate

        basename = (row.get("source_file") or os.path.basename(source)).lower()
        if not basename:
            return None

        self._build_basename_index()
        matches = self._basename_index.get(basename, [])
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        grade = (row.get("grade") or "").lower().replace(" ", "-")
        for match in matches:
            normalized = str(match).lower().replace(" ", "-")
            if grade and grade in normalized:
                return match
        return matches[0]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _source_identity(row: Dict[str, Any], resolved_path: Optional[Path] = None) -> str:
    source = (row.get("source") or "").replace("\\", "/").strip()
    if source:
        return os.path.normpath(source).replace("\\", "/")
    if resolved_path is not None:
        return str(resolved_path.resolve()).replace("\\", "/")
    basename = row.get("source_file") or os.path.basename(source)
    return basename or "unknown-source"


def _tokenize_query(text: str) -> List[str]:
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
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 1 and token not in stopwords
    ]


def _json_from_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


class PageIndexRefiner:
    def __init__(
        self,
        client,
        source_resolver: PdfSourceResolver,
        cache_dir: str,
    ) -> None:
        self.client = client
        self.source_resolver = source_resolver
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.repo_dir = Path(
            os.getenv(
                "PAGEINDEX_REPO_DIR",
                first_existing_path(
                    [
                        ".tools/external/PageIndex",
                        "../.tools/external/PageIndex",
                        "/mnt/f/SchoolWorks/BigData/MultiAgent/repo/.tools/external/PageIndex",
                    ],
                    ".tools/external/PageIndex",
                ),
            )
        )
        model_list = os.getenv("PAGEINDEX_MODELS", "gpt-4o-mini,gpt-4o")
        self.model_candidates = [part.strip() for part in model_list.split(",") if part.strip()]

    @property
    def enabled(self) -> bool:
        return self.client is not None and self.repo_dir.exists() and available_api_key() is not None

    def _cache_path(self, pdf_path: Path) -> Path:
        return self.cache_dir / f"{pdf_path.stem}_structure.json"

    def ensure_tree(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        pdf_path = pdf_path.resolve()
        cache_path = self._cache_path(pdf_path)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        result_dir = self.repo_dir / "results"
        result_dir.mkdir(exist_ok=True)
        result_file = result_dir / f"{pdf_path.stem}_structure.json"
        if result_file.exists():
            shutil.copyfile(result_file, cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        env = dict(os.environ)
        env["CHATGPT_API_KEY"] = available_api_key() or ""

        last_error = None
        for model_name in self.model_candidates:
            cmd = [
                sys.executable,
                "run_pageindex.py",
                "--pdf_path",
                str(pdf_path),
                "--model",
                model_name,
            ]
            proc = subprocess.run(
                cmd,
                cwd=self.repo_dir,
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and result_file.exists():
                shutil.copyfile(result_file, cache_path)
                return json.loads(cache_path.read_text(encoding="utf-8"))
            last_error = proc.stderr or proc.stdout

        if last_error:
            print(f"[PageIndex] failed for {pdf_path.name}: {last_error[:500]}")
        return None

    def _flatten_nodes(self, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []

        def walk(nodes: Iterable[Dict[str, Any]]) -> None:
            for node in nodes:
                flattened.append(
                    {
                        "node_id": node.get("node_id"),
                        "title": node.get("title"),
                        "summary": node.get("summary"),
                        "start_index": node.get("start_index"),
                        "end_index": node.get("end_index"),
                    }
                )
                walk(node.get("nodes", []))

        walk(tree.get("structure", []))
        return [node for node in flattened if node.get("node_id")]

    def _select_nodes(self, query: str, tree: Dict[str, Any], model: str) -> List[Dict[str, Any]]:
        nodes = self._flatten_nodes(tree)
        if not nodes:
            return []

        packed = nodes[:40]
        if self.client is None:
            tokens = _tokenize_query(query)
            scored = []
            for node in packed:
                haystack = _normalize_text(
                    f"{node.get('title', '')} {node.get('summary', '')}"
                )
                overlap = sum(1 for token in tokens if token in haystack)
                scored.append((overlap, node))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [node for score, node in scored if score > 0][:2]

        prompt = f"""
You are selecting the most relevant sections from a PageIndex document tree.

Question:
{query}

Document:
{tree.get("doc_name")}

Candidate nodes:
{json.dumps(packed, ensure_ascii=False)}

Rules:
- Prefer nodes that directly answer the question.
- For high-level curriculum questions, prioritize overview/introduction nodes and early pages.
- Return ONLY valid JSON with key "node_ids".
- Choose at most 3 node ids.
""".strip()

        try:
            data = _json_from_text(
                complete_text(
                    self.client,
                    prompt,
                    model=model,
                    reasoning_effort="low",
                    temperature=0.0,
                )
            )
            chosen = [node_id for node_id in data.get("node_ids", []) if isinstance(node_id, str)]
        except Exception:
            chosen = []

        by_id = {node["node_id"]: node for node in nodes}
        return [by_id[node_id] for node_id in chosen if node_id in by_id]

    def collect_page_boosts(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        model: str,
        max_docs: int = 2,
    ) -> Dict[Tuple[str, int], float]:
        if not self.enabled:
            return {}

        boosts: Dict[Tuple[str, int], float] = {}
        seen_paths = set()
        normalized_query = _normalize_text(query)
        grade_match = re.search(r"\bgrade\s*(\d{1,2})\b", normalized_query)
        module_match = re.search(r"\bmodule\s*(\d{1,2})\b", normalized_query)
        query_grade = f"grade {grade_match.group(1)}" if grade_match else ""
        query_module = module_match.group(1) if module_match else ""

        def doc_priority(row: Dict[str, Any]) -> Tuple[int, float]:
            source_file = _normalize_text(row.get("source_file") or "")
            doc_type = _normalize_text(row.get("doc_type") or "")
            row_grade = _normalize_text(row.get("grade") or "")
            score = float(row.get("score", 0.0))
            mismatch_penalty = 0
            if query_grade and row_grade and row_grade != query_grade:
                mismatch_penalty += 2
            if query_module and f"m{query_module}" not in source_file and f"module-{query_module}" not in source_file:
                mismatch_penalty += 1
            if any(
                term in normalized_query
                for term in [
                    "how many questions",
                    "total questions",
                    "four parts",
                    "tools",
                    "materials",
                    "allowed to use",
                    "graphing calculator",
                    "straightedge",
                    "ruler",
                    "compass",
                ]
            ):
                if doc_type in {"exam", "exam_questions"} or source_file == "exam.pdf":
                    return (0 + mismatch_penalty, -score)
                if any(term in source_file for term in ["rating_guide", "scoring_key", "model_responses"]):
                    return (2 + mismatch_penalty, -score)
            if any(term in normalized_query for term in ["title", "overview", "introduction", "essential question", "how many lessons", "what texts"]):
                if "module-overview" in source_file:
                    return (0 + mismatch_penalty, -score)
                if "unit-overview" in source_file:
                    return (1 + mismatch_penalty, -score)
                if any(term in source_file for term in ["rubric", "checklist", "assessment", "lesson"]):
                    return (3 + mismatch_penalty, -score)
            return (2 + mismatch_penalty, -score)

        for candidate in sorted(candidates, key=doc_priority):
            pdf_path = self.source_resolver.resolve(candidate)
            if pdf_path is None or pdf_path in seen_paths:
                continue
            seen_paths.add(pdf_path)
            if len(seen_paths) > max_docs:
                break

            tree = self.ensure_tree(pdf_path)
            if not tree:
                continue

            selected_nodes = self._select_nodes(query, tree, model=model)
            source_id = _source_identity(candidate, pdf_path)
            for node in selected_nodes:
                start = max(0, int(node.get("start_index", 1)) - 1)
                end = max(start, int(node.get("end_index", start + 1)) - 1)
                for page in range(start, end + 1):
                    boosts[(source_id, page)] = max(boosts.get((source_id, page), 0.0), 0.35)
        return boosts


class OpenVikingRefiner:
    def __init__(
        self,
        source_resolver: PdfSourceResolver,
        cache_dir: str,
    ) -> None:
        self.source_resolver = source_resolver
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir = self.cache_dir / "db"
        self.manifest_path = self.cache_dir / "manifest.json"
        self._client = None
        self._manifest: Dict[str, str] = {}
        if self.manifest_path.exists():
            try:
                self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                self._manifest = {}

    @property
    def enabled(self) -> bool:
        try:
            import openviking  # noqa: F401
        except Exception:
            return False
        return available_api_key() is not None

    def _config_path(self) -> Path:
        explicit = os.getenv("OPENVIKING_CONFIG_FILE")
        if explicit:
            return Path(explicit)

        config_path = self.cache_dir / "ov.conf"
        if config_path.exists():
            return config_path

        config = {
            "storage": {
                "workspace": str(self.cache_dir / "workspace"),
                "vectordb": {"name": "context", "backend": "local", "project": "default"},
                "agfs": {
                    "port": int(os.getenv("OPENVIKING_AGFS_PORT", "1833")),
                    "log_level": "warn",
                    "backend": "local",
                    "timeout": 10,
                    "retry_times": 3,
                },
            },
            "embedding": {
                "dense": {
                    "model": os.getenv("OPENVIKING_EMBED_MODEL", "text-embedding-3-small"),
                    "api_key": available_api_key(),
                    "api_base": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    "dimension": int(os.getenv("OPENVIKING_EMBED_DIM", "1536")),
                    "provider": "openai",
                    "input": "text",
                }
            },
            "vlm": {
                "model": os.getenv("OPENVIKING_VLM_MODEL", "gpt-4o-mini"),
                "api_key": available_api_key(),
                "api_base": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "temperature": 0.0,
                "max_retries": 2,
                "provider": "openai",
                "thinking": False,
            },
            "auto_generate_l0": True,
            "auto_generate_l1": True,
            "default_search_mode": "thinking",
            "default_search_limit": 3,
            "enable_memory_decay": False,
            "log": {"level": "INFO", "output": "stdout"},
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config_path

    def _client_instance(self):
        if self._client is not None:
            return self._client
        import openviking as ov

        os.environ["OPENVIKING_CONFIG_FILE"] = str(self._config_path())
        self._client = ov.SyncOpenViking(path=str(self.db_dir))
        self._client.initialize()
        return self._client

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")

    def ensure_indexed(self, pdf_path: Path) -> Optional[str]:
        if not self.enabled:
            return None
        key = str(pdf_path.resolve())
        if key in self._manifest:
            return self._manifest[key]

        client = self._client_instance()
        result = client.add_resource(path=key)
        root_uri = result.get("root_uri")
        client.wait_processed()
        if root_uri:
            self._manifest[key] = root_uri
            self._save_manifest()
        return root_uri

    def collect_page_boosts(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        max_docs: int = 2,
    ) -> Dict[Tuple[str, int], float]:
        if not self.enabled:
            return {}

        client = self._client_instance()
        boosts: Dict[Tuple[str, int], float] = {}
        seen_paths = set()
        normalized_query = _normalize_text(query)
        grade_match = re.search(r"\bgrade\s*(\d{1,2})\b", normalized_query)
        module_match = re.search(r"\bmodule\s*(\d{1,2})\b", normalized_query)
        query_grade = f"grade {grade_match.group(1)}" if grade_match else ""
        query_module = module_match.group(1) if module_match else ""

        def doc_priority(row: Dict[str, Any]) -> Tuple[int, float]:
            source_file = _normalize_text(row.get("source_file") or "")
            doc_type = _normalize_text(row.get("doc_type") or "")
            row_grade = _normalize_text(row.get("grade") or "")
            score = float(row.get("score", 0.0))
            mismatch_penalty = 0
            if query_grade and row_grade and row_grade != query_grade:
                mismatch_penalty += 2
            if query_module and f"m{query_module}" not in source_file and f"module-{query_module}" not in source_file:
                mismatch_penalty += 1
            if any(
                term in normalized_query
                for term in [
                    "how many questions",
                    "total questions",
                    "four parts",
                    "tools",
                    "materials",
                    "allowed to use",
                    "graphing calculator",
                    "straightedge",
                    "ruler",
                    "compass",
                ]
            ):
                if doc_type in {"exam", "exam_questions"} or source_file == "exam.pdf":
                    return (0 + mismatch_penalty, -score)
                if any(term in source_file for term in ["rating_guide", "scoring_key", "model_responses"]):
                    return (2 + mismatch_penalty, -score)
            if any(term in normalized_query for term in ["title", "overview", "introduction", "essential question", "how many lessons", "what texts"]):
                if "module-overview" in source_file:
                    return (0 + mismatch_penalty, -score)
                if "unit-overview" in source_file:
                    return (1 + mismatch_penalty, -score)
                if any(term in source_file for term in ["rubric", "checklist", "assessment", "lesson"]):
                    return (3 + mismatch_penalty, -score)
            return (2 + mismatch_penalty, -score)

        for candidate in sorted(candidates, key=doc_priority):
            pdf_path = self.source_resolver.resolve(candidate)
            if pdf_path is None or pdf_path in seen_paths:
                continue
            seen_paths.add(pdf_path)
            if len(seen_paths) > max_docs:
                break

            root_uri = self.ensure_indexed(pdf_path)
            if not root_uri:
                continue

            results = client.find(query, target_uri=root_uri, limit=4)
            source_id = _source_identity(candidate, pdf_path)
            for item in results.resources[:4]:
                uri = getattr(item, "uri", "")
                match = re.search(r"_(\d+)\.md$", uri)
                if not match:
                    continue
                page = max(0, int(match.group(1)) - 1)
                boosts[(source_id, page)] = max(boosts.get((source_id, page), 0.0), 0.20)
        return boosts
