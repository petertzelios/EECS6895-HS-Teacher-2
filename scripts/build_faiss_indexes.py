"""
Build and save FAISS indexes for the HS teacher assistant corpus.

This script is adapted from the FAISS index-building portion of:
- midterm_hsteacher2_v3-2.ipynb

It walks a PDF corpus, classifies files into four buckets, chunks the PDF text,
embeds the chunks with sentence-transformers/all-MiniLM-L6-v2, and saves:

    <index_name>.faiss
    <index_name>.meta.jsonl

Buckets:
- curriculum_overview
- exam_questions
- exam_scoring
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def first_existing_path(candidates, fallback: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    return fallback


ROOT_DIR = os.getenv(
    "ROOT_DIR",
    first_existing_path(
        [
            "./midterm_data_clean",
            "../midterm_data_clean",
            "/mnt/f/SchoolWorks/BigData/MultiAgent/midterm_data_clean",
        ],
        "./midterm_data_clean",
    ),
)
INDEX_DIR = os.getenv(
    "INDEX_DIR",
    first_existing_path(
        [
            "./indexes_hs_teacher_clean",
            "../indexes_hs_teacher_clean",
            "/mnt/f/SchoolWorks/BigData/MultiAgent/indexes_hs_teacher_clean",
        ],
        "./indexes_hs_teacher_clean",
    ),
)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE_STANDARDS = int(os.getenv("CHUNK_SIZE_STANDARDS", "1000"))
CHUNK_OVERLAP_STANDARDS = int(os.getenv("CHUNK_OVERLAP_STANDARDS", "150"))
CHUNK_SIZE_EXAMS = int(os.getenv("CHUNK_SIZE_EXAMS", "750"))
CHUNK_OVERLAP_EXAMS = int(os.getenv("CHUNK_OVERLAP_EXAMS", "180"))


def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def classify_file(path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    norm = path.replace("\\", "/")
    fname = os.path.basename(norm).lower()

    if "/regents_raw_selected/" in norm:
        parts = norm.split("/")
        try:
            i = parts.index("regents_raw_selected")
            subject = parts[i + 1]
            exam_admin = parts[i + 2]
        except Exception:
            subject = "Unknown"
            exam_admin = "Unknown"

        meta = {
            "collection": "regents",
            "subject": subject,
            "admin": exam_admin,
            "source_file": os.path.basename(path),
        }

        if fname == "exam.pdf":
            meta["doc_type"] = "exam_questions"
            return "exam_questions", meta
        if fname in {"rating_guide.pdf", "scoring_key.pdf", "model_responses.pdf"} or fname.startswith("rating_guide"):
            meta["doc_type"] = "exam_scoring"
            return "exam_scoring", meta

        meta["doc_type"] = "exam_scoring"
        return "exam_scoring", meta

    if "/ela_curriculum_selected/" in norm:
        parts = norm.split("/")
        try:
            i = parts.index("ela_curriculum_selected")
            grade = parts[i + 1]
        except Exception:
            grade = "Unknown"

        meta = {
            "collection": "ela_curriculum",
            "subject": "ELA",
            "grade": grade,
            "source_file": os.path.basename(path),
        }

        if (
            "module-overview" in fname
            or "unit-overview" in fname
            or "performance-assessment" in fname
            or "assessment" in fname
            or "rubric" in fname
            or "checklist" in fname
        ):
            meta["doc_type"] = "curriculum_overview"
            return "curriculum_overview", meta
        return None, None

    if "/math_curriculum_selected/" in norm:
        parts = norm.split("/")
        try:
            i = parts.index("math_curriculum_selected")
            subject = parts[i + 1] if len(parts) > i + 1 else "Math"
            course = parts[i + 2] if len(parts) > i + 2 else "Unknown"
        except Exception:
            subject = "Math"
            course = "Unknown"

        meta = {
            "collection": "math_curriculum",
            "subject": subject,
            "course": course,
            "source_file": os.path.basename(path),
        }

        if (
            "module-overview" in fname
            or ("topic-" in fname and "overview" in fname)
            or "overview" in fname
            or "assessment" in fname
            or "copy-ready-materials" in fname
        ):
            meta["doc_type"] = "curriculum_overview"
            return "curriculum_overview", meta
        return None, None

    return None, None


def normalize_ws(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_ws(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]

        if end < n:
            back = max(chunk.rfind("\n\n"), chunk.rfind("\n"), chunk.rfind(". "))
            if back > int(0.6 * len(chunk)):
                end = start + back + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if len(chunk) > 80:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def iter_pdf_pages(path: str):
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = normalize_ws(txt)
        if txt:
            yield i, txt


def make_doc_id(path: str, page: Optional[int], chunk_idx: int) -> str:
    base = os.path.basename(path)
    p = f"p{page}" if page is not None else "pNA"
    return f"{base}_{p}_c{chunk_idx:03d}"


@dataclass
class Chunk:
    text: str
    doc_id: str
    source: str
    page: Optional[int]
    meta: Dict[str, Any]


def walk_pdf_files(root_dir: str) -> List[str]:
    pdfs = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            path = os.path.join(root, f)
            if is_pdf(path):
                pdfs.append(path)
    return sorted(pdfs)


def build_chunks_for_file(path: str, index_name: str, base_meta: Dict[str, Any]) -> List[Chunk]:
    if index_name in {"exam_questions", "exam_scoring"}:
        chunk_size = CHUNK_SIZE_EXAMS
        overlap = CHUNK_OVERLAP_EXAMS
    else:
        chunk_size = CHUNK_SIZE_STANDARDS
        overlap = CHUNK_OVERLAP_STANDARDS

    chunks: List[Chunk] = []
    for page_num, page_text in iter_pdf_pages(path):
        parts = split_text(page_text, chunk_size, overlap)
        for j, part in enumerate(parts):
            doc_id = make_doc_id(path, page_num, j)
            meta = dict(base_meta)
            meta["index_name"] = index_name
            chunks.append(
                Chunk(
                    text=part,
                    doc_id=doc_id,
                    source=path,
                    page=page_num,
                    meta=meta,
                )
            )
    return chunks


def ingest_all_chunks(root_dir: str) -> Dict[str, List[Chunk]]:
    buckets: Dict[str, List[Chunk]] = {
        "curriculum_overview": [],
        "exam_questions": [],
        "exam_scoring": [],
    }

    pdf_files = walk_pdf_files(root_dir)
    print("Found PDFs:", len(pdf_files))

    skipped = 0
    for path in pdf_files:
        idx_name, meta = classify_file(path)
        if idx_name is None:
            skipped += 1
            continue

        file_chunks = build_chunks_for_file(path, idx_name, meta)
        if file_chunks:
            buckets[idx_name].extend(file_chunks)

    print("Skipped PDFs (by rules):", skipped)
    for k, v in buckets.items():
        print(f"{k}: {len(v)} chunks")
    return buckets


def embed_texts(embed_model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = embed_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return np.asarray(vecs, dtype=np.float32)


def build_faiss_ip_index(vecs: np.ndarray) -> faiss.Index:
    if vecs.shape[0] == 0:
        return faiss.IndexFlatIP(384)
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index


def save_index(name: str, index: faiss.Index, meta_rows: List[Dict[str, Any]], index_dir: str) -> None:
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, f"{name}.faiss"))
    with open(os.path.join(index_dir, f"{name}.meta.jsonl"), "w", encoding="utf-8") as f:
        for r in meta_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_and_save_all_indexes(root_dir: str, index_dir: str, force_rebuild: bool = False) -> None:
    needed = ["curriculum_overview", "exam_questions", "exam_scoring"]

    if not force_rebuild:
        ok = True
        for n in needed:
            if not (
                os.path.exists(os.path.join(index_dir, f"{n}.faiss"))
                and os.path.exists(os.path.join(index_dir, f"{n}.meta.jsonl"))
            ):
                ok = False
                break
        if ok:
            print("Indexes already exist. Pass --force-rebuild to rebuild.")
            return

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    buckets = ingest_all_chunks(root_dir)

    for name in needed:
        chunks = buckets[name]
        texts = [c.text for c in chunks]
        print(f"\nEmbedding '{name}' with {len(texts)} chunks...")

        vecs = (
            embed_texts(embed_model, texts)
            if texts
            else np.zeros((0, 384), dtype=np.float32)
        )
        index = build_faiss_ip_index(vecs)

        meta_rows = []
        for c in chunks:
            meta_rows.append(
                {
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "source": c.source,
                    "page": c.page,
                    **c.meta,
                }
            )

        save_index(name, index, meta_rows, index_dir)
        print(f"Saved {name}: {index.ntotal} vectors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/save FAISS indexes for the HS teacher assistant.")
    parser.add_argument("--root-dir", default=ROOT_DIR)
    parser.add_argument("--index-dir", default=INDEX_DIR)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    build_and_save_all_indexes(
        root_dir=args.root_dir,
        index_dir=args.index_dir,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
