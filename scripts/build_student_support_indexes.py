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

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))

# Defaults mirror the notebook teammate's structure, but are script-friendly.
ROOT_DIR = os.getenv("ROOT_DIR", "./midterm_data")
STUDENT_SUPPORT_ROOT = os.getenv("STUDENT_SUPPORT_ROOT", ROOT_DIR)
STUDENT_SUPPORT_INDEX_DIR = os.getenv("STUDENT_SUPPORT_INDEX_DIR", "./indexes_student_support")


@dataclass
class Chunk:
    text: str
    doc_id: str
    source: str
    page: Optional[int]
    meta: Dict[str, Any]


def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def normalize_ws(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
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


def walk_pdf_files(root_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            path = os.path.join(root, fname)
            if is_pdf(path):
                out.append(path)
    out.sort()
    return out


def iter_pdf_pages(path: str):
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = normalize_ws(txt)
        if txt:
            yield i, txt


def make_doc_id(path: str, page_num: int, chunk_idx: int) -> str:
    # Use relative path + page + chunk to avoid collisions across repeated filenames.
    base = os.path.normpath(path).replace("\\", "/")
    return f"{base}::p{page_num}::c{chunk_idx}"


def classify_file(path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    norm = path.replace("\\", "/")

    if "/College Application/" in norm:
        meta = {
            "collection": "student_support",
            "category": "college_application",
            "doc_type": "college_info",
            "source_file": os.path.basename(path),
        }
        return "college_info", meta

    # Preserve the notebook teammate's folder spelling, but also allow the corrected spelling.
    if "/Time_managemet/" in norm or "/Time_management/" in norm:
        meta = {
            "collection": "student_support",
            "category": "time_management",
            "doc_type": "time_management",
            "source_file": os.path.basename(path),
        }
        return "time_management", meta

    return None, None


def build_chunks_for_file(path: str, index_name: str, base_meta: Dict[str, Any]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_num, page_text in iter_pdf_pages(path):
        parts = split_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
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
        "college_info": [],
        "time_management": [],
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


def build_and_save_all_student_support_indexes(root_dir: str, index_dir: str, force_rebuild: bool = False) -> None:
    needed = ["college_info", "time_management"]

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
            print("Student-support indexes already exist. Pass --force-rebuild to rebuild.")
            return

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    buckets = ingest_all_chunks(root_dir)

    for name in needed:
        chunks = buckets[name]
        texts = [c.text for c in chunks]
        print(f"\nEmbedding '{name}' with {len(texts)} chunks...")

        vecs = embed_texts(embed_model, texts) if texts else np.zeros((0, 384), dtype=np.float32)
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
    parser = argparse.ArgumentParser(description="Build/save FAISS indexes for student-support documents.")
    parser.add_argument("--root-dir", default=STUDENT_SUPPORT_ROOT)
    parser.add_argument("--index-dir", default=STUDENT_SUPPORT_INDEX_DIR)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    build_and_save_all_student_support_indexes(
        root_dir=args.root_dir,
        index_dir=args.index_dir,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
