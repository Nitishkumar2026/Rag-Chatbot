"""
Document Processor: Cleans and chunks raw documents for the RAG pipeline.

Usage:
    python src/document_processor.py
    python src/document_processor.py --doc data/myfile.txt --chunk-size 600
"""

import os
import re
import json
import argparse
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter

import pypdf

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / "data"
CHUNKS_DIR = Path(__file__).parent.parent / "chunks"


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def extract_pdf_text(file_stream) -> str:
    """
    Extract raw text from a PDF byte stream using pypdf.
    """
    reader = pypdf.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def clean_text(text: str) -> str:
    """
    Clean raw document text:
      - Strip HTML/XML tags
      - Normalise whitespace and newlines
      - Remove decorative separator lines (━, ─, =, -)
      - Collapse multiple blank lines to one
    """
    # Remove HTML / XML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove markdown / document decorators (e.g. ━━━, ----, ====)
    text = re.sub(r"[━─=\-]{4,}", " ", text)

    # Remove URLs (keep domain for reference)
    text = re.sub(r"https?://\S+", "[URL]", text)

    # Normalise unicode whitespace characters
    text = re.sub(r"[\u2000-\u200f\u2028\u2029\xa0]", " ", text)

    # Collapse multiple spaces / tabs into one space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[dict]:
    """
    Split text into sentence-aware overlapping chunks (~100-300 words each).

    Args:
        text:          Cleaned document text.
        chunk_size:    Target chunk size in characters (600 chars ≈ ~100 words).
        chunk_overlap: Overlap between adjacent chunks in characters.

    Returns:
        List of chunk dicts with id, text, word_count, char_count.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, raw in enumerate(raw_chunks):
        raw = raw.strip()
        if len(raw.split()) < 10:          # skip near-empty chunks
            continue
        chunks.append({
            "id":         i,
            "text":       raw,
            "word_count": len(raw.split()),
            "char_count": len(raw),
        })

    return chunks


# ── Persistence ───────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], output_path: Path = None) -> Path:
    """Serialise chunks to a JSON file and return the path."""
    if output_path is None:
        output_path = CHUNKS_DIR / "chunks.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, indent=2, ensure_ascii=False)

    print(f"[OK] Saved {len(chunks)} chunks -> {output_path}")
    return output_path


def load_chunks(chunks_path: Path = None) -> list[dict]:
    """Load chunks from a JSON file."""
    if chunks_path is None:
        chunks_path = CHUNKS_DIR / "chunks.json"

    with open(chunks_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def process_document(doc_path: Path = None, chunk_size: int = 600) -> list[dict]:
    """
    End-to-end document processing pipeline:
      1. Load raw text from file
      2. Clean text
      3. Chunk into segments
      4. Save chunks to disk

    Returns the list of chunk dicts.
    """
    if doc_path is None:
        doc_path = DATA_DIR / "document.txt"

    print(f"[DOC] Loading document: {doc_path}")
    with open(doc_path, "r", encoding="utf-8") as fh:
        raw_text = fh.read()

    original_words = len(raw_text.split())
    print(f"[STATS] Original size  : {original_words:,} words | {len(raw_text):,} chars")

    print("[CLEAN] Cleaning text ...")
    cleaned = clean_text(raw_text)
    print(f"    After clean    : {len(cleaned.split()):,} words")

    print(f"[CHUNK] Chunking (size={chunk_size} chars, overlap=80) ...")
    chunks = chunk_text(cleaned, chunk_size=chunk_size)

    word_counts = [c["word_count"] for c in chunks]
    avg_words   = sum(word_counts) / len(word_counts) if word_counts else 0
    print(f"[INFO] Chunks created : {len(chunks)}")
    print(f"    Avg words/chunk: {avg_words:.0f}")
    print(f"    Min / Max words: {min(word_counts)} / {max(word_counts)}")

    save_chunks(chunks)
    return chunks


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process & chunk a document for RAG")
    parser.add_argument("--doc",        type=str, default=None,
                        help="Path to the input document (default: data/document.txt)")
    parser.add_argument("--chunk-size", type=int, default=600,
                        help="Chunk size in characters (default: 600 ≈ 100 words)")
    args = parser.parse_args()

    path   = Path(args.doc) if args.doc else None
    chunks = process_document(path, chunk_size=args.chunk_size)
    print(f"\n[OK] Processing complete! {len(chunks)} chunks ready in chunks/chunks.json")
