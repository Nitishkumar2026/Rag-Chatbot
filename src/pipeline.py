"""
RAG Pipeline: Orchestrates retriever + generator for end-to-end inference.

Usage:
    python src/pipeline.py --query "What is the refund policy?"
    python src/pipeline.py --query "How is data encrypted?" --top-k 5
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Generator

# Ensure root is on sys.path for direct-script invocation
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Suppress HuggingFace transformers warnings BEFORE importing sentence_transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from src.retriever import FAISSRetriever
from src.generator import build_prompt, stream_response, get_model_info
from src.document_processor import clean_text, chunk_text
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.

    Flow:
        user query
            ↓
        FAISSRetriever  → top-k semantically similar chunks
            ↓
        build_prompt    → context + query → prompt string
            ↓
        stream_response → token generator (Ollama or HuggingFace)
    """

    def __init__(self, top_k: int = 3):
        """
        Initialise RAG pipeline.

        Args:
            top_k: Default number of chunks to retrieve per query.
        """
        print("[PIPELINE] Initialising RAG Pipeline ...")
        self.top_k      = top_k
        self.retriever  = FAISSRetriever()
        self.model_info = get_model_info()
        print(f"[LLM] Model: {self.model_info['display_name']}")
        print("[OK] Pipeline ready!\n")

    # ── Static Methods ────────────────────────────────────────────────────────

    def build_temp_retriever(self, text: str) -> FAISSRetriever:
        """
        Create an in-memory retriever for a small text document (e.g. uploaded PDF).
        """
        print("[CLEAN] Cleaning text ...")
        cleaned = clean_text(text)
        
        print("[CHUNK] Chunking ...")
        chunks = chunk_text(cleaned, chunk_size=640)
        
        print("[EMBED] Generating embeddings ...")
        model   = SentenceTransformer("all-MiniLM-L6-v2")
        texts   = [c["text"] for c in chunks]
        ebs     = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        
        dim     = ebs.shape[1]
        index   = faiss.IndexFlatIP(dim)
        index.add(ebs.astype(np.float32))
        
        return FAISSRetriever(index=index, chunks=chunks)

    # ── Core API ──────────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        top_k: int = None,
        retriever: FAISSRetriever = None,
    ) -> tuple[Generator, list[dict]]:
        """
        Execute the full RAG pipeline for a single user query.

        Returns:
            (token_generator, source_chunks)
            • token_generator  : yields str tokens for streaming
            • source_chunks    : list of retrieved chunk dicts
        """
        k = top_k or self.top_k
        
        # Use provided retriever (uploaded file) or default (Global KB)
        active_retriever = retriever or self.retriever

        # ── Step 1 : Retrieve ───────────────────────────────────────────────
        source_chunks = active_retriever.retrieve(query, top_k=k)

        if not source_chunks:
            def _empty():
                yield "I couldn't find any relevant information."
            return _empty(), []

        # ── Step 2 : Build context ──────────────────────────────────────────
        context = active_retriever.get_context_string(source_chunks)

        # ── Step 3 : Build prompt ───────────────────────────────────────────
        prompt = build_prompt(context, query)

        # ── Step 4 : Stream response ────────────────────────────────────────
        return stream_response(prompt), source_chunks

    # ── Meta / Diagnostics ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Pipeline statistics for display in the UI sidebar."""
        return {
            "model":       self.model_info["display_name"],
            "chunk_count": self.retriever.chunk_count,
            "top_k":       self.top_k,
        }


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline from CLI")
    parser.add_argument("--query", type=str, required=True,
                        help="User question to answer")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of chunks to retrieve (default: 3)")
    args = parser.parse_args()

    pipeline = RAGPipeline(top_k=args.top_k)

    print(f"QUERY: {args.query}\n")
    print("ANSWER:\n" + "-" * 60)

    token_gen, sources = pipeline.run(args.query)

    for token in token_gen:
        print(token, end="", flush=True)

    print(f"\n{'─'*60}")
    print(f"\nSOURCES ({len(sources)} chunks retrieved):")
    for chunk in sources:
        score = chunk["similarity_score"]
        text  = chunk["text"][:250].replace("\n", " ")
        print(f"\n  [Chunk {chunk['rank']} | Score: {score:.4f}]")
        print(f"  {text}…")
