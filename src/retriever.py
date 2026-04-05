"""
Retriever: Semantic search over the FAISS vector database.

Usage:
    python src/retriever.py --query "What is the refund policy?"
    python src/retriever.py --query "How is data encrypted?" --top-k 5
"""

import sys
import argparse
import numpy as np
import faiss
from pathlib import Path

# Ensure project root is on sys.path so this module works both as a
# direct script (python src/retriever.py) and as an import (from src.retriever import ...)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sentence_transformers import SentenceTransformer
from src.embeddings import load_index, EMBEDDING_MODEL


class FAISSRetriever:
    """
    Semantic retriever backed by a FAISS flat index.

    Cosine similarity is used (inner-product on L2-normalised vectors).
    """

    def __init__(
        self,
        index_dir:  Path = None,
        model_name: str  = EMBEDDING_MODEL,
        index:      faiss.Index = None,
        chunks:     list[dict]  = None,
    ):
        """
        Initialise retriever. Either load from disk (index_dir) or use provided objects.
        """
        self.model = SentenceTransformer(model_name)
        
        if index is not None and chunks is not None:
            self.index  = index
            self.chunks = chunks
            print(f"[OK] Retriever ready (In-memory) | {len(self.chunks)} chunks")
        else:
            print("[LOAD] Loading retriever from disk ...")
            self.index, self.chunks = load_index(index_dir)
            print(f"[OK] Retriever ready (Disk) | {len(self.chunks)} chunks indexed")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Return the top-k most relevant chunks for *query*.

        Each result dict extends the original chunk dict with:
            - similarity_score (float, 0-1)
            - rank             (int, 1-based)
        """
        # Encode and normalise the query
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # FAISS search
        k      = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:          # FAISS returns -1 when ntotal < k
                continue
            chunk = dict(self.chunks[idx])   # shallow copy
            chunk["similarity_score"] = float(score)
            chunk["rank"]             = rank
            results.append(chunk)

        return results

    def get_context_string(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a numbered context string
        ready to be injected into the LLM prompt.
        """
        parts = [
            f"[Source {c['rank']}]:\n{c['text']}"
            for c in results
        ]
        return "\n\n".join(parts)

    @property
    def chunk_count(self) -> int:
        """Total number of indexed chunks."""
        return len(self.chunks)


# ── CLI / Quick-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the FAISS retriever")
    parser.add_argument("--query", type=str, required=True,
                        help="Natural-language query to search for")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of chunks to retrieve (default: 3)")
    args = parser.parse_args()

    retriever = FAISSRetriever()
    results   = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\n[SEARCH] Query : {args.query}")
    print(f"[STATS] Top-{len(results)} results:\n")
    for r in results:
        score = r["similarity_score"]
        words = r["word_count"]
        text  = r["text"][:300].replace("\n", " ")
        print(f"  ┌─ Rank {r['rank']}  |  Score: {score:.4f}  |  Words: {words}")
        print(f"  │  {text}…")
        print(f"  └{'─'*60}")
