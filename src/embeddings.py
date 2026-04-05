"""
Embeddings: Generate sentence embeddings and build a FAISS vector index.

Usage:
    python src/embeddings.py
    python src/embeddings.py --chunks chunks/chunks.json
"""

import json
import pickle
import argparse
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss

# ── Paths & Config ────────────────────────────────────────────────────────────
CHUNKS_DIR    = Path(__file__).parent.parent / "chunks"
VECTORDB_DIR  = Path(__file__).parent.parent / "vectordb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Model Loader ──────────────────────────────────────────────────────────────

def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Download (if needed) and load the SentenceTransformer embedding model."""
    print(f"[MODEL] Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    dim   = model.get_sentence_embedding_dimension()
    print(f"[OK] Model ready  |  embedding dim = {dim}")
    return model


# ── Embedding Generation ──────────────────────────────────────────────────────

def generate_embeddings(
    chunks: list[dict],
    model:  SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode all chunk texts into L2-normalised embeddings.

    Normalised embeddings allow cosine similarity via inner-product (IndexFlatIP).
    """
    texts = [c["text"] for c in chunks]
    print(f"[EMBED] Embedding {len(texts)} chunks ...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-norm → cosine sim = inner product
        convert_to_numpy=True,
    )

    print(f"[OK] Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ── FAISS Index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat inner-product index.

    Because vectors are L2-normalised, inner product = cosine similarity.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Exact search — good for < 100 k chunks
    index.add(embeddings)
    print(f"[OK] FAISS index  |  {index.ntotal} vectors  |  dim = {dim}")
    return index


# ── Persistence ───────────────────────────────────────────────────────────────

def save_index(
    index:      faiss.Index,
    chunks:     list[dict],
    output_dir: Path = None,
) -> None:
    """Persist the FAISS index (.faiss) and chunk metadata (.pkl) to disk."""
    if output_dir is None:
        output_dir = VECTORDB_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    faiss_path    = output_dir / "index.faiss"
    metadata_path = output_dir / "index.pkl"

    faiss.write_index(index, str(faiss_path))
    print(f"[OK] FAISS index saved  -> {faiss_path}")

    with open(metadata_path, "wb") as fh:
        pickle.dump(chunks, fh)
    print(f"[OK] Metadata saved     -> {metadata_path}")


def load_index(index_dir: Path = None) -> tuple[faiss.Index, list[dict]]:
    """Load a previously saved FAISS index and its chunk metadata."""
    if index_dir is None:
        index_dir = VECTORDB_DIR

    faiss_path    = index_dir / "index.faiss"
    metadata_path = index_dir / "index.pkl"

    if not faiss_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{index_dir}'.\n"
            "Please run the setup pipeline first:\n"
            "  python src/document_processor.py\n"
            "  python src/embeddings.py"
        )

    index = faiss.read_index(str(faiss_path))

    with open(metadata_path, "rb") as fh:
        chunks = pickle.load(fh)

    print(f"[OK] Loaded FAISS index  |  {index.ntotal} vectors")
    return index, chunks


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def build_vector_store(chunks_path: Path = None) -> None:
    """
    End-to-end embedding pipeline:
      1. Load chunks from JSON
      2. Generate embeddings
      3. Build FAISS index
      4. Save to vectordb/
    """
    if chunks_path is None:
        chunks_path = CHUNKS_DIR / "chunks.json"

    print(f"[INFO] Loading chunks: {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)
    print(f"    {len(chunks)} chunks loaded")

    model      = load_embedding_model()
    embeddings = generate_embeddings(chunks, model)
    index      = build_faiss_index(embeddings)
    save_index(index, chunks)

    print(f"\n[OK] Vector store ready!  {len(chunks)} chunks indexed in vectordb/")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vector store from chunks")
    parser.add_argument(
        "--chunks", type=str, default=None,
        help="Path to chunks.json (default: chunks/chunks.json)"
    )
    args = parser.parse_args()

    chunks_path = Path(args.chunks) if args.chunks else None
    build_vector_store(chunks_path)
