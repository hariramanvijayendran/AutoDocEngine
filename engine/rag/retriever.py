"""
RAG Retriever — semantic search over the FAISS vector store.
Uses faiss-cpu, numpy, and ollama SDK directly (no langchain).
Python 3.14 compatible.
"""
import json
import numpy as np
from typing import List, Optional
from engine.config import VECTOR_STORE_DIR, EMBEDDING_MODEL, RETRIEVER_K

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
FAISS_META_PATH  = VECTOR_STORE_DIR / "faiss_meta.json"


def _embed_query(query: str) -> np.ndarray:
    import ollama
    resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    return np.array([resp["embedding"]], dtype=np.float32)


def retrieve(query: str, document_id: Optional[str] = None, k: int = RETRIEVER_K) -> List[str]:
    """Retrieve top-k relevant chunks. Optionally filter by document_id."""
    import faiss

    if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        print("[Retriever] ⚠️  No FAISS index found.")
        return []

    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        query_vec = _embed_query(query)
        fetch_k = min(k * 5 if document_id else k, index.ntotal)
        if fetch_k == 0:
            return []

        distances, indices = index.search(query_vec, fetch_k)
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(meta):
                continue
            entry = meta[idx]
            if document_id and entry["document_id"] != document_id:
                continue
            results.append(entry["text"])
            if len(results) >= k:
                break

        return results
    except Exception as e:
        print(f"[Retriever] ❌ Retrieval failed: {e}")
        return []
