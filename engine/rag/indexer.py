"""
RAG Indexer — builds and updates the FAISS vector store from document chunks.
Uses faiss-cpu, numpy, and ollama Python SDK directly.
Zero langchain dependencies for Python 3.14 compatibility.
"""
import json
import numpy as np
from engine.config import VECTOR_STORE_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
FAISS_META_PATH  = VECTOR_STORE_DIR / "faiss_meta.json"


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Pure Python recursive text splitter."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break on a newline or space near the end
        if end < len(text):
            for sep in ["\n\n", "\n", " "]:
                idx = text.rfind(sep, start, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return [c.strip() for c in chunks if c.strip()]


def _get_embeddings(texts):
    """Call Ollama embedding model via SDK and return numpy array."""
    import ollama
    vecs = []
    for text in texts:
        resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        vecs.append(resp["embedding"])
    return np.array(vecs, dtype=np.float32)


def _load_index():
    import faiss
    if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    return None, []


def _save_index(index, meta: list):
    import faiss
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def index_document(document_id: str, filename: str, text: str) -> bool:
    """Chunk, embed, and add document to persistent FAISS index."""
    import faiss
    try:
        chunks = _split_text(text)
        if not chunks:
            print(f"[Indexer] ⚠️  No chunks from '{filename}'")
            return False

        print(f"[Indexer] Embedding {len(chunks)} chunks for '{filename}'…")
        vecs = _get_embeddings(chunks)

        index, meta = _load_index()
        if index is None:
            index = faiss.IndexFlatL2(vecs.shape[1])

        index.add(vecs)
        for i, chunk in enumerate(chunks):
            meta.append({"document_id": document_id, "filename": filename, "chunk_idx": i, "text": chunk})

        _save_index(index, meta)
        print(f"[Indexer] ✅ Indexed {len(chunks)} chunks for '{filename}'")
        return True
    except Exception as e:
        print(f"[Indexer] ❌ Failed to index '{filename}': {e}")
        return False
