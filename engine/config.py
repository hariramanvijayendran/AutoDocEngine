"""
Central configuration for the Autonomous Document Workflow Engine.
All tunable parameters live here — change them without touching agent code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
EVENTS_DB_PATH = DATA_DIR / "events.db"

# Create directories if they don't exist
for d in [UPLOADS_DIR, VECTOR_STORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ollama / LLM Settings ───────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MAIN_LLM_MODEL = "llama3.1:8b"           # For classification, extraction, summary
EMBEDDING_MODEL = "nomic-embed-text"      # For FAISS vector indexing

# ── RAG Settings ────────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVER_K = 4                           # Top-k docs returned by retriever

# ── SOP Genetic Evolution Engine Settings ───────────────────────────────────
EVOLUTION_POPULATION_SIZE = 8             # Genomes per generation
EVOLUTION_GENERATIONS = 5                 # Number of evolution cycles
EVOLUTION_TOP_K = 3                       # Survivors kept per generation (elitism)
EVOLUTION_MUTATION_RATE = 0.3             # Per-gene mutation probability
EVOLUTION_PARALLEL_WORKERS = 4           # Max concurrent evaluations
EVOLUTION_DB_PATH = DATA_DIR / "evolution.db"  # Separate DB for evolution runs

# ── Document Classification Labels ─────────────────────────────────────────
DOCUMENT_TYPES = [
    "contract",
    "invoice",
    "research_paper",
    "legal_document",
    "report",
    "email",
    "general",
]

# ── API Settings ────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".doc"}
