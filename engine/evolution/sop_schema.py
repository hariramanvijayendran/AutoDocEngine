"""
SOP Genome Schema — typed, validated configuration that the Genetic Evolution Engine
can safely mutate, crossover, and persist. Uses dataclasses for Python 3.14 compat
(avoids Pydantic v1/v2 conflicts that exist in the rest of the project).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import datetime


# ── Prompt Variant Libraries ─────────────────────────────────────────────────
# Instead of mutating raw prompt strings (brittle), the engine picks from
# these curated, semantically-different variants. Index = gene value.

SUMMARY_PROMPT_VARIANTS = [
    # Variant 0 — concise, 3-5 sentences (current default)
    "You are a professional document summarizer. Write a clear, concise summary in 3-5 sentences. "
    "Focus on the main purpose, key points, and important conclusions. Return ONLY the summary text.",

    # Variant 1 — structured bullets
    "You are an expert analyst. Summarize this document as exactly 3 bullet points: "
    "(1) Main Purpose (2) Key Facts (3) Required Actions. Be precise and factual.",

    # Variant 2 — executive brief, single paragraph, max 80 words
    "You are a C-suite briefing writer. Summarize the document in ONE dense paragraph of at most 80 words. "
    "Every word must carry weight. No filler. Return ONLY the paragraph.",

    # Variant 3 — structured with header sections
    "You are a document analyst. Return a summary with these labeled sections: "
    "OVERVIEW: (1-2 sentences), KEY POINTS: (3 bullets), ACTION ITEMS: (if any, else 'None'). "
    "Return only this structured text.",
]

EXTRACTION_PROMPT_VARIANTS = [
    # Variant 0 — standard extraction (current default)
    "You are an expert document analyst. Extract the requested fields from the document. "
    "Respond ONLY with a valid JSON object. Use null for missing fields.",

    # Variant 1 — chain-of-thought then extract
    "You are an expert document analyst. First, briefly identify what type of document this is. "
    "Then extract the requested fields. Respond ONLY with a valid JSON object. Use null for missing fields.",

    # Variant 2 — high-confidence only
    "You are a precise information extractor. Extract ONLY fields you are highly confident about. "
    "FOR UNCERTAIN FIELDS USE null. Respond ONLY with a valid JSON object.",
]

CLASSIFICATION_PROMPT_VARIANTS = [
    # Variant 0 — standard (current default)
    "You are a document classification expert. Classify the document into exactly ONE category. "
    "Respond ONLY with JSON: {\"label\": \"<category>\", \"confidence\": \"<high|medium|low>\", \"reasoning\": \"<one sentence>\"}",

    # Variant 1 — multi-step reasoning
    "You are a document classification expert. First identify 3 textual signals, then classify. "
    "Respond ONLY with JSON: {\"label\": \"<category>\", \"confidence\": \"<high|medium|low>\", \"reasoning\": \"<one sentence>\"}",
]

VALID_MODELS = ["llama3.1:8b", "qwen2:7b"]


@dataclass
class SOPGenome:
    """
    A complete, evolvable Standard Operating Procedure.
    Every field is a gene that the evolution engine can mutate or crossover.
    """
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── Chunking Genes ──────────────────────────────────────────────────────
    chunk_size: int = 800         # Range: 400–1200
    chunk_overlap: int = 100      # Range: 50–200 (must be < chunk_size)

    # ── Retrieval Genes ──────────────────────────────────────────────────────
    retriever_k: int = 4          # Range: 2–8

    # ── Model Genes ──────────────────────────────────────────────────────────
    llm_model: str = "llama3.1:8b"

    # ── Prompt Variant Genes (index into variant library) ───────────────────
    summary_prompt_variant: int = 0        # 0–3
    extraction_prompt_variant: int = 0     # 0–2
    classification_prompt_variant: int = 0 # 0–1

    # ── Feature Flag Genes ───────────────────────────────────────────────────
    use_extraction: bool = True
    use_summary: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SOPGenome:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get_summary_prompt(self) -> str:
        idx = max(0, min(self.summary_prompt_variant, len(SUMMARY_PROMPT_VARIANTS) - 1))
        return SUMMARY_PROMPT_VARIANTS[idx]

    def get_extraction_prompt_prefix(self) -> str:
        idx = max(0, min(self.extraction_prompt_variant, len(EXTRACTION_PROMPT_VARIANTS) - 1))
        return EXTRACTION_PROMPT_VARIANTS[idx]

    def get_classification_prompt(self, doc_types_str: str) -> str:
        idx = max(0, min(self.classification_prompt_variant, len(CLASSIFICATION_PROMPT_VARIANTS) - 1))
        return CLASSIFICATION_PROMPT_VARIANTS[idx] + f"\nValid categories: {doc_types_str}"

    def clamp(self) -> SOPGenome:
        """Enforce gene boundary constraints after mutation."""
        self.chunk_size = max(400, min(1200, self.chunk_size))
        self.chunk_overlap = max(50, min(200, min(self.chunk_size - 50, self.chunk_overlap)))
        self.retriever_k = max(2, min(8, self.retriever_k))
        self.llm_model = self.llm_model if self.llm_model in VALID_MODELS else "llama3.1:8b"
        self.summary_prompt_variant = max(0, min(len(SUMMARY_PROMPT_VARIANTS) - 1, self.summary_prompt_variant))
        self.extraction_prompt_variant = max(0, min(len(EXTRACTION_PROMPT_VARIANTS) - 1, self.extraction_prompt_variant))
        self.classification_prompt_variant = max(0, min(len(CLASSIFICATION_PROMPT_VARIANTS) - 1, self.classification_prompt_variant))
        return self


@dataclass
class FitnessResult:
    """
    Multi-dimensional score for one SOPGenome evaluation run.
    Each dimension is 0.0–1.0, higher is better.
    """
    genome_id: str
    completeness: float = 0.0    # Are key entities from source present in output?
    conciseness: float = 0.0     # Is the output an appropriate length?
    coherence: float = 0.0       # Is the output readable and well-structured?
    extraction_coverage: float = 0.0  # % of schema fields successfully filled
    weighted_total: float = 0.0  # Final fitness score
    error: Optional[str] = None
    eval_time_seconds: float = 0.0

    def compute_weighted_total(
        self,
        w_completeness: float = 0.35,
        w_conciseness: float = 0.20,
        w_coherence: float = 0.25,
        w_extraction: float = 0.20,
    ) -> float:
        self.weighted_total = (
            self.completeness * w_completeness
            + self.conciseness * w_conciseness
            + self.coherence * w_coherence
            + self.extraction_coverage * w_extraction
        )
        return self.weighted_total

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationRecord:
    """Snapshot of one generation's population and scores."""
    generation: int
    population: List[dict]       # List of SOPGenome.to_dict()
    fitness_results: List[dict]  # List of FitnessResult.to_dict()
    best_fitness: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EvolutionRun:
    """Top-level record for a complete evolution run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    status: str = "pending"          # pending | running | complete | error
    test_document_id: str = ""
    population_size: int = 8
    generations: int = 5
    current_generation: int = 0
    best_fitness: float = 0.0
    best_genome: Optional[dict] = None
    pareto_front: List[dict] = field(default_factory=list)
    generation_history: List[dict] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
