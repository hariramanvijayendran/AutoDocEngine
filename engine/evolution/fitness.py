"""
Fitness Evaluator — runs the pipeline with a given SOPGenome configuration and
produces a multi-dimensional FitnessResult. No ground truth labels needed:
scoring is done through heuristics + a lightweight LLM coherence check.

Dimensions:
  - completeness (0–1): Do key terms from the source appear in the output?
  - conciseness  (0–1): Is output length appropriate relative to input?
  - coherence    (0–1): LLM rates readability/structure on a 1–5 scale
  - extraction_coverage (0–1): % of schema fields non-null in extracted dict
"""
from __future__ import annotations

import time
import math
import re
import json
import ollama
from pathlib import Path
from typing import Optional

from engine.evolution.sop_schema import SOPGenome, FitnessResult
from engine.config import MAIN_LLM_MODEL


# ── Coherence Check via Ollama ────────────────────────────────────────────────

_COHERENCE_SYSTEM = """You are an objective text quality evaluator.
Rate the QUALITY of the following text on a scale from 1 to 5:
  1 = Incoherent, incomplete, or off-topic
  2 = Partially useful but missing key information
  3 = Adequate — covers the main points
  4 = Good — clear, structured, relevant
  5 = Excellent — precise, complete, and well-written

Respond ONLY with a JSON object: {"score": <integer 1-5>, "reason": "<brief reason>"}"""


def _score_coherence(text: str, model: str = MAIN_LLM_MODEL) -> float:
    """Ask the LLM to rate the output quality. Returns 0.0–1.0."""
    if not text or len(text.strip()) < 20:
        return 0.0
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _COHERENCE_SYSTEM},
                {"role": "user", "content": f"Text to evaluate:\n\n{text[:1500]}"},
            ],
            format="json",
        )
        parsed = json.loads(response.message.content)
        raw_score = int(parsed.get("score", 3))
        return (raw_score - 1) / 4.0  # normalize 1–5 → 0.0–1.0
    except Exception:
        return 0.5  # neutral fallback on LLM error


# ── Completeness Check ────────────────────────────────────────────────────────

def _extract_key_terms(text: str, top_n: int = 30) -> list[str]:
    """
    Simple TF-IDF-like heuristic: extract frequent, meaningful tokens from source.
    Avoids stop words and short tokens.
    """
    stop = {
        "the", "a", "an", "is", "in", "of", "to", "and", "or", "for",
        "this", "that", "be", "are", "was", "were", "it", "on", "at",
        "by", "with", "as", "from", "not", "but", "if", "so", "do",
        "its", "all", "he", "she", "we", "they", "can", "has", "had",
        "have", "will", "shall", "may", "been", "their", "than", "then",
    }
    tokens = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq: dict[str, int] = {}
    for t in tokens:
        if t not in stop:
            freq[t] = freq.get(t, 0) + 1
    sorted_terms = sorted(freq.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_terms[:top_n]]


def _score_completeness(source_text: str, output_text: str) -> float:
    """Fraction of key source terms present in output."""
    if not output_text:
        return 0.0
    key_terms = _extract_key_terms(source_text, top_n=25)
    if not key_terms:
        return 0.5
    out_lower = output_text.lower()
    hits = sum(1 for t in key_terms if t in out_lower)
    return hits / len(key_terms)


# ── Conciseness Check ──────────────────────────────────────────────────────────

def _score_conciseness(source_text: str, output_text: str) -> float:
    """
    Ideal summary length is ~10–20% of source. Score peaks at ~15%, drops at extremes.
    Uses a Gaussian curve centered on 0.15.
    """
    if not output_text or not source_text:
        return 0.0
    ratio = len(output_text) / max(len(source_text), 1)
    # Gaussian: peaks at ratio=0.15, σ=0.10 → score=1 at ideal, ~0.6 at extremes
    score = math.exp(-((ratio - 0.15) ** 2) / (2 * 0.10**2))
    return min(1.0, max(0.0, score))


# ── Extraction Coverage ────────────────────────────────────────────────────────

def _score_extraction_coverage(extracted: Optional[dict]) -> float:
    """Fraction of schema fields that are non-null, non-empty."""
    if not extracted or not isinstance(extracted, dict):
        return 0.0
    filled = sum(
        1 for v in extracted.values()
        if v is not None and v != "" and v != [] and v != "null"
    )
    return filled / max(len(extracted), 1)


# ── Main Evaluator ─────────────────────────────────────────────────────────────

def evaluate_fitness(
    genome: SOPGenome,
    test_doc_path: str,
    coherence_model: Optional[str] = None,
) -> FitnessResult:
    """
    Run the pipeline with the given SOPGenome and compute a FitnessResult.

    Strategy:
    - Temporarily patch the global engine config with this genome's parameters.
    - Run the pipeline in-process.
    - Extract summary, extracted fields from the final state.
    - Score all 4 dimensions.
    - Restore original config values.
    """
    start = time.time()
    result = FitnessResult(genome_id=genome.genome_id)
    model = coherence_model or genome.llm_model

    # ── Patch config at runtime ──────────────────────────────────────────────
    import engine.config as cfg
    _orig_chunk_size = cfg.CHUNK_SIZE
    _orig_chunk_overlap = cfg.CHUNK_OVERLAP
    _orig_retriever_k = cfg.RETRIEVER_K
    _orig_main_model = cfg.MAIN_LLM_MODEL

    try:
        cfg.CHUNK_SIZE = genome.chunk_size
        cfg.CHUNK_OVERLAP = genome.chunk_overlap
        cfg.RETRIEVER_K = genome.retriever_k
        cfg.MAIN_LLM_MODEL = genome.llm_model

        # ── Run pipeline ──────────────────────────────────────────────────────
        from engine.pipeline import run_pipeline_with_sop
        path = Path(test_doc_path)
        if not path.exists():
            result.error = f"Test document not found: {test_doc_path}"
            return result

        doc_id = f"evo_{genome.genome_id}"
        state = run_pipeline_with_sop(doc_id, str(path), genome)

        if state.get("error"):
            result.error = state["error"]
            result.weighted_total = 0.0
            return result

        source_text = state.get("text", "")
        summary = state.get("summary", "") or ""
        extracted = state.get("extracted") or {}

        # ── Score dimensions ──────────────────────────────────────────────────
        result.completeness = _score_completeness(source_text, summary)
        result.conciseness = _score_conciseness(source_text, summary)
        result.coherence = _score_coherence(summary, model=model)
        result.extraction_coverage = _score_extraction_coverage(extracted)
        result.compute_weighted_total()

    except Exception as e:
        result.error = str(e)
        result.weighted_total = 0.0

    finally:
        # ── Restore config ────────────────────────────────────────────────────
        cfg.CHUNK_SIZE = _orig_chunk_size
        cfg.CHUNK_OVERLAP = _orig_chunk_overlap
        cfg.RETRIEVER_K = _orig_retriever_k
        cfg.MAIN_LLM_MODEL = _orig_main_model
        result.eval_time_seconds = round(time.time() - start, 2)

    return result
