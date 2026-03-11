"""
Tests for the FitnessResult scoring logic (heuristic functions only).
These tests do NOT make LLM calls — they test the math/heuristics directly.
"""
import pytest
from engine.evolution.fitness import (
    _extract_key_terms,
    _score_completeness,
    _score_conciseness,
    _score_extraction_coverage,
)


class TestKeyTermExtraction:
    def test_returns_list(self):
        terms = _extract_key_terms("hello world test document content")
        assert isinstance(terms, list)

    def test_filters_short_words(self):
        terms = _extract_key_terms("this is a test of the extraction function")
        assert "this" not in terms  # stop word
        assert "extraction" in terms or "function" in terms

    def test_respects_top_n(self):
        terms = _extract_key_terms("word " * 100, top_n=5)
        assert len(terms) <= 5


class TestCompletenessScore:
    def test_perfect_overlap(self):
        source = "The patient underwent surgery for cardiac treatment"
        summary = "The patient underwent surgery for cardiac treatment and recovery"
        score = _score_completeness(source, summary)
        assert score > 0.7

    def test_zero_overlap(self):
        source = "quantum entanglement phenomenon"
        summary = "The weather today is sunny and warm outside"
        score = _score_completeness(source, summary)
        assert score < 0.4

    def test_empty_output(self):
        assert _score_completeness("some source text here", "") == 0.0

    def test_range(self):
        score = _score_completeness("document with many words", "partial output")
        assert 0.0 <= score <= 1.0


class TestConcisenessScore:
    def test_ideal_ratio_scores_high(self):
        # ~15% ratio: 100-word source, 15-word output
        source = "word " * 100
        output = "word " * 15
        score = _score_conciseness(source, output)
        assert score > 0.8

    def test_too_long_output_scores_lower(self):
        source = "short source"
        output = "very very very very very very very very very long output " * 20
        score_long = _score_conciseness(source, output)
        output_ideal = "short output summary"
        score_ideal = _score_conciseness(source, output_ideal)
        assert score_ideal > score_long

    def test_empty_returns_zero(self):
        assert _score_conciseness("source text", "") == 0.0

    def test_range(self):
        score = _score_conciseness("some document text here", "summary text")
        assert 0.0 <= score <= 1.0


class TestExtractionCoverage:
    def test_all_filled(self):
        extracted = {"parties": "Alice and Bob", "date": "2024-01-01", "amount": "1000"}
        assert _score_extraction_coverage(extracted) == 1.0

    def test_none_values_count_as_empty(self):
        extracted = {"parties": None, "date": None, "amount": None}
        assert _score_extraction_coverage(extracted) == 0.0

    def test_partial_fill(self):
        extracted = {"parties": "Alice", "date": None, "amount": "500"}
        score = _score_extraction_coverage(extracted)
        assert abs(score - 2/3) < 0.01

    def test_empty_dict(self):
        assert _score_extraction_coverage({}) == 0.0

    def test_none_input(self):
        assert _score_extraction_coverage(None) == 0.0
