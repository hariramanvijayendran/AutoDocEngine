"""
Tests for evolution genetic operators:
- mutation always produces a valid, different genome
- crossover produces a child whose genes come from one of its two parents
"""
import pytest
from engine.evolution.sop_schema import SOPGenome, VALID_MODELS
from engine.evolution.operators import mutate, crossover, initialize_population


class TestMutation:
    def test_mutation_produces_valid_genome(self):
        base = SOPGenome()
        child = mutate(base, mutation_rate=1.0)  # force mutation of all genes
        assert isinstance(child, SOPGenome)
        assert child.genome_id != base.genome_id
        assert 400 <= child.chunk_size <= 1200
        assert 50 <= child.chunk_overlap <= 200
        assert child.chunk_overlap < child.chunk_size
        assert 2 <= child.retriever_k <= 8
        assert child.llm_model in VALID_MODELS

    def test_mutation_increases_generation(self):
        base = SOPGenome(generation=3)
        child = mutate(base)
        assert child.generation == 4

    def test_mutation_tracks_parent(self):
        base = SOPGenome(genome_id="parent01")
        child = mutate(base)
        assert "parent01" in child.parent_ids

    def test_mutation_always_differs(self):
        """Forced mutation_rate=1.0 must produce a different genome."""
        base = SOPGenome()
        children = {mutate(base, mutation_rate=1.0).genome_id for _ in range(10)}
        # All 10 children should have unique IDs
        assert len(children) == 10


class TestCrossover:
    def test_crossover_genes_from_parents(self):
        pa = SOPGenome(chunk_size=400, retriever_k=2, llm_model="llama3.1:8b")
        pb = SOPGenome(chunk_size=1200, retriever_k=8, llm_model="qwen2:7b")
        child = crossover(pa, pb)
        assert child.chunk_size in (400, 1200)
        assert child.retriever_k in (2, 8)
        assert child.llm_model in ("llama3.1:8b", "qwen2:7b")

    def test_crossover_tracks_both_parents(self):
        pa = SOPGenome(genome_id="aaaa")
        pb = SOPGenome(genome_id="bbbb")
        child = crossover(pa, pb)
        assert "aaaa" in child.parent_ids
        assert "bbbb" in child.parent_ids

    def test_crossover_increments_generation(self):
        pa = SOPGenome(generation=2)
        pb = SOPGenome(generation=4)
        child = crossover(pa, pb)
        assert child.generation == 5  # max(2, 4) + 1

    def test_crossover_valid_boundaries(self):
        pa = SOPGenome(chunk_size=400)
        pb = SOPGenome(chunk_size=800)
        for _ in range(20):
            child = crossover(pa, pb)
            assert 400 <= child.chunk_size <= 1200
            assert child.chunk_overlap < child.chunk_size


class TestPopulationInit:
    def test_init_correct_size(self):
        pop = initialize_population(8, seed=42)
        assert len(pop) == 8

    def test_first_is_baseline(self):
        """First genome is always the production baseline."""
        pop = initialize_population(4, seed=0)
        assert pop[0].genome_id == "baseline"
        assert pop[0].chunk_size == 800  # default

    def test_all_valid_genomes(self):
        pop = initialize_population(10, seed=7)
        for g in pop:
            assert 400 <= g.chunk_size <= 1200
            assert 2 <= g.retriever_k <= 8
            assert g.llm_model in VALID_MODELS
