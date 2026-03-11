"""
Genetic Operators — Mutation and Crossover for SOPGenome evolution.

Mutation:
  Randomly tweaks 1–2 numeric/discrete genes by ±15–30%.
  Prompt variant genes are flipped to a random neighbour index.
  Boolean genes are flipped with 20% probability.

Crossover (Uniform):
  For each gene, picks randomly from parent A or parent B.
  On average, 50% genes from each parent.
"""
from __future__ import annotations

import random
import uuid
from copy import deepcopy

from engine.evolution.sop_schema import SOPGenome, VALID_MODELS, SUMMARY_PROMPT_VARIANTS, EXTRACTION_PROMPT_VARIANTS, CLASSIFICATION_PROMPT_VARIANTS


# ── Mutation ──────────────────────────────────────────────────────────────────

_NUMERIC_GENES = {
    "chunk_size":                    (400,  1200, 50),   # (min, max, step)
    "chunk_overlap":                 (50,   200,  25),
    "retriever_k":                   (2,    8,    1),
    "summary_prompt_variant":        (0,    len(SUMMARY_PROMPT_VARIANTS) - 1,    1),
    "extraction_prompt_variant":     (0,    len(EXTRACTION_PROMPT_VARIANTS) - 1, 1),
    "classification_prompt_variant": (0,    len(CLASSIFICATION_PROMPT_VARIANTS) - 1, 1),
}

_BOOLEAN_GENES = ["use_extraction", "use_summary"]
_MODEL_GENE = "llm_model"


def mutate(genome: SOPGenome, mutation_rate: float = 0.3, seed: int | None = None) -> SOPGenome:
    """
    Return a new, mutated SOPGenome offspring.
    Each gene mutates independently with probability = mutation_rate.
    """
    rng = random.Random(seed)
    child = deepcopy(genome)
    child.genome_id = str(uuid.uuid4())[:8]
    child.generation = genome.generation + 1
    child.parent_ids = [genome.genome_id]

    mutated_any = False

    for gene, (lo, hi, step) in _NUMERIC_GENES.items():
        if rng.random() < mutation_rate:
            current = getattr(child, gene)
            delta = rng.choice([-2, -1, 1, 2]) * step
            new_val = max(lo, min(hi, current + delta))
            setattr(child, gene, new_val)
            mutated_any = True

    for gene in _BOOLEAN_GENES:
        if rng.random() < 0.20:   # lower flip rate for feature flags
            setattr(child, gene, not getattr(child, gene))
            mutated_any = True

    if rng.random() < mutation_rate * 0.5:  # model swap is costly, lower rate
        child.llm_model = rng.choice(VALID_MODELS)
        mutated_any = True

    # Force at least one mutation so the child is always different
    if not mutated_any:
        gene = rng.choice(list(_NUMERIC_GENES.keys()))
        lo, hi, step = _NUMERIC_GENES[gene]
        current = getattr(child, gene)
        delta = rng.choice([-1, 1]) * step
        setattr(child, gene, max(lo, min(hi, current + delta)))

    return child.clamp()


# ── Crossover ──────────────────────────────────────────────────────────────────

def crossover(parent_a: SOPGenome, parent_b: SOPGenome, seed: int | None = None) -> SOPGenome:
    """
    Uniform crossover: each gene is drawn from parent_a or parent_b with equal probability.
    Returns a new child genome.
    """
    rng = random.Random(seed)
    child = SOPGenome(
        genome_id=str(uuid.uuid4())[:8],
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parent_ids=[parent_a.genome_id, parent_b.genome_id],
    )

    all_genes = list(_NUMERIC_GENES.keys()) + _BOOLEAN_GENES + [_MODEL_GENE]
    for gene in all_genes:
        src = parent_a if rng.random() < 0.5 else parent_b
        setattr(child, gene, getattr(src, gene))

    return child.clamp()


# ── Population Initialization ─────────────────────────────────────────────────

def initialize_population(size: int, seed: int | None = None) -> list[SOPGenome]:
    """
    Create an initial diverse population. The first genome is always the
    current production defaults; remaining genomes are randomly initialized.
    """
    rng = random.Random(seed)
    population = []

    # Individual 0: production defaults
    population.append(SOPGenome(genome_id="baseline", generation=0))

    for _ in range(size - 1):
        g = SOPGenome(
            genome_id=str(uuid.uuid4())[:8],
            generation=0,
            chunk_size=rng.choice([400, 600, 800, 1000, 1200]),
            chunk_overlap=rng.choice([50, 75, 100, 150, 200]),
            retriever_k=rng.randint(2, 8),
            llm_model=rng.choice(VALID_MODELS),
            summary_prompt_variant=rng.randint(0, len(SUMMARY_PROMPT_VARIANTS) - 1),
            extraction_prompt_variant=rng.randint(0, len(EXTRACTION_PROMPT_VARIANTS) - 1),
            classification_prompt_variant=rng.randint(0, len(CLASSIFICATION_PROMPT_VARIANTS) - 1),
            use_extraction=rng.random() > 0.2,
            use_summary=True,   # always keep summary enabled for scoring
        )
        population.append(g.clamp())

    return population


# ── Pareto Front ──────────────────────────────────────────────────────────────

def compute_pareto_front(
    scored: list[tuple[SOPGenome, "FitnessResult"]]  # type: ignore[name-defined]
) -> list[tuple[SOPGenome, "FitnessResult"]]:  # type: ignore[name-defined]
    """
    Identify Pareto-optimal solutions (non-dominated set) over two objectives:
      - quality_score (completeness + coherence)
      - speed_proxy   (inversely proportional to retriever_k + chunk_size)

    A solution is Pareto-optimal if no other solution is strictly better in
    BOTH objectives simultaneously.
    """
    def objectives(pair):
        g, r = pair
        quality = (r.completeness + r.coherence) / 2.0
        # Smaller genomes = faster → higher speed score
        speed = 1.0 - ((g.retriever_k - 2) / 6.0 * 0.6 + (g.chunk_size - 400) / 800.0 * 0.4)
        return quality, speed

    pareto = []
    for i, pair_i in enumerate(scored):
        q_i, s_i = objectives(pair_i)
        dominated = False
        for j, pair_j in enumerate(scored):
            if i == j:
                continue
            q_j, s_j = objectives(pair_j)
            if q_j >= q_i and s_j >= s_i and (q_j > q_i or s_j > s_i):
                dominated = True
                break
        if not dominated:
            pareto.append(pair_i)
    return pareto
