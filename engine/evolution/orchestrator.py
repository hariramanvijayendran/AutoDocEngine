"""
Evolution Orchestrator — main loop that runs the full Genetic Evolution cycle.

Algorithm per generation:
  1. Evaluate population fitness in parallel (ThreadPoolExecutor)
  2. Rank by fitness score
  3. Select top-k survivors (elitism)
  4. Produce next generation via crossover + mutation of survivors
  5. Inject one fresh random genome (diversity injection)
  6. Persist results + update EvolutionRun record

Final output: Pareto-optimal SOPs over quality vs. speed objectives.
"""
from __future__ import annotations

import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional

from engine.evolution.sop_schema import SOPGenome, FitnessResult, EvolutionRun, GenerationRecord
from engine.evolution.operators import (
    initialize_population, mutate, crossover, compute_pareto_front
)
from engine.evolution.fitness import evaluate_fitness
from engine.evolution.persistence import evolution_store
import engine.config as cfg


def _evaluate_one(
    genome: SOPGenome,
    test_doc_path: str,
) -> tuple[SOPGenome, FitnessResult]:
    """Evaluate a single genome. Designed to run in a thread pool."""
    print(f"  [Evo] Evaluating genome {genome.genome_id} "
          f"(chunk={genome.chunk_size}, k={genome.retriever_k}, "
          f"model={genome.llm_model}) ...")
    fitness = evaluate_fitness(genome, test_doc_path)
    print(f"  [Evo] ✓ {genome.genome_id}: fitness={fitness.weighted_total:.3f} "
          f"(complete={fitness.completeness:.2f}, concise={fitness.conciseness:.2f}, "
          f"cohere={fitness.coherence:.2f}, extract={fitness.extraction_coverage:.2f})")
    return genome, fitness


def run_evolution(
    test_document_id: str,
    test_doc_path: str,
    population_size: int = 8,
    generations: int = 5,
    top_k: int = 3,
    mutation_rate: float = 0.3,
    parallel_workers: int = 4,
    run_id: Optional[str] = None,
) -> EvolutionRun:
    """
    Execute the full genetic evolution loop and return an EvolutionRun
    with the Pareto-optimal SOPs.

    Args:
        test_document_id: ID of the already-ingested test document.
        test_doc_path:     Absolute path to the test document file.
        population_size:   Number of SOPs per generation.
        generations:       Number of evolution cycles.
        top_k:             Number of survivors to keep per generation.
        mutation_rate:     Per-gene mutation probability (0–1).
        parallel_workers:  Max concurrent evaluations.
        run_id:            Optional fixed run ID (for resuming).
    """
    # ── Initialise run record ─────────────────────────────────────────────────
    run = EvolutionRun(
        run_id=run_id or str(uuid.uuid4())[:12],
        status="running",
        test_document_id=test_document_id,
        population_size=population_size,
        generations=generations,
        started_at=datetime.utcnow().isoformat(),
    )
    if run_id is not None:
        # API already created the stub record — update status to running
        run.run_id = run_id
        evolution_store.update_run(run)
    else:
        evolution_store.create_run(run)
    print(f"\n{'='*60}")
    print(f"  🧬 SOP Genetic Evolution Engine  (run_id={run.run_id})")
    print(f"  Population={population_size}  Generations={generations}  Workers={parallel_workers}")
    print(f"{'='*60}\n")

    # ── Seed random state reproducibly ────────────────────────────────────────
    seed = int(time.time())
    rng = random.Random(seed)

    # ── Initial population ────────────────────────────────────────────────────
    population: List[SOPGenome] = initialize_population(population_size, seed=seed)
    all_scored: List[tuple[SOPGenome, FitnessResult]] = []

    try:
        for gen in range(1, generations + 1):
            print(f"\n--- Generation {gen}/{generations} ---")
            run.current_generation = gen
            gen_results: List[tuple[SOPGenome, FitnessResult]] = []

            # ── Parallel fitness evaluation ────────────────────────────────
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {
                    executor.submit(_evaluate_one, g, test_doc_path): g
                    for g in population
                }
                for future in as_completed(futures):
                    try:
                        genome, fitness = future.result()
                        gen_results.append((genome, fitness))
                        evolution_store.save_genome_result(run.run_id, genome, fitness)
                    except Exception as e:
                        g = futures[future]
                        err_fitness = FitnessResult(genome_id=g.genome_id, error=str(e))
                        gen_results.append((g, err_fitness))
                        print(f"  [Evo] ❌ Genome {g.genome_id} failed: {e}")

            # ── Rank ──────────────────────────────────────────────────────────
            gen_results.sort(key=lambda x: x[1].weighted_total, reverse=True)
            best_genome, best_fitness = gen_results[0]
            all_scored.extend(gen_results)

            print(f"\n  ✨ Gen {gen} best: genome={best_genome.genome_id} "
                  f"fitness={best_fitness.weighted_total:.4f}")

            # ── Update run best ───────────────────────────────────────────────
            if best_fitness.weighted_total > run.best_fitness:
                run.best_fitness = best_fitness.weighted_total
                run.best_genome = best_genome.to_dict()

            # ── Record generation snapshot ────────────────────────────────────
            gen_record = GenerationRecord(
                generation=gen,
                population=[g.to_dict() for g, _ in gen_results],
                fitness_results=[f.to_dict() for _, f in gen_results],
                best_fitness=best_fitness.weighted_total,
            )
            run.generation_history.append(gen_record.__dict__)
            evolution_store.update_run(run)

            # ── Select survivors (elitism) ────────────────────────────────────
            if gen == generations:
                break  # No need to breed final generation

            survivors = [g for g, _ in gen_results[:top_k]]

            # ── Breed next generation ─────────────────────────────────────────
            next_pop: List[SOPGenome] = list(survivors)  # keep survivors

            # Fill rest via crossover + mutation
            while len(next_pop) < population_size - 1:
                if len(survivors) >= 2:
                    pa, pb = rng.sample(survivors, 2)
                    child = crossover(pa, pb)
                else:
                    child = mutate(survivors[0], mutation_rate=mutation_rate)

                if rng.random() < 0.5:
                    child = mutate(child, mutation_rate=mutation_rate)
                next_pop.append(child)

            # Diversity injection: one fresh random genome per generation
            fresh = initialize_population(2, seed=rng.randint(0, 99999))[1]
            next_pop.append(fresh)
            population = next_pop

    except Exception as e:
        run.status = "error"
        run.error = str(e)
        run.completed_at = datetime.utcnow().isoformat()
        evolution_store.update_run(run)
        print(f"\n[Evo] ❌ Evolution run failed: {e}")
        return run

    # ── Pareto Front ─────────────────────────────────────────────────────────
    pareto = compute_pareto_front(all_scored)
    run.pareto_front = [
        {
            "genome": g.to_dict(),
            "fitness": f.to_dict(),
            "score": f.weighted_total,
        }
        for g, f in pareto
    ][:10]  # cap at 10 for UI display

    run.status = "complete"
    run.completed_at = datetime.utcnow().isoformat()
    evolution_store.update_run(run)

    print(f"\n{'='*60}")
    print(f"  🏆 Evolution complete! Best fitness: {run.best_fitness:.4f}")
    print(f"  Pareto front: {len(pareto)} non-dominated solutions")
    print(f"{'='*60}\n")

    return run


def apply_sop_to_config(genome_dict: dict):
    """
    Apply a winning SOPGenome's parameters to the live engine config.
    Called from the API's 'Apply Best SOP' action.
    """
    cfg.CHUNK_SIZE = genome_dict.get("chunk_size", cfg.CHUNK_SIZE)
    cfg.CHUNK_OVERLAP = genome_dict.get("chunk_overlap", cfg.CHUNK_OVERLAP)
    cfg.RETRIEVER_K = genome_dict.get("retriever_k", cfg.RETRIEVER_K)
    cfg.MAIN_LLM_MODEL = genome_dict.get("llm_model", cfg.MAIN_LLM_MODEL)
    print(f"[Evo] ✅ Active SOP updated: chunk={cfg.CHUNK_SIZE}, "
          f"k={cfg.RETRIEVER_K}, model={cfg.MAIN_LLM_MODEL}")
