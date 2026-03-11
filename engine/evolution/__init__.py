"""
SOP Genetic Evolution Engine
Self-optimizing pipeline that evolves Standard Operating Procedures (SOPs)
using genetic algorithms to maximize document processing quality.
"""
from engine.evolution.sop_schema import SOPGenome, FitnessResult, EvolutionRun
from engine.evolution.orchestrator import run_evolution

__all__ = ["SOPGenome", "FitnessResult", "EvolutionRun", "run_evolution"]
