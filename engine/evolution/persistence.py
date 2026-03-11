"""
Evolution Persistence — SQLite-backed storage for all evolution run data.
Stores: EvolutionRun metadata, per-generation records, individual genome scores.
Allows the API to serve live status, resume interrupted runs, and audit history.

Uses the same DB file as the main event_store (events.db) to keep things simple,
but in dedicated tables prefixed with 'evo_'.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional, List

from engine.config import EVENTS_DB_PATH
from engine.evolution.sop_schema import SOPGenome, FitnessResult, EvolutionRun


class EvolutionStore:
    def __init__(self, db_path: str = str(EVENTS_DB_PATH)):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS evo_runs (
                    run_id          TEXT PRIMARY KEY,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    test_document_id TEXT NOT NULL,
                    population_size INTEGER NOT NULL,
                    generations     INTEGER NOT NULL,
                    current_generation INTEGER NOT NULL DEFAULT 0,
                    best_fitness    REAL NOT NULL DEFAULT 0.0,
                    best_genome     TEXT,
                    pareto_front    TEXT,
                    generation_history TEXT,
                    started_at      TEXT,
                    completed_at    TEXT,
                    error           TEXT
                );

                CREATE TABLE IF NOT EXISTS evo_genomes (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT NOT NULL,
                    generation      INTEGER NOT NULL,
                    genome_id       TEXT NOT NULL,
                    genome_json     TEXT NOT NULL,
                    fitness_json    TEXT,
                    weighted_total  REAL DEFAULT 0.0,
                    FOREIGN KEY (run_id) REFERENCES evo_runs(run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_evo_run ON evo_genomes(run_id);
                CREATE INDEX IF NOT EXISTS idx_evo_gen ON evo_genomes(run_id, generation);
            """)

    # ── Run CRUD ──────────────────────────────────────────────────────────────

    def create_run(self, run: EvolutionRun):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO evo_runs
                   (run_id, status, test_document_id, population_size, generations,
                    current_generation, best_fitness, started_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (run.run_id, run.status, run.test_document_id,
                 run.population_size, run.generations,
                 run.current_generation, run.best_fitness, run.started_at),
            )

    def update_run(self, run: EvolutionRun):
        with self._conn() as conn:
            conn.execute(
                """UPDATE evo_runs SET
                   status=?, current_generation=?, best_fitness=?,
                   best_genome=?, pareto_front=?, generation_history=?,
                   completed_at=?, error=?
                   WHERE run_id=?""",
                (
                    run.status, run.current_generation, run.best_fitness,
                    json.dumps(run.best_genome) if run.best_genome else None,
                    json.dumps(run.pareto_front),
                    json.dumps(run.generation_history),
                    run.completed_at, run.error,
                    run.run_id,
                ),
            )

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM evo_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        for key in ("best_genome", "pareto_front", "generation_history"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        return d

    def list_runs(self, limit: int = 20) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT run_id, status, test_document_id, generations, current_generation, "
                "best_fitness, started_at, completed_at FROM evo_runs ORDER BY rowid DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Genome / Fitness Records ──────────────────────────────────────────────

    def save_genome_result(self, run_id: str, genome: SOPGenome, fitness: FitnessResult):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO evo_genomes
                   (run_id, generation, genome_id, genome_json, fitness_json, weighted_total)
                   VALUES (?,?,?,?,?,?)""",
                (
                    run_id, genome.generation, genome.genome_id,
                    json.dumps(genome.to_dict()),
                    json.dumps(fitness.to_dict()),
                    fitness.weighted_total,
                ),
            )

    def get_generation_results(self, run_id: str, generation: int) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT genome_json, fitness_json, weighted_total FROM evo_genomes "
                "WHERE run_id=? AND generation=? ORDER BY weighted_total DESC",
                (run_id, generation),
            ).fetchall()
        results = []
        for row in rows:
            genome_d = json.loads(row["genome_json"])
            fitness_d = json.loads(row["fitness_json"]) if row["fitness_json"] else {}
            results.append({"genome": genome_d, "fitness": fitness_d, "score": row["weighted_total"]})
        return results

    def get_all_results(self, run_id: str) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT genome_json, fitness_json, weighted_total, generation FROM evo_genomes "
                "WHERE run_id=? ORDER BY weighted_total DESC",
                (run_id,),
            ).fetchall()
        return [
            {
                "genome": json.loads(r["genome_json"]),
                "fitness": json.loads(r["fitness_json"]) if r["fitness_json"] else {},
                "score": r["weighted_total"],
                "generation": r["generation"],
            }
            for r in rows
        ]


# Shared singleton
evolution_store = EvolutionStore()
