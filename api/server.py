"""
FastAPI server — REST API for the Document Workflow Engine.
v2.0 adds the SOP Genetic Evolution Engine endpoints.
"""
import uuid
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.config import (
    UPLOADS_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB,
    EVOLUTION_POPULATION_SIZE, EVOLUTION_GENERATIONS,
    EVOLUTION_TOP_K, EVOLUTION_MUTATION_RATE, EVOLUTION_PARALLEL_WORKERS,
)
from engine.event_store import event_store
from engine.pipeline import run_pipeline

app = FastAPI(
    title="Autonomous Document Workflow Engine",
    description="Event-driven multi-agent document processing with RAG + SOP Genetic Evolution Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the web dashboard."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Document Endpoints ───────────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and trigger the full processing pipeline asynchronously.
    Returns immediately with the document_id to track progress.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE_MB}MB)")

    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}{suffix}"
    dest = UPLOADS_DIR / safe_name
    dest.write_bytes(content)

    event_store.register_document(doc_id, file.filename)

    def _run():
        try:
            run_pipeline(doc_id, str(dest))
        except Exception as e:
            print(f"[API] Pipeline error for {doc_id}: {e}")

    threading.Thread(target=_run, daemon=True).start()

    return JSONResponse({
        "document_id": doc_id,
        "filename": file.filename,
        "status": "processing",
        "message": "Document received. Processing started."
    })


@app.get("/documents")
async def list_documents():
    """List all documents and their processing status."""
    return event_store.get_documents()


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get full result for a specific document."""
    doc = event_store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/events")
async def list_events(document_id: Optional[str] = None, limit: int = 100):
    """Get the event log. Optionally filter by document_id."""
    return event_store.get_events(document_id=document_id, limit=limit)


@app.get("/events/{document_id}")
async def get_document_events(document_id: str):
    """Get all events for a specific document."""
    return event_store.get_events(document_id=document_id, limit=200)


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "Autonomous Document Workflow Engine v2.0"}


# ── SOP Genetic Evolution Engine Endpoints ────────────────────────────────────

class EvolutionRequest(BaseModel):
    test_document_id: str
    population_size: Optional[int] = None
    generations: Optional[int] = None
    top_k: Optional[int] = None
    mutation_rate: Optional[float] = None
    parallel_workers: Optional[int] = None


@app.post("/evolve")
async def start_evolution(req: EvolutionRequest):
    """
    Start a new SOP Genetic Evolution run in the background.
    Returns a run_id immediately — poll /evolve/{run_id}/status for live updates.
    """
    from engine.evolution.persistence import evolution_store
    from engine.evolution.orchestrator import run_evolution
    from engine.evolution.sop_schema import EvolutionRun
    from datetime import datetime

    # Resolve the test document path
    doc = event_store.get_document(req.test_document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Test document not found. Upload a document first.")

    # Find the file on disk
    candidates = list(UPLOADS_DIR.glob(f"{req.test_document_id}.*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="Test document file not found on disk.")
    doc_path = str(candidates[0])

    # Create the run record immediately so the caller has a run_id
    run_id = str(uuid.uuid4())[:12]
    stub_run = EvolutionRun(
        run_id=run_id,
        status="pending",
        test_document_id=req.test_document_id,
        population_size=req.population_size or EVOLUTION_POPULATION_SIZE,
        generations=req.generations or EVOLUTION_GENERATIONS,
        started_at=datetime.utcnow().isoformat(),
    )
    evolution_store.create_run(stub_run)

    pop_size = req.population_size or EVOLUTION_POPULATION_SIZE
    gens = req.generations or EVOLUTION_GENERATIONS
    top_k = req.top_k or EVOLUTION_TOP_K
    mut_rate = req.mutation_rate or EVOLUTION_MUTATION_RATE
    workers = req.parallel_workers or EVOLUTION_PARALLEL_WORKERS

    def _run_evolution():
        try:
            run_evolution(
                test_document_id=req.test_document_id,
                test_doc_path=doc_path,
                population_size=pop_size,
                generations=gens,
                top_k=top_k,
                mutation_rate=mut_rate,
                parallel_workers=workers,
                run_id=run_id,
            )
        except Exception as e:
            print(f"[API/Evolution] Run {run_id} failed: {e}")

    threading.Thread(target=_run_evolution, daemon=True).start()

    return JSONResponse({
        "run_id": run_id,
        "status": "started",
        "population_size": pop_size,
        "generations": gens,
        "message": f"Evolution run {run_id} started. Poll /evolve/{run_id}/status for updates."
    })


@app.get("/evolve")
async def list_evolution_runs(limit: int = 20):
    """List recent evolution runs with their status and best fitness."""
    from engine.evolution.persistence import evolution_store
    return evolution_store.list_runs(limit=limit)


@app.get("/evolve/{run_id}/status")
async def get_evolution_status(run_id: str):
    """
    Get live status of an evolution run.
    Returns current generation, best fitness, Pareto front, and generation history.
    """
    from engine.evolution.persistence import evolution_store
    run = evolution_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Evolution run '{run_id}' not found.")
    return run


@app.post("/evolve/{run_id}/apply-best")
async def apply_best_sop(run_id: str):
    """
    Apply the best-performing genome from a completed evolution run to the live engine config.
    Subsequent pipeline runs will use the optimized parameters immediately.
    """
    from engine.evolution.persistence import evolution_store
    from engine.evolution.orchestrator import apply_sop_to_config
    import engine.config as cfg

    run = evolution_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Evolution run '{run_id}' not found.")
    if run["status"] != "complete":
        raise HTTPException(status_code=400, detail="Evolution run is not complete yet.")
    if not run.get("best_genome"):
        raise HTTPException(status_code=400, detail="No best genome found in this run.")

    genome = run["best_genome"]
    apply_sop_to_config(genome)

    return JSONResponse({
        "message": "✅ Best SOP applied to live engine config.",
        "applied_genome": genome,
        "new_config": {
            "chunk_size": cfg.CHUNK_SIZE,
            "chunk_overlap": cfg.CHUNK_OVERLAP,
            "retriever_k": cfg.RETRIEVER_K,
            "llm_model": cfg.MAIN_LLM_MODEL,
        }
    })


@app.get("/evolve/{run_id}/results")
async def get_evolution_results(run_id: str):
    """Get all individual genome results from an evolution run, sorted by fitness."""
    from engine.evolution.persistence import evolution_store
    run = evolution_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Evolution run '{run_id}' not found.")
    return evolution_store.get_all_results(run_id)

