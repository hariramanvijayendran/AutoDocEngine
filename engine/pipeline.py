"""
Document Workflow Pipeline — custom sequential executor.
Replaces LangGraph StateGraph to avoid pydantic v1 compatibility issues on Python 3.14+.
Implements the same conditional routing: ingestion → indexing → classification
→ router → (extraction and/or summary) → notification.

Evolution-aware: run_pipeline_with_sop() accepts an optional SOPGenome and temporarily
overrides prompts / config on a per-run basis without touching the shared config file.
"""
from typing import Dict, Any, Optional, TYPE_CHECKING
from engine.agents.ingestion_agent import ingestion_agent
from engine.agents.classification_agent import classification_agent
from engine.agents.router_agent import router_agent
from engine.agents.extraction_agent import extraction_agent
from engine.agents.summary_agent import summary_agent
from engine.agents.notification_agent import notification_agent
from engine.rag.indexer import index_document
from engine.event_bus import bus, Event, EventType

if TYPE_CHECKING:
    from engine.evolution.sop_schema import SOPGenome


def _indexing_step(state: dict) -> dict:
    """Index document into FAISS after successful ingestion."""
    if state.get("error") or not state.get("text"):
        return state
    index_document(state["document_id"], state["filename"], state["text"])
    return state


def run_pipeline(document_id: str, file_path: str) -> Dict[str, Any]:
    """
    Run the full document workflow pipeline sequentially with conditional routing.
    Returns the final state dict.
    """
    return run_pipeline_with_sop(document_id, file_path, sop=None)


def run_pipeline_with_sop(
    document_id: str,
    file_path: str,
    sop: Optional["SOPGenome"] = None,
) -> Dict[str, Any]:
    """
    Evolution-aware pipeline variant.
    When `sop` is provided, agent prompts and flags are overridden per the genome.
    Config (CHUNK_SIZE, RETRIEVER_K etc.) must be patched by the caller before
    calling this function (the fitness evaluator does this).
    """
    # Emit opening event — suppress for silent evolution evaluations
    silent = sop is not None
    if not silent:
        bus.emit(Event(
            event_type=EventType.DOCUMENT_RECEIVED,
            document_id=document_id,
            payload={"file_path": file_path}
        ))

    # Initial state
    state: Dict[str, Any] = {
        "document_id": document_id,
        "file_path": file_path,
        "filename": None,
        "text": None,
        "doc_type": None,
        "classification_detail": None,
        "run_extraction": False,
        "run_summary": False,
        "extracted": None,
        "summary": None,
        "result": None,
        "error": None,
        # SOP override fields (used by agents that check them)
        "_sop": sop,
        "_summary_prompt_override": sop.get_summary_prompt() if sop else None,
        "_extraction_prompt_override": sop.get_extraction_prompt_prefix() if sop else None,
    }

    # Step 1: Ingestion
    state = ingestion_agent(state)

    # Step 2: Index into FAISS (skip during evolution to avoid polluting the index)
    if not state.get("error") and not silent:
        state = _indexing_step(state)

    # Step 3: Classification
    if not state.get("error"):
        state = classification_agent(state)

    # Step 4: Router (sets run_extraction / run_summary flags)
    if not state.get("error"):
        state = router_agent(state)

    # Step 5: Respect SOP feature flags
    if sop is not None:
        state["run_extraction"] = state.get("run_extraction") and sop.use_extraction
        state["run_summary"] = state.get("run_summary") and sop.use_summary

    # Step 6: Extraction (conditional)
    if not state.get("error") and state.get("run_extraction"):
        state = extraction_agent(state)

    # Step 7: Summary (conditional)
    if not state.get("error") and state.get("run_summary"):
        state = summary_agent(state)

    # Step 8: Notification (skip for silent evolution runs)
    if not silent:
        state = notification_agent(state)

    return state
