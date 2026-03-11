"""
Router Agent — decides which downstream agents to engage based on doc type.
In LangGraph this drives conditional routing after classification.
Emits ROUTING_COMPLETE.
"""
from engine.event_bus import bus, Event, EventType


# Defines which processing steps each document type should run
_ROUTING_MAP = {
    "contract":       {"run_extraction": True,  "run_summary": True},
    "invoice":        {"run_extraction": True,  "run_summary": False},
    "research_paper": {"run_extraction": True,  "run_summary": True},
    "legal_document": {"run_extraction": True,  "run_summary": True},
    "report":         {"run_extraction": True,  "run_summary": True},
    "email":          {"run_extraction": True,  "run_summary": False},
    "general":        {"run_extraction": False, "run_summary": True},
}


def router_agent(state: dict) -> dict:
    doc_id = state["document_id"]

    if state.get("error"):
        return state

    doc_type = state.get("doc_type", "general")
    routing = _ROUTING_MAP.get(doc_type, _ROUTING_MAP["general"])

    print(f"[RouterAgent] 🔀 Routing '{state['filename']}' (type={doc_type}) → {routing}")

    bus.emit(Event(
        event_type=EventType.ROUTING_COMPLETE,
        document_id=doc_id,
        payload={"doc_type": doc_type, "routing": routing}
    ))
    return {**state, **routing}


def get_next_step(state: dict) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node to run.
    """
    if state.get("error"):
        return "notification"
    if state.get("run_extraction", False):
        return "extraction"
    if state.get("run_summary", False):
        return "summary"
    return "notification"


def get_post_extraction_step(state: dict) -> str:
    """After extraction, decide whether to also summarize."""
    if state.get("run_summary", False):
        return "summary"
    return "notification"
