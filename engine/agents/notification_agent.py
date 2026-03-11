"""
Notification Agent — collects all outputs into a final structured result.
Emits WORKFLOW_COMPLETE (or WORKFLOW_ERROR on failure).
"""
import time
from engine.event_bus import bus, Event, EventType


def notification_agent(state: dict) -> dict:
    doc_id = state["document_id"]

    if state.get("error"):
        bus.emit(Event(
            event_type=EventType.WORKFLOW_ERROR,
            document_id=doc_id,
            payload={"error": state["error"], "filename": state.get("filename", "unknown")}
        ))
        return state

    result = {
        "document_id": doc_id,
        "filename": state.get("filename"),
        "doc_type": state.get("doc_type", "general"),
        "classification_detail": state.get("classification_detail", {}),
        "extracted": state.get("extracted", {}),
        "summary": state.get("summary", ""),
        "completed_at": time.time()
    }

    print(f"[NotificationAgent] ✅ Workflow complete for '{state.get('filename')}'")

    bus.emit(Event(
        event_type=EventType.WORKFLOW_COMPLETE,
        document_id=doc_id,
        payload={"result": result}
    ))
    return {**state, "result": result}
