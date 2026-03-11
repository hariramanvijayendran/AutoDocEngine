"""
Classification Agent — uses Ollama SDK directly (no langchain dependency).
Emits CLASSIFICATION_COMPLETE with the label and a short reasoning.
"""
import json
import ollama
from engine.event_bus import bus, Event, EventType
from engine.config import MAIN_LLM_MODEL, DOCUMENT_TYPES

_SYSTEM_PROMPT = f"""You are a document classification expert.
Given the beginning of a document, classify it into exactly ONE of these categories:
{', '.join(DOCUMENT_TYPES)}

Respond ONLY with a valid JSON object in this exact format:
{{"label": "<category>", "confidence": "<high|medium|low>", "reasoning": "<one sentence why>"}}
"""


def classification_agent(state: dict) -> dict:
    doc_id = state["document_id"]

    if state.get("error"):
        return state

    text = state["text"]
    snippet = text[:2000]

    print(f"[ClassificationAgent] 🔍 Classifying '{state['filename']}'")

    try:
        response = ollama.chat(
            model=MAIN_LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Document content:\n\n{snippet}"}
            ],
            format="json"
        )
        content = response["message"]["content"]
        parsed = json.loads(content)
        label = parsed.get("label", "general")
        if label not in DOCUMENT_TYPES:
            label = "general"

        bus.emit(Event(
            event_type=EventType.CLASSIFICATION_COMPLETE,
            document_id=doc_id,
            payload={
                "label": label,
                "confidence": parsed.get("confidence", "medium"),
                "reasoning": parsed.get("reasoning", "")
            }
        ))
        return {**state, "doc_type": label, "classification_detail": parsed}

    except Exception as e:
        print(f"[ClassificationAgent] ❌ Error: {e}")
        bus.emit(Event(
            event_type=EventType.CLASSIFICATION_COMPLETE,
            document_id=doc_id,
            payload={"label": "general", "confidence": "low", "reasoning": f"Classification failed: {e}"}
        ))
        return {**state, "doc_type": "general", "classification_detail": {}}
