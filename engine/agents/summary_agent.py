"""
Summary Agent — generates a concise document summary using Ollama SDK directly.
Emits SUMMARY_COMPLETE.
"""
import ollama
from engine.event_bus import bus, Event, EventType
from engine.config import MAIN_LLM_MODEL

_DEFAULT_SYSTEM_PROMPT = """You are a professional document summarizer.
Write a clear, concise summary of the document in 3-5 sentences.
Focus on the main purpose, key points, and any important conclusions or actions required.
Return ONLY the summary text with no preamble or labels."""


def summary_agent(state: dict) -> dict:
    doc_id = state["document_id"]

    if state.get("error"):
        return state

    filename = state["filename"]
    text = state["text"]
    snippet = text[:4000]

    # Support SOP prompt override from evolution engine
    system_prompt = state.get("_summary_prompt_override") or _DEFAULT_SYSTEM_PROMPT

    print(f"[SummaryAgent] 📝 Summarizing '{filename}'")

    try:
        response = ollama.chat(
            model=MAIN_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document:\n\n{snippet}"}
            ]
        )
        summary = response["message"]["content"].strip()
    except Exception as e:
        print(f"[SummaryAgent] ❌ Error: {e}")
        summary = f"Summary unavailable: {e}"

    bus.emit(Event(
        event_type=EventType.SUMMARY_COMPLETE,
        document_id=doc_id,
        payload={"summary": summary}
    ))
    return {**state, "summary": summary}
