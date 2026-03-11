"""
Extraction Agent — RAG-based key information extraction using Ollama SDK directly.
No langchain dependencies. Retrieves relevant chunks from FAISS store, then
uses ollama to extract structured entities based on document type.
Emits EXTRACTION_COMPLETE with the extracted fields.
"""
import json
import ollama
from engine.event_bus import bus, Event, EventType
from engine.config import MAIN_LLM_MODEL
from engine.rag.retriever import retrieve

# Extraction schemas per document type
_EXTRACTION_SCHEMAS = {
    "contract": {
        "parties": "Names of all parties involved",
        "effective_date": "Contract start/effective date",
        "expiry_date": "Contract end/expiry date",
        "key_obligations": "List of main obligations or deliverables",
        "payment_terms": "Payment amounts and schedule",
        "penalties": "Penalties or breach clauses"
    },
    "invoice": {
        "vendor": "Vendor or supplier name",
        "invoice_number": "Invoice ID or number",
        "invoice_date": "Date of the invoice",
        "due_date": "Payment due date",
        "line_items": "List of items/services and their costs",
        "total_amount": "Total amount due"
    },
    "research_paper": {
        "title": "Full title of the paper",
        "authors": "List of authors",
        "objective": "Main research objective",
        "methods": "Key methodology used",
        "findings": "Key findings or results",
        "conclusion": "Main conclusion"
    },
    "legal_document": {
        "document_type": "Type of legal document (e.g., NDA, motion, ruling)",
        "parties": "All named parties",
        "jurisdiction": "Jurisdiction or governing law",
        "key_provisions": "Most important legal provisions",
        "dates": "Key dates or deadlines"
    },
    "report": {
        "title": "Report title",
        "author_or_org": "Author or organization",
        "period_covered": "Time period the report covers",
        "key_metrics": "Key performance indicators or metrics mentioned",
        "recommendations": "Recommendations or next steps"
    },
    "email": {
        "sender": "Sender name/email",
        "recipients": "Recipients",
        "subject": "Email subject",
        "key_action_items": "Action items or requests",
        "deadline": "Any mentioned deadlines"
    },
    "general": {
        "topic": "Main topic or subject",
        "key_points": "Top 3-5 key points from the document",
        "entities": "Key people, organizations, or places mentioned",
        "dates": "Significant dates mentioned"
    }
}


def extraction_agent(state: dict) -> dict:
    doc_id = state["document_id"]

    if state.get("error"):
        return state

    doc_type = state.get("doc_type", "general")
    schema = _EXTRACTION_SCHEMAS.get(doc_type, _EXTRACTION_SCHEMAS["general"])
    filename = state["filename"]

    print(f"[ExtractionAgent] 🔎 Extracting from '{filename}' (type={doc_type})")

    # Retrieve top-k relevant chunks for this specific document
    query = f"key information entities dates parties from this {doc_type}"
    chunks = retrieve(query, document_id=doc_id)

    # Fall back to raw text snippet if RAG yields nothing
    if not chunks:
        chunks = [state["text"][:3000]]

    context = "\n\n---\n\n".join(chunks[:4])

    schema_str = "\n".join(f'  "{k}": "<{v}>"' for k, v in schema.items())
    system_prompt = f"""You are an expert document analyst specializing in {doc_type} documents.
Extract the following information from the document and respond ONLY with a valid JSON object.
If a field cannot be found, use null for its value.

Required JSON format:
{{
{schema_str}
}}"""

    try:
        response = ollama.chat(
            model=MAIN_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document excerpts:\n\n{context}"}
            ],
            format="json"
        )
        extracted = json.loads(response["message"]["content"])
    except Exception as e:
        print(f"[ExtractionAgent] ❌ LLM extraction failed: {e}")
        extracted = {"error": str(e)}

    bus.emit(Event(
        event_type=EventType.EXTRACTION_COMPLETE,
        document_id=doc_id,
        payload={"doc_type": doc_type, "extracted_fields": extracted}
    ))
    return {**state, "extracted": extracted}
