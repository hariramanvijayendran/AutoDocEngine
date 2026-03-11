"""
Ingestion Agent — parses file content from PDF, DOCX, or TXT.
Emits INGESTION_COMPLETE on success, INGESTION_FAILED on error.
"""
from pathlib import Path
from engine.event_bus import bus, Event, EventType


def ingestion_agent(state: dict) -> dict:
    doc_id = state["document_id"]
    file_path = Path(state["file_path"])
    filename = file_path.name
    ext = file_path.suffix.lower()

    print(f"[IngestionAgent] 📄 Processing '{filename}' (type={ext})")

    try:
        text = _extract_text(file_path, ext)
        if not text or not text.strip():
            raise ValueError("Extracted text is empty")

        bus.emit(Event(
            event_type=EventType.INGESTION_COMPLETE,
            document_id=doc_id,
            payload={"filename": filename, "char_count": len(text)}
        ))
        return {**state, "text": text, "filename": filename, "error": None}

    except Exception as e:
        bus.emit(Event(
            event_type=EventType.INGESTION_FAILED,
            document_id=doc_id,
            payload={"filename": filename, "error": str(e)}
        ))
        return {**state, "text": "", "filename": filename, "error": str(e)}


def _extract_text(path: Path, ext: str) -> str:
    if ext == ".pdf":
        return _read_pdf(path)
    elif ext in (".docx", ".doc"):
        return _read_docx(path)
    elif ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(para.text for para in doc.paragraphs)
