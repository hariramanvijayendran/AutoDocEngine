"""
SQLite-backed event store.
Every event emitted by any agent is persisted here with full payload.
The API reads from this store to serve the live event feed & document history.
"""
import sqlite3
import json
import time
from typing import List, Dict, Any, Optional
from engine.event_bus import Event, EventType, bus
from engine.config import EVENTS_DB_PATH


class EventStore:
    def __init__(self, db_path=EVENTS_DB_PATH):
        self.db_path = str(db_path)
        self._init_db()
        # Subscribe to ALL event types so every emitted event is saved
        for et in EventType:
            bus.subscribe(et, self._on_event)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type  TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    payload     TEXT NOT NULL,
                    timestamp   REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename    TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'processing',
                    created_at  REAL NOT NULL,
                    result      TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON events(document_id)")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _on_event(self, event: Event):
        """Auto-called by the bus for every event."""
        self.save_event(event)
        # Update document status based on terminal events
        if event.event_type == EventType.WORKFLOW_COMPLETE:
            self.update_document_status(
                event.document_id, "complete",
                result=json.dumps(event.payload.get("result", {}))
            )
        elif event.event_type == EventType.WORKFLOW_ERROR:
            self.update_document_status(event.document_id, "error")

    def save_event(self, event: Event):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events (event_type, document_id, payload, timestamp) VALUES (?,?,?,?)",
                (event.event_type.value, event.document_id,
                 json.dumps(event.payload), event.timestamp)
            )

    def register_document(self, document_id: str, filename: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents (document_id, filename, status, created_at) VALUES (?,?,?,?)",
                (document_id, filename, "processing", time.time())
            )

    def update_document_status(self, document_id: str, status: str, result: Optional[str] = None):
        with self._conn() as conn:
            if result:
                conn.execute(
                    "UPDATE documents SET status=?, result=? WHERE document_id=?",
                    (status, result, document_id)
                )
            else:
                conn.execute(
                    "UPDATE documents SET status=? WHERE document_id=?",
                    (status, document_id)
                )

    def get_events(self, document_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            if document_id:
                rows = conn.execute(
                    "SELECT * FROM events WHERE document_id=? ORDER BY timestamp DESC LIMIT ?",
                    (document_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
        return [
            {**dict(r), "payload": json.loads(r["payload"])}
            for r in rows
        ]

    def get_documents(self) -> List[Dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("result"):
                try:
                    d["result"] = json.loads(d["result"])
                except Exception:
                    pass
            result.append(d)
        return result

    def get_document(self, document_id: str) -> Optional[Dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id=?", (document_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("result"):
            try:
                d["result"] = json.loads(d["result"])
            except Exception:
                pass
        return d


# Shared singleton
event_store = EventStore()
