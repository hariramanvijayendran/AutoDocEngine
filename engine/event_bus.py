"""
Event Bus — defines all event types and a simple synchronous dispatcher.
Every agent emits events here; the event store listens and persists them.
"""
from enum import Enum
from typing import Callable, Dict, List, Any
from dataclasses import dataclass, field
import time


class EventType(str, Enum):
    DOCUMENT_RECEIVED      = "DOCUMENT_RECEIVED"
    INGESTION_COMPLETE     = "INGESTION_COMPLETE"
    INGESTION_FAILED       = "INGESTION_FAILED"
    CLASSIFICATION_COMPLETE = "CLASSIFICATION_COMPLETE"
    ROUTING_COMPLETE       = "ROUTING_COMPLETE"
    EXTRACTION_COMPLETE    = "EXTRACTION_COMPLETE"
    SUMMARY_COMPLETE       = "SUMMARY_COMPLETE"
    WORKFLOW_COMPLETE      = "WORKFLOW_COMPLETE"
    WORKFLOW_ERROR         = "WORKFLOW_ERROR"


@dataclass
class Event:
    event_type: EventType
    document_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """Simple synchronous event bus with subscribe/emit pattern."""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Register a callback to be called when an event_type is emitted."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def emit(self, event: Event) -> None:
        """Emit an event to all registered subscribers."""
        print(f"[EventBus] 📡 {event.event_type.value} | doc={event.document_id}")
        for callback in self._subscribers.get(event.event_type, []):
            try:
                callback(event)
            except Exception as e:
                print(f"[EventBus] ⚠️  Subscriber error for {event.event_type}: {e}")
        # Always also fire wildcard subscribers
        for callback in self._subscribers.get("*", []):
            try:
                callback(event)
            except Exception as e:
                print(f"[EventBus] ⚠️  Wildcard subscriber error: {e}")


# Single shared instance — import this everywhere
bus = EventBus()
