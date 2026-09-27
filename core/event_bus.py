"""
core/event_bus.py

Thread-safe event bus with async dispatch via a dedicated worker thread.

Usage::

    bus = EventBus()
    bus.subscribe(EventType.FIRE_DETECTED, my_handler)
    bus.publish(FireDetected(result={...}))
    bus.flush()   # blocks until all queued events are dispatched (useful in tests)
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EventType registry
# ---------------------------------------------------------------------------

class EventType(Enum):
    # Frame / camera
    FRAME_READY = auto()
    PROCESSED_FRAME = auto()
    CAMERA_CONNECTED = auto()
    CAMERA_DISCONNECTED = auto()
    # Gesture
    GESTURE_CANDIDATE = auto()
    GESTURE_NEAR_MISS = auto()
    GESTURE_CONFIRMED = auto()
    GESTURE_REJECTED = auto()
    GESTURE_PROGRESS = auto()
    # Auth
    AUTH_SUCCESS = auto()
    AUTH_FAILED = auto()
    # Presence
    PRESENCE_UPDATED = auto()
    # Detection
    FIRE_DETECTED = auto()
    INJURY_DETECTED = auto()
    SUSPICIOUS_ACTIVITY = auto()
    DETECTION_COMPLETE = auto()
    ALERT_DISPATCHED = auto()
    # State machine
    STATE_TRANSITION = auto()
    # System lifecycle
    SYSTEM_STARTUP = auto()
    SYSTEM_SHUTDOWN = auto()
    # Model lifecycle
    MODEL_LOADED = auto()
    MODEL_UNLOADED = auto()


# ---------------------------------------------------------------------------
# Legacy generic event (kept for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """Generic envelope. Prefer typed event dataclasses from core.events."""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _event_type(event: Any) -> EventType:
    """Extract EventType from either a typed event dataclass or a legacy Event."""
    # Typed dataclasses declare event_type as a ClassVar
    et = getattr(type(event), "event_type", None)
    if isinstance(et, EventType):
        return et
    # Legacy Event has an instance field 'type'
    return event.type  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Async event bus.

    ``publish()`` enqueues the event immediately and returns.  A dedicated
    daemon thread dequeues and dispatches to subscribers.  The queue is
    unbounded so that subscribers which publish new events from inside their
    handlers cannot deadlock.

    Thread-safety
    -------------
    ``subscribe`` / ``unsubscribe`` use a ``threading.Lock``.
    ``publish`` is safe to call from any thread.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable[[Any], None]]] = {
            et: [] for et in EventType
        }
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()  # unbounded

        self._worker = threading.Thread(target=self._dispatch_loop, daemon=True, name="EventBus-worker")
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        with self._lock:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        with self._lock:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass

    def publish(self, event: Any) -> None:
        """Enqueue *event* for async dispatch. Returns immediately."""
        et = _event_type(event)
        ts = getattr(event, "timestamp", None)
        logger.debug("publish %s ts=%s", type(event).__name__, ts)
        self._queue.put((et, event))

    def flush(self) -> None:
        """Block until every previously published event has been dispatched.

        Useful in tests and orderly shutdown sequences.
        """
        self._queue.join()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        while True:
            event_type, event = self._queue.get()
            try:
                self._dispatch(event_type, event)
            finally:
                self._queue.task_done()

    def _dispatch(self, event_type: EventType, event: Any) -> None:
        with self._lock:
            callbacks = list(self._subscribers[event_type])
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception(
                    "Exception in event handler %s for event type %s", cb, event_type
                )
