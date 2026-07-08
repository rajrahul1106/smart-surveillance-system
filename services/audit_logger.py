from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from core.event_bus import EventBus, EventType
from core.events import StateTransition
from services.storage_service import StorageService

logger = logging.getLogger(__name__)


class AuditLogger:

    def __init__(self, event_bus: EventBus, storage_service: StorageService) -> None:
        self._event_bus = event_bus
        self._storage = storage_service
        self._handlers: Dict[EventType, Callable] = {}

    def start(self) -> None:
        for et in EventType:
            handler = self._make_handler(et)
            self._handlers[et] = handler
            self._event_bus.subscribe(et, handler)
        logger.info("AuditLogger started — subscribed to %d event types", len(self._handlers))

    def stop(self) -> None:
        for et, handler in self._handlers.items():
            self._event_bus.unsubscribe(et, handler)
        self._handlers.clear()
        logger.info("AuditLogger stopped")

    def _make_handler(self, et: EventType) -> Callable:
        def _handle(event: Any) -> None:
            try:
                self._storage.save_event(event)
            except Exception:
                logger.warning("Failed to save event %s", et.name, exc_info=True)

            if et == EventType.STATE_TRANSITION and isinstance(event, StateTransition):
                try:
                    self._storage.save_transition(
                        from_state=event.from_state,
                        to_state=event.to_state,
                        trigger=getattr(event, "trigger", ""),
                        timestamp=event.timestamp,
                    )
                except Exception:
                    logger.warning("Failed to save state transition", exc_info=True)

        return _handle
