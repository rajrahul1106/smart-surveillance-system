from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from core.states.base_state import AbstractState

logger = logging.getLogger(__name__)


class CooldownState(AbstractState):

    def __init__(self, context: Dict[str, Any]) -> None:
        self._gesture_model = context["gesture_model"]
        self._config = context["config"]
        self._timer: Optional[threading.Timer] = None
        self._trigger_callback: Optional[Any] = None

    @property
    def active_models(self) -> List[Any]:
        return []

    @property
    def subscriptions(self) -> Dict[str, Any]:
        return {}

    def on_enter(self, context: Dict[str, Any]) -> None:
        self._trigger_callback = context.get("trigger_callback")
        # Reduce frame processing during cooldown — no models running.
        fm = context.get("frame_manager")
        if fm is not None:
            cfg = context["config"]
            idle_skip = getattr(getattr(cfg, "frame", None), "skip_rate_idle", 3)
            fm.set_skip_rate(idle_skip)
        duration = self._config.detection.cooldown_seconds
        self._timer = threading.Timer(duration, self._on_cooldown_expired)
        self._timer.daemon = True
        self._timer.start()

    def _on_cooldown_expired(self) -> None:
        if self._trigger_callback is not None:
            self._trigger_callback("cooldown_expired")

    def on_exit(self, context: Dict[str, Any]) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def on_frame(self, frame: Any, context: Dict[str, Any]) -> Optional[str]:
        return None
