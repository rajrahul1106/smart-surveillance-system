from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.event_bus import EventType
from core.events import GestureCandidate, GestureNearMiss
from core.states.base_state import AbstractState
from core.states.presence import reset_presence

logger = logging.getLogger(__name__)


class IdleState(AbstractState):

    def __init__(self, context: Dict[str, Any]) -> None:
        self._gesture_model = context["gesture_model"]
        self._config = context["config"]
        self._frame_count: int = 0

    @property
    def active_models(self) -> List[Any]:
        return [self._gesture_model]

    @property
    def subscriptions(self) -> Dict[str, Any]:
        return {EventType.PROCESSED_FRAME: self.on_frame}

    def on_enter(self, context: Dict[str, Any]) -> None:
        self._gesture_model.load()
        for name in ("face_model", "fire_model", "injury_model", "activity_model"):
            model = context.get(name)
            if model is not None:
                model.unload()
        # Restore idle frame skip rate for lower CPU usage.
        fm = context.get("frame_manager")
        if fm is not None:
            cfg = context["config"]
            idle_skip = getattr(getattr(cfg, "frame", None), "skip_rate_idle", 3)
            fm.set_skip_rate(idle_skip)
        # Clear stale face auth data from previous detection cycle so the
        # dashboard no longer shows AUTHORIZED after cooldown ends.
        detections = context.get("last_detections") or {}
        for key in ("face_bbox", "face_name", "face_confidence",
                     "face_authorized", "face_frame_size",
                     "faces", "faces_frame_size", "presence",
                     "authenticated_user", "authenticated_users"):
            detections.pop(key, None)
        context["last_detections"] = detections
        # Presence monitoring is over; the next session announces its own.
        reset_presence(context)

    def on_exit(self, context: Dict[str, Any]) -> None:
        pass

    def on_frame(self, frame: Any, context: Dict[str, Any]) -> Optional[str]:
        # COOLDOWN -> IDLE fires on a timer thread, so a frame can arrive
        # while on_enter is still loading the gesture model. Skip it.
        if not self._gesture_model.is_loaded:
            return None
        result = self._gesture_model.predict(frame)
        confidence = result.get("confidence", 0.0)

        self._frame_count += 1
        if self._frame_count % 30 == 0:
            logger.debug(
                "IdleState frame #%d — gesture=%s confidence=%.3f",
                self._frame_count, result.get("gesture", "?"), confidence,
            )

        threshold = self._config.gesture.confidence_threshold
        near_miss = self._config.gesture.near_miss_threshold

        context["last_detections"] = {
            "landmarks": result.get("landmarks"),
            "landmarks_frame_size": result.get("frame_size"),
        }

        if confidence >= threshold:
            context["event_bus"].publish(GestureCandidate(confidence=confidence, frame=frame))
            return "gesture_candidate"

        if confidence >= near_miss:
            context["event_bus"].publish(GestureNearMiss(confidence=confidence, frame=frame))

        return None
