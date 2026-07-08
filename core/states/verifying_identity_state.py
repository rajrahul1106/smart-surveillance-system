from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.event_bus import EventType
from core.events import AuthFailed, AuthSuccess
from core.states.base_state import AbstractState

logger = logging.getLogger(__name__)


class VerifyingIdentityState(AbstractState):

    def __init__(self, context: Dict[str, Any]) -> None:
        self._gesture_model = context["gesture_model"]
        self._face_model = context["face_model"]
        self._config = context["config"]
        self._attempt_count: int = 0
        self._entry_time: float = 0.0

    @property
    def active_models(self) -> List[Any]:
        return [self._gesture_model, self._face_model]

    @property
    def subscriptions(self) -> Dict[str, Any]:
        return {EventType.PROCESSED_FRAME: self.on_frame}

    def on_enter(self, context: Dict[str, Any]) -> None:
        self._face_model.load()
        self._attempt_count = 0
        self._entry_time = time.monotonic()
        # Process every camera frame for face detection accuracy.
        fm = context.get("frame_manager")
        if fm is not None:
            fm.set_skip_rate(1)

    def on_exit(self, context: Dict[str, Any]) -> None:
        next_state = context.get("next_state")
        if next_state is not None and next_state.name != "ACTIVE_DETECTION":
            self._face_model.unload()

    def on_frame(self, frame: Any, context: Dict[str, Any]) -> Optional[str]:
        result = self._face_model.predict(frame)
        # The face model gates authorization on cosine similarity >= 0.5
        # (see FaceModel.AUTHORIZE_THRESHOLD) and emits BOTH "is_authorized"
        # and the legacy "authorized" alias with the same value. Accept
        # either so the state machine doesn't depend on which key is used.
        authorized = bool(result.get("is_authorized", result.get("authorized", False)))
        is_live = result.get("is_live", False)

        detections = context.get("last_detections", {}) or {}
        detections.pop("landmarks", None)
        detections.pop("landmarks_frame_size", None)
        detections["face_bbox"] = result.get("bbox")
        # Use the model's pre-formatted label (handles AUTHORIZED / POSSIBLE
        # / UNKNOWN tiers) so the dashboard never shows a name without the
        # corresponding confidence prefix.
        detections["face_name"] = result.get("label") or result.get("name") or result.get("user_id")
        detections["face_confidence"] = result.get("confidence", 0.0)
        detections["face_authorized"] = authorized
        # The face bbox is in the coordinates of *this* frame; the annotator
        # uses this size to scale to its output resolution. Prefer the model's
        # own frame_size (handles the insightface (x1,y1,x2,y2) origin), and
        # fall back to the frame's shape when available.
        face_size = result.get("frame_size")
        if face_size is None:
            shape = getattr(frame, "shape", None)
            if shape and len(shape) >= 2:
                face_size = (int(shape[1]), int(shape[0]))
        if face_size is not None:
            detections["face_frame_size"] = face_size
        context["last_detections"] = detections

        if authorized and is_live:
            context["event_bus"].publish(AuthSuccess(result=result))
            return "auth_success"

        self._attempt_count += 1

        if self._attempt_count >= self._config.face_auth.max_attempts:
            context["event_bus"].publish(AuthFailed(reason="max_attempts_exceeded"))
            return "auth_failed"

        elapsed = time.monotonic() - self._entry_time
        if elapsed > self._config.face_auth.attempt_timeout_seconds:
            context["event_bus"].publish(AuthFailed(reason="timeout"))
            return "auth_failed"

        return None
