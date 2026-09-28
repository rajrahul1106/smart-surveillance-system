from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.event_bus import EventType
from core.events import AuthFailed, AuthSuccess
from core.states.base_state import AbstractState
from core.states.presence import publish_presence_if_changed

logger = logging.getLogger(__name__)

# A face must be AUTHORIZED on this many consecutive processed frames before
# authentication succeeds, so a single lucky match can't open the session.
AUTH_CONFIRM_FRAMES = 3


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
        # Centre crop + upscale: authenticate the (possibly distant) person
        # who just made the gesture.
        self._face_model.set_mode("verify")
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
        # The face model gets the full-resolution frame when the pipeline
        # provides it; enrollment embeddings are taken from that same frame.
        face_frame = context.get("original_frame")
        if face_frame is None:
            face_frame = frame
        result = self._face_model.predict(face_frame)
        # The face model emits BOTH "is_authorized" and the legacy
        # "authorized" alias with the same value. Accept either so the state
        # machine doesn't depend on which key is used.
        authorized = bool(result.get("is_authorized", result.get("authorized", False)))
        is_live = result.get("is_live", False)
        faces: Optional[List[Dict[str, Any]]] = result.get("faces")

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
        # The face bbox is in the coordinates of the frame the model saw; the
        # annotator uses this size to scale to its output resolution. Prefer
        # the model's own frame_size, and fall back to the frame's shape.
        face_size = result.get("frame_size")
        if face_size is None:
            shape = getattr(face_frame, "shape", None)
            if shape and len(shape) >= 2:
                face_size = (int(shape[1]), int(shape[0]))
        if face_size is not None:
            detections["face_frame_size"] = face_size
        if faces is not None:
            # One box per person; the annotator prefers these over face_bbox.
            detections["faces"] = faces
            detections["faces_frame_size"] = face_size
            detections["presence"] = result.get("presence")
        context["last_detections"] = detections
        publish_presence_if_changed(context, result.get("presence"))

        if faces is None:
            # Result without per-face tracking: one authorized frame is enough.
            confirmed: List[Dict[str, Any]] = []
            success = authorized and is_live
        else:
            confirmed = [
                f for f in faces
                if f.get("status") == "AUTHORIZED"
                and f.get("auth_streak", 0) >= AUTH_CONFIRM_FRAMES
            ]
            success = bool(confirmed)

        if success:
            if confirmed:
                result = self._record_session(result, faces or [], confirmed, detections)
            context["event_bus"].publish(AuthSuccess(result=result))
            return "auth_success"

        # A frame where someone is AUTHORIZED but still building their streak
        # is progress towards success, not a failed attempt.
        if not any(f.get("status") == "AUTHORIZED" for f in faces or ()):
            self._attempt_count += 1

        if self._attempt_count >= self._config.face_auth.max_attempts:
            context["event_bus"].publish(AuthFailed(reason="max_attempts_exceeded"))
            return "auth_failed"

        elapsed = time.monotonic() - self._entry_time
        if elapsed > self._config.face_auth.attempt_timeout_seconds:
            context["event_bus"].publish(AuthFailed(reason="timeout"))
            return "auth_failed"

        return None

    @staticmethod
    def _record_session(
        result: Dict[str, Any],
        faces: List[Dict[str, Any]],
        confirmed: List[Dict[str, Any]],
        detections: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Store who authenticated in the session context; return the AuthSuccess payload.

        The primary user is the confirmed face with the highest smoothed
        confidence; every enrolled person recognised in the frame is recorded.
        """
        primary = max(confirmed, key=lambda f: f.get("confidence", 0.0))
        name = primary["identity"]
        confidence = float(primary.get("confidence", 0.0)) / 100.0
        everyone = sorted({
            f["identity"] for f in faces
            if f.get("status") == "AUTHORIZED" and f.get("identity")
        })

        detections.update({
            "face_bbox": primary["bbox"],
            "face_name": name,
            "face_confidence": confidence,
            "face_authorized": True,
            "authenticated_user": name,
            "authenticated_users": everyone,
        })
        logger.info(
            "Identity verified: %s (%.0f%%)%s", name, confidence * 100.0,
            f"; also recognised: {[n for n in everyone if n != name]}" if len(everyone) > 1 else "",
        )

        auth_result = dict(result)
        auth_result.update({
            "user_id": name,
            "name": name,
            "label": name,
            "confidence": confidence,
            "authorized": True,
            "is_authorized": True,
            "bbox": primary["bbox"],
            "authenticated_users": everyone,
        })
        return auth_result
