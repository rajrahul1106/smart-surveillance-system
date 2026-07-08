from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.event_bus import EventType
from core.events import GestureConfirmed, GestureProgress, GestureRejected
from core.states.base_state import AbstractState

logger = logging.getLogger(__name__)


class VerifyingGestureState(AbstractState):

    def __init__(self, context: Dict[str, Any]) -> None:
        self._gesture_model = context["gesture_model"]
        self._config = context["config"]
        self._rolling_buffer: List[str] = []
        self._matched_steps: List[str] = []
        self._entry_time: float = 0.0

    @property
    def active_models(self) -> List[Any]:
        return [self._gesture_model]

    @property
    def subscriptions(self) -> Dict[str, Any]:
        return {EventType.PROCESSED_FRAME: self.on_frame}

    def on_enter(self, context: Dict[str, Any]) -> None:
        self._entry_time = time.monotonic()
        self._rolling_buffer.clear()
        self._matched_steps.clear()

        # Process every camera frame for responsive gesture timing.
        fm = context.get("frame_manager")
        if fm is not None:
            fm.set_skip_rate(1)

        expected = self._config.gesture.sequence
        context["last_detections"] = context.get("last_detections", {}) or {}
        context["last_detections"]["sos_progress"] = {
            "step": 0,
            "total": len(expected),
            "sequence": list(expected),
            "matched": [],
        }

    def on_exit(self, context: Dict[str, Any]) -> None:
        self._rolling_buffer.clear()
        self._matched_steps.clear()
        if "last_detections" in context:
            context["last_detections"].pop("sos_progress", None)
            context["last_detections"].pop("landmarks", None)
            context["last_detections"].pop("landmarks_frame_size", None)
            context["last_detections"].pop("gesture_label", None)
            context["last_detections"].pop("gesture_confidence", None)

    def on_frame(self, frame: Any, context: Dict[str, Any]) -> Optional[str]:
        result = self._gesture_model.predict(frame)
        gesture = result.get("gesture", "")

        detections = context.get("last_detections", {}) or {}
        if result.get("landmarks"):
            detections["landmarks"] = result["landmarks"]
            detections["landmarks_frame_size"] = result.get("frame_size")
        else:
            detections.pop("landmarks", None)
            detections.pop("landmarks_frame_size", None)
        detections["gesture_label"] = gesture if gesture and gesture != "none" else None
        detections["gesture_confidence"] = result.get("confidence")
        context["last_detections"] = detections

        expected = self._config.gesture.sequence

        # Append ONLY on a real gesture change. Skip "none" / empty detections
        # (no hand visible) and skip repeats while a gesture is held — otherwise
        # holding "palm" for 10 frames would fill the buffer with palms and the
        # state would time out without ever seeing the alternating pattern.
        if gesture and gesture != "none":
            last = self._rolling_buffer[-1] if self._rolling_buffer else None
            if gesture != last:
                self._rolling_buffer.append(gesture)
                if len(self._rolling_buffer) > len(expected):
                    self._rolling_buffer = self._rolling_buffer[-len(expected):]

                logger.debug(
                    "Gesture buffer: %s (%d/%d)",
                    self._rolling_buffer,
                    len(self._rolling_buffer),
                    len(expected),
                )

                # Recompute matched-prefix progress on every transition.
                self._matched_steps = self._compute_matched_steps(
                    self._rolling_buffer, expected,
                )
                self._publish_progress(context, expected)

        if self._rolling_buffer == expected:
            logger.info("SOS sequence COMPLETE - triggering gesture_confirmed")
            context["event_bus"].publish(GestureConfirmed())
            return "gesture_confirmed"

        elapsed = time.monotonic() - self._entry_time
        if elapsed > self._config.gesture.sequence_timeout_seconds:
            context["event_bus"].publish(GestureRejected())
            return "gesture_rejected"

        return None

    @staticmethod
    def _compute_matched_steps(buffer: List[str], expected: List[str]) -> List[str]:
        """Longest prefix of *buffer* that matches *expected*."""
        matched: List[str] = []
        for i, g in enumerate(buffer):
            if i < len(expected) and g == expected[i]:
                matched.append(g)
            else:
                break
        return matched

    def _publish_progress(self, context: Dict[str, Any], expected: List[str]) -> None:
        step = len(self._matched_steps)
        total = len(expected)
        current = self._matched_steps[-1] if self._matched_steps else ""

        context["event_bus"].publish(GestureProgress(
            step=step,
            total=total,
            current_gesture=current,
            sequence=list(expected),
        ))

        detections = context.get("last_detections", {}) or {}
        detections["sos_progress"] = {
            "step": step,
            "total": total,
            "sequence": list(expected),
            "matched": list(self._matched_steps),
        }
        context["last_detections"] = detections
