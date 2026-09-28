from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from core.event_bus import EventType
from core.events import (
    DetectionComplete,
    FireDetected,
    InjuryDetected,
    SuspiciousActivity,
)
from core.states.base_state import AbstractState
from core.states.presence import publish_presence_if_changed

logger = logging.getLogger(__name__)

_MODEL_EVENT_FACTORIES = {
    "fire_model": lambda result: FireDetected(result=result),
    "injury_model": lambda result: InjuryDetected(result=result),
    "activity_model": lambda result: SuspiciousActivity(result=result),
}

# Map model name → key used in the ``last_detections`` dict that the
# annotator and dashboard read.
_DETECTION_KEY = {
    "fire_model": "fire",
    "injury_model": "injury",
    "activity_model": "suspicious",
}

# Staggered-inference schedule. Running all 5 models on every frame
# saturates the CPU and pins process FPS around 10. We stagger fire /
# injury / activity so most frames only run 0-1 of them.
#
#   fire     — every 3rd frame, offset 1  → frames 1, 4, 7, 10, …
#   injury   — every 3rd frame, offset 2  → frames 2, 5, 8, 11, …
#                                          (never co-runs with fire)
#   activity — every 5th frame, offset 0  → frames 5, 10, 15, …
#   face     — every ``face_auth.presence_interval_frames`` (default 10),
#              offset 3 → frames 3, 13, 23, …  Presence monitoring costs
#              ~200-300 ms per call on the M2 CPU (detector at 960×960), far
#              more than a 33 ms frame, so it runs on its own worker thread:
#              the frame never waits for it, a slot is skipped if the
#              previous run is still going, and between runs the last result
#              is reused.
#
# Frame 1 (state entry) seeds the cache by running everything once so
# the annotator has something to show immediately and existing
# integration tests that expect all three models on a single on_frame
# call still pass.
_FIRE_PERIOD, _FIRE_OFFSET = 3, 1
_INJURY_PERIOD, _INJURY_OFFSET = 3, 2
_ACTIVITY_PERIOD, _ACTIVITY_OFFSET = 5, 0
_FACE_PERIOD_DEFAULT, _FACE_OFFSET = 10, 3


class ActiveDetectionState(AbstractState):

    def __init__(self, context: Dict[str, Any]) -> None:
        self._fire_model = context["fire_model"]
        self._injury_model = context["injury_model"]
        self._activity_model = context["activity_model"]
        self._face_model = context["face_model"]
        self._gesture_model = context["gesture_model"]
        self._config = context["config"]
        self._entry_time: float = 0.0
        self._detection_results: List[Dict[str, Any]] = []

        # Staggered-inference state
        self._frame_counter: int = 0
        self._cached_results: Dict[str, Dict[str, Any]] = {}

        # Presence monitoring (face model on a worker thread)
        self._face_period: int = _FACE_PERIOD_DEFAULT
        self._face_executor: Optional[ThreadPoolExecutor] = None
        self._face_future: Optional[Future] = None

    @property
    def active_models(self) -> List[Any]:
        return [self._fire_model, self._injury_model, self._activity_model, self._face_model]

    @property
    def subscriptions(self) -> Dict[str, Any]:
        return {EventType.PROCESSED_FRAME: self.on_frame}

    def on_enter(self, context: Dict[str, Any]) -> None:
        # The face model stays loaded from VERIFYING_IDENTITY and switches to
        # full-frame presence monitoring; only the gesture model is released.
        self._gesture_model.unload()
        self._face_model.load()
        self._face_model.set_mode("presence")
        self._fire_model.load()
        self._injury_model.load()
        self._activity_model.load()
        self._entry_time = time.monotonic()
        self._detection_results.clear()
        self._frame_counter = 0
        self._cached_results = {}
        self._face_period = max(1, int(getattr(
            self._config.face_auth, "presence_interval_frames", _FACE_PERIOD_DEFAULT,
        )))
        self._face_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="face-presence",
        )
        self._face_future = None
        # Process every camera frame — staggered inference handles CPU load.
        fm = context.get("frame_manager")
        if fm is not None:
            fm.set_skip_rate(1)
        detections = context.get("last_detections", {}) or {}
        detections.pop("landmarks", None)
        detections.pop("landmarks_frame_size", None)
        context["last_detections"] = detections

    def on_exit(self, context: Dict[str, Any]) -> None:
        self._stop_face_worker()
        self._fire_model.unload()
        self._injury_model.unload()
        self._activity_model.unload()
        self._face_model.unload()
        self._frame_counter = 0
        self._cached_results = {}
        detections = context.get("last_detections")
        if detections is not None:
            # Presence monitoring stops here; don't leave last-seen boxes on
            # screen through COOLDOWN.
            detections["faces"] = []
            detections.pop("presence", None)

    # ------------------------------------------------------------------
    # Frame handling
    # ------------------------------------------------------------------

    def on_frame(self, frame: Any, context: Dict[str, Any]) -> Optional[str]:
        self._frame_counter += 1
        counter = self._frame_counter
        event_bus = context["event_bus"]

        models = {
            "fire_model": self._fire_model,
            "injury_model": self._injury_model,
            "activity_model": self._activity_model,
        }

        face_result = self._collect_face_result()
        if face_result is not None:
            self._cached_results["face_model"] = face_result
            publish_presence_if_changed(context, face_result.get("presence"))

        scheduled = self._schedule_for(counter)
        if "face_model" in scheduled:
            scheduled.remove("face_model")
            face_frame = context.get("original_frame")
            self._submit_face_run(face_frame if face_frame is not None else frame)
        results = self._run_models(models, scheduled, frame)

        # Update cache and publish events for fresh detections only.
        for name, result in results.items():
            self._cached_results[name] = result
            if result.get("detected", False):
                self._detection_results.append({"model": name, "result": result})
                event_bus.publish(_MODEL_EVENT_FACTORIES[name](result))

        # Build the detections dict from the latest cached state per model
        # — fresh from this frame where available, otherwise from the most
        # recent prior run. Models that ran fresh and said "not detected"
        # clear any stale entry; models that haven't run yet leave any
        # pre-existing detection alone.
        detections = context.get("last_detections", {}) or {}
        detections.pop("landmarks", None)
        detections.pop("landmarks_frame_size", None)
        for name, dkey in _DETECTION_KEY.items():
            cached = self._cached_results.get(name)
            if cached is None:
                continue
            if cached.get("detected"):
                detections[dkey] = cached
            else:
                detections.pop(dkey, None)

        presence_result = self._cached_results.get("face_model")
        if presence_result is not None:
            detections["faces"] = presence_result.get("faces", [])
            detections["faces_frame_size"] = presence_result.get("frame_size")
            detections["presence"] = presence_result.get("presence")

        # Frame size the detections were computed on — used by the
        # annotator to scale bounding boxes for display.
        shape = getattr(frame, "shape", None)
        if shape and len(shape) >= 2:
            detections["detection_frame_size"] = (int(shape[1]), int(shape[0]))
        context["last_detections"] = detections

        elapsed = time.monotonic() - self._entry_time
        if elapsed > self._config.detection.active_timeout_seconds:
            event_bus.publish(DetectionComplete(results=list(self._detection_results)))
            return "detection_complete"

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _schedule_for(self, counter: int) -> List[str]:
        """Return the list of model names that should run on this tick.

        Frame 1 seeds the cache by running everything; subsequent frames
        follow the staggered schedule defined at the top of this module.
        """
        if counter == 1:
            return ["fire_model", "injury_model", "activity_model", "face_model"]

        scheduled: List[str] = []
        if counter % _FIRE_PERIOD == _FIRE_OFFSET:
            scheduled.append("fire_model")
        if counter % _INJURY_PERIOD == _INJURY_OFFSET:
            scheduled.append("injury_model")
        if counter % _ACTIVITY_PERIOD == _ACTIVITY_OFFSET:
            scheduled.append("activity_model")
        if counter % self._face_period == _FACE_OFFSET % self._face_period:
            scheduled.append("face_model")
        return scheduled

    def _submit_face_run(self, face_frame: Any) -> None:
        """Start a presence run on the worker unless one is still in flight."""
        if self._face_executor is None or not self._face_model.is_loaded:
            return
        if self._face_future is not None and not self._face_future.done():
            return  # previous run still going; the next slot will try again
        self._face_future = self._face_executor.submit(self._face_model.predict, face_frame)

    def _collect_face_result(self) -> Optional[Dict[str, Any]]:
        """Return the finished presence run's result, if one has completed."""
        future = self._face_future
        if future is None or not future.done():
            return None
        self._face_future = None
        try:
            return future.result()
        except Exception:
            logger.exception("Presence face run failed")
            return None

    def _stop_face_worker(self) -> None:
        if self._face_executor is not None:
            # Don't wait: an in-flight run finishes in the background and
            # FaceModel.unload() waits for it under the model's lock.
            self._face_executor.shutdown(wait=False, cancel_futures=True)
            self._face_executor = None
        self._face_future = None

    @staticmethod
    def _run_models(
        models: Dict[str, Any], scheduled: List[str], frame: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Run *scheduled* models against *frame* and return name → result."""
        if not scheduled:
            return {}

        if len(scheduled) == 1:
            # Single-model frames are the common case under staggering;
            # skip the ThreadPoolExecutor overhead for them.
            name = scheduled[0]
            return {name: models[name].predict(frame)}

        results: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(scheduled)) as executor:
            futures = {
                name: executor.submit(models[name].predict, frame)
                for name in scheduled
            }
            for name, fut in futures.items():
                results[name] = fut.result()
        return results
