"""Tests for EventBus and typed event dataclasses."""

from __future__ import annotations

import sys
import os
import threading
import time
from dataclasses import dataclass, field
from typing import ClassVar, List

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.event_bus import Event, EventBus, EventType
from core.events import (
    AlertDispatched,
    AuthFailed,
    AuthSuccess,
    CameraConnected,
    CameraDisconnected,
    DetectionComplete,
    FireDetected,
    GestureCandidate,
    GestureConfirmed,
    GestureNearMiss,
    GestureRejected,
    InjuryDetected,
    ModelLoaded,
    ModelUnloaded,
    ProcessedFrame,
    StateTransition,
    SuspiciousActivity,
    SystemShutdown,
    SystemStartup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bus() -> EventBus:
    return EventBus()


# ---------------------------------------------------------------------------
# Basic subscribe / publish
# ---------------------------------------------------------------------------

class TestSubscribePublish:
    def test_subscriber_receives_event(self):
        bus = _bus()
        received = []
        bus.subscribe(EventType.GESTURE_CANDIDATE, received.append)

        bus.publish(GestureCandidate(confidence=0.9))
        bus.flush()

        assert len(received) == 1
        assert isinstance(received[0], GestureCandidate)
        assert received[0].confidence == 0.9

    def test_multiple_subscribers_all_receive(self):
        bus = _bus()
        r1, r2, r3 = [], [], []
        bus.subscribe(EventType.FIRE_DETECTED, r1.append)
        bus.subscribe(EventType.FIRE_DETECTED, r2.append)
        bus.subscribe(EventType.FIRE_DETECTED, r3.append)

        bus.publish(FireDetected(result={"zone": "A"}))
        bus.flush()

        assert len(r1) == len(r2) == len(r3) == 1

    def test_subscriber_for_different_type_not_called(self):
        bus = _bus()
        received = []
        bus.subscribe(EventType.FIRE_DETECTED, received.append)

        bus.publish(InjuryDetected(result={}))
        bus.flush()

        assert len(received) == 0

    def test_legacy_event_dispatched(self):
        """Legacy Event(type=..., data=...) still works."""
        bus = _bus()
        received = []
        bus.subscribe(EventType.GESTURE_CONFIRMED, received.append)

        bus.publish(Event(type=EventType.GESTURE_CONFIRMED, data={"extra": 1}))
        bus.flush()

        assert len(received) == 1
        assert received[0].data["extra"] == 1


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------

class TestUnsubscribe:
    def test_removed_subscriber_does_not_receive(self):
        bus = _bus()
        received = []

        def handler(e):
            received.append(e)

        bus.subscribe(EventType.AUTH_FAILED, handler)
        bus.publish(AuthFailed(reason="timeout"))
        bus.flush()
        assert len(received) == 1

        bus.unsubscribe(EventType.AUTH_FAILED, handler)
        bus.publish(AuthFailed(reason="again"))
        bus.flush()
        assert len(received) == 1  # still 1, not 2

    def test_unsubscribe_nonexistent_is_safe(self):
        bus = _bus()
        # Should not raise
        bus.unsubscribe(EventType.AUTH_SUCCESS, lambda e: None)

    def test_only_matching_subscriber_removed(self):
        bus = _bus()
        r1, r2 = [], []

        def h1(e): r1.append(e)
        def h2(e): r2.append(e)

        bus.subscribe(EventType.GESTURE_REJECTED, h1)
        bus.subscribe(EventType.GESTURE_REJECTED, h2)
        bus.unsubscribe(EventType.GESTURE_REJECTED, h1)

        bus.publish(GestureRejected())
        bus.flush()

        assert len(r1) == 0
        assert len(r2) == 1


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------

class TestExceptionIsolation:
    def test_exception_in_one_subscriber_does_not_block_others(self):
        bus = _bus()
        good_received = []

        def bad_handler(e):
            raise RuntimeError("intentional test error")

        def good_handler(e):
            good_received.append(e)

        bus.subscribe(EventType.AUTH_SUCCESS, bad_handler)
        bus.subscribe(EventType.AUTH_SUCCESS, good_handler)

        bus.publish(AuthSuccess(result={}))
        bus.flush()

        assert len(good_received) == 1

    def test_multiple_exceptions_all_other_subscribers_still_run(self):
        bus = _bus()
        good = []

        def raiser(e): raise ValueError("boom")

        bus.subscribe(EventType.DETECTION_COMPLETE, raiser)
        bus.subscribe(EventType.DETECTION_COMPLETE, raiser)
        bus.subscribe(EventType.DETECTION_COMPLETE, good.append)

        bus.publish(DetectionComplete(results=[]))
        bus.flush()

        assert len(good) == 1


# ---------------------------------------------------------------------------
# No deadlock when publishing from inside a subscriber
# ---------------------------------------------------------------------------

class TestNoDeadlock:
    def test_publish_from_inside_subscriber(self):
        """Subscriber publishes a second event — must not deadlock."""
        bus = _bus()
        secondary = []

        def chained_handler(e):
            # Publishing from inside a subscriber callback
            bus.publish(AuthSuccess(result={"chained": True}))

        bus.subscribe(EventType.AUTH_FAILED, chained_handler)
        bus.subscribe(EventType.AUTH_SUCCESS, secondary.append)

        bus.publish(AuthFailed(reason="trigger"))
        bus.flush()
        bus.flush()  # second flush to drain the chained event

        assert len(secondary) == 1
        assert secondary[0].result["chained"] is True

    def test_concurrent_publishers_no_deadlock(self):
        bus = _bus()
        received = []
        bus.subscribe(EventType.MODEL_LOADED, received.append)

        def publisher():
            for i in range(10):
                bus.publish(ModelLoaded(model_name=f"model_{i}"))

        threads = [threading.Thread(target=publisher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        bus.flush()

        assert len(received) == 50


# ---------------------------------------------------------------------------
# Event ordering
# ---------------------------------------------------------------------------

class TestEventOrdering:
    def test_events_delivered_in_publish_order(self):
        bus = _bus()
        order = []

        bus.subscribe(EventType.MODEL_LOADED, lambda e: order.append(e.model_name))

        for name in ["A", "B", "C", "D", "E"]:
            bus.publish(ModelLoaded(model_name=name))
        bus.flush()

        assert order == ["A", "B", "C", "D", "E"]

    def test_mixed_event_types_maintain_internal_order(self):
        bus = _bus()
        log = []

        bus.subscribe(EventType.MODEL_LOADED, lambda e: log.append(("loaded", e.model_name)))
        bus.subscribe(EventType.MODEL_UNLOADED, lambda e: log.append(("unloaded", e.model_name)))

        bus.publish(ModelLoaded(model_name="X"))
        bus.publish(ModelUnloaded(model_name="Y"))
        bus.publish(ModelLoaded(model_name="Z"))
        bus.flush()

        assert log == [("loaded", "X"), ("unloaded", "Y"), ("loaded", "Z")]


# ---------------------------------------------------------------------------
# Typed event dataclasses
# ---------------------------------------------------------------------------

class TestTypedEventDataclasses:
    def test_all_events_have_timestamp(self):
        before = time.time()
        events = [
            ProcessedFrame(frame=None),
            GestureCandidate(confidence=0.5),
            GestureNearMiss(confidence=0.4),
            GestureConfirmed(),
            GestureRejected(),
            AuthSuccess(),
            AuthFailed(),
            FireDetected(),
            InjuryDetected(),
            SuspiciousActivity(),
            DetectionComplete(),
            AlertDispatched(),
            StateTransition(),
            SystemStartup(),
            SystemShutdown(),
            ModelLoaded(),
            ModelUnloaded(),
            CameraConnected(camera_id="cam0", index=0),
            CameraDisconnected(camera_id="cam0"),
        ]
        after = time.time()
        for ev in events:
            assert hasattr(ev, "timestamp"), f"{type(ev).__name__} missing timestamp"
            assert before <= ev.timestamp <= after

    def test_event_type_classvar_matches_enum(self):
        mapping = {
            ProcessedFrame: EventType.PROCESSED_FRAME,
            GestureCandidate: EventType.GESTURE_CANDIDATE,
            GestureNearMiss: EventType.GESTURE_NEAR_MISS,
            GestureConfirmed: EventType.GESTURE_CONFIRMED,
            GestureRejected: EventType.GESTURE_REJECTED,
            AuthSuccess: EventType.AUTH_SUCCESS,
            AuthFailed: EventType.AUTH_FAILED,
            FireDetected: EventType.FIRE_DETECTED,
            InjuryDetected: EventType.INJURY_DETECTED,
            SuspiciousActivity: EventType.SUSPICIOUS_ACTIVITY,
            DetectionComplete: EventType.DETECTION_COMPLETE,
            AlertDispatched: EventType.ALERT_DISPATCHED,
            StateTransition: EventType.STATE_TRANSITION,
            SystemStartup: EventType.SYSTEM_STARTUP,
            SystemShutdown: EventType.SYSTEM_SHUTDOWN,
            ModelLoaded: EventType.MODEL_LOADED,
            ModelUnloaded: EventType.MODEL_UNLOADED,
            CameraConnected: EventType.CAMERA_CONNECTED,
            CameraDisconnected: EventType.CAMERA_DISCONNECTED,
        }
        for cls, expected_type in mapping.items():
            assert cls.event_type == expected_type, (
                f"{cls.__name__}.event_type should be {expected_type}"
            )

    def test_processed_frame_fields(self):
        pf = ProcessedFrame(frame="data", camera_id="cam1", resolution=(1920, 1080))
        assert pf.frame == "data"
        assert pf.camera_id == "cam1"
        assert pf.resolution == (1920, 1080)

    def test_auth_failed_reason(self):
        ev = AuthFailed(reason="max_attempts_exceeded")
        assert ev.reason == "max_attempts_exceeded"

    def test_detection_complete_results(self):
        results = [{"model": "fire", "result": {"detected": True}}]
        ev = DetectionComplete(results=results)
        assert ev.results == results

    def test_model_loaded_fields(self):
        ev = ModelLoaded(model_name="gesture_model", load_time_ms=42.5)
        assert ev.model_name == "gesture_model"
        assert ev.load_time_ms == 42.5

    def test_camera_connected_fields(self):
        ev = CameraConnected(camera_id="cam0", index=0, resolution=(640, 480))
        assert ev.camera_id == "cam0"
        assert ev.index == 0
        assert ev.resolution == (640, 480)
