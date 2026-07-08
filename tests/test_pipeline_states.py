"""Tests for the state-driven pipeline architecture."""

from __future__ import annotations

import sys
import os
import time
import threading
from unittest.mock import MagicMock

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.event_bus import EventBus, EventType
from core.events import (
    AuthFailed,
    AuthSuccess,
    DetectionComplete,
    FireDetected,
    GestureCandidate,
    GestureConfirmed,
    GestureNearMiss,
    GestureRejected,
    InjuryDetected,
    ProcessedFrame,
    SuspiciousActivity,
)
from core.config import PipelineConfig, GestureConfig, FaceAuthConfig, DetectionConfig
from core.state_machine import State, StateMachine
from core.states.idle_state import IdleState
from core.states.verifying_gesture_state import VerifyingGestureState
from core.states.verifying_identity_state import VerifyingIdentityState
from core.states.active_detection_state import ActiveDetectionState
from core.states.cooldown_state import CooldownState
from core.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(**predict_return):
    model = MagicMock()
    model.predict.return_value = predict_return
    return model


def _make_context(**overrides):
    ctx = {
        "event_bus": EventBus(),
        "config": PipelineConfig(),
        "gesture_model": _make_model(confidence=0.0, gesture="none"),
        "face_model": _make_model(authorized=False, is_live=False),
        "fire_model": _make_model(detected=False),
        "injury_model": _make_model(detected=False),
        "activity_model": _make_model(detected=False),
        "camera_manager": None,
        "trigger_callback": MagicMock(),
    }
    ctx.update(overrides)
    return ctx


def _publish_frame(event_bus: EventBus, frame: str = "frame_data", camera_id: str = "") -> None:
    event_bus.publish(ProcessedFrame(frame=frame, camera_id=camera_id))
    event_bus.flush()


# ---------------------------------------------------------------------------
# IdleState
# ---------------------------------------------------------------------------

class TestIdleState:
    def test_gesture_candidate_above_threshold(self):
        ctx = _make_context()
        ctx["gesture_model"].predict.return_value = {"confidence": 0.9, "gesture": "palm"}
        state = IdleState(ctx)
        state.on_enter(ctx)

        published = []
        ctx["event_bus"].subscribe(EventType.GESTURE_CANDIDATE, published.append)

        trigger = state.on_frame("frame_data", ctx)
        ctx["event_bus"].flush()

        assert trigger == "gesture_candidate"
        assert len(published) == 1
        assert published[0].confidence == 0.9

    def test_near_miss_between_thresholds(self):
        ctx = _make_context(config=PipelineConfig(
            gesture=GestureConfig(confidence_threshold=0.85, near_miss_threshold=0.6)
        ))
        ctx["gesture_model"].predict.return_value = {"confidence": 0.7, "gesture": "palm"}
        state = IdleState(ctx)
        state.on_enter(ctx)

        published = []
        ctx["event_bus"].subscribe(EventType.GESTURE_NEAR_MISS, published.append)

        trigger = state.on_frame("frame_data", ctx)
        ctx["event_bus"].flush()

        assert trigger is None
        assert len(published) == 1
        assert isinstance(published[0], GestureNearMiss)

    def test_below_near_miss_no_event(self):
        ctx = _make_context()
        ctx["gesture_model"].predict.return_value = {"confidence": 0.1, "gesture": "none"}
        state = IdleState(ctx)
        state.on_enter(ctx)

        published = []
        ctx["event_bus"].subscribe(EventType.GESTURE_CANDIDATE, published.append)
        ctx["event_bus"].subscribe(EventType.GESTURE_NEAR_MISS, published.append)

        trigger = state.on_frame("frame_data", ctx)

        assert trigger is None
        assert len(published) == 0

    def test_on_enter_unloads_other_models(self):
        ctx = _make_context()
        state = IdleState(ctx)
        state.on_enter(ctx)

        ctx["gesture_model"].load.assert_called_once()
        ctx["face_model"].unload.assert_called_once()
        ctx["fire_model"].unload.assert_called_once()
        ctx["injury_model"].unload.assert_called_once()
        ctx["activity_model"].unload.assert_called_once()


# ---------------------------------------------------------------------------
# VerifyingGestureState
# ---------------------------------------------------------------------------

class TestVerifyingGestureState:
    def test_correct_sequence_triggers_confirmed(self):
        ctx = _make_context()
        state = VerifyingGestureState(ctx)
        state.on_enter(ctx)

        published = []
        ctx["event_bus"].subscribe(EventType.GESTURE_CONFIRMED, published.append)

        gestures = ["palm", "fist", "palm", "fist"]
        trigger = None
        for g in gestures:
            ctx["gesture_model"].predict.return_value = {"gesture": g}
            trigger = state.on_frame("frame", ctx)
        ctx["event_bus"].flush()

        assert trigger == "gesture_confirmed"
        assert len(published) == 1
        assert isinstance(published[0], GestureConfirmed)

    def test_timeout_triggers_rejected(self):
        ctx = _make_context(config=PipelineConfig(
            gesture=GestureConfig(sequence_timeout_seconds=0.01)
        ))
        state = VerifyingGestureState(ctx)
        state.on_enter(ctx)

        published = []
        ctx["event_bus"].subscribe(EventType.GESTURE_REJECTED, published.append)

        time.sleep(0.02)
        ctx["gesture_model"].predict.return_value = {"gesture": "palm"}
        trigger = state.on_frame("frame", ctx)
        ctx["event_bus"].flush()

        assert trigger == "gesture_rejected"
        assert len(published) == 1
        assert isinstance(published[0], GestureRejected)

    def test_on_exit_clears_buffer(self):
        ctx = _make_context()
        state = VerifyingGestureState(ctx)
        state.on_enter(ctx)
        ctx["gesture_model"].predict.return_value = {"gesture": "palm"}
        state.on_frame("frame", ctx)
        assert len(state._rolling_buffer) == 1

        state.on_exit(ctx)
        assert len(state._rolling_buffer) == 0


# ---------------------------------------------------------------------------
# VerifyingIdentityState
# ---------------------------------------------------------------------------

class TestVerifyingIdentityState:
    def test_auth_success(self):
        ctx = _make_context()
        ctx["face_model"].predict.return_value = {"authorized": True, "is_live": True}
        state = VerifyingIdentityState(ctx)
        state.on_enter(ctx)

        published = []
        ctx["event_bus"].subscribe(EventType.AUTH_SUCCESS, published.append)

        trigger = state.on_frame("frame", ctx)
        ctx["event_bus"].flush()

        assert trigger == "auth_success"
        assert len(published) == 1
        assert isinstance(published[0], AuthSuccess)

    def test_max_attempts_exceeded(self):
        ctx = _make_context(config=PipelineConfig(
            face_auth=FaceAuthConfig(max_attempts=3)
        ))
        ctx["face_model"].predict.return_value = {"authorized": False, "is_live": False}
        state = VerifyingIdentityState(ctx)
        state.on_enter(ctx)

        published = []
        ctx["event_bus"].subscribe(EventType.AUTH_FAILED, published.append)

        trigger = None
        for _ in range(3):
            trigger = state.on_frame("frame", ctx)
        ctx["event_bus"].flush()

        assert trigger == "auth_failed"
        assert isinstance(published[-1], AuthFailed)
        assert published[-1].reason == "max_attempts_exceeded"

    def test_timeout(self):
        ctx = _make_context(config=PipelineConfig(
            face_auth=FaceAuthConfig(attempt_timeout_seconds=0.01, max_attempts=100)
        ))
        ctx["face_model"].predict.return_value = {"authorized": False, "is_live": False}
        state = VerifyingIdentityState(ctx)
        state.on_enter(ctx)

        published = []
        ctx["event_bus"].subscribe(EventType.AUTH_FAILED, published.append)

        time.sleep(0.02)
        trigger = state.on_frame("frame", ctx)
        ctx["event_bus"].flush()

        assert trigger == "auth_failed"
        assert published[-1].reason == "timeout"

    def test_on_enter_loads_face_model(self):
        ctx = _make_context()
        state = VerifyingIdentityState(ctx)
        state.on_enter(ctx)
        ctx["face_model"].load.assert_called_once()

    def test_on_exit_unloads_face_model_when_not_going_to_active(self):
        ctx = _make_context()
        state = VerifyingIdentityState(ctx)
        ctx["next_state"] = State.IDLE
        state.on_exit(ctx)
        ctx["face_model"].unload.assert_called_once()


# ---------------------------------------------------------------------------
# ActiveDetectionState
# ---------------------------------------------------------------------------

class TestActiveDetectionState:
    def test_runs_models_in_parallel(self):
        call_threads = []
        barrier = threading.Barrier(3, timeout=2)

        def _record_thread(frame):
            call_threads.append(threading.current_thread().ident)
            barrier.wait()
            return {"detected": False}

        ctx = _make_context(config=PipelineConfig(
            detection=DetectionConfig(active_timeout_seconds=100.0)
        ))
        for name in ("fire_model", "injury_model", "activity_model"):
            ctx[name].predict.side_effect = _record_thread

        state = ActiveDetectionState(ctx)
        state.on_enter(ctx)
        state.on_frame("frame", ctx)

        assert len(call_threads) == 3
        assert len(set(call_threads)) >= 2

    def test_publishes_detection_events(self):
        ctx = _make_context(config=PipelineConfig(
            detection=DetectionConfig(active_timeout_seconds=100.0)
        ))
        ctx["fire_model"].predict.return_value = {"detected": True, "location": "zone_a"}
        ctx["injury_model"].predict.return_value = {"detected": False}
        ctx["activity_model"].predict.return_value = {"detected": True, "type": "loitering"}

        state = ActiveDetectionState(ctx)
        state.on_enter(ctx)

        fire_events = []
        activity_events = []
        ctx["event_bus"].subscribe(EventType.FIRE_DETECTED, fire_events.append)
        ctx["event_bus"].subscribe(EventType.SUSPICIOUS_ACTIVITY, activity_events.append)

        state.on_frame("frame", ctx)
        ctx["event_bus"].flush()

        assert len(fire_events) == 1
        assert isinstance(fire_events[0], FireDetected)
        assert len(activity_events) == 1
        assert isinstance(activity_events[0], SuspiciousActivity)

    def test_timeout_triggers_detection_complete(self):
        ctx = _make_context(config=PipelineConfig(
            detection=DetectionConfig(active_timeout_seconds=0.01)
        ))
        ctx["fire_model"].predict.return_value = {"detected": False}
        ctx["injury_model"].predict.return_value = {"detected": False}
        ctx["activity_model"].predict.return_value = {"detected": False}

        state = ActiveDetectionState(ctx)
        state.on_enter(ctx)

        published = []
        ctx["event_bus"].subscribe(EventType.DETECTION_COMPLETE, published.append)

        time.sleep(0.02)
        trigger = state.on_frame("frame", ctx)
        ctx["event_bus"].flush()

        assert trigger == "detection_complete"
        assert len(published) == 1
        assert isinstance(published[0], DetectionComplete)

    def test_on_exit_unloads_models(self):
        ctx = _make_context()
        state = ActiveDetectionState(ctx)
        state.on_exit(ctx)

        ctx["fire_model"].unload.assert_called_once()
        ctx["injury_model"].unload.assert_called_once()
        ctx["activity_model"].unload.assert_called_once()
        ctx["face_model"].unload.assert_called_once()


# ---------------------------------------------------------------------------
# CooldownState
# ---------------------------------------------------------------------------

class TestCooldownState:
    def test_timer_fires_trigger(self):
        ctx = _make_context(config=PipelineConfig(
            detection=DetectionConfig(cooldown_seconds=0.05)
        ))
        callback = MagicMock()
        ctx["trigger_callback"] = callback

        state = CooldownState(ctx)
        state.on_enter(ctx)

        time.sleep(0.15)

        callback.assert_called_once_with("cooldown_expired")

    def test_on_frame_returns_none(self):
        ctx = _make_context()
        state = CooldownState(ctx)
        assert state.on_frame("frame", ctx) is None

    def test_on_exit_cancels_timer(self):
        ctx = _make_context(config=PipelineConfig(
            detection=DetectionConfig(cooldown_seconds=10.0)
        ))
        state = CooldownState(ctx)
        state.on_enter(ctx)

        assert state._timer is not None
        state.on_exit(ctx)
        assert state._timer is None


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_full_cycle_idle_to_cooldown(self):
        event_bus = EventBus()
        sm = StateMachine(active_detection_timeout=100.0, cooldown_duration=100.0)
        config = PipelineConfig(
            gesture=GestureConfig(confidence_threshold=0.8, near_miss_threshold=0.5),
            face_auth=FaceAuthConfig(max_attempts=5, attempt_timeout_seconds=10.0),
            detection=DetectionConfig(active_timeout_seconds=0.01, cooldown_seconds=0.05),
        )

        gesture_model = _make_model(confidence=0.0, gesture="none")
        face_model = _make_model(authorized=False, is_live=False)
        fire_model = _make_model(detected=False)
        injury_model = _make_model(detected=False)
        activity_model = _make_model(detected=False)

        pipeline = Pipeline(
            event_bus=event_bus,
            state_machine=sm,
            config=config,
            gesture_model=gesture_model,
            face_model=face_model,
            fire_model=fire_model,
            injury_model=injury_model,
            activity_model=activity_model,
        )

        assert sm.state == State.IDLE
        gesture_model.load.assert_called()

        # Step 1: Gesture candidate detected
        gesture_model.predict.return_value = {"confidence": 0.9, "gesture": "palm"}
        _publish_frame(event_bus, "f1")
        assert sm.state == State.VERIFYING_GESTURE

        # Step 2: Complete gesture sequence
        for g in ["palm", "fist", "palm", "fist"]:
            gesture_model.predict.return_value = {"gesture": g}
            _publish_frame(event_bus)
        assert sm.state == State.VERIFYING_IDENTITY

        # Step 3: Auth success
        face_model.predict.return_value = {"authorized": True, "is_live": True}
        _publish_frame(event_bus)
        assert sm.state == State.ACTIVE_DETECTION
        fire_model.load.assert_called()
        injury_model.load.assert_called()
        activity_model.load.assert_called()

        # Step 4: Detection timeout
        time.sleep(0.02)
        _publish_frame(event_bus)
        assert sm.state == State.COOLDOWN

        # Step 5: Cooldown expires → back to IDLE
        time.sleep(0.15)
        assert sm.state == State.IDLE

    def test_model_load_unload_lifecycle(self):
        event_bus = EventBus()
        sm = StateMachine(active_detection_timeout=100.0, cooldown_duration=100.0)
        config = PipelineConfig(
            gesture=GestureConfig(confidence_threshold=0.8),
            face_auth=FaceAuthConfig(max_attempts=1),
        )

        gesture_model = _make_model(confidence=0.0, gesture="none")
        face_model = _make_model(authorized=False, is_live=False)
        fire_model = _make_model(detected=False)
        injury_model = _make_model(detected=False)
        activity_model = _make_model(detected=False)

        Pipeline(
            event_bus=event_bus,
            state_machine=sm,
            config=config,
            gesture_model=gesture_model,
            face_model=face_model,
            fire_model=fire_model,
            injury_model=injury_model,
            activity_model=activity_model,
        )

        # Trigger gesture → verifying gesture
        gesture_model.predict.return_value = {"confidence": 0.9, "gesture": "palm"}
        _publish_frame(event_bus)
        assert sm.state == State.VERIFYING_GESTURE

        # Complete sequence → verifying identity
        for g in ["palm", "fist", "palm", "fist"]:
            gesture_model.predict.return_value = {"gesture": g}
            _publish_frame(event_bus)
        assert sm.state == State.VERIFYING_IDENTITY
        face_model.load.assert_called()

        # Auth fails → IDLE, face_model unloaded
        face_model.predict.return_value = {"authorized": False, "is_live": False}
        _publish_frame(event_bus)
        assert sm.state == State.IDLE
        face_model.unload.assert_called()
