"""Tests for the state-driven pipeline architecture."""

from __future__ import annotations

import sys
import os
import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
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
    PresenceUpdated,
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
from models.face_model import FaceModel


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


_ENROLLED = np.ones(512, dtype=np.float32) / np.sqrt(512)
# Orthogonal to _ENROLLED (cosine similarity 0): a stranger.
_STRANGER = np.array([1.0, -1.0] * 256, dtype=np.float32) / np.sqrt(512)


def _insightface_modules(faces):
    """sys.modules entries that make FaceModel.load() use a fake insightface app."""
    app = MagicMock()
    app.get.return_value = list(faces)
    package = MagicMock()
    package.app.FaceAnalysis.return_value = app
    return {"insightface": package, "insightface.app": package.app}


def _insightface_face(bbox, embedding):
    face = MagicMock()
    face.bbox = np.asarray(bbox, dtype=np.float32)
    face.normed_embedding = np.asarray(embedding, dtype=np.float32)
    return face


def _tracked_face(track_id=1, status="UNKNOWN", identity=None, confidence=5.0, auth_streak=0):
    """One entry of FaceModel's ``faces`` list."""
    return {
        "track_id": track_id, "bbox": (10 + 60 * track_id, 10, 50, 50),
        "status": status, "identity": identity,
        "confidence": confidence, "auth_streak": auth_streak,
    }


def _drain_face_worker(state):
    """Wait for the presence run the last on_frame started (it runs on a worker thread)."""
    if state._face_future is not None:
        state._face_future.result(timeout=5)


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

    def test_skips_frames_until_gesture_model_is_loaded(self):
        # COOLDOWN -> IDLE runs on a timer thread, so a frame can arrive
        # while IdleState.on_enter is still loading the gesture model.
        ctx = _make_context()
        ctx["gesture_model"].is_loaded = False
        state = IdleState(ctx)

        assert state.on_frame("frame_data", ctx) is None
        ctx["gesture_model"].predict.assert_not_called()

    def test_on_enter_clears_presence_session(self):
        ctx = _make_context()
        ctx["last_presence"] = (("Rahul Raj",), 0, 0, 1)
        ctx["last_detections"] = {
            "faces": [_tracked_face()], "presence": {"total": 1},
            "authenticated_user": "Rahul Raj", "authenticated_users": ["Rahul Raj"],
        }
        IdleState(ctx).on_enter(ctx)

        assert "last_presence" not in ctx
        for key in ("faces", "presence", "authenticated_user", "authenticated_users"):
            assert key not in ctx["last_detections"]


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

    def test_on_enter_uses_verify_mode(self):
        ctx = _make_context()
        VerifyingIdentityState(ctx).on_enter(ctx)
        ctx["face_model"].set_mode.assert_called_once_with("verify")

    def test_auth_succeeds_with_enrolled_and_unknown_face_in_frame(self):
        faces = [
            _insightface_face((200, 150, 300, 270), _ENROLLED),  # the gesture-doer
            _insightface_face((20, 150, 120, 270), _STRANGER),   # a bystander
        ]
        face_model = FaceModel(encodings_path="/nonexistent/enc.pkl")
        ctx = _make_context(face_model=face_model)
        state = VerifyingIdentityState(ctx)
        with patch.dict("sys.modules", _insightface_modules(faces)):
            state.on_enter(ctx)
        face_model.add_encoding("Rahul Raj", _ENROLLED)

        published = []
        ctx["event_bus"].subscribe(EventType.AUTH_SUCCESS, published.append)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frames 1-2 build the AUTHORIZED streak; they aren't failed attempts.
        triggers = [state.on_frame(frame, ctx) for _ in range(3)]
        ctx["event_bus"].flush()

        assert triggers == [None, None, "auth_success"]
        assert len(published) == 1
        assert published[0].result["user_id"] == "Rahul Raj"
        assert published[0].result["authenticated_users"] == ["Rahul Raj"]
        session = ctx["last_detections"]
        assert session["face_authorized"] is True
        assert session["authenticated_user"] == "Rahul Raj"
        assert sorted(f["status"] for f in session["faces"]) == ["AUTHORIZED", "UNCERTAIN"]
        face_model.unload()

    def test_confirmation_frames_do_not_use_up_attempts(self):
        ctx = _make_context(config=PipelineConfig(face_auth=FaceAuthConfig(max_attempts=1)))
        state = VerifyingIdentityState(ctx)
        state.on_enter(ctx)

        triggers = []
        for streak in (1, 2, 3):
            face = _tracked_face(status="AUTHORIZED", identity="Rahul Raj",
                                 confidence=91.0, auth_streak=streak)
            ctx["face_model"].predict.return_value = {
                "is_authorized": True, "is_live": True, "faces": [face],
            }
            triggers.append(state.on_frame("frame", ctx))
        assert triggers == [None, None, "auth_success"]

    def test_primary_user_is_highest_confidence_and_all_names_recorded(self):
        ctx = _make_context()
        state = VerifyingIdentityState(ctx)
        state.on_enter(ctx)
        published = []
        ctx["event_bus"].subscribe(EventType.AUTH_SUCCESS, published.append)

        ctx["face_model"].predict.return_value = {
            "is_authorized": True, "is_live": True,
            "faces": [
                _tracked_face(1, "AUTHORIZED", "Meghna Lal", 74.0, auth_streak=3),
                _tracked_face(2, "AUTHORIZED", "Rahul Raj", 88.0, auth_streak=3),
                _tracked_face(3, "UNKNOWN"),
            ],
        }
        assert state.on_frame("frame", ctx) == "auth_success"
        ctx["event_bus"].flush()

        assert published[0].result["user_id"] == "Rahul Raj"
        assert published[0].result["confidence"] == pytest.approx(0.88)
        assert ctx["last_detections"]["authenticated_user"] == "Rahul Raj"
        assert ctx["last_detections"]["authenticated_users"] == ["Meghna Lal", "Rahul Raj"]

    def test_unknown_faces_count_as_failed_attempts(self):
        ctx = _make_context(config=PipelineConfig(face_auth=FaceAuthConfig(max_attempts=3)))
        ctx["face_model"].predict.return_value = {
            "is_authorized": False, "is_live": True, "faces": [_tracked_face()],
        }
        state = VerifyingIdentityState(ctx)
        state.on_enter(ctx)

        triggers = [state.on_frame("frame", ctx) for _ in range(3)]
        assert triggers == [None, None, "auth_failed"]


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


class TestActiveDetectionPresence:
    @staticmethod
    def _context(face_result=None):
        ctx = _make_context(config=PipelineConfig(
            detection=DetectionConfig(active_timeout_seconds=100.0)
        ))
        if face_result is not None:
            ctx["face_model"].predict.return_value = face_result
        return ctx

    def test_face_model_stays_loaded_in_presence_mode(self):
        ctx = self._context()
        state = ActiveDetectionState(ctx)
        state.on_enter(ctx)

        ctx["face_model"].unload.assert_not_called()
        ctx["face_model"].set_mode.assert_called_once_with("presence")
        ctx["gesture_model"].unload.assert_called_once()
        state.on_exit(ctx)

    def test_face_runs_every_presence_interval_and_result_is_reused(self):
        face = _tracked_face(status="AUTHORIZED", identity="Rahul Raj",
                             confidence=88.0, auth_streak=9)
        presence = {"authorized": ["Rahul Raj"], "unknown_count": 0,
                    "uncertain_count": 0, "total": 1}
        ctx = self._context({"faces": [face], "presence": presence, "frame_size": (640, 480)})
        state = ActiveDetectionState(ctx)
        state.on_enter(ctx)

        for _ in range(25):
            state.on_frame("frame", ctx)
            _drain_face_worker(state)

        # Frame 1 seeds every model, then every 10th frame at offset 3: 3, 13, 23.
        assert ctx["face_model"].predict.call_count == 4
        # Frame 25 didn't run the face model; the last result is reused.
        detections = ctx["last_detections"]
        assert detections["faces"] == [face]
        assert detections["presence"] == presence
        assert detections["faces_frame_size"] == (640, 480)
        state.on_exit(ctx)

    def test_presence_change_published_once(self):
        presence = {"authorized": [], "unknown_count": 1, "uncertain_count": 0, "total": 1}
        ctx = self._context({"faces": [_tracked_face()], "presence": presence})
        state = ActiveDetectionState(ctx)
        state.on_enter(ctx)
        events = []
        ctx["event_bus"].subscribe(EventType.PRESENCE_UPDATED, events.append)

        for _ in range(15):
            state.on_frame("frame", ctx)
            _drain_face_worker(state)
        ctx["event_bus"].flush()

        assert len(events) == 1
        assert events[0].unknown_count == 1
        state.on_exit(ctx)

    def test_on_exit_stops_worker_and_clears_faces(self):
        presence = {"authorized": [], "unknown_count": 1, "uncertain_count": 0, "total": 1}
        ctx = self._context({"faces": [_tracked_face()], "presence": presence})
        state = ActiveDetectionState(ctx)
        state.on_enter(ctx)
        state.on_frame("frame", ctx)
        _drain_face_worker(state)
        state.on_frame("frame", ctx)
        assert ctx["last_detections"]["faces"]

        state.on_exit(ctx)
        assert state._face_executor is None
        assert ctx["last_detections"]["faces"] == []
        assert "presence" not in ctx["last_detections"]
        ctx["face_model"].unload.assert_called_once()


class TestPresenceUpdates:
    def test_published_on_change_only_not_every_frame(self):
        ctx = _make_context(config=PipelineConfig(face_auth=FaceAuthConfig(max_attempts=100)))
        state = VerifyingIdentityState(ctx)
        state.on_enter(ctx)
        events = []
        ctx["event_bus"].subscribe(EventType.PRESENCE_UPDATED, events.append)

        one = {"faces": [_tracked_face(1)],
               "presence": {"authorized": [], "unknown_count": 1, "uncertain_count": 0, "total": 1}}
        two = {"faces": [_tracked_face(1), _tracked_face(2)],
               "presence": {"authorized": [], "unknown_count": 2, "uncertain_count": 0, "total": 2}}
        for result in (one, one, one, two, two, one):
            ctx["face_model"].predict.return_value = result
            state.on_frame("frame", ctx)
        ctx["event_bus"].flush()

        assert all(isinstance(e, PresenceUpdated) for e in events)
        assert [(e.unknown_count, e.total) for e in events] == [(1, 1), (2, 2), (1, 1)]

    def test_same_people_not_reannounced_when_detection_starts(self):
        presence = {"authorized": ["Rahul Raj"], "unknown_count": 0, "uncertain_count": 0, "total": 1}
        face = _tracked_face(status="AUTHORIZED", identity="Rahul Raj",
                             confidence=90.0, auth_streak=3)
        ctx = _make_context(config=PipelineConfig(
            detection=DetectionConfig(active_timeout_seconds=100.0)
        ))
        ctx["face_model"].predict.return_value = {
            "is_authorized": True, "is_live": True, "faces": [face], "presence": presence,
        }
        events = []
        ctx["event_bus"].subscribe(EventType.PRESENCE_UPDATED, events.append)

        verifying = VerifyingIdentityState(ctx)
        verifying.on_enter(ctx)
        assert verifying.on_frame("frame", ctx) == "auth_success"
        active = ActiveDetectionState(ctx)
        active.on_enter(ctx)
        for _ in range(5):
            active.on_frame("frame", ctx)
            _drain_face_worker(active)
        active.on_exit(ctx)
        ctx["event_bus"].flush()

        assert len(events) == 1


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

    def test_face_model_receives_full_resolution_frame(self):
        event_bus = EventBus()
        sm = StateMachine(active_detection_timeout=100.0, cooldown_duration=100.0)
        face_model = _make_model(authorized=False, is_live=False)
        Pipeline(
            event_bus=event_bus,
            state_machine=sm,
            config=PipelineConfig(face_auth=FaceAuthConfig(max_attempts=5)),
            gesture_model=_make_model(confidence=0.0, gesture="none"),
            face_model=face_model,
            fire_model=_make_model(detected=False),
            injury_model=_make_model(detected=False),
            activity_model=_make_model(detected=False),
        )
        sm.transition("gesture_candidate")
        sm.transition("gesture_confirmed")
        assert sm.state == State.VERIFYING_IDENTITY

        event_bus.publish(ProcessedFrame(frame="320x240", original_frame="640x480"))
        event_bus.flush()
        face_model.predict.assert_called_with("640x480")
