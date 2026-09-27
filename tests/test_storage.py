"""Tests for StorageService, AuditLogger, and JSON serialization."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time

import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.event_bus import EventBus, EventType
from core.events import (
    AuthFailed,
    AuthSuccess,
    CameraConnected,
    DetectionComplete,
    FireDetected,
    FrameReady,
    GestureCandidate,
    GestureConfirmed,
    ModelLoaded,
    PresenceUpdated,
    ProcessedFrame,
    StateTransition,
    SystemStartup,
)
from services.storage_service import StorageService, _EventEncoder, _serialise_event
from services.audit_logger import AuditLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_events.db")


@pytest.fixture
def storage(db_path):
    svc = StorageService(db_path=db_path)
    yield svc
    svc.close()


# ---------------------------------------------------------------------------
# Database creation and schema
# ---------------------------------------------------------------------------

class TestDatabaseCreation:
    def test_creates_db_file(self, db_path):
        svc = StorageService(db_path=db_path)
        assert os.path.isfile(db_path)
        svc.close()

    def test_creates_parent_directory(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "test.db")
        svc = StorageService(db_path=nested)
        assert os.path.isfile(nested)
        svc.close()

    def test_events_table_schema(self, storage):
        cursor = storage._conn.execute("PRAGMA table_info(events)")
        cols = {row[1] for row in cursor.fetchall()}
        assert cols == {"id", "event_type", "camera_id", "timestamp", "confidence", "payload", "created_at"}

    def test_state_transitions_table_schema(self, storage):
        cursor = storage._conn.execute("PRAGMA table_info(state_transitions)")
        cols = {row[1] for row in cursor.fetchall()}
        assert cols == {"id", "from_state", "to_state", "trigger", "timestamp", "created_at"}

    def test_wal_journal_mode(self, storage):
        cursor = storage._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"


# ---------------------------------------------------------------------------
# save_event
# ---------------------------------------------------------------------------

class TestSaveEvent:
    def test_save_gesture_candidate(self, storage):
        evt = GestureCandidate(confidence=0.9)
        storage.save_event(evt)
        rows = storage.query_events()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "GESTURE_CANDIDATE"
        assert rows[0]["confidence"] == 0.9

    def test_save_fire_detected(self, storage):
        evt = FireDetected(result={"severity": "hazardous"})
        storage.save_event(evt)
        rows = storage.query_events()
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["result"]["severity"] == "hazardous"

    def test_save_auth_failed(self, storage):
        evt = AuthFailed(reason="timeout")
        storage.save_event(evt)
        rows = storage.query_events()
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["reason"] == "timeout"

    def test_save_camera_connected(self, storage):
        evt = CameraConnected(camera_id="cam0", index=0, resolution=(1920, 1080))
        storage.save_event(evt)
        rows = storage.query_events()
        assert len(rows) == 1
        assert rows[0]["camera_id"] == "cam0"
        payload = json.loads(rows[0]["payload"])
        assert payload["resolution"] == [1920, 1080]

    def test_save_model_loaded(self, storage):
        evt = ModelLoaded(model_name="gesture", load_time_ms=42.5)
        storage.save_event(evt)
        rows = storage.query_events()
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["model_name"] == "gesture"
        assert payload["load_time_ms"] == 42.5

    def test_save_gesture_confirmed_no_confidence(self, storage):
        evt = GestureConfirmed()
        storage.save_event(evt)
        rows = storage.query_events()
        assert len(rows) == 1
        assert rows[0]["confidence"] is None

    def test_save_system_startup(self, storage):
        evt = SystemStartup(config_summary={"fps": 30, "models": ["gesture", "face"]})
        storage.save_event(evt)
        rows = storage.query_events()
        payload = json.loads(rows[0]["payload"])
        assert payload["config_summary"]["fps"] == 30

    def test_timestamp_stored_correctly(self, storage):
        ts = 1700000000.123
        evt = AuthFailed(reason="test", timestamp=ts)
        storage.save_event(evt)
        rows = storage.query_events()
        assert abs(rows[0]["timestamp"] - ts) < 0.001


# ---------------------------------------------------------------------------
# save_transition
# ---------------------------------------------------------------------------

class TestSaveTransition:
    def test_basic_transition(self, storage):
        storage.save_transition("IDLE", "VERIFYING_GESTURE", "gesture_candidate", 1700000000.0)
        rows = storage.query_transitions()
        assert len(rows) == 1
        assert rows[0]["from_state"] == "IDLE"
        assert rows[0]["to_state"] == "VERIFYING_GESTURE"
        assert rows[0]["trigger"] == "gesture_candidate"

    def test_multiple_transitions(self, storage):
        for i, (f, t, tr) in enumerate([
            ("IDLE", "VERIFYING_GESTURE", "gesture_candidate"),
            ("VERIFYING_GESTURE", "VERIFYING_IDENTITY", "gesture_confirmed"),
            ("VERIFYING_IDENTITY", "ACTIVE_DETECTION", "auth_success"),
        ]):
            storage.save_transition(f, t, tr, 1700000000.0 + i)
        rows = storage.query_transitions()
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# query_events
# ---------------------------------------------------------------------------

class TestQueryEvents:
    def test_no_filter_returns_all(self, storage):
        for _ in range(5):
            storage.save_event(GestureConfirmed())
        rows = storage.query_events()
        assert len(rows) == 5

    def test_event_type_filter(self, storage):
        storage.save_event(GestureConfirmed())
        storage.save_event(AuthFailed(reason="test"))
        storage.save_event(GestureConfirmed())

        rows = storage.query_events(event_type="GESTURE_CONFIRMED")
        assert len(rows) == 2
        assert all(r["event_type"] == "GESTURE_CONFIRMED" for r in rows)

    def test_since_filter(self, storage):
        storage.save_event(AuthFailed(reason="old", timestamp=1000.0))
        storage.save_event(AuthFailed(reason="new", timestamp=2000.0))
        storage.save_event(AuthFailed(reason="newest", timestamp=3000.0))

        rows = storage.query_events(since=1500.0)
        assert len(rows) == 2
        timestamps = [r["timestamp"] for r in rows]
        assert all(t >= 1500.0 for t in timestamps)

    def test_limit(self, storage):
        for i in range(10):
            storage.save_event(GestureConfirmed(timestamp=float(i)))
        rows = storage.query_events(limit=3)
        assert len(rows) == 3

    def test_combined_filters(self, storage):
        storage.save_event(AuthFailed(reason="a", timestamp=1000.0))
        storage.save_event(GestureConfirmed(timestamp=2000.0))
        storage.save_event(AuthFailed(reason="b", timestamp=3000.0))

        rows = storage.query_events(event_type="AUTH_FAILED", since=1500.0)
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["reason"] == "b"


# ---------------------------------------------------------------------------
# query_transitions
# ---------------------------------------------------------------------------

class TestQueryTransitions:
    def test_since_filter(self, storage):
        storage.save_transition("A", "B", "t1", 1000.0)
        storage.save_transition("B", "C", "t2", 2000.0)
        rows = storage.query_transitions(since=1500.0)
        assert len(rows) == 1
        assert rows[0]["from_state"] == "B"

    def test_limit(self, storage):
        for i in range(10):
            storage.save_transition("A", "B", "t", float(i))
        rows = storage.query_transitions(limit=4)
        assert len(rows) == 4


# ---------------------------------------------------------------------------
# JSON serialization edge cases
# ---------------------------------------------------------------------------

class TestJsonSerialization:
    def test_numpy_array_excluded(self, storage):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        evt = ProcessedFrame(frame=frame, camera_id="cam0", original_frame=frame)
        storage.save_event(evt)
        rows = storage.query_events()
        payload = json.loads(rows[0]["payload"])
        assert "frame" not in payload
        assert "original_frame" not in payload

    def test_numpy_floats_converted(self, storage):
        evt = GestureCandidate(confidence=np.float64(0.85))
        storage.save_event(evt)
        rows = storage.query_events()
        payload = json.loads(rows[0]["payload"])
        assert isinstance(payload["confidence"], float)
        assert abs(payload["confidence"] - 0.85) < 1e-6

    def test_numpy_ints_converted(self):
        data = {"count": np.int64(42)}
        result = json.dumps(data, cls=_EventEncoder)
        assert json.loads(result)["count"] == 42

    def test_tuple_converted_to_list(self):
        data = {"resolution": (1920, 1080)}
        result = json.dumps(data, cls=_EventEncoder)
        assert json.loads(result)["resolution"] == [1920, 1080]

    def test_serialise_event_drops_frame_fields(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        evt = FrameReady(frame=frame, camera_id="cam0")
        payload = _serialise_event(evt)
        assert "frame" not in payload
        assert "camera_id" in payload

    def test_landmarks_with_numpy_floats(self, storage):
        landmarks = [(np.float32(0.5), np.float32(0.6), np.float32(0.0))] * 21
        result = {"landmarks": landmarks}
        evt = GestureCandidate(confidence=0.9)
        storage.save_event(evt)
        # Verify the encoder handles it when used directly
        dumped = json.dumps(result, cls=_EventEncoder)
        loaded = json.loads(dumped)
        assert len(loaded["landmarks"]) == 21
        assert isinstance(loaded["landmarks"][0][0], float)

    def test_detection_complete_with_nested_dicts(self, storage):
        results = [
            {"model": "fire", "result": {"detected": True, "confidence": np.float64(0.95)}},
            {"model": "injury", "result": {"detected": False}},
        ]
        evt = DetectionComplete(results=results)
        storage.save_event(evt)
        rows = storage.query_events()
        payload = json.loads(rows[0]["payload"])
        assert len(payload["results"]) == 2
        assert payload["results"][0]["result"]["confidence"] == 0.95


# ---------------------------------------------------------------------------
# Concurrent writes
# ---------------------------------------------------------------------------

class TestConcurrentWrites:
    def test_concurrent_writes_no_corruption(self, storage):
        errors = []

        def writer(n: int):
            try:
                for i in range(20):
                    storage.save_event(AuthFailed(reason=f"thread-{n}-{i}", timestamp=float(i)))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        rows = storage.query_events(limit=200)
        assert len(rows) == 100  # 5 threads × 20 events

    def test_concurrent_transitions_no_corruption(self, storage):
        def writer(n: int):
            for i in range(10):
                storage.save_transition(f"S{n}", f"S{n+1}", f"t{i}", float(i))

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rows = storage.query_transitions(limit=200)
        assert len(rows) == 50


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

class TestConnectionLifecycle:
    def test_close_releases_connection(self, db_path):
        svc = StorageService(db_path=db_path)
        svc.save_event(GestureConfirmed())
        svc.close()

        with pytest.raises(Exception):
            svc.save_event(GestureConfirmed())

    def test_can_reopen_after_close(self, db_path):
        svc = StorageService(db_path=db_path)
        svc.save_event(GestureConfirmed())
        svc.close()

        svc2 = StorageService(db_path=db_path)
        rows = svc2.query_events()
        assert len(rows) == 1
        svc2.close()


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class TestAuditLogger:
    def test_logs_events_to_storage(self, storage):
        bus = EventBus()
        al = AuditLogger(bus, storage)
        al.start()

        bus.publish(GestureConfirmed())
        bus.publish(AuthFailed(reason="denied"))
        bus.flush()

        al.stop()

        rows = storage.query_events()
        assert len(rows) == 2
        types = {r["event_type"] for r in rows}
        assert "GESTURE_CONFIRMED" in types
        assert "AUTH_FAILED" in types

    def test_logs_state_transition_to_both_tables(self, storage):
        bus = EventBus()
        al = AuditLogger(bus, storage)
        al.start()

        bus.publish(StateTransition(from_state="IDLE", to_state="VERIFYING_GESTURE"))
        bus.flush()

        al.stop()

        events = storage.query_events(event_type="STATE_TRANSITION")
        assert len(events) == 1

        transitions = storage.query_transitions()
        assert len(transitions) == 1
        assert transitions[0]["from_state"] == "IDLE"

    def test_logs_presence_updated(self, storage):
        bus = EventBus()
        al = AuditLogger(bus, storage)
        al.start()

        bus.publish(PresenceUpdated(
            authorized=["Rahul Raj"], unknown_count=1, uncertain_count=0, total=2,
        ))
        bus.flush()

        al.stop()

        rows = storage.query_events(event_type="PRESENCE_UPDATED")
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["authorized"] == ["Rahul Raj"]
        assert payload["unknown_count"] == 1
        assert payload["uncertain_count"] == 0
        assert payload["total"] == 2

    def test_stop_unsubscribes(self, storage):
        bus = EventBus()
        al = AuditLogger(bus, storage)
        al.start()
        al.stop()

        bus.publish(GestureConfirmed())
        bus.flush()

        rows = storage.query_events()
        assert len(rows) == 0

    def test_save_failure_does_not_crash(self, storage):
        bus = EventBus()
        al = AuditLogger(bus, storage)
        al.start()

        storage.close()

        bus.publish(GestureConfirmed())
        bus.flush()

        al.stop()
