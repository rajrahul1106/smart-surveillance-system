"""Tests for the FastAPI application — routes and status endpoints."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, PropertyMock

import pytest
from fastapi.testclient import TestClient

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.state_machine import State, StateMachine
from core.config import AppConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_deps():
    event_bus = MagicMock()
    sm = StateMachine()
    storage = MagicMock()
    storage.query_events.return_value = []
    storage.query_transitions.return_value = []
    camera = MagicMock()
    camera.get_status.return_value = {
        "camera_index": 0,
        "is_connected": True,
        "measured_fps": 29.5,
        "total_frame_count": 1200,
    }
    config = AppConfig()

    def _make_model(loaded=False):
        m = MagicMock()
        m.is_loaded = loaded
        return m

    return {
        "event_bus": event_bus,
        "state_machine": sm,
        "storage_service": storage,
        "camera_manager": camera,
        "config": config,
        "gesture_model": _make_model(),
        "face_model": _make_model(),
        "fire_model": _make_model(),
        "injury_model": _make_model(),
        "activity_model": _make_model(),
    }


@pytest.fixture
def client(mock_deps):
    import api.routes as routes_mod
    import api.ws as ws_mod

    routes_mod._deps.clear()
    routes_mod._start_time = 0.0
    ws_mod._deps.clear()
    ws_mod._subscribed = False

    from api.app import create_app
    app = create_app(**mock_deps)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_returns_200(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_json_structure(self, client):
        data = client.get("/api/status").json()
        assert "state" in data
        assert "camera" in data
        assert "models" in data
        assert "uptime_seconds" in data
        assert "config" in data

    def test_state_is_idle(self, client):
        data = client.get("/api/status").json()
        assert data["state"] == "IDLE"

    def test_camera_fields(self, client):
        cam = client.get("/api/status").json()["camera"]
        assert cam["index"] == 0
        assert cam["is_connected"] is True
        assert cam["fps"] == 29.5
        assert cam["frame_count"] == 1200

    def test_models_all_unloaded(self, client):
        models = client.get("/api/status").json()["models"]
        for name in ("gesture", "face", "fire", "injury", "activity"):
            assert name in models
            assert models[name] == "unloaded"

    def test_model_loaded_status(self, client, mock_deps):
        mock_deps["gesture_model"].is_loaded = True
        models = client.get("/api/status").json()["models"]
        assert models["gesture"] == "loaded"
        assert models["face"] == "unloaded"

    def test_config_subset(self, client):
        cfg = client.get("/api/status").json()["config"]
        assert "camera_index" in cfg
        assert "gesture_threshold" in cfg
        assert "detection_timeout" in cfg
        assert "alerts_dry_run" in cfg
        assert "twilio" not in str(cfg).lower()


# ---------------------------------------------------------------------------
# GET /api/events
# ---------------------------------------------------------------------------

class TestGetEvents:
    def test_returns_200_with_list(self, client):
        resp = client.get("/api/events")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_type_filter_passed_through(self, client, mock_deps):
        mock_deps["storage_service"].query_events.return_value = [
            {"event_type": "FIRE_DETECTED", "timestamp": 1.0}
        ]
        resp = client.get("/api/events?type=FIRE_DETECTED")
        assert resp.status_code == 200
        mock_deps["storage_service"].query_events.assert_called_with(
            event_type="FIRE_DETECTED", since=None, limit=50,
        )

    def test_since_and_limit(self, client, mock_deps):
        client.get("/api/events?since=1000.0&limit=10")
        mock_deps["storage_service"].query_events.assert_called_with(
            event_type=None, since=1000.0, limit=10,
        )


# ---------------------------------------------------------------------------
# GET /api/transitions
# ---------------------------------------------------------------------------

class TestGetTransitions:
    def test_returns_200_with_list(self, client):
        resp = client.get("/api/transitions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_since_filter(self, client, mock_deps):
        client.get("/api/transitions?since=500.0&limit=5")
        mock_deps["storage_service"].query_transitions.assert_called_with(
            since=500.0, limit=5,
        )


# ---------------------------------------------------------------------------
# POST /api/override
# ---------------------------------------------------------------------------

class TestPostOverride:
    def test_reset_to_idle_success(self, client, mock_deps):
        sm = mock_deps["state_machine"]
        sm.transition("gesture_candidate")
        assert sm.state == State.VERIFYING_GESTURE

        resp = client.post("/api/override", json={"action": "reset_to_idle"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["new_state"] == "IDLE"

    def test_invalid_action_returns_400(self, client):
        resp = client.post("/api/override", json={"action": "launch_missiles"})
        assert resp.status_code == 400

    def test_reset_from_idle_still_succeeds(self, client):
        resp = client.post("/api/override", json={"action": "reset_to_idle"})
        assert resp.status_code == 200
        assert resp.json()["new_state"] == "IDLE"
