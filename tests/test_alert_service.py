"""Tests for AlertService — channels, rate limiting, lifecycle."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.config import AlertConfig
from core.event_bus import EventBus, EventType
from core.events import (
    AlertDispatched,
    FireDetected,
    InjuryDetected,
    SuspiciousActivity,
)
from services.alert_service import AlertService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus():
    b = EventBus()
    yield b


@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "alerts.log")


@pytest.fixture
def dry_config(log_path):
    return AlertConfig(dry_run=True, alerts_log_path=log_path, rate_limit_seconds=0.0)


@pytest.fixture
def live_config(log_path):
    return AlertConfig(dry_run=False, alerts_log_path=log_path, rate_limit_seconds=0.0)


@pytest.fixture
def alert_service(bus, dry_config):
    svc = AlertService(bus, dry_config)
    svc.start()
    yield svc
    svc.stop()


# ---------------------------------------------------------------------------
# Subscription / lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_subscribes_to_three_events(self, bus, dry_config):
        svc = AlertService(bus, dry_config)
        svc.start()
        assert len(svc._handlers) == 3
        assert EventType.FIRE_DETECTED in svc._handlers
        assert EventType.INJURY_DETECTED in svc._handlers
        assert EventType.SUSPICIOUS_ACTIVITY in svc._handlers
        svc.stop()

    def test_stop_unsubscribes(self, bus, dry_config):
        svc = AlertService(bus, dry_config)
        svc.start()
        svc.stop()
        assert len(svc._handlers) == 0

        dispatched = []
        bus.subscribe(EventType.ALERT_DISPATCHED, dispatched.append)

        bus.publish(FireDetected(result={"severity_level": "hazardous"}))
        bus.flush()

        assert len(dispatched) == 0


# ---------------------------------------------------------------------------
# Console channel
# ---------------------------------------------------------------------------

class TestConsoleChannel:
    def test_dry_run_prints_prefix(self, bus, dry_config, capsys):
        svc = AlertService(bus, dry_config)
        svc.start()

        bus.publish(FireDetected(result={"severity_level": "hazardous"}))
        bus.flush()

        captured = capsys.readouterr()
        assert "[DRY RUN]" in captured.out
        assert "FIRE" in captured.out
        assert "hazardous" in captured.out
        svc.stop()

    def test_live_mode_no_prefix(self, bus, live_config, capsys):
        svc = AlertService(bus, live_config)
        svc.start()

        bus.publish(FireDetected(result={"severity_level": "small"}))
        bus.flush()

        captured = capsys.readouterr()
        assert "[DRY RUN]" not in captured.out
        assert "FIRE" in captured.out
        svc.stop()


# ---------------------------------------------------------------------------
# File channel
# ---------------------------------------------------------------------------

class TestFileChannel:
    def test_writes_jsonl(self, bus, dry_config, log_path):
        svc = AlertService(bus, dry_config)
        svc.start()

        bus.publish(FireDetected(result={"severity_level": "hazardous"}))
        bus.flush()

        svc.stop()

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["alert_type"] == "FIRE"
        assert entry["dry_run"] is True

    def test_live_mode_no_dry_run_key(self, bus, live_config, log_path):
        svc = AlertService(bus, live_config)
        svc.start()

        bus.publish(InjuryDetected(result={"posture_type": "collapsed"}))
        bus.flush()

        svc.stop()

        with open(log_path) as f:
            entry = json.loads(f.readline())
        assert "dry_run" not in entry
        assert entry["alert_type"] == "INJURY"

    def test_multiple_alerts_append(self, bus, dry_config, log_path):
        svc = AlertService(bus, dry_config)
        svc.start()

        bus.publish(FireDetected(result={"severity_level": "small"}))
        bus.publish(InjuryDetected(result={"posture_type": "lying"}))
        bus.publish(SuspiciousActivity(result={"activity_type": "loitering"}))
        bus.flush()

        svc.stop()

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 3
        types = {json.loads(l)["alert_type"] for l in lines}
        assert types == {"FIRE", "INJURY", "SUSPICIOUS_ACTIVITY"}


# ---------------------------------------------------------------------------
# Webhook channel
# ---------------------------------------------------------------------------

class TestWebhookChannel:
    def test_dry_run_skips_webhook(self, bus, log_path):
        config = AlertConfig(
            dry_run=True, alerts_log_path=log_path,
            rate_limit_seconds=0.0, webhook_url="http://example.com/hook",
        )
        svc = AlertService(bus, config)
        svc.start()

        with patch.object(svc, "_post_webhook") as mock_post:
            bus.publish(FireDetected(result={}))
            bus.flush()
            time.sleep(0.05)
            mock_post.assert_not_called()

        svc.stop()

    def test_live_mode_calls_webhook(self, bus, log_path):
        config = AlertConfig(
            dry_run=False, alerts_log_path=log_path,
            rate_limit_seconds=0.0, webhook_url="http://example.com/hook",
        )
        svc = AlertService(bus, config)
        svc.start()

        with patch.object(svc, "_post_webhook") as mock_post:
            bus.publish(FireDetected(result={"severity_level": "hazardous"}))
            bus.flush()
            time.sleep(0.1)
            mock_post.assert_called_once()
            args = mock_post.call_args[0]
            assert args[0] == "http://example.com/hook"

        svc.stop()

    def test_no_url_skips_webhook(self, bus, live_config):
        svc = AlertService(bus, live_config)
        svc.start()

        with patch.object(svc, "_post_webhook") as mock_post:
            bus.publish(FireDetected(result={}))
            bus.flush()
            time.sleep(0.05)
            mock_post.assert_not_called()

        svc.stop()


# ---------------------------------------------------------------------------
# SMS channel
# ---------------------------------------------------------------------------

class TestSmsChannel:
    def test_dry_run_skips_sms(self, bus, log_path):
        config = AlertConfig(
            dry_run=True, alerts_log_path=log_path,
            rate_limit_seconds=0.0,
            twilio_account_sid="AC_test", twilio_auth_token="token",
            twilio_from_number="+1111", twilio_to_number="+2222",
        )
        svc = AlertService(bus, config)
        svc.start()

        with patch.object(svc, "_channel_sms") as mock_sms:
            bus.publish(FireDetected(result={}))
            bus.flush()
            mock_sms.assert_not_called()

        svc.stop()

    def test_twilio_import_failure_graceful(self, bus, log_path):
        config = AlertConfig(
            dry_run=False, alerts_log_path=log_path,
            rate_limit_seconds=0.0,
            twilio_account_sid="AC_test", twilio_auth_token="token",
        )
        with patch.dict("sys.modules", {"twilio": None, "twilio.rest": None}):
            svc = AlertService(bus, config)
        assert svc._twilio_client is None
        svc.start()
        svc.stop()

    def test_sms_called_in_live_mode(self, bus, log_path):
        config = AlertConfig(
            dry_run=False, alerts_log_path=log_path,
            rate_limit_seconds=0.0,
        )
        svc = AlertService(bus, config)
        mock_client = MagicMock()
        svc._twilio_client = mock_client
        svc.start()

        bus.publish(FireDetected(result={"severity_level": "hazardous"}))
        bus.flush()

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        assert "FIRE" in call_kwargs.kwargs.get("body", call_kwargs[1].get("body", ""))
        svc.stop()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_rate_limits_same_type(self, bus, log_path):
        config = AlertConfig(
            dry_run=True, alerts_log_path=log_path,
            rate_limit_seconds=10.0,
        )
        svc = AlertService(bus, config)
        svc.start()

        bus.publish(FireDetected(result={"severity_level": "small"}))
        bus.publish(FireDetected(result={"severity_level": "hazardous"}))
        bus.flush()

        svc.stop()

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

    def test_different_types_not_rate_limited(self, bus, log_path):
        config = AlertConfig(
            dry_run=True, alerts_log_path=log_path,
            rate_limit_seconds=10.0,
        )
        svc = AlertService(bus, config)
        svc.start()

        bus.publish(FireDetected(result={}))
        bus.publish(InjuryDetected(result={}))
        bus.publish(SuspiciousActivity(result={}))
        bus.flush()

        svc.stop()

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_rate_limit_window_expires(self, bus, log_path):
        config = AlertConfig(
            dry_run=True, alerts_log_path=log_path,
            rate_limit_seconds=0.1,
        )
        svc = AlertService(bus, config)
        svc.start()

        bus.publish(FireDetected(result={"severity_level": "small"}))
        bus.flush()
        time.sleep(0.15)
        bus.publish(FireDetected(result={"severity_level": "hazardous"}))
        bus.flush()

        svc.stop()

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Payload extraction
# ---------------------------------------------------------------------------

class TestPayloadExtraction:
    def test_fire_extracts_severity(self, bus, dry_config, log_path):
        svc = AlertService(bus, dry_config)
        svc.start()

        bus.publish(FireDetected(result={"severity_level": "uncontrollable", "confidence": 0.9}))
        bus.flush()
        svc.stop()

        with open(log_path) as f:
            entry = json.loads(f.readline())
        assert "uncontrollable" in entry["summary"]
        assert entry["details"]["severity_level"] == "uncontrollable"

    def test_injury_extracts_posture(self, bus, dry_config, log_path):
        svc = AlertService(bus, dry_config)
        svc.start()

        bus.publish(InjuryDetected(result={"posture_type": "collapsed", "confidence": 0.8}))
        bus.flush()
        svc.stop()

        with open(log_path) as f:
            entry = json.loads(f.readline())
        assert "collapsed" in entry["summary"]

    def test_suspicious_extracts_activity(self, bus, dry_config, log_path):
        svc = AlertService(bus, dry_config)
        svc.start()

        bus.publish(SuspiciousActivity(result={"activity_type": "loitering"}))
        bus.flush()
        svc.stop()

        with open(log_path) as f:
            entry = json.loads(f.readline())
        assert "loitering" in entry["summary"]


# ---------------------------------------------------------------------------
# AlertDispatched event
# ---------------------------------------------------------------------------

class TestAlertDispatched:
    def test_publishes_alert_dispatched(self, bus, dry_config):
        dispatched = []
        bus.subscribe(EventType.ALERT_DISPATCHED, dispatched.append)

        svc = AlertService(bus, dry_config)
        svc.start()

        bus.publish(FireDetected(result={"severity_level": "hazardous"}))
        bus.flush()

        svc.stop()
        bus.flush()

        assert len(dispatched) == 1
        assert isinstance(dispatched[0], AlertDispatched)
        assert dispatched[0].alert_type == "FIRE"

    def test_dispatched_contains_details(self, bus, dry_config):
        dispatched = []
        bus.subscribe(EventType.ALERT_DISPATCHED, dispatched.append)

        svc = AlertService(bus, dry_config)
        svc.start()

        bus.publish(InjuryDetected(result={"posture_type": "lying", "confidence": 0.7}))
        bus.flush()

        svc.stop()
        bus.flush()

        assert dispatched[0].details["posture_type"] == "lying"
