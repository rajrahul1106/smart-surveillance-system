from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from core.config import AlertConfig
from core.event_bus import EventBus, EventType
from core.events import (
    AlertDispatched,
    FireDetected,
    InjuryDetected,
    SuspiciousActivity,
)

logger = logging.getLogger(__name__)


class AlertService:

    def __init__(self, event_bus: EventBus, config: Optional[AlertConfig] = None) -> None:
        self._event_bus = event_bus
        self._config = config or AlertConfig()
        self._handlers: Dict[EventType, Callable] = {}
        self._last_dispatch: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._twilio_client: Any = None

        parent = os.path.dirname(self._config.alerts_log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if not self._config.dry_run:
            self._init_twilio()

    def _init_twilio(self) -> None:
        if not self._config.twilio_account_sid:
            return
        try:
            from twilio.rest import Client
            self._twilio_client = Client(
                self._config.twilio_account_sid,
                self._config.twilio_auth_token,
            )
        except ImportError:
            logger.warning("twilio package not installed — SMS alerts disabled")
        except Exception:
            logger.warning("Failed to initialise Twilio client", exc_info=True)

    def start(self) -> None:
        subscriptions = [
            (EventType.FIRE_DETECTED, self._on_fire),
            (EventType.INJURY_DETECTED, self._on_injury),
            (EventType.SUSPICIOUS_ACTIVITY, self._on_suspicious),
        ]
        for et, handler in subscriptions:
            self._handlers[et] = handler
            self._event_bus.subscribe(et, handler)
        logger.info("AlertService started — dry_run=%s", self._config.dry_run)

    def stop(self) -> None:
        for et, handler in self._handlers.items():
            self._event_bus.unsubscribe(et, handler)
        self._handlers.clear()
        logger.info("AlertService stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_fire(self, event: FireDetected) -> None:
        result = event.result or {}
        severity = result.get("severity_level", "unknown")
        self._dispatch(
            alert_type="FIRE",
            summary=f"Fire detected — severity: {severity}",
            details=result,
        )

    def _on_injury(self, event: InjuryDetected) -> None:
        result = event.result or {}
        posture = result.get("posture_type", "unknown")
        self._dispatch(
            alert_type="INJURY",
            summary=f"Possible injury — posture: {posture}",
            details=result,
        )

    def _on_suspicious(self, event: SuspiciousActivity) -> None:
        result = event.result or {}
        activity = result.get("activity_type", "unknown")
        self._dispatch(
            alert_type="SUSPICIOUS_ACTIVITY",
            summary=f"Suspicious activity — type: {activity}",
            details=result,
        )

    # ------------------------------------------------------------------
    # Core dispatch with rate limiting
    # ------------------------------------------------------------------

    def _dispatch(self, alert_type: str, summary: str, details: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            last = self._last_dispatch.get(alert_type, 0.0)
            if now - last < self._config.rate_limit_seconds:
                logger.debug("Rate-limited alert %s (%.1fs remaining)", alert_type,
                             self._config.rate_limit_seconds - (now - last))
                return
            self._last_dispatch[alert_type] = now

        payload = {
            "alert_type": alert_type,
            "summary": summary,
            "details": details,
            "timestamp": now,
        }

        if self._config.dry_run:
            self._channel_console(payload, dry_run=True)
            self._channel_file(payload, dry_run=True)
        else:
            self._channel_console(payload)
            self._channel_file(payload)
            self._channel_webhook(payload)
            self._channel_sms(payload)

        self._event_bus.publish(AlertDispatched(
            alert_type=alert_type,
            details=details,
        ))

    # ------------------------------------------------------------------
    # Alert channels
    # ------------------------------------------------------------------

    def _channel_console(self, payload: Dict[str, Any], *, dry_run: bool = False) -> None:
        prefix = "[DRY RUN] " if dry_run else ""
        print(f"{prefix}ALERT [{payload['alert_type']}]: {payload['summary']}")

    def _channel_file(self, payload: Dict[str, Any], *, dry_run: bool = False) -> None:
        try:
            entry = dict(payload)
            if dry_run:
                entry["dry_run"] = True
            with open(self._config.alerts_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.warning("Failed to write alert to file", exc_info=True)

    def _channel_webhook(self, payload: Dict[str, Any]) -> None:
        url = self._config.webhook_url
        if not url:
            return
        threading.Thread(target=self._post_webhook, args=(url, payload), daemon=True).start()

    def _post_webhook(self, url: str, payload: Dict[str, Any]) -> None:
        try:
            data = json.dumps(payload).encode()
            req = Request(url, data=data, headers={"Content-Type": "application/json"})
            urlopen(req, timeout=5)
        except Exception:
            logger.warning("Webhook POST to %s failed", url, exc_info=True)

    def _channel_sms(self, payload: Dict[str, Any]) -> None:
        if self._twilio_client is None:
            return
        try:
            self._twilio_client.messages.create(
                body=f"ALERT [{payload['alert_type']}]: {payload['summary']}",
                from_=self._config.twilio_from_number,
                to=self._config.twilio_to_number,
            )
        except Exception:
            logger.warning("Twilio SMS failed", exc_info=True)
