from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.storage_service import _EventEncoder

logger = logging.getLogger(__name__)

router = APIRouter()

_connections: Set[WebSocket] = set()
_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_deps: Dict[str, Any] = {}
_subscribed = False


# Only these event types are forwarded to the dashboard. Frame-level events
# (FRAME_READY, PROCESSED_FRAME) are intentionally excluded — they fire many
# times per second and would flood the WebSocket.
_BROADCAST_ALLOWLIST = {
    "GESTURE_CANDIDATE",
    "GESTURE_NEAR_MISS",
    "GESTURE_CONFIRMED",
    "GESTURE_REJECTED",
    "GESTURE_PROGRESS",
    "AUTH_SUCCESS",
    "AUTH_FAILED",
    "PRESENCE_UPDATED",
    "FIRE_DETECTED",
    "INJURY_DETECTED",
    "SUSPICIOUS_ACTIVITY",
    "DETECTION_COMPLETE",
    "ALERT_DISPATCHED",
    "STATE_TRANSITION",
    "CAMERA_CONNECTED",
    "CAMERA_DISCONNECTED",
    "MODEL_LOADED",
    "MODEL_UNLOADED",
    "SYSTEM_STARTUP",
    "SYSTEM_SHUTDOWN",
}


def init_ws(event_bus: Any) -> None:
    _deps["event_bus"] = event_bus


def _subscribe_to_events() -> None:
    global _subscribed
    if _subscribed:
        return
    _subscribed = True

    from core.event_bus import EventType

    event_bus = _deps["event_bus"]
    for et in EventType:
        if et.name not in _BROADCAST_ALLOWLIST:
            continue
        event_bus.subscribe(et, _make_handler(et))


def _make_handler(et: Any) -> Any:
    def handler(event: Any) -> None:
        _broadcast_event(et, event)
    return handler


def _serialize_event(et: Any, event: Any) -> str:
    import dataclasses
    payload: Dict[str, Any] = {"event_type": et.name}
    if dataclasses.is_dataclass(event) and not isinstance(event, type):
        for f in dataclasses.fields(event):
            if f.name in ("frame", "original_frame"):
                continue
            val = getattr(event, f.name)
            import numpy as np
            if isinstance(val, np.ndarray):
                continue
            payload[f.name] = val
    return json.dumps(payload, cls=_EventEncoder)


def _broadcast_event(et: Any, event: Any) -> None:
    with _lock:
        if not _connections or _loop is None:
            return
        conns = set(_connections)
        loop = _loop

    msg = _serialize_event(et, event)

    async def _send_all() -> None:
        stale = []
        for ws in conns:
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        if stale:
            with _lock:
                for ws in stale:
                    _connections.discard(ws)

    try:
        asyncio.run_coroutine_threadsafe(_send_all(), loop)
    except Exception:
        pass


@router.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket) -> None:
    global _loop
    await ws.accept()

    with _lock:
        _connections.add(ws)
        _loop = asyncio.get_event_loop()

    _subscribe_to_events()

    try:
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=10.0)
            except asyncio.TimeoutError:
                try:
                    await ws.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with _lock:
            _connections.discard(ws)
