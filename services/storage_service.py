from __future__ import annotations

import dataclasses
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from core.event_bus import EventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON encoder that handles numpy / opencv / tuple edge cases
# ---------------------------------------------------------------------------

class _EventEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, EventType):
            return obj.name
        return super().default(obj)


def _serialise_event(event: Any) -> Dict[str, Any]:
    """Convert a dataclass event to a JSON-safe dict, dropping frame data."""
    if not dataclasses.is_dataclass(event) or isinstance(event, type):
        return {}

    out: Dict[str, Any] = {}
    for f in dataclasses.fields(event):
        val = getattr(event, f.name)
        if f.name in ("frame", "original_frame"):
            continue
        if isinstance(val, np.ndarray):
            continue
        out[f.name] = val
    return out


# ---------------------------------------------------------------------------
# StorageService
# ---------------------------------------------------------------------------

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT    NOT NULL,
    camera_id   TEXT,
    timestamp   REAL    NOT NULL,
    confidence  REAL,
    payload     TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_TRANSITIONS = """
CREATE TABLE IF NOT EXISTS state_transitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    from_state  TEXT NOT NULL,
    to_state    TEXT NOT NULL,
    trigger     TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


class StorageService:

    def __init__(self, db_path: str = "data/events.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()

        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_EVENTS)
        self._conn.execute(_CREATE_TRANSITIONS)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write operations (lock-protected)
    # ------------------------------------------------------------------

    def save_event(self, event: Any) -> None:
        et = getattr(type(event), "event_type", None)
        if isinstance(et, EventType):
            event_type_name = et.name
        elif hasattr(event, "type") and isinstance(event.type, EventType):
            event_type_name = event.type.name
        else:
            event_type_name = type(event).__name__

        payload = _serialise_event(event)
        ts = getattr(event, "timestamp", time.time())
        camera_id = getattr(event, "camera_id", None)
        confidence = getattr(event, "confidence", None)

        payload_json = json.dumps(payload, cls=_EventEncoder)

        with self._lock:
            self._conn.execute(
                "INSERT INTO events (event_type, camera_id, timestamp, confidence, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (event_type_name, camera_id, ts, confidence, payload_json),
            )
            self._conn.commit()

    def save_transition(
        self,
        from_state: str,
        to_state: str,
        trigger: str,
        timestamp: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO state_transitions (from_state, to_state, trigger, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (from_state, to_state, trigger, timestamp),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def query_events(
        self,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT id, event_type, camera_id, timestamp, confidence, payload, created_at FROM events"
        conditions: List[str] = []
        params: List[Any] = []

        if event_type is not None:
            conditions.append("event_type = ?")
            params.append(event_type)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def query_transitions(
        self,
        since: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT id, from_state, to_state, trigger, timestamp, created_at FROM state_transitions"
        params: List[Any] = []

        if since is not None:
            sql += " WHERE timestamp >= ?"
            params.append(since)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()
