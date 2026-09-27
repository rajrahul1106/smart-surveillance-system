"""
core/events.py

Typed event dataclasses for the surveillance pipeline.

Every event has a ``timestamp`` field (float, defaults to time.time()) and a
``event_type`` ClassVar that the EventBus uses as the subscription key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from core.event_bus import EventType


# ---------------------------------------------------------------------------
# Frame / camera events
# ---------------------------------------------------------------------------

@dataclass
class FrameReady:
    event_type: ClassVar[EventType] = EventType.FRAME_READY
    frame: Any
    camera_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessedFrame:
    event_type: ClassVar[EventType] = EventType.PROCESSED_FRAME
    frame: Any
    camera_id: str = ""
    resolution: Tuple[int, int] = field(default_factory=tuple)
    original_frame: Any = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CameraConnected:
    event_type: ClassVar[EventType] = EventType.CAMERA_CONNECTED
    camera_id: str
    index: int
    resolution: Tuple[int, int] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CameraDisconnected:
    event_type: ClassVar[EventType] = EventType.CAMERA_DISCONNECTED
    camera_id: str
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Gesture events
# ---------------------------------------------------------------------------

@dataclass
class GestureCandidate:
    event_type: ClassVar[EventType] = EventType.GESTURE_CANDIDATE
    confidence: float
    frame: Any = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class GestureNearMiss:
    event_type: ClassVar[EventType] = EventType.GESTURE_NEAR_MISS
    confidence: float
    frame: Any = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class GestureConfirmed:
    event_type: ClassVar[EventType] = EventType.GESTURE_CONFIRMED
    timestamp: float = field(default_factory=time.time)


@dataclass
class GestureRejected:
    event_type: ClassVar[EventType] = EventType.GESTURE_REJECTED
    timestamp: float = field(default_factory=time.time)


@dataclass
class GestureProgress:
    event_type: ClassVar[EventType] = EventType.GESTURE_PROGRESS
    step: int = 0
    total: int = 0
    current_gesture: str = ""
    sequence: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Auth events
# ---------------------------------------------------------------------------

@dataclass
class AuthSuccess:
    event_type: ClassVar[EventType] = EventType.AUTH_SUCCESS
    result: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AuthFailed:
    event_type: ClassVar[EventType] = EventType.AUTH_FAILED
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Presence events
# ---------------------------------------------------------------------------

@dataclass
class PresenceUpdated:
    """Who is in view changed.  Published on change only, never per frame."""
    event_type: ClassVar[EventType] = EventType.PRESENCE_UPDATED
    authorized: List[str] = field(default_factory=list)
    unknown_count: int = 0
    uncertain_count: int = 0
    total: int = 0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Detection events
# ---------------------------------------------------------------------------

@dataclass
class FireDetected:
    event_type: ClassVar[EventType] = EventType.FIRE_DETECTED
    result: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class InjuryDetected:
    event_type: ClassVar[EventType] = EventType.INJURY_DETECTED
    result: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SuspiciousActivity:
    event_type: ClassVar[EventType] = EventType.SUSPICIOUS_ACTIVITY
    result: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DetectionComplete:
    event_type: ClassVar[EventType] = EventType.DETECTION_COMPLETE
    results: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlertDispatched:
    event_type: ClassVar[EventType] = EventType.ALERT_DISPATCHED
    alert_type: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# State machine events
# ---------------------------------------------------------------------------

@dataclass
class StateTransition:
    event_type: ClassVar[EventType] = EventType.STATE_TRANSITION
    from_state: str = ""
    to_state: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# System lifecycle events
# ---------------------------------------------------------------------------

@dataclass
class SystemStartup:
    event_type: ClassVar[EventType] = EventType.SYSTEM_STARTUP
    config_summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemShutdown:
    event_type: ClassVar[EventType] = EventType.SYSTEM_SHUTDOWN
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Model lifecycle events
# ---------------------------------------------------------------------------

@dataclass
class ModelLoaded:
    event_type: ClassVar[EventType] = EventType.MODEL_LOADED
    model_name: str = ""
    load_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelUnloaded:
    event_type: ClassVar[EventType] = EventType.MODEL_UNLOADED
    model_name: str = ""
    timestamp: float = field(default_factory=time.time)
