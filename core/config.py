from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30

    def validate(self) -> None:
        if self.index < 0:
            raise ValueError("camera.index must be >= 0")
        if self.fps <= 0:
            raise ValueError("camera.fps must be > 0")


@dataclass
class GestureConfig:
    confidence_threshold: float = 0.90
    near_miss_threshold: float = 0.60
    sequence: List[str] = field(default_factory=lambda: ["palm", "fist", "palm", "fist"])
    sequence_timeout_seconds: float = 5.0

    def validate(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("gesture.confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.near_miss_threshold <= 1.0:
            raise ValueError("gesture.near_miss_threshold must be between 0.0 and 1.0")
        if self.sequence_timeout_seconds <= 0:
            raise ValueError("gesture.sequence_timeout_seconds must be > 0")


@dataclass
class FaceAuthConfig:
    max_attempts: int = 3
    attempt_timeout_seconds: float = 10.0
    encodings_path: str = "data/face_encodings.pkl"
    # Multi-person identification. Defaults mirror FaceModel's class constants.
    auth_threshold: float = 0.60
    unknown_threshold: float = 0.30
    ema_alpha: float = 0.3
    unknown_confirm_frames: int = 5
    track_max_missed: int = 10
    presence_interval_frames: int = 10
    max_faces: int = 6

    def validate(self) -> None:
        if self.attempt_timeout_seconds <= 0:
            raise ValueError("face_auth.attempt_timeout_seconds must be > 0")
        if not 0.0 <= self.unknown_threshold <= self.auth_threshold <= 1.0:
            raise ValueError(
                "face_auth thresholds must satisfy "
                "0 <= unknown_threshold <= auth_threshold <= 1"
            )
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("face_auth.ema_alpha must be in (0, 1]")
        if self.unknown_confirm_frames < 1:
            raise ValueError("face_auth.unknown_confirm_frames must be >= 1")
        if self.track_max_missed < 0:
            raise ValueError("face_auth.track_max_missed must be >= 0")
        if self.presence_interval_frames < 1:
            raise ValueError("face_auth.presence_interval_frames must be >= 1")
        if self.max_faces < 1:
            raise ValueError("face_auth.max_faces must be >= 1")


@dataclass
class DetectionConfig:
    active_timeout_seconds: float = 30.0
    cooldown_seconds: float = 5.0
    loitering_threshold_seconds: float = 30.0
    loitering_movement_pixels: float = 20.0

    def validate(self) -> None:
        if self.active_timeout_seconds <= 0:
            raise ValueError("detection.active_timeout_seconds must be > 0")
        if self.cooldown_seconds <= 0:
            raise ValueError("detection.cooldown_seconds must be > 0")
        if self.loitering_threshold_seconds <= 0:
            raise ValueError("detection.loitering_threshold_seconds must be > 0")
        if self.loitering_movement_pixels < 0:
            raise ValueError("detection.loitering_movement_pixels must be >= 0")


@dataclass
class AlertsConfig:
    dry_run: bool = True
    webhook_url: str = ""
    twilio_enabled: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_to_number: str = ""
    rate_limit_seconds: float = 60.0
    log_path: str = "data/alerts.log"

    def validate(self) -> None:
        if self.rate_limit_seconds < 0:
            raise ValueError("alerts.rate_limit_seconds must be >= 0")


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_path: str = "data/app.log"

    def validate(self) -> None:
        pass


@dataclass
class FrameConfig:
    skip_rate_idle: int = 3
    skip_rate_active: int = 1
    gesture_resolution: List[int] = field(default_factory=lambda: [320, 240])
    detection_resolution: List[int] = field(default_factory=lambda: [640, 480])

    def validate(self) -> None:
        pass


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    face_auth: FaceAuthConfig = field(default_factory=FaceAuthConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)

    def validate(self) -> None:
        self.camera.validate()
        self.gesture.validate()
        self.face_auth.validate()
        self.detection.validate()
        self.alerts.validate()
        self.logging.validate()
        self.frame.validate()


def load_config(path: str = "config.yaml") -> AppConfig:
    raw: Dict[str, Any] = {}
    if os.path.isfile(path):
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    cfg = AppConfig(
        camera=_build(CameraConfig, raw.get("camera")),
        gesture=_build(GestureConfig, raw.get("gesture")),
        face_auth=_build(FaceAuthConfig, raw.get("face_auth")),
        detection=_build(DetectionConfig, raw.get("detection")),
        alerts=_build(AlertsConfig, raw.get("alerts")),
        logging=_build(LoggingConfig, raw.get("logging")),
        frame=_build(FrameConfig, raw.get("frame")),
    )
    cfg.validate()
    return cfg


def _build(cls: type, data: Optional[Dict[str, Any]]) -> Any:
    if not data:
        return cls()
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


# ---------------------------------------------------------------------------
# Backward-compatible aliases used by existing modules
# ---------------------------------------------------------------------------

PipelineConfig = AppConfig
FrameManagerConfig = None  # replaced — see note below


class _FrameManagerConfig:
    """Thin adapter so existing FrameManager code keeps working."""

    def __init__(self, *, frame_skip: int = 1, scale_width: int = 320, scale_height: int = 240) -> None:
        self.frame_skip = frame_skip
        self.scale_width = scale_width
        self.scale_height = scale_height


FrameManagerConfig = _FrameManagerConfig  # type: ignore[assignment,misc]


class _AlertConfig:
    """Thin adapter so existing AlertService code keeps working."""

    def __init__(
        self,
        *,
        rate_limit_seconds: float = 60.0,
        webhook_url: str = "",
        twilio_account_sid: str = "",
        twilio_auth_token: str = "",
        twilio_from_number: str = "",
        twilio_to_number: str = "",
        dry_run: bool = True,
        alerts_log_path: str = "data/alerts.log",
    ) -> None:
        self.rate_limit_seconds = rate_limit_seconds
        self.webhook_url = webhook_url
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.twilio_from_number = twilio_from_number
        self.twilio_to_number = twilio_to_number
        self.dry_run = dry_run
        self.alerts_log_path = alerts_log_path


AlertConfig = _AlertConfig  # type: ignore[assignment,misc]
