"""Tests for AppConfig, load_config, and validation."""

from __future__ import annotations

import os
import sys

import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.config import (
    AppConfig,
    AlertsConfig,
    CameraConfig,
    DetectionConfig,
    FaceAuthConfig,
    FireConfig,
    FrameConfig,
    GestureConfig,
    LoggingConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# load_config — no file
# ---------------------------------------------------------------------------

class TestLoadConfigNoFile:
    def test_returns_all_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(cfg, AppConfig)
        assert cfg.camera.index == 0
        assert cfg.camera.width == 640
        assert cfg.camera.height == 480
        assert cfg.camera.fps == 30
        assert cfg.gesture.confidence_threshold == 0.90
        assert cfg.gesture.near_miss_threshold == 0.60
        assert cfg.gesture.sequence == ["palm", "fist", "palm", "fist"]
        assert cfg.gesture.sequence_timeout_seconds == 5.0
        assert cfg.face_auth.max_attempts == 3
        assert cfg.face_auth.attempt_timeout_seconds == 10.0
        assert cfg.face_auth.encodings_path == "data/face_encodings.pkl"
        assert cfg.detection.active_timeout_seconds == 30.0
        assert cfg.detection.cooldown_seconds == 5.0
        assert cfg.alerts.dry_run is True
        assert cfg.alerts.rate_limit_seconds == 60.0
        assert cfg.logging.level == "INFO"
        assert cfg.frame.skip_rate_idle == 3
        assert cfg.frame.skip_rate_active == 1


# ---------------------------------------------------------------------------
# load_config — full YAML
# ---------------------------------------------------------------------------

class TestLoadConfigFullYaml:
    def test_loads_all_values(self, tmp_path):
        yaml_content = """\
camera:
  index: 2
  width: 1920
  height: 1080
  fps: 60

gesture:
  confidence_threshold: 0.80
  near_miss_threshold: 0.50
  sequence: ["palm", "fist"]
  sequence_timeout_seconds: 5.0

face_auth:
  max_attempts: 5
  attempt_timeout_seconds: 15.0
  encodings_path: "custom/encodings.pkl"

detection:
  active_timeout_seconds: 60.0
  cooldown_seconds: 10.0

alerts:
  dry_run: false
  webhook_url: "http://example.com/hook"
  twilio_enabled: true
  twilio_account_sid: "AC_test"
  twilio_auth_token: "token"
  twilio_from_number: "+1111"
  twilio_to_number: "+2222"
  rate_limit_seconds: 30.0
  log_path: "custom/alerts.log"

logging:
  level: "DEBUG"
  log_to_file: false
  log_path: "custom/app.log"

frame:
  skip_rate_idle: 5
  skip_rate_active: 2
  gesture_resolution: [160, 120]
  detection_resolution: [800, 600]
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        cfg = load_config(str(path))

        assert cfg.camera.index == 2
        assert cfg.camera.width == 1920
        assert cfg.camera.fps == 60
        assert cfg.gesture.confidence_threshold == 0.80
        assert cfg.gesture.sequence == ["palm", "fist"]
        assert cfg.face_auth.max_attempts == 5
        assert cfg.face_auth.encodings_path == "custom/encodings.pkl"
        assert cfg.detection.active_timeout_seconds == 60.0
        assert cfg.alerts.dry_run is False
        assert cfg.alerts.webhook_url == "http://example.com/hook"
        assert cfg.alerts.twilio_account_sid == "AC_test"
        assert cfg.alerts.rate_limit_seconds == 30.0
        assert cfg.logging.level == "DEBUG"
        assert cfg.logging.log_to_file is False
        assert cfg.frame.skip_rate_idle == 5
        assert cfg.frame.gesture_resolution == [160, 120]
        assert cfg.frame.detection_resolution == [800, 600]


# ---------------------------------------------------------------------------
# load_config — partial YAML
# ---------------------------------------------------------------------------

class TestLoadConfigPartialYaml:
    def test_fills_defaults_for_missing_sections(self, tmp_path):
        yaml_content = """\
camera:
  index: 1
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        cfg = load_config(str(path))

        assert cfg.camera.index == 1
        assert cfg.camera.width == 640  # default
        assert cfg.gesture.confidence_threshold == 0.90  # entire section default
        assert cfg.detection.active_timeout_seconds == 30.0
        assert cfg.alerts.dry_run is True

    def test_fills_defaults_for_missing_fields_within_section(self, tmp_path):
        yaml_content = """\
gesture:
  confidence_threshold: 0.75
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        cfg = load_config(str(path))

        assert cfg.gesture.confidence_threshold == 0.75
        assert cfg.gesture.near_miss_threshold == 0.60  # default
        assert cfg.gesture.sequence == ["palm", "fist", "palm", "fist"]  # default


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_confidence_above_1(self, tmp_path):
        yaml_content = """\
gesture:
  confidence_threshold: 1.5
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        with pytest.raises(ValueError, match="gesture.confidence_threshold"):
            load_config(str(path))

    def test_negative_timeout(self, tmp_path):
        yaml_content = """\
detection:
  active_timeout_seconds: -5
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        with pytest.raises(ValueError, match="detection.active_timeout_seconds"):
            load_config(str(path))

    def test_negative_camera_index(self, tmp_path):
        yaml_content = """\
camera:
  index: -1
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        with pytest.raises(ValueError, match="camera.index"):
            load_config(str(path))

    def test_fps_zero(self, tmp_path):
        yaml_content = """\
camera:
  fps: 0
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        with pytest.raises(ValueError, match="camera.fps"):
            load_config(str(path))

    def test_negative_near_miss_threshold(self):
        cfg = AppConfig(gesture=GestureConfig(near_miss_threshold=-0.1))
        with pytest.raises(ValueError, match="gesture.near_miss_threshold"):
            cfg.validate()

    def test_negative_cooldown(self):
        cfg = AppConfig(detection=DetectionConfig(cooldown_seconds=-1))
        with pytest.raises(ValueError, match="detection.cooldown_seconds"):
            cfg.validate()

    def test_unknown_threshold_above_auth_threshold(self):
        cfg = AppConfig(face_auth=FaceAuthConfig(auth_threshold=0.5, unknown_threshold=0.6))
        with pytest.raises(ValueError, match="face_auth thresholds"):
            cfg.validate()

    def test_ema_alpha_zero(self):
        cfg = AppConfig(face_auth=FaceAuthConfig(ema_alpha=0.0))
        with pytest.raises(ValueError, match="face_auth.ema_alpha"):
            cfg.validate()

    def test_presence_interval_zero(self):
        cfg = AppConfig(face_auth=FaceAuthConfig(presence_interval_frames=0))
        with pytest.raises(ValueError, match="face_auth.presence_interval_frames"):
            cfg.validate()

    def test_fire_score_cap_below_threshold(self):
        cfg = AppConfig(fire=FireConfig(score_threshold=3.0, score_cap=2.0))
        with pytest.raises(ValueError, match="fire.score_cap"):
            cfg.validate()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_camera_config_defaults(self):
        cfg = CameraConfig()
        assert cfg.index == 0
        assert cfg.width == 640
        assert cfg.height == 480
        assert cfg.fps == 30

    def test_alerts_config_dry_run_defaults_true(self):
        cfg = AlertsConfig()
        assert cfg.dry_run is True

    def test_face_auth_defaults_match_face_model_constants(self):
        from models.face_model import FaceModel

        cfg = FaceAuthConfig()
        assert cfg.auth_threshold == FaceModel.AUTHORIZE_THRESHOLD
        assert cfg.unknown_threshold == FaceModel.UNKNOWN_THRESHOLD
        assert cfg.ema_alpha == FaceModel.EMA_ALPHA
        assert cfg.unknown_confirm_frames == FaceModel.UNKNOWN_CONFIRM_FRAMES
        assert cfg.track_max_missed == FaceModel.TRACK_MAX_MISSED
        assert cfg.max_faces == FaceModel.MAX_FACES
        assert cfg.presence_interval_frames == 10

    def test_repo_config_loads_multi_person_face_keys(self):
        cfg = load_config(os.path.join(_project_root, "config.yaml"))
        assert cfg.face_auth.auth_threshold == 0.60
        assert cfg.face_auth.unknown_threshold == 0.30
        assert cfg.face_auth.ema_alpha == 0.3
        assert cfg.face_auth.unknown_confirm_frames == 5
        assert cfg.face_auth.track_max_missed == 10
        assert cfg.face_auth.presence_interval_frames == 10
        assert cfg.face_auth.max_faces == 6

    def test_fire_defaults_match_fire_model(self):
        from models.fire_model import FireModel

        cfg = FireConfig()
        assert cfg.model_path == FireModel.MODEL_PATH
        assert cfg.labels_path == FireModel.LABELS_PATH
        assert cfg.confidence_threshold == FireModel.CONFIDENCE_THRESHOLD
        assert cfg.iou_threshold == FireModel.IOU_THRESHOLD
        assert cfg.input_size == FireModel.INPUT_SIZE
        assert cfg.score_threshold == FireModel.SCORE_THRESHOLD
        assert cfg.score_cap == FireModel.SCORE_CAP

    def test_repo_config_uses_yolo11s_at_conf_035(self):
        cfg = load_config(os.path.join(_project_root, "config.yaml"))
        assert cfg.fire.model_path.endswith("fire_yolo11s_480.onnx")
        assert cfg.fire.labels_path.endswith("fire_yolo11s_labels.json")
        assert cfg.fire.confidence_threshold == 0.35
        assert cfg.fire.input_size == 480

    def test_app_config_all_sections_populated(self):
        cfg = AppConfig()
        assert isinstance(cfg.camera, CameraConfig)
        assert isinstance(cfg.gesture, GestureConfig)
        assert isinstance(cfg.face_auth, FaceAuthConfig)
        assert isinstance(cfg.detection, DetectionConfig)
        assert isinstance(cfg.alerts, AlertsConfig)
        assert isinstance(cfg.logging, LoggingConfig)
        assert isinstance(cfg.frame, FrameConfig)
