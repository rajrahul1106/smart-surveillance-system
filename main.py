"""Smart Surveillance System — single entry point."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main() -> None:
    # ------------------------------------------------------------------ args
    parser = argparse.ArgumentParser(description="Smart Surveillance System")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--camera", type=int, default=None, help="Override camera index")
    args = parser.parse_args()

    # ------------------------------------------------------------------ config
    from core.config import load_config

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg.camera.index = args.camera

    # ------------------------------------------------------------------ logging
    os.makedirs("data", exist_ok=True)

    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.logging.log_to_file:
        log_dir = os.path.dirname(cfg.logging.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.logging.log_path))

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format=log_fmt,
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)
    logger.info("Config loaded from %s", args.config)

    # ------------------------------------------------------------------ event bus
    from core.event_bus import EventBus

    event_bus = EventBus()

    # ------------------------------------------------------------------ state machine
    from core.state_machine import StateMachine

    state_machine = StateMachine()

    # ------------------------------------------------------------------ models
    from models.gesture_model import GestureModel
    from models.face_model import FaceModel
    from models.fire_model import FireModel
    from models.injury_model import InjuryModel
    from models.activity_model import ActivityModel

    gesture_model = GestureModel()
    # face_auth.presence_interval_frames is read by ActiveDetectionState
    # from the same config.
    face_auth = cfg.face_auth
    face_model = FaceModel(
        encodings_path=face_auth.encodings_path,
        match_threshold=face_auth.auth_threshold,
        unknown_threshold=face_auth.unknown_threshold,
        ema_alpha=face_auth.ema_alpha,
        unknown_confirm_frames=face_auth.unknown_confirm_frames,
        track_max_missed=face_auth.track_max_missed,
        max_faces=face_auth.max_faces,
    )
    fire_cfg = cfg.fire
    fire_model = FireModel(
        model_path=fire_cfg.model_path,
        labels_path=fire_cfg.labels_path,
        confidence_threshold=fire_cfg.confidence_threshold,
        iou_threshold=fire_cfg.iou_threshold,
        input_size=fire_cfg.input_size,
        score_threshold=fire_cfg.score_threshold,
        score_cap=fire_cfg.score_cap,
    )
    injury_model = InjuryModel()
    activity_model = ActivityModel(
        loitering_threshold_seconds=cfg.detection.loitering_threshold_seconds,
        loitering_movement_pixels=cfg.detection.loitering_movement_pixels,
    )

    # Wire ModelLoaded / ModelUnloaded events
    for m in (gesture_model, face_model, fire_model, injury_model, activity_model):
        if hasattr(m, "set_event_bus"):
            m.set_event_bus(event_bus)

    # ------------------------------------------------------------------ camera
    from camera.cam_manager import CameraManager
    from camera.cam_config import CameraConfig as CamCfg

    cam_config = CamCfg(
        index=cfg.camera.index,
        resolution_width=cfg.camera.width,
        resolution_height=cfg.camera.height,
        fps=cfg.camera.fps,
    )
    camera_manager = CameraManager(event_bus, cam_config)

    # ------------------------------------------------------------------ frame manager
    from core.config import FrameManagerConfig
    from core.frame_manager import FrameManager

    frame_cfg = FrameManagerConfig(
        frame_skip=cfg.frame.skip_rate_idle,
        scale_width=cfg.frame.gesture_resolution[0],
        scale_height=cfg.frame.gesture_resolution[1],
    )
    frame_manager = FrameManager(event_bus, frame_cfg)

    # ------------------------------------------------------------------ shared frame
    from core.shared_frame import SharedFrame

    shared_frame = SharedFrame()

    # ------------------------------------------------------------------ pipeline
    from core.pipeline import Pipeline

    pipeline = Pipeline(
        event_bus=event_bus,
        state_machine=state_machine,
        config=cfg,
        gesture_model=gesture_model,
        face_model=face_model,
        fire_model=fire_model,
        injury_model=injury_model,
        activity_model=activity_model,
        camera_manager=camera_manager,
        shared_frame=shared_frame,
        frame_manager=frame_manager,
    )

    # ------------------------------------------------------------------ storage
    from services.storage_service import StorageService

    storage = StorageService("data/events.db")

    # ------------------------------------------------------------------ audit logger
    from services.audit_logger import AuditLogger

    audit_logger = AuditLogger(event_bus, storage)
    audit_logger.start()

    # ------------------------------------------------------------------ alert service
    from core.config import AlertConfig
    from services.alert_service import AlertService

    alert_cfg = AlertConfig(
        rate_limit_seconds=cfg.alerts.rate_limit_seconds,
        webhook_url=cfg.alerts.webhook_url,
        twilio_account_sid=cfg.alerts.twilio_account_sid,
        twilio_auth_token=cfg.alerts.twilio_auth_token,
        twilio_from_number=cfg.alerts.twilio_from_number,
        twilio_to_number=cfg.alerts.twilio_to_number,
        dry_run=cfg.alerts.dry_run,
        alerts_log_path=cfg.alerts.log_path,
    )
    alert_service = AlertService(event_bus, alert_cfg)
    alert_service.start()

    # ------------------------------------------------------------------ enrollment service
    from services.enrollment_service import EnrollmentService

    enrollment_service = EnrollmentService(
        shared_frame=shared_frame,
        face_model=face_model,
    )

    # ------------------------------------------------------------------ API server
    from api.app import create_app, start_server

    app = create_app(
        event_bus, state_machine, storage, camera_manager, cfg,
        gesture_model=gesture_model,
        face_model=face_model,
        fire_model=fire_model,
        injury_model=injury_model,
        activity_model=activity_model,
        shared_frame=shared_frame,
        enrollment_service=enrollment_service,
    )
    start_server(app, host="0.0.0.0", port=8000)

    # ------------------------------------------------------------------ startup event
    from core.events import SystemStartup, SystemShutdown

    event_bus.publish(SystemStartup(config_summary={
        "camera_index": cfg.camera.index,
        "gesture_threshold": cfg.gesture.confidence_threshold,
        "alerts_dry_run": cfg.alerts.dry_run,
    }))

    # ------------------------------------------------------------------ start camera
    camera_manager.start()

    # ------------------------------------------------------------------ banner
    dry_label = "DRY RUN" if cfg.alerts.dry_run else "LIVE"
    print(f"""
============================================
Smart Surveillance System - RUNNING
============================================
Camera:    index {cfg.camera.index}
State:     {state_machine.state.name}
Dashboard: http://localhost:8000/dashboard/
API:       http://localhost:8000/api/status
Alerts:    {dry_label}
Config:    {args.config}
============================================
Press Ctrl+C to stop
""")

    # ------------------------------------------------------------------ keep alive
    try:
        signal.pause()
    except KeyboardInterrupt:
        pass

    # ------------------------------------------------------------------ shutdown
    print("\nShutting down...")
    camera_manager.stop()
    gesture_model.unload()
    face_model.unload()
    fire_model.unload()
    injury_model.unload()
    activity_model.unload()
    event_bus.publish(SystemShutdown(reason="user_interrupt"))
    event_bus.flush()
    alert_service.stop()
    audit_logger.stop()
    storage.close()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()
