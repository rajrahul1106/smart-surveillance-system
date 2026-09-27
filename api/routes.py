from __future__ import annotations

import time
from typing import Any, Dict, Optional

import cv2
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.frame_annotator import FrameAnnotator

router = APIRouter(prefix="/api")

_deps: Dict[str, Any] = {}
_start_time: float = 0.0

_VIDEO_FPS = 15
_VIDEO_INTERVAL = 1.0 / _VIDEO_FPS
_JPEG_QUALITY = 60
_TARGET_HEIGHT = 480


def init_routes(
    event_bus: Any,
    state_machine: Any,
    storage_service: Any,
    camera_manager: Any,
    config: Any,
    *,
    gesture_model: Any = None,
    face_model: Any = None,
    fire_model: Any = None,
    injury_model: Any = None,
    activity_model: Any = None,
    shared_frame: Any = None,
    enrollment_service: Any = None,
) -> None:
    _deps["event_bus"] = event_bus
    _deps["state_machine"] = state_machine
    _deps["storage_service"] = storage_service
    _deps["camera_manager"] = camera_manager
    _deps["config"] = config
    _deps["gesture_model"] = gesture_model
    _deps["face_model"] = face_model
    _deps["fire_model"] = fire_model
    _deps["injury_model"] = injury_model
    _deps["activity_model"] = activity_model
    _deps["shared_frame"] = shared_frame
    _deps["enrollment_service"] = enrollment_service
    global _start_time
    _start_time = time.time()


class OverrideRequest(BaseModel):
    action: str


class EnrollStartRequest(BaseModel):
    name: str


def _presence_summary(snapshot: Any) -> Dict[str, Any]:
    """Latest presence summary from the pipeline snapshot (empty when not monitoring)."""
    presence = snapshot.detections.get("presence") if snapshot is not None else None
    if not isinstance(presence, dict):
        presence = {}
    return {
        "authorized": list(presence.get("authorized") or []),
        "unknown_count": int(presence.get("unknown_count", 0)),
        "uncertain_count": int(presence.get("uncertain_count", 0)),
        "total": int(presence.get("total", 0)),
    }


@router.get("/status")
def get_status() -> Dict[str, Any]:
    sm = _deps["state_machine"]
    cam = _deps["camera_manager"]
    cfg = _deps["config"]

    cam_status = cam.get_status() if cam else {
        "camera_index": 0, "is_connected": False, "measured_fps": 0.0, "total_frame_count": 0,
    }

    models_status = {}
    for name, key in [
        ("gesture", "gesture_model"),
        ("face", "face_model"),
        ("fire", "fire_model"),
        ("injury", "injury_model"),
        ("activity", "activity_model"),
    ]:
        model = _deps.get(key)
        if model is None:
            models_status[name] = "OFFLINE"
        else:
            st = getattr(model, "status", None)
            if isinstance(st, str):
                models_status[name] = st
            elif getattr(model, "is_loaded", False):
                models_status[name] = "loaded"
            else:
                models_status[name] = "unloaded"

    shared = _deps.get("shared_frame")
    snapshot = None
    if shared is not None:
        try:
            snapshot = shared.get_snapshot()
        except Exception:
            snapshot = None

    # While an auth session is active (face_authorized=True in the latest
    # snapshot during ACTIVE_DETECTION / COOLDOWN) report the face slot as
    # AUTHORIZED. The face model stays loaded through ACTIVE_DETECTION for
    # presence monitoring, so its own status would otherwise hide the auth.
    if sm.state.name in ("ACTIVE_DETECTION", "COOLDOWN") and snapshot is not None:
        if snapshot.detections.get("face_authorized"):
            models_status["face"] = "AUTHORIZED"

    processing_fps = 0.0
    if shared is not None:
        try:
            processing_fps = float(shared.get_fps())
        except Exception:
            processing_fps = 0.0

    return {
        "state": sm.state.name,
        "camera": {
            "index": cam_status.get("camera_index", 0),
            "is_connected": cam_status.get("is_connected", False),
            "fps": cam_status.get("measured_fps", 0.0),
            "frame_count": cam_status.get("total_frame_count", 0),
            "resolution": cam_status.get("resolution", [
                getattr(cfg.camera, "width", 0) if hasattr(cfg, "camera") else 0,
                getattr(cfg.camera, "height", 0) if hasattr(cfg, "camera") else 0,
            ]),
        },
        "processing_fps": round(processing_fps, 1),
        "models": models_status,
        "presence": _presence_summary(snapshot),
        "uptime_seconds": round(time.time() - _start_time, 2),
        "config": {
            "camera_index": getattr(cfg.camera, "index", 0) if hasattr(cfg, "camera") else 0,
            "gesture_threshold": getattr(cfg.gesture, "confidence_threshold", 0.0) if hasattr(cfg, "gesture") else 0.0,
            "detection_timeout": getattr(cfg.detection, "active_timeout_seconds", 0.0) if hasattr(cfg, "detection") else 0.0,
            "alerts_dry_run": getattr(cfg.alerts, "dry_run", True) if hasattr(cfg, "alerts") else True,
        },
    }


@router.get("/events")
def get_events(
    type: Optional[str] = Query(None),
    since: Optional[float] = Query(None),
    limit: int = Query(50),
) -> list:
    storage = _deps["storage_service"]
    return storage.query_events(event_type=type, since=since, limit=limit)


@router.get("/transitions")
def get_transitions(
    since: Optional[float] = Query(None),
    limit: int = Query(50),
) -> list:
    storage = _deps["storage_service"]
    return storage.query_transitions(since=since, limit=limit)


@router.post("/override")
def post_override(body: OverrideRequest) -> Dict[str, Any]:
    if body.action != "reset_to_idle":
        raise HTTPException(status_code=400, detail=f"Unknown action: {body.action}")

    sm = _deps["state_machine"]
    success = sm.transition("manual_override")
    return {"success": success, "new_state": sm.state.name}


# ---------------------------------------------------------------------------
# Enrollment endpoints
# ---------------------------------------------------------------------------

@router.post("/enroll/start")
def post_enroll_start(body: EnrollStartRequest) -> Dict[str, Any]:
    svc = _deps.get("enrollment_service")
    if svc is None:
        raise HTTPException(status_code=503, detail="enrollment service unavailable")
    res = svc.start(body.name)
    if not res.get("ok"):
        raise HTTPException(status_code=409, detail=res.get("error", "enrollment failed"))
    return {"session_id": res["session_id"], "status": res["status"]}


@router.get("/enroll/status/{session_id}")
def get_enroll_status(session_id: str) -> Dict[str, Any]:
    svc = _deps.get("enrollment_service")
    if svc is None:
        raise HTTPException(status_code=503, detail="enrollment service unavailable")
    sess = svc.get_status(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    return sess


@router.get("/enroll/list")
def get_enroll_list() -> list:
    svc = _deps.get("enrollment_service")
    if svc is None:
        return []
    return svc.list_enrolled()


@router.delete("/enroll/{name}")
def delete_enroll(name: str) -> Dict[str, Any]:
    svc = _deps.get("enrollment_service")
    if svc is None:
        raise HTTPException(status_code=503, detail="enrollment service unavailable")
    ok = svc.remove(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"no enrollment for: {name}")
    return {"success": True, "name": name}


# ---------------------------------------------------------------------------
# MJPEG video feed (throttled, downscaled, lower JPEG quality)
# ---------------------------------------------------------------------------

_annotator = FrameAnnotator(target_height=_TARGET_HEIGHT)


def _mjpeg_generator():
    shared = _deps.get("shared_frame")
    if shared is None:
        return

    next_frame_at = time.monotonic()
    while True:
        now = time.monotonic()
        sleep_for = next_frame_at - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        next_frame_at = max(now, next_frame_at) + _VIDEO_INTERVAL

        # Read the entire snapshot atomically so the frame and the
        # detections we draw on it always come from the same pipeline tick.
        snapshot = shared.get_snapshot()
        frame = snapshot.frame
        if frame is None:
            continue

        # The snapshot's detections dict is already a defensive copy made
        # at set()-time; we can safely add transient render-only fields.
        detections = dict(snapshot.detections)
        try:
            detections["processing_fps"] = shared.get_fps()
        except Exception:
            pass

        try:
            annotated = _annotator.annotate(frame, snapshot.state, detections)
        except Exception:
            continue

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


@router.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
