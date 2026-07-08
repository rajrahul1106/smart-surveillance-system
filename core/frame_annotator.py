from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LINE = cv2.LINE_AA

# Colour palette (BGR for OpenCV). Hex shown is the dashboard equivalent.
# Kept in sync with the CSS variables in dashboard/web/index.html.
_C_GREEN = (136, 255, 0)       # #00ff88 — pose / skeleton / OK
_C_CYAN = (255, 212, 0)        # #00d4ff — face
_C_ORANGE = (53, 107, 255)     # #ff6b35 — weapon / suspicious_object
_C_RED = (99, 46, 255)         # #ff2e63 — fire / danger
_C_YELLOW = (0, 183, 255)      # #ffb700 — possible_concern / suspicious / loitering
_C_TEXT = (235, 231, 229)      # #e5e7eb — primary text
_C_DIM = (175, 161, 156)       # #9ca3af — secondary text
_C_RED_TEXT = (99, 46, 255)    # text colour for low-FPS warning
_C_BG = (32, 24, 17)           # neutral semi-transparent background

_STATE_COLORS = {
    "IDLE": _C_GREEN,
    "VERIFYING_GESTURE": _C_ORANGE,
    "VERIFYING_IDENTITY": _C_ORANGE,
    "ACTIVE_DETECTION": _C_RED,
    "COOLDOWN": _C_CYAN,
}

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_LOW_FPS_THRESHOLD = 15.0
_LABEL_SHIFT_PX = 22  # how far down to nudge a colliding label

# Display thresholds — labels are suppressed below these confidence values
# (in percent). Mirrors the per-detector tier thresholds so the dashboard
# never shows labels weaker than the detector itself would surface.
_DISPLAY_THRESHOLDS = {
    "fire": 40,            # FireModel.MIN_DISPLAY_CONFIDENCE
    "face": 35,            # FaceModel.POSSIBLE_THRESHOLD * 100
    "injury": 30,
    "loitering": 50,       # ActivityModel _loiter_confidence at 1× threshold
    "masked_face": 50,
    "possible_concern": 60,
    "suspicious_object": 80,
    # Default fallback for any other suspicious activity_type
    "_default": 30,
}

# Universal alpha for label / overlay backgrounds.
_ALPHA = 0.6

# Font scales — tightened per spec to reduce overlay clutter.
_SCALE_CHROME = 0.4    # CAM / FPS in the corners
_SCALE_STATE = 0.5     # state pill at bottom-left
_SCALE_LABEL = 0.5     # detection labels
_SCALE_TIMESTAMP = 0.42


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def scale_bbox(
    bbox: Optional[Tuple[int, int, int, int]],
    detection_size: Optional[Tuple[int, int]],
    display_size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Scale a bounding box from the resolution it was detected on to the
    resolution being annotated.

    Parameters
    ----------
    bbox:
        ``(x, y, w, h)`` in *detection_size*'s coordinate system.
    detection_size:
        ``(width, height)`` of the frame the model ran on. If ``None`` (or
        either dimension is non-positive) the bbox is returned with its
        coordinates unchanged.
    display_size:
        ``(width, height)`` of the frame currently being annotated.

    The returned bbox has its x and width multiplied by
    ``display_width / detection_width``, and its y and height multiplied
    by ``display_height / detection_height``. This keeps detections
    aligned with the displayed frame even when detection runs on a
    downscaled copy.
    """
    if not bbox:
        return None
    if detection_size is None:
        x, y, w, h = bbox
        return (int(x), int(y), int(w), int(h))
    fw, fh = detection_size
    tw, th = display_size
    if fw <= 0 or fh <= 0:
        return tuple(int(v) for v in bbox)  # type: ignore[return-value]
    sx = tw / float(fw)
    sy = th / float(fh)
    x, y, w, h = bbox
    return (
        int(round(x * sx)),
        int(round(y * sy)),
        int(round(w * sx)),
        int(round(h * sy)),
    )


def _alpha_rect(frame: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int],
                color: Tuple[int, int, int], alpha: float = _ALPHA) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, p1, p2, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def _label_size(text: str, scale: float, pad: int) -> Tuple[int, int, int]:
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, 1)
    width = tw + pad * 2
    height = th + pad + baseline
    return width, height, baseline


def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _draw_corner_brackets(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                          color: Tuple[int, int, int], length: int = 18,
                          thickness: int = 2) -> None:
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    L = max(8, min(length, w // 3, h // 3))
    cv2.line(frame, (x, y), (x + L, y), color, thickness, _LINE)
    cv2.line(frame, (x, y), (x, y + L), color, thickness, _LINE)
    cv2.line(frame, (x2, y), (x2 - L, y), color, thickness, _LINE)
    cv2.line(frame, (x2, y), (x2, y + L), color, thickness, _LINE)
    cv2.line(frame, (x, y2), (x + L, y2), color, thickness, _LINE)
    cv2.line(frame, (x, y2), (x, y2 - L), color, thickness, _LINE)
    cv2.line(frame, (x2, y2), (x2 - L, y2), color, thickness, _LINE)
    cv2.line(frame, (x2, y2), (x2, y2 - L), color, thickness, _LINE)


def _conf_pct(value: Any) -> Optional[int]:
    """Normalize a confidence value (either 0-1 or 0-100) into 0-100 int."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0.0:
        return None
    if v <= 1.0:
        v *= 100.0
    return int(round(min(99.0, max(0.0, v))))


def _meets_threshold(kind: str, conf_pct: Optional[int]) -> bool:
    """True when *conf_pct* is high enough to surface a label for *kind*."""
    if conf_pct is None:
        return False
    threshold = _DISPLAY_THRESHOLDS.get(kind, _DISPLAY_THRESHOLDS["_default"])
    return conf_pct >= threshold


# ---------------------------------------------------------------------------
# FrameAnnotator
# ---------------------------------------------------------------------------

class FrameAnnotator:

    def __init__(self, target_height: int = 480) -> None:
        self._target_height = target_height

    def annotate(self, frame: Any, state: str, detections: Dict[str, Any]) -> Any:
        if frame is None:
            return frame

        out = self._maybe_resize(frame)
        out_h, out_w = out.shape[:2]
        out_size = (out_w, out_h)
        src_size = (int(frame.shape[1]), int(frame.shape[0]))

        # Default fallback when a detection result didn't include its source
        # frame size: assume the bbox was generated from the same-sized frame
        # we just received from the pipeline.
        detection_size = detections.get("detection_frame_size") or src_size

        # Track regions already covered by labels so subsequent labels can
        # be nudged down to avoid overlapping them.
        placed: List[Tuple[int, int, int, int]] = []

        # ------------------------------------------------------------------
        # Fixed corner chrome (CAM TL, FPS TR) — small + low-key
        # ------------------------------------------------------------------
        camera_label = detections.get("camera_label") or "CAM 0"
        self._draw_chrome_corner(out, camera_label, "tl", placed)
        self._draw_fps_corner(out, detections.get("processing_fps"), out_w, placed)

        landmarks = detections.get("landmarks")
        if landmarks:
            self._draw_landmarks(
                out, landmarks, out_w, out_h,
                detection_size=detections.get("landmarks_frame_size"),
            )

        # ------------------------------------------------------------------
        # FACE — visible cyan rectangle + label directly above
        # Skipped entirely if no bbox or confidence is below the face tier.
        # ------------------------------------------------------------------
        face_bbox = detections.get("face_bbox")
        face_name = detections.get("face_name")
        face_conf = _conf_pct(detections.get("face_confidence"))
        face_src = detections.get("face_frame_size") or src_size
        if face_bbox and _meets_threshold("face", face_conf):
            scaled = scale_bbox(face_bbox, face_src, out_size)
            logger.debug(
                "Face bbox: raw=%s, face_src=%s, out_size=%s, scaled=%s",
                face_bbox, face_src, out_size, scaled,
            )
            if scaled and scaled[2] > 0 and scaled[3] > 0:
                x, y, w, h = scaled
                cv2.rectangle(out, (x, y), (x + w, y + h), _C_CYAN, 2, _LINE)
                _draw_corner_brackets(out, scaled, _C_CYAN, length=18, thickness=1)

                label = face_name or "UNKNOWN"
                if face_conf is not None and face_name:
                    label = f"{label} {face_conf}%"
                self._safe_label(out, label, (x, max(20, y - 10)),
                                 _C_TEXT, _C_CYAN, scale=_SCALE_LABEL, placed=placed)

        # ------------------------------------------------------------------
        # FIRE — pulsing red border + label, bbox-anchored only
        # ------------------------------------------------------------------
        fire = detections.get("fire")
        if fire and fire.get("detected"):
            fire_src = fire.get("frame_size") or detection_size
            fire_bbox = fire.get("bbox")
            scaled = scale_bbox(fire_bbox, fire_src, out_size)
            conf = _conf_pct(fire.get("confidence"))
            if scaled and _meets_threshold("fire", conf):
                self._draw_pulsing(out, scaled, _C_RED, out_w, out_h)
                severity = str(fire.get("severity_level", "?")).upper()
                label = f"FIRE: {severity}" + (f" {conf}%" if conf is not None else "")
                origin = self._origin_above(scaled, default=None)
                if origin is not None:
                    self._safe_label(out, label, origin, _C_TEXT, _C_RED,
                                     scale=_SCALE_LABEL, placed=placed)

        # ------------------------------------------------------------------
        # INJURY — orange brackets + label
        # ------------------------------------------------------------------
        injury = detections.get("injury")
        if injury and injury.get("detected"):
            injury_src = injury.get("frame_size") or detection_size
            injury_bbox = injury.get("bbox")
            scaled = scale_bbox(injury_bbox, injury_src, out_size)
            conf = _conf_pct(injury.get("confidence"))
            if scaled and _meets_threshold("injury", conf):
                _draw_corner_brackets(out, scaled, _C_ORANGE, length=22, thickness=2)
                posture = str(injury.get("posture_type", "?")).upper()
                label = f"INJURY: {posture}" + (f" {conf}%" if conf is not None else "")
                origin = self._origin_above(scaled, default=None)
                if origin is not None:
                    self._safe_label(out, label, origin, _C_TEXT, _C_ORANGE,
                                     scale=_SCALE_LABEL, placed=placed)

        # ------------------------------------------------------------------
        # SUSPICIOUS — yellow / orange depending on the activity tier
        # ------------------------------------------------------------------
        suspicious = detections.get("suspicious")
        if suspicious and suspicious.get("detected"):
            sus_src = suspicious.get("frame_size") or detection_size
            sus_bbox = suspicious.get("bbox")
            scaled = scale_bbox(sus_bbox, sus_src, out_size)
            activity_lower = str(suspicious.get("activity_type", "")).lower()
            activity = activity_lower.upper().replace("_", " ")
            conf = _conf_pct(suspicious.get("confidence"))

            # "suspicious_object" is the high-confidence (≥80%) tier; orange.
            # "possible_concern" is the medium tier; yellow.
            # Anything else (loitering, masked_face) defaults to yellow.
            is_high_tier = activity_lower == "suspicious_object"
            box_color = _C_ORANGE if is_high_tier else _C_YELLOW
            text_color = _C_TEXT if is_high_tier else (0, 0, 0)

            if scaled and _meets_threshold(activity_lower, conf):
                _draw_corner_brackets(out, scaled, box_color, length=22, thickness=2)
                label = f"{activity}" + (f" {conf}%" if conf is not None else "")
                origin = self._origin_above(scaled, default=None)
                if origin is not None:
                    self._safe_label(out, label, origin, text_color, box_color,
                                     scale=_SCALE_LABEL, placed=placed)

        # ------------------------------------------------------------------
        # Gesture label (top-center during VERIFYING_GESTURE)
        # ------------------------------------------------------------------
        gesture_label = detections.get("gesture_label")
        gesture_conf = detections.get("gesture_confidence")
        if gesture_label and state == "VERIFYING_GESTURE":
            g_text = gesture_label.upper()
            if gesture_conf is not None:
                g_pct = _conf_pct(gesture_conf)
                if g_pct is not None:
                    g_text = f"{g_text} ({g_pct}%)"
            gx = out_w // 2 - 60
            self._safe_label(out, g_text, (max(4, gx), 50),
                             _C_TEXT, _C_ORANGE, scale=_SCALE_LABEL, placed=placed)

        # ------------------------------------------------------------------
        # SOS gesture progress (fixed bottom-center)
        # ------------------------------------------------------------------
        sos = detections.get("sos_progress")
        if sos:
            self._draw_sos_progress(out, sos, out_w, out_h, placed)

        # ------------------------------------------------------------------
        # Bottom-left chrome: state pill above the timestamp.
        # ------------------------------------------------------------------
        ts_rect = self._draw_timestamp(out, out_h, placed)
        display_state = self._display_state(state, detections)
        self._draw_state_pill_bl(out, display_state, ts_rect, placed)

        return out

    # ------------------------------------------------------------------
    # Fixed-position elements
    # ------------------------------------------------------------------

    def _draw_chrome_corner(
        self,
        frame: np.ndarray,
        text: str,
        anchor: str,
        placed: List[Tuple[int, int, int, int]],
    ) -> None:
        """Small low-key label in a top corner ('tl' or 'tr')."""
        scale = _SCALE_CHROME
        pad = 4
        w_label, h_label, baseline = _label_size(text, scale, pad)
        h_frame, w_frame = frame.shape[:2]
        if anchor == "tr":
            x = w_frame - w_label - 8
        else:
            x = 8
        y = h_label + 4

        rect = (x, y - h_label + baseline, x + w_label, y + baseline)
        _alpha_rect(frame, (rect[0], rect[1]), (rect[2], rect[3]),
                    _C_BG, alpha=_ALPHA)
        cv2.putText(frame, text, (x + pad, y - 2),
                    _FONT, scale, _C_TEXT, 1, _LINE)
        placed.append(rect)

    def _draw_fps_corner(
        self, frame: np.ndarray, fps: Any, width: int,
        placed: List[Tuple[int, int, int, int]],
    ) -> None:
        try:
            fps_val = float(fps) if fps is not None else 0.0
        except (TypeError, ValueError):
            fps_val = 0.0

        text_color = _C_RED_TEXT if 0 < fps_val < _LOW_FPS_THRESHOLD else _C_TEXT
        text = f"{fps_val:.1f} FPS"
        scale = _SCALE_CHROME
        pad = 4
        w_label, h_label, baseline = _label_size(text, scale, pad)
        x = width - w_label - 8
        y = h_label + 4

        rect = (x, y - h_label + baseline, x + w_label, y + baseline)
        _alpha_rect(frame, (rect[0], rect[1]), (rect[2], rect[3]),
                    _C_BG, alpha=_ALPHA)
        cv2.putText(frame, text, (x + pad, y - 2),
                    _FONT, scale, text_color, 1, _LINE)
        placed.append(rect)

    def _display_state(self, state: str, detections: Dict[str, Any]) -> str:
        if state == "ACTIVE_DETECTION":
            has_threat = any(
                isinstance(detections.get(k), dict) and detections[k].get("detected")
                for k in ("fire", "injury", "suspicious")
            )
            return "THREAT DETECTED" if has_threat else "MONITORING"
        return state

    def _draw_state_pill_bl(
        self, frame: np.ndarray, state: str,
        ts_rect: Optional[Tuple[int, int, int, int]],
        placed: List[Tuple[int, int, int, int]],
    ) -> None:
        """Compact state pill at the bottom-left, just above the timestamp."""
        color = _STATE_COLORS.get(state)
        if color is None:
            if state == "MONITORING":
                color = _C_GREEN
            elif state == "THREAT DETECTED":
                color = _C_RED
            else:
                color = _C_TEXT
        scale = _SCALE_STATE
        pad = 5
        w_label, h_label, baseline = _label_size(state, scale, pad)
        x = 6
        # Sit just above the timestamp if it was drawn, otherwise pin to
        # the bottom edge.
        h_frame = frame.shape[0]
        bottom_y = (ts_rect[1] - 4) if ts_rect else (h_frame - 6)
        y = bottom_y - baseline

        rect = (x, y - h_label + baseline, x + w_label, y + baseline)
        _alpha_rect(frame, (rect[0], rect[1]), (rect[2], rect[3]),
                    color, alpha=_ALPHA)
        cv2.putText(frame, state, (x + pad, y - 2),
                    _FONT, scale, _C_TEXT, 1, _LINE)
        placed.append(rect)

    def _draw_timestamp(
        self, frame: np.ndarray, height: int,
        placed: List[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        ts_text = time.strftime("%H:%M:%S")
        scale = _SCALE_TIMESTAMP
        (tw, th), _ = cv2.getTextSize(ts_text, _FONT, scale, 1)
        rect = (6, height - th - 10, tw + 18, height - 4)
        _alpha_rect(frame, (rect[0], rect[1]), (rect[2], rect[3]),
                    _C_BG, alpha=_ALPHA)
        cv2.putText(frame, ts_text, (12, height - 10),
                    _FONT, scale, _C_DIM, 1, _LINE)
        placed.append(rect)
        return rect

    # ------------------------------------------------------------------
    # Overlap-aware label placement
    # ------------------------------------------------------------------

    def _safe_label(
        self,
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        fg: Tuple[int, int, int],
        bg: Tuple[int, int, int],
        scale: float,
        placed: List[Tuple[int, int, int, int]],
        pad: int = 5,
    ) -> Tuple[int, int, int, int]:
        h_frame = frame.shape[0]
        w_frame = frame.shape[1]
        w_label, h_label, baseline = _label_size(text, scale, pad)
        x, y = origin
        x = max(2, min(w_frame - w_label - 2, x))

        for _ in range(20):
            rect = (x, y - h_label + baseline, x + w_label, y + baseline)
            if not any(_rects_overlap(rect, r) for r in placed):
                break
            y += _LABEL_SHIFT_PX
            if y + baseline >= h_frame - 4:
                break

        rect = (x, y - h_label + baseline, x + w_label, y + baseline)
        _alpha_rect(frame, (rect[0], rect[1]), (rect[2], rect[3]), bg, alpha=_ALPHA)
        cv2.putText(frame, text, (x + pad, y - 2), _FONT, scale, fg, 1, _LINE)
        placed.append(rect)
        return rect

    # ------------------------------------------------------------------

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if h <= self._target_height:
            return frame.copy()
        ratio = self._target_height / float(h)
        new_w = int(round(w * ratio))
        return cv2.resize(frame, (new_w, self._target_height), interpolation=cv2.INTER_AREA)

    def _origin_above(
        self, bbox: Optional[Tuple[int, int, int, int]],
        default: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        if bbox is None:
            return default
        x, y, _, _ = bbox
        return (x, max(20, y - 10))

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: List,
        w: int,
        h: int,
        detection_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Project normalized landmarks (0..1) onto the display frame.

        Only draws the green hand skeleton for 21-point hand landmarks
        (from gesture_model). Pose landmarks (33 points from injury_model)
        are drawn as tiny subtle gray dots to avoid confusion.
        """
        try:
            pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
        except Exception:
            return

        if detection_size is not None and pts:
            try:
                lm0 = landmarks[0]
                if not (0.0 <= lm0[0] <= 1.0 and 0.0 <= lm0[1] <= 1.0):
                    import logging as _logging
                    _logging.getLogger(__name__).debug(
                        "Landmark out of normalized range: lm0=(%.3f, %.3f), "
                        "detection_size=%s, display_size=(%d, %d)",
                        lm0[0], lm0[1], detection_size, w, h,
                    )
            except Exception:
                pass

        if len(pts) == 21:
            for x, y in pts:
                cv2.circle(frame, (x, y), 4, _C_GREEN, -1, _LINE)
                cv2.circle(frame, (x, y), 6, (40, 80, 30), 1, _LINE)
            for i, j in _HAND_CONNECTIONS:
                if i < len(pts) and j < len(pts):
                    cv2.line(frame, pts[i], pts[j], _C_GREEN, 2, _LINE)
        else:
            for x, y in pts:
                cv2.circle(frame, (x, y), 2, (120, 120, 120), -1, _LINE)

    def _draw_pulsing(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]],
                      color: Tuple[int, int, int], w: int, h: int) -> None:
        pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 4.0)
        thickness = 2 + int(round(pulse * 2))
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, thickness, _LINE)
        else:
            cv2.rectangle(frame, (4, 4), (w - 4, h - 4), color, thickness, _LINE)

    def _draw_sos_progress(
        self, frame: np.ndarray, sos: Dict[str, Any], w: int, h: int,
        placed: List[Tuple[int, int, int, int]],
    ) -> None:
        total = int(sos.get("total", 4)) or 4
        step = int(sos.get("step", 0))
        sequence = sos.get("sequence", []) or []

        radius = 10
        spacing = 36
        bar_w = (total - 1) * spacing + radius * 2 + 80
        x0 = (w - bar_w) // 2
        y0 = h - 60

        rect = (x0 - 12, y0 - 24, x0 + bar_w + 12, y0 + 22)
        _alpha_rect(frame, (rect[0], rect[1]), (rect[2], rect[3]),
                    _C_BG, alpha=_ALPHA)
        placed.append(rect)

        text = f"SOS SEQUENCE: {step}/{total}"
        cv2.putText(frame, text, (x0, y0 - 8), _FONT, 0.5, _C_TEXT, 1, _LINE)

        cx0 = x0 + 80 + radius
        for i in range(total):
            cx = cx0 + i * spacing
            cy = y0 + 8
            done = i < step
            color = _C_GREEN if done else (90, 90, 90)
            cv2.circle(frame, (cx, cy), radius, color, -1, _LINE)
            cv2.circle(frame, (cx, cy), radius + 2, _C_TEXT, 1, _LINE)
            if i < len(sequence):
                tag = sequence[i][:1].upper() if sequence[i] else "?"
                cv2.putText(frame, tag, (cx - 4, cy + 4), _FONT, 0.4,
                            (10, 10, 10), 1, _LINE)
