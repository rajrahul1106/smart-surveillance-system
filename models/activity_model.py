from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ActivityModel(BaseModel):
    """Suspicious activity detection.

    - Masked faces:    Haar cascade + lower-half texture/edge analysis
    - Weapon-like:     elongated foreground contours (MOG2 background subtraction)
    - Loitering:       a person whose center-of-mass barely moves over a
                       configurable wall-clock duration (default 30 s).
    """

    HOG_WIN_SIZE = (64, 128)
    HOG_BLOCK_SIZE = (16, 16)
    HOG_BLOCK_STRIDE = (8, 8)
    HOG_CELL_SIZE = (8, 8)
    HOG_NBINS = 9

    # Elongated object detection (formerly "weapon-like")
    ELONGATION_RATIO = 4.0
    MIN_ELONGATED_AREA = 500
    # Confidence gates for the suspicious-object pipeline:
    #   conf >= SUSPICIOUS_OBJECT_THRESHOLD  → activity_type="suspicious_object"
    #   POSSIBLE_CONCERN_THRESHOLD <= conf <  threshold → "possible_concern"
    #   conf <  POSSIBLE_CONCERN_THRESHOLD → suppressed entirely
    SUSPICIOUS_OBJECT_THRESHOLD = 0.80
    POSSIBLE_CONCERN_THRESHOLD = 0.60
    # Object must persist for this many consecutive frames before reporting.
    SUSPICIOUS_OBJECT_MIN_CONSECUTIVE = 3

    # Face/mask detection
    MIN_FACE_SIZE = (30, 30)
    SCALE_FACTOR = 1.1
    MIN_NEIGHBORS = 4

    # Loitering
    LOITER_THRESHOLD_SECONDS_DEFAULT = 30.0
    LOITER_MOVEMENT_PIXELS_DEFAULT = 20.0
    LOITER_MIN_PERSON_AREA = 1500  # need a real-sized blob to count

    def __init__(
        self,
        loitering_threshold_seconds: float = LOITER_THRESHOLD_SECONDS_DEFAULT,
        loitering_movement_pixels: float = LOITER_MOVEMENT_PIXELS_DEFAULT,
    ) -> None:
        super().__init__()
        self._loiter_threshold_s = float(loitering_threshold_seconds)
        self._loiter_movement_px = float(loitering_movement_pixels)

        self._hog: Optional[cv2.HOGDescriptor] = None
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._bg_subtractor: Optional[Any] = None
        self._prev_gray: Optional[np.ndarray] = None

        # Loitering state
        self._anchor_centroid: Optional[Tuple[float, float]] = None
        self._anchor_time: float = 0.0  # monotonic time the anchor was set
        self._last_centroid: Optional[Tuple[float, float]] = None

        # Suspicious-object temporal validation
        self._suspicious_obj_count: int = 0

    def _do_load(self) -> None:
        self._hog = cv2.HOGDescriptor(
            self.HOG_WIN_SIZE, self.HOG_BLOCK_SIZE,
            self.HOG_BLOCK_STRIDE, self.HOG_CELL_SIZE, self.HOG_NBINS,
        )
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=40, detectShadows=False,
        )

        self._prev_gray = None
        self._anchor_centroid = None
        self._anchor_time = 0.0
        self._last_centroid = None
        self._suspicious_obj_count = 0

    def _do_predict(self, frame: Any) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        frame_size = (int(w), int(h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Always update background model so it tracks lighting drift
        fg_mask = self._bg_subtractor.apply(gray)

        masked = self._detect_masked_face(frame, gray)
        if masked["detected"]:
            self._reset_loiter()
            self._suspicious_obj_count = 0
            return _result("masked_face", masked, frame_size)

        suspicious_obj = self._detect_suspicious_object(fg_mask)
        if suspicious_obj is not None:
            self._reset_loiter()
            return _result(suspicious_obj["activity_type"], suspicious_obj, frame_size)

        loiter = self._detect_loitering(frame, fg_mask)
        if loiter["detected"]:
            return _result("loitering", loiter, frame_size)

        self._prev_gray = gray.copy()
        return {
            "detected": False,
            "suspicious": False,
            "activity_type": "none",
            "confidence": 0.0,
            "bbox": None,
            "frame_size": frame_size,
        }

    # ------------------------------------------------------------------
    # Masked-face detection
    # ------------------------------------------------------------------

    def _detect_masked_face(self, frame: Any, gray: np.ndarray) -> Dict[str, Any]:
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=self.SCALE_FACTOR,
            minNeighbors=self.MIN_NEIGHBORS, minSize=self.MIN_FACE_SIZE,
        )
        for (x, y, w, h) in faces:
            lower_face = gray[y + h // 2 : y + h, x : x + w]
            if lower_face.size == 0:
                continue
            variance = float(np.var(lower_face))
            edge_density = float(np.mean(cv2.Canny(lower_face, 50, 150)))
            if variance < 400 and edge_density < 15:
                confidence = max(0.5, 1.0 - variance / 800)
                return {
                    "detected": True,
                    "confidence": round(confidence, 4),
                    "bbox": (int(x), int(y), int(w), int(h)),
                }
        return {"detected": False, "confidence": 0.0, "bbox": None}

    # ------------------------------------------------------------------
    # Suspicious object (formerly "weapon_like_object") detection
    # ------------------------------------------------------------------

    def _detect_suspicious_object(
        self, fg_mask: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Detect elongated foreground objects.

        The HOG+SVM-style "weapon" classifier was prone to confusing fingers
        and hand edges with weapons. We now apply two safeguards:

        1. **Confidence tiers**: only call it ``suspicious_object`` above 80%;
           60-80% is reported as ``possible_concern``; anything weaker is
           dropped entirely.
        2. **Temporal validation**: a candidate object must be seen for 3
           consecutive frames before any suspicious result is reported.
        """
        candidate = self._best_elongated_candidate(fg_mask)
        if candidate is None:
            self._suspicious_obj_count = 0
            return None

        confidence = candidate["confidence"]
        # Anything below the "possible_concern" floor is treated as no
        # detection at all and resets the temporal counter.
        if confidence < self.POSSIBLE_CONCERN_THRESHOLD:
            self._suspicious_obj_count = 0
            return None

        self._suspicious_obj_count += 1
        if self._suspicious_obj_count < self.SUSPICIOUS_OBJECT_MIN_CONSECUTIVE:
            return None

        if confidence >= self.SUSPICIOUS_OBJECT_THRESHOLD:
            activity = "suspicious_object"
        else:
            activity = "possible_concern"

        return {
            "detected": True,
            "activity_type": activity,
            "confidence": round(float(confidence), 4),
            "bbox": candidate["bbox"],
            "consecutive_frames": self._suspicious_obj_count,
        }

    def _best_elongated_candidate(
        self, fg_mask: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Return the highest-confidence elongated foreground contour, if any."""
        _, mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[Dict[str, Any]] = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_ELONGATED_AREA:
                continue
            rect = cv2.minAreaRect(cnt)
            (_, (rw, rh), _) = rect
            if rw < 1 or rh < 1:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect < self.ELONGATION_RATIO:
                continue
            confidence = min(1.0, aspect / 8.0)
            if best is None or confidence > best["confidence"]:
                x, y, w, h = cv2.boundingRect(cnt)
                best = {
                    "confidence": float(confidence),
                    "bbox": (int(x), int(y), int(w), int(h)),
                }
        return best

    # ------------------------------------------------------------------
    # Loitering — wall-clock + centroid tracking
    # ------------------------------------------------------------------

    def _detect_loitering(self, frame: np.ndarray, fg_mask: np.ndarray) -> Dict[str, Any]:
        person_bbox = self._find_person_bbox(frame, fg_mask)
        if person_bbox is None:
            # No person visible — clear the anchor so a future appearance starts fresh.
            self._reset_loiter()
            return {"detected": False, "confidence": 0.0, "bbox": None}

        cx = person_bbox[0] + person_bbox[2] / 2.0
        cy = person_bbox[1] + person_bbox[3] / 2.0
        centroid = (cx, cy)
        self._last_centroid = centroid
        now = time.monotonic()

        if self._anchor_centroid is None:
            self._anchor_centroid = centroid
            self._anchor_time = now
            return {"detected": False, "confidence": 0.0, "bbox": person_bbox}

        ax, ay = self._anchor_centroid
        movement = float(np.hypot(cx - ax, cy - ay))

        if movement > self._loiter_movement_px:
            # Person moved enough — restart the timer at the new position.
            self._anchor_centroid = centroid
            self._anchor_time = now
            return {"detected": False, "confidence": 0.0, "bbox": person_bbox}

        elapsed = now - self._anchor_time
        if elapsed >= self._loiter_threshold_s:
            return {
                "detected": True,
                "confidence": round(
                    _loiter_confidence(elapsed, self._loiter_threshold_s), 4,
                ),
                "bbox": person_bbox,
                "duration_seconds": round(elapsed, 2),
            }

        return {"detected": False, "confidence": 0.0, "bbox": person_bbox}

    def _find_person_bbox(
        self, frame: np.ndarray, fg_mask: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return (x, y, w, h) of the most-likely person, or None.

        Strategy: try HOG people detector first (specific but slower); if it
        finds nothing usable, fall back to the largest sufficiently-sized
        foreground blob from the MOG2 background subtractor.
        """
        try:
            rects, _ = self._hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(8, 8), scale=1.05,
            )
        except cv2.error:
            rects = []

        if len(rects):
            x, y, w, h = max(
                rects, key=lambda r: r[2] * r[3],
            )
            return (int(x), int(y), int(w), int(h))

        # Foreground-mask fallback
        _, mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.LOITER_MIN_PERSON_AREA:
            return None
        x, y, w, h = cv2.boundingRect(largest)
        return (int(x), int(y), int(w), int(h))

    def _reset_loiter(self) -> None:
        self._anchor_centroid = None
        self._anchor_time = 0.0
        self._last_centroid = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _do_unload(self) -> None:
        self._hog = None
        self._face_cascade = None
        self._bg_subtractor = None
        self._prev_gray = None
        self._suspicious_obj_count = 0
        self._reset_loiter()


def _loiter_confidence(elapsed_seconds: float, threshold_seconds: float) -> float:
    """Confidence as a function of how long the person has lingered.

    Scaled to the configured threshold so the spec values
    (1x threshold → 0.50, 2x → 0.75, 3x+ → 0.90, then asymptote to 0.99)
    apply at any threshold setting.

    With the default 30s threshold this matches the spec exactly:
    30s → 0.50, 60s → 0.75, 90s → 0.90, longer → up to 0.99.
    """
    if threshold_seconds <= 0:
        return 0.0
    if elapsed_seconds < threshold_seconds:
        return 0.0
    ratio = elapsed_seconds / float(threshold_seconds)
    if ratio < 2.0:
        # 1x → 2x threshold: linear 0.50 → 0.75
        return 0.50 + (ratio - 1.0) * 0.25
    if ratio < 3.0:
        # 2x → 3x threshold: linear 0.75 → 0.90
        return 0.75 + (ratio - 2.0) * 0.15
    # Beyond 3x: asymptote toward 0.99
    return min(0.99, 0.90 + (ratio - 3.0) / 20.0 * 0.09)


def _result(activity: str, hit: Dict[str, Any], frame_size: Tuple[int, int]) -> Dict[str, Any]:
    out = {
        "detected": True,
        "suspicious": True,
        "activity_type": activity,
        "confidence": hit["confidence"],
        "bbox": hit.get("bbox"),
        "frame_size": frame_size,
    }
    if "duration_seconds" in hit:
        out["duration_seconds"] = hit["duration_seconds"]
    return out
