from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class GestureModel(BaseModel):
    """Single-frame gesture classifier using MediaPipe Hands.

    Returns palm/fist/none classification for each frame.  Sequence
    validation (palm→fist→palm→fist) is handled by the state layer,
    not here.
    """

    CONFIDENCE_THRESHOLD = 0.6
    NEAR_MISS_LOW = 0.3
    NEAR_MISS_HIGH = 0.6

    FIST_CURL_THRESHOLD = 0.15
    MIN_DETECTION_CONFIDENCE = 0.3
    MIN_TRACKING_CONFIDENCE = 0.3

    # Centre-crop ratio for distance detection.  A 60 % crop effectively
    # gives ~1.67× zoom, making hands at 2-5 m large enough for MediaPipe.
    CROP_RATIO = 0.6

    def __init__(
        self,
        near_miss_dir: str = "data/near_miss_dataset",
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        super().__init__()
        self._near_miss_dir = near_miss_dir
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._hands: Optional[Any] = None
        self._mp: Optional[Any] = None

    def _do_load(self) -> None:
        # MediaPipe 0.10.14+ removed the legacy ``mp.solutions.hands`` namespace.
        # Use the Tasks API HandLandmarker, with the .task bundle cached under
        # data/model_artifacts/.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        from models._mediapipe_assets import hand_landmarker_path

        self._mp = mp
        model_path = hand_landmarker_path()

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=self._min_detection_confidence,
            min_hand_presence_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._hands = vision.HandLandmarker.create_from_options(options)

        if os.path.isfile(self._near_miss_dir):
            os.remove(self._near_miss_dir)
        os.makedirs(self._near_miss_dir, exist_ok=True)

    def _do_predict(self, frame: Any) -> Dict[str, Any]:
        if frame is None:
            return {"gesture": "none", "confidence": 0.0, "landmarks": None,
                    "frame_size": None}

        bgr = frame
        if len(bgr.shape) == 3 and bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

        h, w = bgr.shape[:2]
        frame_size = (int(w), int(h))

        # Centre-crop + upscale so distant hands appear larger.
        zoomed = self._prepare_frame_for_distance(bgr)

        rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        results = self._hands.detect(mp_image)

        if not results.hand_landmarks:
            return {"gesture": "none", "confidence": 0.0, "landmarks": None,
                    "frame_size": frame_size}

        hand = results.hand_landmarks[0]
        # Raw landmarks in the cropped image's normalised [0,1] space.
        raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand]

        if logger.isEnabledFor(logging.DEBUG):
            lm0 = raw_landmarks[0]
            logger.debug(
                "GestureModel: frame_shape=%s, wrist normalized=(%.3f, %.3f, %.3f)",
                rgb.shape, lm0[0], lm0[1], lm0[2],
            )

        # Classification uses raw landmarks — curl ratios are scale-invariant.
        gesture, confidence = self._classify_gesture(raw_landmarks)

        # Remap landmarks back to the original (uncropped) normalised space
        # so the annotator draws them at the correct screen position.
        landmarks = self._remap_landmarks(raw_landmarks)

        if self.NEAR_MISS_LOW <= confidence < self.NEAR_MISS_HIGH:
            self._save_near_miss(frame)

        return {
            "gesture": gesture,
            "confidence": confidence,
            "landmarks": landmarks,
            "frame_size": frame_size,
        }

    # ------------------------------------------------------------------
    # Distance helpers (centre-crop + upscale)
    # ------------------------------------------------------------------

    def _prepare_frame_for_distance(self, frame: np.ndarray) -> np.ndarray:
        """Centre-crop *frame* by ``CROP_RATIO`` and upscale back.

        For a 320×240 frame with CROP_RATIO=0.6 the centre 192×144 region
        is extracted and resized to 320×240, giving an effective ~1.67× zoom.
        """
        h, w = frame.shape[:2]
        cr = self.CROP_RATIO
        cw, ch = int(w * cr), int(h * cr)
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        crop = frame[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def _remap_landmarks(
        self, landmarks: List[tuple],
    ) -> List[tuple]:
        """Map normalised landmarks from the cropped space to the original.

        Each coordinate is transformed:  ``orig = lm * CROP_RATIO + (1 - CROP_RATIO) / 2``
        so that the annotator renders them at the correct position on the
        full (uncropped) display frame.
        """
        cr = self.CROP_RATIO
        offset = (1.0 - cr) / 2.0
        return [
            (lm[0] * cr + offset, lm[1] * cr + offset, lm[2])
            for lm in landmarks
        ]

    def _classify_gesture(self, landmarks: List[tuple]) -> tuple:
        """Determine palm vs fist from 21 hand landmarks.

        Strategy: measure how curled the fingers are by comparing each
        fingertip's distance to the wrist against the corresponding MCP
        joint's distance to the wrist.  If all four fingers are curled
        inward the hand is a fist; if all are extended it is a palm.
        """
        wrist = np.array(landmarks[0][:2])

        # Fingertip and MCP indices for index, middle, ring, pinky
        tip_ids = [8, 12, 16, 20]
        mcp_ids = [5, 9, 13, 17]

        curl_ratios: List[float] = []
        for tip_id, mcp_id in zip(tip_ids, mcp_ids):
            tip = np.array(landmarks[tip_id][:2])
            mcp = np.array(landmarks[mcp_id][:2])
            tip_dist = float(np.linalg.norm(tip - wrist))
            mcp_dist = float(np.linalg.norm(mcp - wrist))
            if mcp_dist < 1e-6:
                curl_ratios.append(1.0)
            else:
                curl_ratios.append(tip_dist / mcp_dist)

        avg_curl = float(np.mean(curl_ratios))

        if avg_curl < (1.0 + self.FIST_CURL_THRESHOLD):
            # Tips are close to or behind MCPs → fist
            confidence = max(0.0, min(1.0, 1.0 - (avg_curl - 0.6) / 0.5))
            return "fist", confidence
        else:
            # Tips are well past MCPs → open palm
            confidence = max(0.0, min(1.0, (avg_curl - 1.0) / 0.6))
            return "palm", confidence

    def _save_near_miss(self, frame: Any) -> None:
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int(time.time() * 1000) % 1000
            filename = f"near_miss_{ts}_{ms:03d}.jpg"
            path = os.path.join(self._near_miss_dir, filename)
            cv2.imwrite(path, frame)
            logger.debug("Near-miss frame saved: %s", path)
        except Exception:
            logger.exception("Failed to save near-miss frame")

    def _do_unload(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None
        self._mp = None
