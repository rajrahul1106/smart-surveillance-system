from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class InjuryModel(BaseModel):
    """Pose-based injury/collapse detection using MediaPipe Pose.

    Classifies posture as standing, sitting, lying, or collapsed based
    on shoulder-to-hip vertical ratios and landmark positions.
    """

    # If the vertical span of the torso occupies less than this fraction
    # of the body's total vertical span, the person is likely lying down.
    LYING_RATIO_THRESHOLD = 0.3

    # If shoulders are below hips (inverted torso), likely collapsed.
    COLLAPSED_INVERSION_THRESHOLD = 0.05

    # Sitting: torso takes up a moderate fraction of total height.
    SITTING_RATIO_UPPER = 0.55

    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5

    def __init__(
        self,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        super().__init__()
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._pose: Optional[Any] = None
        self._mp: Optional[Any] = None

    def _do_load(self) -> None:
        # MediaPipe 0.10.14+ removed the legacy ``mp.solutions.pose`` namespace.
        # Use the Tasks API PoseLandmarker; the .task bundle is downloaded on
        # first use and cached under data/model_artifacts/.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        from models._mediapipe_assets import pose_landmarker_path

        self._mp = mp
        model_path = pose_landmarker_path()

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self._min_detection_confidence,
            min_pose_presence_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._pose = vision.PoseLandmarker.create_from_options(options)

    def _do_predict(self, frame: Any) -> Dict[str, Any]:
        if frame is None:
            return {
                "detected": False,
                "injury_detected": False,
                "posture_type": "none",
                "confidence": 0.0,
                "pose_landmarks": None,
                "frame_size": None,
                "bbox": None,
            }

        bgr = frame
        if len(bgr.shape) == 3 and bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        h, w = bgr.shape[:2]
        frame_size = (int(w), int(h))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        results = self._pose.detect(mp_image)

        if not results.pose_landmarks:
            return {
                "detected": False,
                "injury_detected": False,
                "posture_type": "none",
                "confidence": 0.0,
                "pose_landmarks": None,
                "frame_size": frame_size,
                "bbox": None,
            }

        landmarks = [
            (lm.x, lm.y, lm.z, lm.visibility)
            for lm in results.pose_landmarks[0]
        ]

        posture, confidence = self._classify_posture(landmarks)
        injury = posture in ("lying", "collapsed")

        # Bounding box from pose-landmark extents in pixel coordinates.
        bbox = self._landmarks_bbox(landmarks, w, h)

        return {
            "detected": injury,
            "injury_detected": injury,
            "posture_type": posture,
            "confidence": round(float(confidence), 4),
            "frame_size": frame_size,
            "bbox": bbox,
            "pose_landmarks": landmarks,
        }

    def _landmarks_bbox(
        self, landmarks: List[tuple], w: int, h: int,
    ) -> Optional[tuple]:
        """Tight bbox in pixel coords around all visible landmarks."""
        xs = [lm[0] for lm in landmarks if lm[3] > 0.3]
        ys = [lm[1] for lm in landmarks if lm[3] > 0.3]
        if not xs or not ys:
            return None
        x1 = max(0, int(round(min(xs) * w)))
        y1 = max(0, int(round(min(ys) * h)))
        x2 = min(w, int(round(max(xs) * w)))
        y2 = min(h, int(round(max(ys) * h)))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2 - x1, y2 - y1)

    def _classify_posture(self, landmarks: List[tuple]) -> tuple:
        """Classify posture from MediaPipe Pose landmarks (33 points).

        Key indices:
            11, 12 = left/right shoulder
            23, 24 = left/right hip
            27, 28 = left/right ankle
        """
        l_shoulder = np.array(landmarks[11][:2])
        r_shoulder = np.array(landmarks[12][:2])
        l_hip = np.array(landmarks[23][:2])
        r_hip = np.array(landmarks[24][:2])
        l_ankle = np.array(landmarks[27][:2])
        r_ankle = np.array(landmarks[28][:2])

        mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2
        mid_ankle_y = (l_ankle[1] + r_ankle[1]) / 2

        # Total vertical span of the body
        all_y = [l_shoulder[1], r_shoulder[1], l_hip[1], r_hip[1],
                 l_ankle[1], r_ankle[1]]
        body_span = max(all_y) - min(all_y)

        if body_span < 1e-6:
            return "standing", 0.3

        torso_span = abs(mid_hip_y - mid_shoulder_y)
        torso_ratio = torso_span / body_span

        # Collapsed: shoulders below or at the same level as hips
        # (in image coords, y increases downward — so shoulder_y > hip_y
        # means shoulders are below hips)
        if mid_shoulder_y > mid_hip_y + self.COLLAPSED_INVERSION_THRESHOLD:
            confidence = min(1.0, (mid_shoulder_y - mid_hip_y) / 0.2)
            return "collapsed", confidence

        # Lying: very small torso-to-body ratio (body is horizontal)
        if torso_ratio < self.LYING_RATIO_THRESHOLD:
            confidence = max(0.5, 1.0 - torso_ratio / self.LYING_RATIO_THRESHOLD)
            return "lying", confidence

        # Sitting: moderate torso ratio (legs folded, ankles near hips)
        ankle_hip_dist = abs(mid_ankle_y - mid_hip_y)
        if torso_ratio < self.SITTING_RATIO_UPPER and ankle_hip_dist < 0.1:
            confidence = 0.6 + 0.3 * (1.0 - torso_ratio)
            return "sitting", min(1.0, confidence)

        # Standing: normal upright posture
        confidence = min(1.0, 0.5 + torso_ratio)
        return "standing", confidence

    def _do_unload(self) -> None:
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
            self._pose = None
        self._mp = None
