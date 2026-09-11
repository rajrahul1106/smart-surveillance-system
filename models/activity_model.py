from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ActivityModel(BaseModel):
    """MobileNetV3-Small ONNX classifier for suspicious-activity detection.

    Replaces the previous HOG+SVM approach.  Classifies each sampled frame as
    one of {normal, robbery, violence} and confirms an alert only after a
    leaky-accumulator temporal filter crosses its threshold, mirroring the
    fire model's stability strategy.

    The classifier looks at the whole frame, so results never carry a bbox.
    The ONNX graph outputs RAW LOGITS — softmax is applied once, in
    ``_do_predict``.  If the model file is absent the detector degrades
    gracefully (every frame returns *not detected*).
    """

    # -- Model artefact paths -----------------------------------------------
    MODEL_PATH = os.path.join(
        "data", "model_artifacts", "models", "sentinel_activity_mnv3.onnx",
    )
    LABEL_MAP_PATH = os.path.join(
        "data", "model_artifacts", "models", "activity_label_map.json",
    )

    # -- Fallback defaults if the label map is missing ----------------------
    DEFAULT_CLASSES = ["normal", "robbery", "violence"]
    DEFAULT_IMG_SIZE = 224
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]

    # A suspicious class must reach this probability for a full boost
    CONFIDENCE_THRESHOLD = 0.60

    # -- Leaky accumulator (same concept as FireModel) ----------------------
    BOOST_RATE = 1.0        # suspicious class on top, >= threshold
    PARTIAL_BOOST = 0.4     # suspicious class on top, below threshold
    DECAY_RATE = 0.3        # normal class on top
    SCORE_THRESHOLD = 3.0   # activity_score must reach this to confirm
    SCORE_CAP = 10.0        # upper clamp on activity_score

    # Every class except this one counts as suspicious
    NORMAL_CLASS = "normal"

    DEBUG_LOG_EVERY_N_FRAMES = 10

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def __init__(
        self,
        model_path: Optional[str] = None,
        label_map_path: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        score_threshold: float = SCORE_THRESHOLD,
        score_cap: float = SCORE_CAP,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # main.py still passes the HOG-era loitering_* settings; accept and
        # ignore them so construction keeps working.
        if kwargs:
            logger.debug("ActivityModel ignoring unused kwargs: %s", sorted(kwargs))

        self._model_path = model_path or self.MODEL_PATH
        self._label_map_path = label_map_path or self.LABEL_MAP_PATH
        self._conf_threshold = confidence_threshold
        self._score_threshold = score_threshold
        self._score_cap = score_cap

        # Runtime state
        self._session: Any = None            # ort.InferenceSession
        self._input_name: str = "input"
        self._classes: List[str] = list(self.DEFAULT_CLASSES)
        self._img_size: int = self.DEFAULT_IMG_SIZE
        self._mean = np.array(self.DEFAULT_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(self.DEFAULT_STD, dtype=np.float32).reshape(3, 1, 1)

        self._activity_score: float = 0.0
        self._frame_count: int = 0
        # Suspicious class that last raised the score, and its probability.
        # Reported while a confirmed alert decays, so it never reads "normal".
        self._active_class: Optional[str] = None
        self._active_conf: float = 0.0

    # -- BaseModel hooks ----------------------------------------------------

    def _do_load(self) -> None:
        self._reset_temporal_state()
        self._load_label_map()

        if not os.path.isfile(self._model_path):
            logger.error(
                "Activity ONNX model not found at %s - activity detection disabled.  "
                "Place sentinel_activity_mnv3.onnx and activity_label_map.json "
                "in data/model_artifacts/models/.",
                self._model_path,
            )
            self._session = None
            return

        import onnxruntime as ort  # noqa: E402 – deferred to avoid import cost

        self._session = ort.InferenceSession(
            self._model_path,
            providers=["CPUExecutionProvider"],
        )

        # Read the real input name and spatial size from the ONNX graph so
        # preprocessing always matches the model.  Shape is [batch, 3, S, S].
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        shape = model_input.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            if shape[2] != self._img_size:
                logger.info(
                    "Overriding activity img_size %d -> %d (from ONNX)",
                    self._img_size, shape[2],
                )
                self._img_size = int(shape[2])

        logger.info(
            "ActivityModel loaded - MobileNetV3 ONNX, classes=%s, size=%d, "
            "conf=%.2f, trigger=%.1f, input=%s",
            self._classes, self._img_size, self._conf_threshold,
            self._score_threshold, self._input_name,
        )

    def _do_unload(self) -> None:
        self._reset_temporal_state()
        if self._session is not None:
            del self._session
            self._session = None

    def _load_label_map(self) -> None:
        """Read classes, input size and normalisation from the label map."""
        if not os.path.isfile(self._label_map_path):
            logger.warning(
                "Activity label map not found at %s; using defaults",
                self._label_map_path,
            )
            return
        try:
            with open(self._label_map_path, "r") as f:
                meta = json.load(f)
            classes = list(meta.get("classes", self.DEFAULT_CLASSES))
            img_size = int(meta.get("img_size", self.DEFAULT_IMG_SIZE))
            mean = np.array(
                meta.get("mean", self.DEFAULT_MEAN), dtype=np.float32,
            ).reshape(3, 1, 1)
            std = np.array(
                meta.get("std", self.DEFAULT_STD), dtype=np.float32,
            ).reshape(3, 1, 1)
        except Exception as exc:
            logger.warning("Activity label map unreadable (%s); using defaults", exc)
            return
        # Assign only once everything parsed, so a bad file can't half-apply.
        self._classes, self._img_size = classes, img_size
        self._mean, self._std = mean, std

    def _reset_temporal_state(self) -> None:
        self._activity_score = 0.0
        self._frame_count = 0
        self._active_class = None
        self._active_conf = 0.0

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def _do_predict(self, frame: Any) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        frame_size = (int(w), int(h))
        self._frame_count += 1

        # If the ONNX model was never loaded, return a safe "nothing detected".
        if self._session is None:
            return self._empty_result(frame_size)

        blob = self._preprocess(frame)
        logits = self._session.run(None, {self._input_name: blob})[0]

        # The model returns RAW LOGITS — softmax is applied here, exactly once.
        probs = self._softmax(np.asarray(logits, dtype=np.float32).reshape(-1))

        top_idx = int(np.argmax(probs))
        top_class = (
            self._classes[top_idx] if top_idx < len(self._classes) else "unknown"
        )
        top_prob = float(probs[top_idx])
        is_suspicious = top_class != self.NORMAL_CLASS

        # ----- Leaky accumulator update ------------------------------------
        if is_suspicious and top_prob >= self._conf_threshold:
            self._activity_score = min(
                self._activity_score + self.BOOST_RATE, self._score_cap,
            )
        elif is_suspicious:
            self._activity_score = min(
                self._activity_score + self.PARTIAL_BOOST, self._score_cap,
            )
        else:
            self._activity_score = max(self._activity_score - self.DECAY_RATE, 0.0)

        if is_suspicious:
            self._active_class = top_class
            self._active_conf = top_prob

        confirmed = self._activity_score >= self._score_threshold

        # A confirmed alert is held through "normal" frames while the score
        # decays; keep reporting the suspicious class that raised it.
        if confirmed:
            activity_type = self._active_class
            confidence = round(min(99.0, self._active_conf * 100.0), 2)
        else:
            activity_type = "none"
            confidence = round(min(99.0, top_prob * 100.0), 2)

        # Per-class probability map (current frame) for the dashboard / logs
        class_probs = {
            self._classes[i]: round(float(probs[i]) * 100.0, 2)
            for i in range(min(len(self._classes), probs.shape[0]))
        }

        # ----- Debug logging -----------------------------------------------
        if self._frame_count % self.DEBUG_LOG_EVERY_N_FRAMES == 0:
            logger.info(
                "Activity debug: top=%s (%.1f%%), score=%.1f/%.1f, "
                "confirmed=%s, probs=%s",
                top_class, top_prob * 100.0, self._activity_score,
                self._score_threshold, confirmed, class_probs,
            )

        return {
            "detected": bool(confirmed),
            "activity_detected": bool(confirmed),
            "suspicious": bool(confirmed),
            "activity_type": activity_type,
            "severity_level": activity_type,
            "confidence": confidence,
            "confidence_pct": confidence,
            "bbox": None,              # whole-frame classifier, no box
            "frame_size": frame_size,
            "activity_score": round(self._activity_score, 2),
            "class_probabilities": class_probs,
        }

    # -----------------------------------------------------------------------
    # Pre / post-processing
    # -----------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize to img_size, BGR->RGB, HWC->CHW, ImageNet-normalise, add batch.

        Plain resize (no letterbox) — this is a classifier, not a detector.
        """
        img = cv2.resize(
            frame, (self._img_size, self._img_size), interpolation=cv2.INTER_LINEAR,
        )
        img = img[:, :, ::-1].transpose(2, 0, 1)        # BGR -> RGB, HWC -> CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = (img - self._mean) / self._std              # ImageNet normalisation
        return img[np.newaxis, ...]                       # (1, 3, S, S)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _empty_result(frame_size: Tuple[int, int]) -> Dict[str, Any]:
        return {
            "detected": False,
            "activity_detected": False,
            "suspicious": False,
            "activity_type": "none",
            "severity_level": "none",
            "confidence": 0.0,
            "confidence_pct": 0.0,
            "bbox": None,
            "frame_size": frame_size,
            "activity_score": 0.0,
            "class_probabilities": {},
        }
