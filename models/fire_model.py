from __future__ import annotations

import ast
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.base_model import BaseModel
from models.ort_options import session_options

logger = logging.getLogger(__name__)


class FireModel(BaseModel):
    """YOLO11s fire and smoke detection via ONNX Runtime.

    A YOLO11s detector trained on D-Fire (including ~9,800 fire-free images
    such as lamps and sun glare) plus two Roboflow fire/smoke sets, exported
    as a static 480×480 ONNX graph.  The leaky-accumulator temporal
    verification means a single stray frame cannot trigger an alert.

    Input size, class names and architecture are read from the ONNX file at
    load time; the graph is authoritative over config and the labels JSON.
    If the model file is absent the detector degrades gracefully (every
    frame returns *not detected*).

    Severity mapping:
        - ``confidence >= 60 %``   -> area-based severity
        - ``25 % <= confidence < 60 %`` -> ``"simulated"``
        - below 25 %               -> not reported
    """

    # -- Model artefact paths -----------------------------------------------
    MODEL_PATH = os.path.join(
        "data", "model_artifacts", "models", "fire_yolo11s_480.onnx",
    )
    LABELS_PATH = os.path.join(
        "data", "model_artifacts", "models", "fire_yolo11s_labels.json",
    )

    # -- YOLO inference parameters ------------------------------------------
    INPUT_SIZE = 480
    CONFIDENCE_THRESHOLD = 0.35
    IOU_THRESHOLD = 0.45
    # ONNX Runtime intra-op threads (0 = runtime default).  Capped via config
    # so fire and face presence inference don't fight over the same cores.
    INTRA_OP_THREADS = 0

    # -- Leaky accumulator (same concept as before) -------------------------
    BOOST_RATE = 1.0        # fire detection  -> strong boost
    PARTIAL_BOOST = 0.4     # smoke-only      -> partial boost
    DECAY_RATE = 0.3        # nothing         -> decay
    SCORE_THRESHOLD = 3.0   # fire_score must reach this to confirm
    SCORE_CAP = 10.0        # upper clamp on fire_score

    # -- Severity thresholds (unchanged) ------------------------------------
    SEVERITY_THRESHOLDS: Dict[str, float] = {
        "uncontrollable": 0.30,
        "hazardous": 0.15,
        "controllable": 0.05,
        "small": 0.005,
    }
    REAL_FIRE_CONFIDENCE = 60.0

    DEBUG_LOG_EVERY_N_FRAMES = 10

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        input_size: int = INPUT_SIZE,
        score_threshold: float = SCORE_THRESHOLD,
        score_cap: float = SCORE_CAP,
        labels_path: Optional[str] = None,
        intra_op_threads: int = INTRA_OP_THREADS,
    ) -> None:
        super().__init__()
        self._model_path = model_path or self.MODEL_PATH
        self._labels_path = labels_path or self.LABELS_PATH
        self._intra_op_threads = intra_op_threads
        self._conf_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._input_size = input_size
        self._score_threshold = score_threshold
        self._score_cap = score_cap

        # Runtime state
        self._session: Any = None            # ort.InferenceSession
        self._input_name: str = ""
        self._class_names: Dict[int, str] = {0: "fire", 1: "smoke"}
        self._fire_score: float = 0.0
        self._frame_count: int = 0

    # -- BaseModel hooks ----------------------------------------------------

    def _do_load(self) -> None:
        self._fire_score = 0.0
        self._frame_count = 0

        if not os.path.isfile(self._model_path):
            logger.error(
                "Fire ONNX model not found at %s — detection disabled.  "
                "Place fire_yolo11s_480.onnx and fire_yolo11s_labels.json "
                "in data/model_artifacts/models/.",
                self._model_path,
            )
            self._session = None
            return

        import onnxruntime as ort  # noqa: E402 – deferred to avoid import cost

        self._session = ort.InferenceSession(
            self._model_path,
            sess_options=session_options(self._intra_op_threads),
            providers=["CPUExecutionProvider"],
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name

        # Read actual input dimensions from the ONNX graph so preprocessing
        # always matches the model, regardless of config or class defaults.
        # Shape is typically [1, 3, H, W] or ['batch', 3, H, W].
        input_shape = model_input.shape
        if len(input_shape) == 4:
            model_h = input_shape[2]
            model_w = input_shape[3]
            if isinstance(model_h, int) and isinstance(model_w, int):
                if model_h != self._input_size:
                    logger.info(
                        "Overriding input_size %d → %d to match ONNX model",
                        self._input_size, model_h,
                    )
                self._input_size = model_h

        # Read class names embedded in the model metadata (Ultralytics
        # stores them as ``{"0": "fire", "1": "smoke", ...}``).
        metadata = self._session.get_modelmeta().custom_metadata_map or {}
        if "names" in metadata:
            try:
                raw = ast.literal_eval(metadata["names"])
                self._class_names = {int(k): v for k, v in raw.items()}
            except Exception:
                pass  # keep defaults

        names_list = [self._class_names.get(i, "?") for i in sorted(self._class_names)]
        self._check_labels_file(names_list)
        logger.info(
            "FireModel loaded - %s ONNX, input=[%s], classes=[%s], "
            "conf=%.2f, iou=%.2f, trigger=%.1f, threads=%s",
            _architecture(metadata),
            ",".join(str(d) for d in input_shape),
            ",".join(repr(n) for n in names_list),
            self._conf_threshold, self._iou_threshold, self._score_threshold,
            self._intra_op_threads or "default",
        )

    def _check_labels_file(self, onnx_classes: List[str]) -> None:
        """Warn if the labels JSON disagrees with the ONNX metadata, which wins."""
        if not os.path.isfile(self._labels_path):
            logger.debug("No fire labels file at %s", self._labels_path)
            return
        try:
            with open(self._labels_path) as f:
                classes = json.load(f).get("classes")
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning(
                "Fire labels file %s is unreadable (%s); using the ONNX metadata",
                self._labels_path, exc,
            )
            return
        if classes is not None and list(classes) != list(onnx_classes):
            logger.warning(
                "Fire labels file %s lists classes %s but the ONNX metadata "
                "says %s; using the ONNX metadata",
                self._labels_path, classes, onnx_classes,
            )

    def _do_unload(self) -> None:
        self._fire_score = 0.0
        self._frame_count = 0
        if self._session is not None:
            del self._session
            self._session = None

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def _do_predict(self, frame: Any) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        frame_size = (int(w), int(h))
        self._frame_count += 1

        # Local reference: the COOLDOWN transition unloads from a timer
        # thread and may clear self._session while this call is running.
        session = self._session
        # If ONNX model was never loaded, return a safe "nothing detected".
        if session is None:
            return self._empty_result(frame_size)

        # Preprocess → infer → postprocess
        blob, ratio, pad = self._preprocess(frame)
        raw_output = session.run(None, {self._input_name: blob})
        detections = self._postprocess(raw_output, ratio, pad, frame.shape)

        # Separate fire vs smoke detections
        fire_dets = [d for d in detections if d["class_name"].lower() == "fire"]
        smoke_dets = [d for d in detections if d["class_name"].lower() == "smoke"]

        # ----- Leaky accumulator update ------------------------------------
        if fire_dets:
            self._fire_score = min(
                self._fire_score + self.BOOST_RATE, self._score_cap,
            )
        elif smoke_dets:
            self._fire_score = min(
                self._fire_score + self.PARTIAL_BOOST, self._score_cap,
            )
        else:
            self._fire_score = max(self._fire_score - self.DECAY_RATE, 0.0)

        confirmed = self._fire_score >= self._score_threshold

        # Best detection (highest confidence across fire + smoke)
        all_fire = fire_dets + smoke_dets
        best: Optional[Dict[str, Any]] = None
        if all_fire:
            best = max(all_fire, key=lambda d: d["confidence"])

        # Confidence in percentage (0-99)
        confidence = round(min(99.0, best["confidence"] * 100.0), 2) if best else 0.0

        # Bbox in (x, y, w, h) format for annotator compatibility
        bbox: Optional[Tuple[int, int, int, int]] = None
        if best is not None:
            x1, y1, x2, y2 = best["bbox"]
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

        fire_ratio = self._compute_fire_ratio(bbox, h * w) if bbox else 0.0
        severity = (
            self._classify_severity_for(confidence, fire_ratio)
            if confirmed else "none"
        )

        fire_type = best["class_name"] if best else None

        # ----- Debug logging -----------------------------------------------
        if self._frame_count % self.DEBUG_LOG_EVERY_N_FRAMES == 0:
            logger.info(
                "Fire debug: detections=%df/%ds, score=%.1f/%.1f, "
                "best_conf=%.1f%%, type=%s, confirmed=%s, severity=%s",
                len(fire_dets), len(smoke_dets),
                self._fire_score, self._score_threshold,
                confidence, fire_type, confirmed, severity,
            )

        return {
            "detected": bool(confirmed),
            "fire_detected": bool(confirmed),
            "severity_level": severity,
            "confidence": confidence,
            "confidence_pct": confidence,
            "bbox": bbox if confirmed else None,
            "frame_size": frame_size,
            "consecutive_frames": 0,
            "fire_score": round(self._fire_score, 2),
            "fire_type": fire_type,
        }

    # -----------------------------------------------------------------------
    # YOLO preprocessing
    # -----------------------------------------------------------------------

    def _preprocess(
        self, frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Letterbox-resize *frame* and normalise for YOLO inference.

        Returns ``(blob, ratio, (pad_w, pad_h))``.
        """
        img = frame.copy()
        oh, ow = img.shape[:2]
        r = min(self._input_size / oh, self._input_size / ow)
        new_w, new_h = int(round(ow * r)), int(round(oh * r))

        dw = (self._input_size - new_w) / 2.0
        dh = (self._input_size - new_h) / 2.0

        if (ow, oh) != (new_w, new_h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # BGR → RGB, HWC → CHW, float32 [0, 1], batch dim
        img = img[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        blob = blob[np.newaxis, ...]

        # Safety: if rounding produced a blob that doesn't match the
        # expected input size, force-resize each channel to fit.
        expected = self._input_size
        actual_h, actual_w = blob.shape[2], blob.shape[3]
        if actual_h != expected or actual_w != expected:
            logger.warning(
                "Blob size mismatch: got %dx%d, expected %dx%d — resizing",
                actual_h, actual_w, expected, expected,
            )
            fixed = np.zeros((1, 3, expected, expected), dtype=np.float32)
            for c in range(3):
                fixed[0, c] = cv2.resize(
                    blob[0, c], (expected, expected),
                    interpolation=cv2.INTER_LINEAR,
                )
            blob = fixed

        return blob, r, (dw, dh)

    # -----------------------------------------------------------------------
    # YOLO postprocessing
    # -----------------------------------------------------------------------

    def _postprocess(
        self,
        output: List[np.ndarray],
        ratio: float,
        pad: Tuple[float, float],
        original_shape: Tuple[int, ...],
    ) -> List[Dict[str, Any]]:
        """Decode YOLO ONNX output ``(1, 4 + classes, N)`` into detection dicts.

        Each dict: ``{"bbox": (x1,y1,x2,y2), "confidence": float,
                      "class_id": int, "class_name": str}``

        Coordinates are in the *original frame*'s pixel space.
        """
        preds = output[0]  # (1, 4+num_classes, N)
        if preds.ndim == 3:
            preds = preds[0]

        # Ensure shape is (N, 4+C).  Ultralytics exports (4+C, N) where
        # N >> (4+C); the first dimension equals the feature count.
        num_classes = len(self._class_names)
        num_features = 4 + num_classes
        if preds.shape[0] == num_features and preds.shape[1] != num_features:
            preds = preds.T
        elif preds.shape[0] < preds.shape[1]:
            preds = preds.T  # fallback heuristic
        boxes = preds[:, :4]                   # cx, cy, w, h
        scores = preds[:, 4:4 + num_classes]   # per-class confidences

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # Confidence filter
        keep = max_scores >= self._conf_threshold
        boxes = boxes[keep]
        max_scores = max_scores[keep]
        class_ids = class_ids[keep]

        if len(boxes) == 0:
            return []

        # cx,cy,w,h  →  x1,y1,x2,y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(), max_scores.tolist(),
            self._conf_threshold, self._iou_threshold,
        )
        if len(indices) == 0:
            return []
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        pad_w, pad_h = pad
        orig_h, orig_w = original_shape[:2]

        results: List[Dict[str, Any]] = []
        for i in indices:
            bx = xyxy[i].copy()
            # Undo padding + rescale to original resolution
            bx[0] = np.clip((bx[0] - pad_w) / ratio, 0, orig_w)
            bx[1] = np.clip((bx[1] - pad_h) / ratio, 0, orig_h)
            bx[2] = np.clip((bx[2] - pad_w) / ratio, 0, orig_w)
            bx[3] = np.clip((bx[3] - pad_h) / ratio, 0, orig_h)

            cid = int(class_ids[i])
            results.append({
                "bbox": (int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])),
                "confidence": float(max_scores[i]),
                "class_id": cid,
                "class_name": self._class_names.get(cid, "unknown"),
            })

        return results

    # -----------------------------------------------------------------------
    # Severity helpers (kept from previous implementation)
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_fire_ratio(
        bbox: Tuple[int, int, int, int], total_pixels: int,
    ) -> float:
        """Approximate fire-area ratio from the bounding box."""
        _, _, bw, bh = bbox
        if total_pixels <= 0:
            return 0.0
        return (bw * bh) / total_pixels

    def _classify_severity_for(self, confidence: float, fire_ratio: float) -> str:
        if confidence >= self.REAL_FIRE_CONFIDENCE:
            return self._classify_severity(fire_ratio)
        return "simulated"

    def _classify_severity(self, fire_ratio: float) -> str:
        for level, threshold in self.SEVERITY_THRESHOLDS.items():
            if fire_ratio >= threshold:
                return level
        return "small"

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _empty_result(
        frame_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        return {
            "detected": False,
            "fire_detected": False,
            "severity_level": "none",
            "confidence": 0.0,
            "confidence_pct": 0.0,
            "bbox": None,
            "frame_size": frame_size,
            "consecutive_frames": 0,
            "fire_score": 0.0,
            "fire_type": None,
        }


def _architecture(metadata: Dict[str, str]) -> str:
    """Model family from Ultralytics metadata, e.g. "YOLO11s" (else "YOLO")."""
    match = re.search(r"\b(YOLO[\w.-]*)", metadata.get("description", ""))
    return match.group(1) if match else "YOLO"
