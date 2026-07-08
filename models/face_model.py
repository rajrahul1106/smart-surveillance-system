"""ArcFace-based face identity model (insightface buffalo_l).

Replaces the previous dlib/face_recognition pipeline. Uses one
InsightFace ``FaceAnalysis`` app for both detection AND 512-d ArcFace
embedding extraction, with cosine-similarity matching against
enrolled encodings.

Storage format::

    data/face_encodings.pkl  →  { name: [np.ndarray(512,), ...], ... }

Predict return contract is unchanged for the rest of the pipeline:
``user_id``, ``confidence``, ``is_live``, ``authorized``/``is_authorized``,
``bbox`` (x, y, w, h tuple of ints).
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class FaceModel(BaseModel):
    """ArcFace face recognition via insightface buffalo_l (CPU)."""

    ENCODINGS_PATH_DEFAULT = "data/face_encodings.pkl"
    MODEL_ROOT_DEFAULT = "data/model_artifacts"

    # ArcFace cosine similarity authorization tiers.
    # Display percentage = cosine * 100 (e.g. 0.68 → "68%").
    #
    # > AUTHORIZE_THRESHOLD (0.60)        → is_authorized=True, "name"
    # POSSIBLE_THRESHOLD .. AUTHORIZE     → is_authorized=False, "POSSIBLE: name"
    # < POSSIBLE_THRESHOLD (0.40)         → is_authorized=False, "UNKNOWN"
    #
    # These thresholds match what an end-user reads on screen: a 60% match
    # confidence is the floor for authorisation; 40-60% surfaces the
    # candidate name as a possible match without granting access.
    AUTHORIZE_THRESHOLD = 0.60
    POSSIBLE_THRESHOLD = 0.40
    # Legacy alias kept for backward compat with callers that pass
    # match_threshold= explicitly; if provided it is treated as the
    # AUTHORIZE_THRESHOLD.
    MATCH_THRESHOLD = AUTHORIZE_THRESHOLD
    DET_SIZE = (960, 960)

    # Centre-crop ratio for distance detection.  A 70 % crop gives ~1.43×
    # zoom, making faces at 2-5 m large enough for reliable detection.
    CROP_RATIO = 0.7

    def __init__(
        self,
        encodings_path: str = ENCODINGS_PATH_DEFAULT,
        model_root: str = MODEL_ROOT_DEFAULT,
        match_threshold: float = MATCH_THRESHOLD,
        possible_threshold: float = POSSIBLE_THRESHOLD,
    ) -> None:
        super().__init__()
        self._encodings_path = encodings_path
        self._model_root = model_root
        self._authorize_threshold = match_threshold
        self._possible_threshold = possible_threshold
        # Legacy attribute name some tests may inspect.
        self._match_threshold = match_threshold

        self._app: Optional[Any] = None
        # { name: [np.ndarray(512,), ...] }
        self._known_encodings: Dict[str, List[np.ndarray]] = {}
        # EMA smoothing for identity confidence
        self._smoothed_confidence: Dict[str, float] = {}
        self._confidence_alpha: float = 0.3
        self._low_conf_streak: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _do_load(self) -> None:
        import insightface

        os.makedirs(self._model_root, exist_ok=True)

        self._app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=self._model_root,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=self.DET_SIZE)

        # Lower the face-detection threshold so smaller / distant faces
        # are picked up.  InsightFace stores it inside each detection
        # sub-model; wrap in try/except for test-mock safety.
        try:
            for m in self._app.models:
                if hasattr(m, "det_thresh"):
                    m.det_thresh = 0.3
        except Exception:
            pass

        self._load_encodings()
        logger.info(
            "FaceModel ready — %d enrolled identit%s",
            len(self._known_encodings),
            "y" if len(self._known_encodings) == 1 else "ies",
        )

    def _do_unload(self) -> None:
        self._app = None
        self._known_encodings = {}
        self._smoothed_confidence = {}
        self._low_conf_streak = 0

    def _do_predict(self, frame: Any) -> Dict[str, Any]:
        if frame is None:
            return _empty_result()

        bgr = frame
        if len(bgr.shape) == 3 and bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        bgr = bgr.astype(np.uint8)
        h, w = bgr.shape[:2]
        frame_size = (int(w), int(h))

        # Centre-crop + upscale so distant faces appear larger.
        zoomed = self._prepare_frame_for_distance(bgr)

        # insightface expects BGR (which OpenCV provides natively).
        faces = self._app.get(zoomed)
        if not faces:
            empty = _empty_result()
            empty["frame_size"] = frame_size
            return empty

        # Use the largest face by bounding-box area.
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        # Remap bbox from the cropped-then-resized space back to the
        # original frame coordinates.
        raw_bbox = face.bbox.tolist()
        raw_bbox = self._remap_bbox(raw_bbox, w, h)
        x1, y1, x2, y2 = [int(round(v)) for v in raw_bbox]
        x1 = max(0, min(x1, w)); y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w)); y2 = max(0, min(y2, h))
        bbox_xywh = (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
        logger.debug(
            "FaceModel bbox: raw=%s, clamped=(%d,%d,%d,%d), "
            "xywh=%s, input_frame=(%d,%d)",
            raw_bbox, x1, y1, x2, y2, bbox_xywh, w, h,
        )

        embedding = np.asarray(face.normed_embedding, dtype=np.float32)

        best_name, best_score = self._match(embedding)
        best_score = self._smooth_confidence(best_name, best_score)
        is_authorized = best_score > self._authorize_threshold

        # Three-tier label/identity assignment.
        if is_authorized:
            label = best_name or "UNKNOWN"
            user_id = best_name
            name_field = best_name
        elif best_score >= self._possible_threshold and best_name is not None:
            # Strong-enough match to surface the candidate name, but not
            # confident enough to authorize.
            label = f"POSSIBLE: {best_name}"
            user_id = f"POSSIBLE: {best_name}"
            name_field = label
        else:
            label = "UNKNOWN"
            user_id = None
            name_field = None

        return {
            "user_id": user_id,
            "name": name_field,
            "label": label,
            "confidence": float(best_score),
            "is_live": True,  # ArcFace + buffalo_l detector implies a real face crop
            "authorized": bool(is_authorized),
            "is_authorized": bool(is_authorized),
            "bbox": bbox_xywh,
            "frame_size": frame_size,
        }

    # ------------------------------------------------------------------
    # Distance helpers (centre-crop + upscale)
    # ------------------------------------------------------------------

    def _prepare_frame_for_distance(self, frame: np.ndarray) -> np.ndarray:
        """Centre-crop *frame* by ``CROP_RATIO`` and upscale back.

        For a 640×480 frame with CROP_RATIO=0.7 the centre 448×336 region
        is extracted and resized to 640×480, giving an effective ~1.43× zoom.
        """
        h, w = frame.shape[:2]
        cr = self.CROP_RATIO
        cw, ch = int(w * cr), int(h * cr)
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        crop = frame[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def _remap_bbox(
        self, bbox: List[float], orig_w: int, orig_h: int,
    ) -> List[float]:
        """Map a bbox from the cropped-then-resized space to original coords.

        ``orig_coord = coord * CROP_RATIO + start_offset``
        """
        cr = self.CROP_RATIO
        x_start = (orig_w - int(orig_w * cr)) / 2.0
        y_start = (orig_h - int(orig_h * cr)) / 2.0
        scale_x = int(orig_w * cr) / orig_w
        scale_y = int(orig_h * cr) / orig_h
        x1, y1, x2, y2 = bbox
        return [
            x1 * scale_x + x_start,
            y1 * scale_y + y_start,
            x2 * scale_x + x_start,
            y2 * scale_y + y_start,
        ]

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._known_encodings:
            return None, 0.0

        emb_norm = float(np.linalg.norm(embedding))
        if emb_norm < 1e-8:
            return None, 0.0

        best_name: Optional[str] = None
        best_score: float = -1.0

        for name, vectors in self._known_encodings.items():
            for known in vectors:
                known = np.asarray(known, dtype=np.float32)
                k_norm = float(np.linalg.norm(known))
                if k_norm < 1e-8:
                    continue
                # Cosine similarity. Embeddings are already L2-normalized
                # by insightface, but recompute defensively.
                score = float(np.dot(embedding, known) / (emb_norm * k_norm))
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < 0:
            best_score = 0.0
        return best_name, best_score

    def _smooth_confidence(self, name: Optional[str], raw: float) -> float:
        if raw < 0.3:
            self._low_conf_streak += 1
            if self._low_conf_streak >= 5:
                self._smoothed_confidence.clear()
                self._low_conf_streak = 0
            return raw
        self._low_conf_streak = 0
        if name is None:
            return raw
        if name not in self._smoothed_confidence:
            self._smoothed_confidence[name] = raw
        else:
            self._smoothed_confidence[name] = (
                (1.0 - self._confidence_alpha) * self._smoothed_confidence[name]
                + self._confidence_alpha * raw
            )
        return self._smoothed_confidence[name]

    # ------------------------------------------------------------------
    # Enrollment helpers (used by EnrollmentService and CLI scripts)
    # ------------------------------------------------------------------

    @property
    def app(self) -> Any:
        """Direct access to the underlying insightface FaceAnalysis app."""
        return self._app

    def add_encoding(self, name: str, embedding: np.ndarray) -> None:
        """Append a single embedding for *name* (in-memory only)."""
        emb = np.asarray(embedding, dtype=np.float32)
        self._known_encodings.setdefault(name, []).append(emb)

    def remove_identity(self, name: str) -> bool:
        """Remove all encodings for *name*. Returns True if anything was removed."""
        if name in self._known_encodings:
            del self._known_encodings[name]
            return True
        return False

    def save_encodings(self) -> None:
        """Persist the encodings dict to ``self._encodings_path``."""
        parent = os.path.dirname(self._encodings_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self._encodings_path, "wb") as f:
            pickle.dump(self._known_encodings, f)

    def get_enrolled_names(self) -> List[str]:
        return list(self._known_encodings.keys())

    def get_sample_count(self, name: str) -> int:
        return len(self._known_encodings.get(name, []))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_encodings(self) -> None:
        if not os.path.isfile(self._encodings_path):
            self._known_encodings = {}
            logger.warning("No face encodings file at %s", self._encodings_path)
            return

        try:
            with open(self._encodings_path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            logger.exception("Failed to read encodings DB; starting fresh")
            self._known_encodings = {}
            return

        # New schema: {name: [vec, ...]}
        if isinstance(data, dict) and not (
            "encodings" in data and "names" in data
        ):
            self._known_encodings = {
                name: [np.asarray(v, dtype=np.float32) for v in vecs]
                for name, vecs in data.items()
                if isinstance(vecs, (list, tuple))
            }
            return

        # Legacy schema from the dlib/face_recognition era — flatten into the
        # new format so old enrollments aren't completely lost (though the
        # 128-d dlib vectors are NOT compatible with 512-d ArcFace; matching
        # against them will simply fail silently).
        if isinstance(data, dict) and "names" in data and "encodings" in data:
            logger.warning(
                "Legacy face_recognition encodings detected at %s — "
                "they are NOT compatible with ArcFace; please re-enroll.",
                self._encodings_path,
            )
            self._known_encodings = {}
            return

        logger.warning("Unknown encodings format at %s; ignoring", self._encodings_path)
        self._known_encodings = {}


def _empty_result() -> Dict[str, Any]:
    return {
        "user_id": None,
        "name": None,
        "confidence": 0.0,
        "is_live": False,
        "authorized": False,
        "is_authorized": False,
        "bbox": None,
    }
