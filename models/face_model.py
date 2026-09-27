"""ArcFace multi-face identity model (insightface buffalo_l).

One InsightFace ``FaceAnalysis`` app detects every face in the frame and
extracts a 512-d ArcFace embedding for each.  All faces are matched against
the enrolled encodings in one matrix multiply, and a :class:`FaceTracker`
gives each person a stable track id plus an AUTHORIZED / UNCERTAIN / UNKNOWN
status smoothed over time.

Modes (see :meth:`FaceModel.set_mode`):

- ``"verify"`` (VERIFYING_IDENTITY): centre crop + upscale, for long-range
  authentication of the person doing the gesture.
- ``"presence"`` (ACTIVE_DETECTION): the full frame, no crop, so people at
  the frame edges are seen too.  ``DET_SIZE`` already upsamples small faces.

Storage format::

    data/face_encodings.pkl  →  { name: [np.ndarray(512,), ...], ... }

A person may have several enrolled samples; their score is the max over them.

Predict return contract: the legacy single-face keys (``user_id``, ``name``,
``label``, ``confidence``, ``is_live``, ``authorized``/``is_authorized``,
``bbox`` (x, y, w, h tuple of ints), ``frame_size``) describe the best
AUTHORIZED face, else the largest face.  ``faces`` lists every face seen in
this frame and ``presence`` summarises who is in view.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from models.base_model import BaseModel
from models.face_tracker import (
    AUTHORIZED,
    FaceDetection,
    FaceTracker,
    Track,
    summarize_presence,
)

logger = logging.getLogger(__name__)

MODE_VERIFY = "verify"
MODE_PRESENCE = "presence"


class FaceModel(BaseModel):
    """ArcFace face recognition via insightface buffalo_l (CPU)."""

    ENCODINGS_PATH_DEFAULT = "data/face_encodings.pkl"
    MODEL_ROOT_DEFAULT = "data/model_artifacts"

    # ArcFace cosine similarity tiers, applied to each track's smoothed score.
    # Display percentage = cosine * 100 (e.g. 0.68 → "68%").
    #
    # >= AUTHORIZE_THRESHOLD (0.60)       → AUTHORIZED, "name"
    # POSSIBLE_THRESHOLD .. AUTHORIZE     → legacy label "POSSIBLE: name"
    # < UNKNOWN_THRESHOLD (0.30), held    → UNKNOWN (see FaceTracker)
    AUTHORIZE_THRESHOLD = 0.60
    POSSIBLE_THRESHOLD = 0.40
    # Legacy alias kept for backward compat with callers that pass
    # match_threshold= explicitly; if provided it is treated as the
    # AUTHORIZE_THRESHOLD.
    MATCH_THRESHOLD = AUTHORIZE_THRESHOLD
    DET_SIZE = (960, 960)

    # Centre-crop ratio for verify mode.  A 70 % crop gives ~1.43× zoom,
    # making faces at 2-5 m large enough for reliable detection.
    CROP_RATIO = 0.7

    # Detector score threshold per mode: low in verify mode so the distant
    # gesture-doer is still found; the library default in presence mode,
    # where every false detection would become an UNKNOWN person.
    VERIFY_DET_THRESH = 0.3
    PRESENCE_DET_THRESH = 0.5

    # Only the detector (bbox + 5-point kps) and ArcFace heads are used;
    # skipping buffalo_l's landmark and gender/age heads saves ~30 ms per face.
    INSIGHTFACE_MODULES = ("detection", "recognition")

    # Multi-person tracking (status rules live in FaceTracker).
    UNKNOWN_THRESHOLD = 0.30
    EMA_ALPHA = 0.3
    UNKNOWN_CONFIRM_FRAMES = 5
    TRACK_MAX_MISSED = 10
    MAX_FACES = 6

    def __init__(
        self,
        encodings_path: str = ENCODINGS_PATH_DEFAULT,
        model_root: str = MODEL_ROOT_DEFAULT,
        match_threshold: float = MATCH_THRESHOLD,
        possible_threshold: float = POSSIBLE_THRESHOLD,
        unknown_threshold: float = UNKNOWN_THRESHOLD,
        ema_alpha: float = EMA_ALPHA,
        unknown_confirm_frames: int = UNKNOWN_CONFIRM_FRAMES,
        track_max_missed: int = TRACK_MAX_MISSED,
        max_faces: int = MAX_FACES,
    ) -> None:
        super().__init__()
        self._encodings_path = encodings_path
        self._model_root = model_root
        self._authorize_threshold = match_threshold
        self._possible_threshold = possible_threshold
        # Legacy attribute name some tests may inspect.
        self._match_threshold = match_threshold
        self._max_faces = max_faces

        self._app: Optional[Any] = None
        # { name: [np.ndarray(512,), ...] }
        self._known_encodings: Dict[str, List[np.ndarray]] = {}
        self._tracker = FaceTracker(
            auth_threshold=match_threshold,
            unknown_threshold=unknown_threshold,
            ema_alpha=ema_alpha,
            unknown_confirm_frames=unknown_confirm_frames,
            track_max_missed=track_max_missed,
        )
        self._mode = MODE_VERIFY
        # Serialises predict with unload/set_mode: presence runs on a worker
        # thread and the COOLDOWN transition unloads from a timer thread.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Switch between ``"verify"`` (centre crop) and ``"presence"`` (full frame).

        Tracks are kept: both modes report boxes in original-frame pixels,
        so the authenticated person keeps their track into presence mode.
        """
        if mode not in (MODE_VERIFY, MODE_PRESENCE):
            raise ValueError(f"unknown face model mode: {mode!r}")
        with self._lock:
            self._mode = mode
            self._apply_det_thresh()

    def _det_thresh(self) -> float:
        return self.VERIFY_DET_THRESH if self._mode == MODE_VERIFY else self.PRESENCE_DET_THRESH

    def _apply_det_thresh(self) -> None:
        det_model = getattr(self._app, "det_model", None) if self._app is not None else None
        if det_model is not None:
            det_model.det_thresh = self._det_thresh()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _do_load(self) -> None:
        import insightface

        os.makedirs(self._model_root, exist_ok=True)

        app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=self._model_root,
            allowed_modules=list(self.INSIGHTFACE_MODULES),
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=self.DET_SIZE, det_thresh=self._det_thresh())
        with self._lock:
            self._app = app
            self._tracker.reset()

        self._load_encodings()
        logger.info(
            "FaceModel ready — %d enrolled identit%s, mode=%s",
            len(self._known_encodings),
            "y" if len(self._known_encodings) == 1 else "ies",
            self._mode,
        )

    def _do_unload(self) -> None:
        with self._lock:
            self._app = None
            self._known_encodings = {}
            self._tracker.reset()
            self._mode = MODE_VERIFY

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _do_predict(self, frame: Any) -> Dict[str, Any]:
        if frame is None:
            return _empty_result()

        bgr = frame
        if len(bgr.shape) == 3 and bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        bgr = bgr.astype(np.uint8)
        h, w = bgr.shape[:2]
        frame_size = (int(w), int(h))

        with self._lock:
            if self._app is None:  # unloaded while this call was waiting
                return _empty_result(frame_size)

            verify = self._mode == MODE_VERIFY
            image = self._prepare_frame_for_distance(bgr) if verify else bgr
            # insightface expects BGR (which OpenCV provides natively).
            faces = self._app.get(image)
            # Largest faces first, capped so a crowd can't stall the pipeline.
            faces = sorted(faces, key=_bbox_area, reverse=True)[: self._max_faces]

            boxes: List[Tuple[int, int, int, int]] = []
            embeddings: List[np.ndarray] = []
            for face in faces:
                raw_bbox = face.bbox.tolist()
                if verify:
                    raw_bbox = self._remap_bbox(raw_bbox, w, h)
                bbox = _clamp_xywh(raw_bbox, w, h)
                if bbox[2] == 0 or bbox[3] == 0:
                    continue
                boxes.append(bbox)
                embeddings.append(face.normed_embedding)

            similarities = self._similarities(embeddings)
            tracks = self._tracker.update([
                FaceDetection(bbox=b, similarities=s) for b, s in zip(boxes, similarities)
            ])
            logger.debug(
                "FaceModel %s: %d face(s) %s, input_frame=(%d,%d)",
                self._mode, len(tracks),
                [(t.track_id, t.status, t.identity) for t in tracks], w, h,
            )
            return self._build_result(tracks, frame_size)

    def _build_result(
        self, tracks: List[Track], frame_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        if not tracks:
            return _empty_result(frame_size)

        faces = [self._face_entry(t) for t in tracks]
        authorized = [t for t in tracks if t.status == AUTHORIZED]
        if authorized:
            primary = max(authorized, key=lambda t: t.confidence)
        else:
            primary = max(tracks, key=lambda t: t.bbox[2] * t.bbox[3])

        result = self._legacy_fields(primary)
        result.update(
            bbox=primary.bbox,
            frame_size=frame_size,
            faces=faces,
            presence=summarize_presence(faces),
        )
        return result

    @staticmethod
    def _face_entry(track: Track) -> Dict[str, Any]:
        return {
            "track_id": track.track_id,
            "bbox": track.bbox,
            "status": track.status,
            "identity": track.identity if track.status == AUTHORIZED else None,
            "confidence": round(track.confidence * 100.0, 1),
            "auth_streak": track.auth_streak,
        }

    def _legacy_fields(self, track: Track) -> Dict[str, Any]:
        """Single-face keys the pre-tracking pipeline read, for one track."""
        is_authorized = track.status == AUTHORIZED
        best_name, best_score = track.best_match()

        # Three-tier label/identity assignment.
        if is_authorized:
            label = user_id = name_field = track.identity
        elif best_name is not None and best_score >= self._possible_threshold:
            # Strong-enough match to surface the candidate name, but not
            # confident enough to authorize.
            label = user_id = name_field = f"POSSIBLE: {best_name}"
        else:
            label, user_id, name_field = "UNKNOWN", None, None

        return {
            "user_id": user_id,
            "name": name_field,
            "label": label,
            "confidence": float(track.confidence),
            "is_live": True,  # ArcFace + buffalo_l detector implies a real face crop
            "authorized": is_authorized,
            "is_authorized": is_authorized,
        }

    # ------------------------------------------------------------------
    # Distance helpers (verify mode: centre-crop + upscale)
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

    def _similarities(self, embeddings: Sequence[np.ndarray]) -> List[Dict[str, float]]:
        """Cosine similarity of each query face to every enrolled identity.

        Queries ``(N, D)`` and enrolled samples ``(M, D)`` are L2-normalised
        and compared in one ``Q @ E.T``; a person with several samples scores
        the max over them.
        """
        if not embeddings:
            return []
        queries = _l2_normalize(np.stack([
            np.asarray(e, dtype=np.float32).ravel() for e in embeddings
        ]))
        names, samples, owner = self._enrolled_matrix(queries.shape[1])
        if not names:
            return [{} for _ in embeddings]

        sims = queries @ samples.T  # (N, M)
        per_identity = np.stack(
            [sims[:, owner == j].max(axis=1) for j in range(len(names))], axis=1,
        )  # (N, identities)
        return [dict(zip(names, row.tolist())) for row in per_identity]

    def _enrolled_matrix(self, dim: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Stack usable enrolled samples: (names, samples (M, dim), owner index per row)."""
        names: List[str] = []
        rows: List[np.ndarray] = []
        owner: List[int] = []
        # Snapshot: EnrollmentService may add identities from another thread.
        for name, vectors in list(self._known_encodings.items()):
            usable = [
                v for v in (np.asarray(v, dtype=np.float32).ravel() for v in vectors)
                if v.shape[0] == dim and float(np.linalg.norm(v)) > 1e-8
            ]
            if not usable:
                continue
            owner.extend([len(names)] * len(usable))
            names.append(name)
            rows.extend(usable)
        if not names:
            return [], np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=int)
        return names, _l2_normalize(np.stack(rows)), np.asarray(owner)

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


def _bbox_area(face: Any) -> float:
    return float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


def _clamp_xywh(bbox: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    """(x1, y1, x2, y2) floats → (x, y, w, h) ints clamped to the frame."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(x1, w)); y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w)); y2 = max(0, min(y2, h))
    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-8)


def _empty_result(frame_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "user_id": None,
        "name": None,
        "confidence": 0.0,
        "is_live": False,
        "authorized": False,
        "is_authorized": False,
        "bbox": None,
        "faces": [],
        "presence": summarize_presence([]),
    }
    if frame_size is not None:
        result["frame_size"] = frame_size
    return result
