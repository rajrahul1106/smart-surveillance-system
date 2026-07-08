"""Background face enrollment service driven by the API.

Captures N samples from the live camera (via ``SharedFrame``), uses the
FaceModel's underlying insightface ``FaceAnalysis`` app to produce a 512-d
ArcFace embedding for each sample, persists individual sample ``.npy``
files under ``data/known_faces/<name>/``, then averages the embeddings
and adds them to ``face_model._known_encodings`` so newly enrolled faces
are immediately matchable.
"""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

KNOWN_FACES_DIR = "data/known_faces"
ENCODINGS_PATH = "data/face_encodings.pkl"

_DEFAULT_SAMPLES = 20
_DEFAULT_INTERVAL = 0.5
_DEFAULT_TIMEOUT = 30.0


def _safe_dirname(name: str) -> str:
    return name.strip().replace(" ", "_").lower()


def _ensure_known_faces_dir() -> None:
    p = Path(KNOWN_FACES_DIR)
    if p.exists() and not p.is_dir():
        p.unlink()
    p.mkdir(parents=True, exist_ok=True)


def _largest_face(faces: List[Any]) -> Optional[Any]:
    if not faces:
        return None
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )


def _read_names_from_disk() -> List[str]:
    """Read enrolled identity names from the on-disk encodings pickle.

    Used by ``list_enrolled`` so it works even when the face_model is
    unloaded (e.g. while the pipeline is in IDLE).
    """
    if not os.path.isfile(ENCODINGS_PATH):
        return []
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
    except Exception:
        logger.exception("Failed to read encodings DB from %s", ENCODINGS_PATH)
        return []
    if isinstance(data, dict) and "names" not in data and "encodings" not in data:
        return list(data.keys())
    return []


def _remove_from_disk(name: str) -> bool:
    """Drop *name* from the encodings pickle and write it back.

    Used by ``remove`` when the face_model is unloaded.
    """
    if not os.path.isfile(ENCODINGS_PATH):
        return False
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
    except Exception:
        logger.exception("Failed to read encodings DB during remove")
        return False
    if not isinstance(data, dict) or "names" in data or "encodings" in data:
        return False
    if name not in data:
        return False
    del data[name]
    try:
        parent = os.path.dirname(ENCODINGS_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception:
        logger.exception("Failed to write encodings DB during remove")
        return False


class EnrollmentService:
    """Single-session face enrollment driven from a SharedFrame source."""

    def __init__(
        self,
        shared_frame: Any,
        face_model: Any,
        samples: int = _DEFAULT_SAMPLES,
        interval: float = _DEFAULT_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._shared_frame = shared_frame
        self._face_model = face_model
        self._samples = samples
        self._interval = interval
        self._timeout = timeout

        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._active_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_enrolled(self) -> List[Dict[str, Any]]:
        _ensure_known_faces_dir()

        # Prefer the face_model when it's loaded; otherwise read encodings
        # straight from disk so the API works even when the pipeline is in
        # IDLE and the model is unloaded.
        names: List[str] = []
        if self._face_model is not None and getattr(self._face_model, "is_loaded", False):
            try:
                names = list(self._face_model.get_enrolled_names())
            except Exception:
                logger.exception("get_enrolled_names failed")
                names = []
        if not names:
            names = _read_names_from_disk()

        result: List[Dict[str, Any]] = []
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            person_dir = Path(KNOWN_FACES_DIR) / _safe_dirname(name)
            sample_count = 0
            enrolled_at = 0.0
            if person_dir.is_dir():
                files = list(person_dir.glob("*.npy"))
                sample_count = len(files)
                if files:
                    enrolled_at = max(p.stat().st_mtime for p in files)
            if sample_count == 0 and self._face_model is not None:
                try:
                    sample_count = self._face_model.get_sample_count(name)
                except Exception:
                    pass
            result.append({
                "name": name,
                "samples": sample_count,
                "enrolled_at": enrolled_at,
            })
        return result

    def remove(self, name: str) -> bool:
        removed = False
        if self._face_model is not None and getattr(self._face_model, "is_loaded", False):
            try:
                if self._face_model.remove_identity(name):
                    self._face_model.save_encodings()
                    removed = True
            except Exception:
                logger.exception("Failed to remove identity from face_model")
        else:
            # Model unloaded — edit the on-disk pickle directly.
            removed = _remove_from_disk(name) or removed

        person_dir = Path(KNOWN_FACES_DIR) / _safe_dirname(name)
        if person_dir.is_dir():
            shutil.rmtree(person_dir, ignore_errors=True)
            removed = True

        return removed

    def start(self, name: str) -> Dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {"ok": False, "error": "name is required"}

        with self._lock:
            for sid, sess in self._sessions.items():
                if sess["status"] == "capturing":
                    return {
                        "ok": False,
                        "error": "another enrollment is already in progress",
                        "session_id": sid,
                    }

            session_id = uuid.uuid4().hex
            self._sessions[session_id] = {
                "session_id": session_id,
                "name": name,
                "progress": 0,
                "total": self._samples,
                "status": "capturing",
                "started_at": time.time(),
                "reason": None,
            }

        thread = threading.Thread(
            target=self._run_session,
            args=(session_id, name),
            daemon=True,
            name=f"enroll-{name}",
        )
        with self._lock:
            self._active_thread = thread
        thread.start()

        return {"ok": True, "session_id": session_id, "status": "capturing"}

    def get_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return None
            return dict(sess)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run_session(self, session_id: str, name: str) -> None:
        if self._face_model is None:
            self._finalize(session_id, "failed", reason="face model not provided")
            return

        # During IDLE the face_model is unloaded — load it temporarily so
        # enrollment works from any pipeline state. Track whether WE loaded
        # it so we don't unload a model that the state machine is using.
        loaded_for_enrollment = False
        if not getattr(self._face_model, "is_loaded", False):
            try:
                self._face_model.load()
                loaded_for_enrollment = True
                logger.info("EnrollmentService: temporarily loaded face_model")
            except Exception:
                logger.exception("EnrollmentService: failed to load face_model")
                self._finalize(session_id, "failed", reason="could not load face model")
                return

        try:
            self._do_capture(session_id, name)
        finally:
            if loaded_for_enrollment and self._face_model is not None:
                try:
                    self._face_model.unload()
                    logger.info("EnrollmentService: unloaded face_model after enrollment")
                except Exception:
                    logger.exception("EnrollmentService: failed to unload face_model")

    def _do_capture(self, session_id: str, name: str) -> None:
        if getattr(self._face_model, "app", None) is None:
            self._finalize(session_id, "failed", reason="face model has no analysis app")
            return

        _ensure_known_faces_dir()
        person_dir = Path(KNOWN_FACES_DIR) / _safe_dirname(name)
        person_dir.mkdir(parents=True, exist_ok=True)

        embeddings: List[np.ndarray] = []
        deadline = time.monotonic() + self._timeout
        last_capture = 0.0

        while True:
            with self._lock:
                sess = self._sessions.get(session_id)
                if sess is None or sess["status"] != "capturing":
                    return
                if sess["progress"] >= self._samples:
                    break

            if time.monotonic() > deadline:
                if not embeddings:
                    self._finalize(session_id, "failed", reason="timeout — no face detected")
                    return
                break

            now = time.monotonic()
            if now - last_capture < self._interval:
                time.sleep(0.05)
                continue

            frame, _state, _det = self._shared_frame.get() if self._shared_frame else (None, "", {})
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                bgr = frame.astype(np.uint8)
                if len(bgr.shape) == 3 and bgr.shape[2] == 4:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
            except Exception:
                time.sleep(0.05)
                continue

            try:
                faces = self._face_model.app.get(bgr)
            except Exception:
                logger.exception("insightface get() failed during enrollment")
                time.sleep(0.05)
                continue

            face = _largest_face(faces)
            if face is None:
                time.sleep(0.05)
                continue

            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            embeddings.append(embedding)
            last_capture = now

            sample_path = person_dir / f"sample_{len(embeddings) - 1:03d}.npy"
            try:
                np.save(sample_path, embedding)
            except Exception:
                logger.exception("Failed to write sample file")

            with self._lock:
                sess = self._sessions.get(session_id)
                if sess is None:
                    return
                sess["progress"] = len(embeddings)

        if not embeddings:
            self._finalize(session_id, "failed", reason="no embeddings extracted")
            return

        # Average and persist via the face_model so it's immediately usable.
        avg = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        try:
            self._face_model.remove_identity(name)
            self._face_model.add_encoding(name, avg)
            self._face_model.save_encodings()
        except Exception:
            logger.exception("Failed to update face_model encodings")
            self._finalize(session_id, "failed", reason="failed to persist encodings")
            return

        self._finalize(session_id, "complete", samples=len(embeddings))

    def _finalize(self, session_id: str, status: str, **extra: Any) -> None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return
            sess["status"] = status
            sess["finished_at"] = time.time()
            for k, v in extra.items():
                sess[k] = v
