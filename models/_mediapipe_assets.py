"""Lazy download + cache for MediaPipe Tasks API .task model bundles.

MediaPipe 0.10.14+ removed the legacy ``mediapipe.solutions`` namespace and
moved to a Tasks API that requires per-task model files.  We cache them under
``data/model_artifacts/`` on first use.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "data/model_artifacts"

_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _ensure_dir() -> None:
    p = Path(ARTIFACTS_DIR)
    if p.exists() and not p.is_dir():
        p.unlink()
    p.mkdir(parents=True, exist_ok=True)


def _ensure_file(name: str, url: str) -> str:
    _ensure_dir()
    path = os.path.join(ARTIFACTS_DIR, name)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    logger.info("Downloading MediaPipe asset %s from %s", name, url)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def hand_landmarker_path() -> str:
    return _ensure_file("hand_landmarker.task", _HAND_LANDMARKER_URL)


def pose_landmarker_path() -> str:
    return _ensure_file("pose_landmarker_lite.task", _POSE_LANDMARKER_URL)
