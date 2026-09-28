"""IoU face tracker with smoothed identity and AUTHORIZED / UNKNOWN / UNCERTAIN status.

Detections arrive once per face-model run.  Each track keeps an exponential
moving average (EMA) of its cosine similarity to every enrolled identity, and
its status comes from the best of those EMAs:

    ema >= auth_threshold      -> AUTHORIZED (identity = best match)
    ema <  unknown_threshold   -> UNKNOWN once this has held for
                                  ``unknown_confirm_frames`` updates in a row;
                                  until then the previous status stays
                                  (UNCERTAIN for a new track)
    otherwise                  -> UNCERTAIN

The hysteresis on the UNKNOWN side stops a real enrolled person from
flickering to UNKNOWN when they turn their head, and the per-identity EMA
stops a track's name from jumping between people frame to frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

AUTHORIZED = "AUTHORIZED"
UNKNOWN = "UNKNOWN"
UNCERTAIN = "UNCERTAIN"

BBox = Tuple[int, int, int, int]  # (x, y, w, h) in original-frame pixels


@dataclass
class FaceDetection:
    """One detected face: its box and cosine similarity to each identity."""

    bbox: BBox
    similarities: Dict[str, float] = field(default_factory=dict)


@dataclass
class Track:
    track_id: int
    bbox: BBox
    missed: int = 0
    ema_similarity: Dict[str, float] = field(default_factory=dict)
    status: str = UNCERTAIN
    identity: Optional[str] = None
    consecutive_unknown: int = 0
    auth_streak: int = 0  # consecutive updates that earned AUTHORIZED

    def best_match(self) -> Tuple[Optional[str], float]:
        """Identity with the highest smoothed similarity, and that similarity."""
        if not self.ema_similarity:
            return None, 0.0
        name = max(self.ema_similarity, key=self.ema_similarity.__getitem__)
        return name, self.ema_similarity[name]

    @property
    def confidence(self) -> float:
        """Smoothed similarity (0-1) behind the current status."""
        if self.status == AUTHORIZED and self.identity is not None:
            return self.ema_similarity.get(self.identity, 0.0)
        return self.best_match()[1]


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = min(ax + aw, bx + bw) - max(ax, bx)
    inter_h = min(ay + ah, by + bh) - max(ay, by)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def summarize_presence(faces: Iterable[dict]) -> dict:
    """Who is in view: unique AUTHORIZED names (sorted) plus per-status counts."""
    faces = list(faces)
    return {
        "authorized": sorted({
            f["identity"] for f in faces
            if f.get("status") == AUTHORIZED and f.get("identity")
        }),
        "unknown_count": sum(1 for f in faces if f.get("status") == UNKNOWN),
        "uncertain_count": sum(1 for f in faces if f.get("status") == UNCERTAIN),
        "total": len(faces),
    }


class FaceTracker:
    """Greedy IoU tracker; call ``update()`` once per face-model run."""

    IOU_THRESHOLD = 0.3
    AUTH_THRESHOLD = 0.60
    UNKNOWN_THRESHOLD = 0.30
    EMA_ALPHA = 0.3
    UNKNOWN_CONFIRM_FRAMES = 5
    TRACK_MAX_MISSED = 10

    def __init__(
        self,
        auth_threshold: float = AUTH_THRESHOLD,
        unknown_threshold: float = UNKNOWN_THRESHOLD,
        ema_alpha: float = EMA_ALPHA,
        unknown_confirm_frames: int = UNKNOWN_CONFIRM_FRAMES,
        track_max_missed: int = TRACK_MAX_MISSED,
        iou_threshold: float = IOU_THRESHOLD,
    ) -> None:
        self._auth_threshold = auth_threshold
        self._unknown_threshold = unknown_threshold
        self._alpha = ema_alpha
        self._unknown_confirm_frames = unknown_confirm_frames
        self._max_missed = track_max_missed
        self._iou_threshold = iou_threshold

        self._tracks: List[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> List[Track]:
        """Every live track, including ones missed in the latest update."""
        return list(self._tracks)

    def reset(self) -> None:
        """Drop all tracks.  Ids keep increasing so they stay unique in logs."""
        self._tracks = []

    def update(self, detections: Sequence[FaceDetection]) -> List[Track]:
        """Match *detections* to tracks; return the tracks seen, in detection order."""
        pairs = sorted(
            (
                (iou(track.bbox, det.bbox), ti, di)
                for ti, track in enumerate(self._tracks)
                for di, det in enumerate(detections)
            ),
            reverse=True,
        )
        seen: List[Optional[Track]] = [None] * len(detections)
        matched = set()
        for overlap, ti, di in pairs:
            if overlap < self._iou_threshold:
                break
            if ti in matched or seen[di] is not None:
                continue
            track = self._tracks[ti]
            matched.add(ti)
            track.bbox = detections[di].bbox
            track.missed = 0
            self._observe(track, detections[di].similarities)
            seen[di] = track

        for ti, track in enumerate(self._tracks):
            if ti not in matched:
                track.missed += 1
                track.auth_streak = 0
        self._tracks = [t for t in self._tracks if t.missed <= self._max_missed]

        for di, det in enumerate(detections):
            if seen[di] is None:
                track = Track(track_id=self._next_id, bbox=det.bbox)
                self._next_id += 1
                self._observe(track, det.similarities)
                self._tracks.append(track)
                seen[di] = track
        return [t for t in seen if t is not None]

    def _observe(self, track: Track, similarities: Dict[str, float]) -> None:
        """Fold one frame's similarities into the EMAs and re-derive the status."""
        for name, sim in similarities.items():
            sim = min(1.0, max(0.0, float(sim)))
            prev = track.ema_similarity.get(name)
            track.ema_similarity[name] = (
                sim if prev is None else self._alpha * sim + (1.0 - self._alpha) * prev
            )

        name, ema = track.best_match()
        earned_auth = ema >= self._auth_threshold
        if earned_auth:
            track.status, track.identity = AUTHORIZED, name
            track.consecutive_unknown = 0
        elif ema < self._unknown_threshold:
            track.consecutive_unknown += 1
            if track.consecutive_unknown >= self._unknown_confirm_frames:
                track.status, track.identity = UNKNOWN, None
            # Otherwise keep the previous status (UNCERTAIN for a new track).
        else:
            track.status, track.identity = UNCERTAIN, None
            track.consecutive_unknown = 0
        track.auth_streak = track.auth_streak + 1 if earned_auth else 0
