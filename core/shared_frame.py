from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional


@dataclass(frozen=True)
class FrameSnapshot:
    """Immutable bundle of one pipeline tick's output.

    Frozen so the video endpoint can never observe a partially-mutated
    snapshot.  The pipeline thread builds a new snapshot each tick;
    consumers read whatever snapshot is current.
    """
    frame: Optional[Any] = None
    state: str = "IDLE"
    detections: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    frame_id: int = 0


class SharedFrame:
    """Thread-safe atomic frame + detections handoff.

    The pipeline calls ``set(frame, state, detections)`` once per processed
    frame; the video endpoint thread calls ``get_snapshot()`` (or the
    legacy ``get()`` tuple form) whenever it streams a JPEG.

    Atomicity guarantee
    -------------------
    ``set()`` shallow-copies the detections dict before storing, then
    publishes a frozen :class:`FrameSnapshot` under the lock.  Subsequent
    in-place mutations of the caller's ``detections`` dict (which the
    pipeline reuses across ticks) cannot leak into the published snapshot.
    Consumers always observe a consistent (frame, detections, state)
    triple from the same pipeline tick.
    """

    _FPS_WINDOW_SECONDS = 1.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: FrameSnapshot = FrameSnapshot()
        self._next_frame_id: int = 0
        self._timestamps: Deque[float] = deque(maxlen=120)

    # ------------------------------------------------------------------
    # Producer API (pipeline thread)
    # ------------------------------------------------------------------

    def set(
        self,
        frame: Any,
        state: str,
        detections: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Atomically publish a new (frame, state, detections) snapshot."""
        # Shallow-copy detections so subsequent mutations of the caller's
        # dict cannot bleed into the published snapshot. The inner detection
        # values (fire/injury/suspicious result dicts, landmarks list, etc.)
        # are produced freshly each tick and never mutated in place by the
        # state machine — they are replaced wholesale — so a top-level copy
        # is sufficient.
        det_copy: Dict[str, Any] = dict(detections) if detections else {}

        now = time.monotonic()
        with self._lock:
            self._next_frame_id += 1
            self._snapshot = FrameSnapshot(
                frame=frame,
                state=state,
                detections=det_copy,
                timestamp=now,
                frame_id=self._next_frame_id,
            )
            self._timestamps.append(now)

    # ------------------------------------------------------------------
    # Consumer API (video endpoint, enrollment, etc.)
    # ------------------------------------------------------------------

    def get_snapshot(self) -> FrameSnapshot:
        """Return the most recent immutable snapshot."""
        with self._lock:
            return self._snapshot

    def get(self) -> tuple:
        """Legacy tuple form: (frame, state, detections-copy).

        Equivalent to unpacking ``get_snapshot()``.  The detections dict
        is copied so callers that mutate the result don't accidentally
        affect the next consumer's view.
        """
        snap = self.get_snapshot()
        return snap.frame, snap.state, dict(snap.detections)

    def get_fps(self) -> float:
        """Pipeline processing FPS over the last second."""
        cutoff = time.monotonic() - self._FPS_WINDOW_SECONDS
        with self._lock:
            recent = sum(1 for t in self._timestamps if t >= cutoff)
        return float(recent)
