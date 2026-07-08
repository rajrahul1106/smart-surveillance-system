"""Live-webcam diagnostic for hand-landmark coordinate space.

Captures one frame from the default camera, applies the same horizontal
flip the surveillance pipeline applies (selfie view), runs the gesture
model, and reports where MediaPipe thinks the wrist is.

After our recent fix, the printed wrist x-coordinate should match where
the hand is on the *display* (selfie view): hand on the left of the
screen → x < 0.5; hand on the right → x > 0.5.

Usage:
    python scripts/diagnose_hand_landmarks.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from models.gesture_model import GestureModel  # noqa: E402


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera 0")
        return 1

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: cap.read() returned no frame")
        return 1

    # Mirror like CameraManager does.
    frame = cv2.flip(frame, 1)

    g = GestureModel()
    g.load()
    result = g.predict(frame)

    if not result.get("landmarks"):
        print("No hand detected — hold your hand up and rerun.")
        g.unload()
        return 1

    lm = result["landmarks"][0]  # wrist = landmark 0
    h, w = frame.shape[:2]
    px = int(lm[0] * w)
    py = int(lm[1] * h)

    print(f"Frame shape:        {frame.shape}")
    print(f"Wrist normalized:   x={lm[0]:.3f}, y={lm[1]:.3f}, z={lm[2]:.3f}")
    print(f"Wrist pixel:        ({px}, {py})")
    print(f"Frame center x:     {w // 2}")

    if lm[0] > 0.5:
        side = "RIGHT half of the frame"
    else:
        side = "LEFT half of the frame"
    print(f"Wrist is in the {side}.")
    print()
    print("Compare to where your hand actually appears in the selfie")
    print("view. If they disagree, the gesture model still needs a flip.")

    g.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
