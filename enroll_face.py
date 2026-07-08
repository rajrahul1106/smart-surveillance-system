"""Enroll a face from webcam — uses insightface ArcFace embeddings."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ENCODINGS_PATH = "data/face_encodings.pkl"
KNOWN_FACES_DIR = "data/known_faces"
MODEL_ROOT = "data/model_artifacts"
NUM_SAMPLES = 20
CAPTURE_INTERVAL = 1.0
BRIGHTNESS_LOW = 50.0
BRIGHTNESS_HIGH = 220.0

PROMPTS = [
    ("Look straight at the camera", 4),
    ("Turn your head slightly LEFT", 4),
    ("Turn your head slightly RIGHT", 4),
    ("Tilt your head UP slightly", 4),
    ("Tilt your head DOWN slightly", 4),
]

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def init_app():
    try:
        import insightface
    except ImportError:
        print("ERROR: insightface is required. Install with: pip install insightface onnxruntime")
        sys.exit(1)

    os.makedirs(MODEL_ROOT, exist_ok=True)
    app = insightface.app.FaceAnalysis(
        name="buffalo_l", root=MODEL_ROOT,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def check_brightness(gray: np.ndarray):
    mean_brightness = float(np.mean(gray))
    if mean_brightness < BRIGHTNESS_LOW:
        return "Too dark — move to better lighting"
    if mean_brightness > BRIGHTNESS_HIGH:
        return "Too bright — reduce lighting"
    return None


def load_encodings_db() -> dict:
    if os.path.isfile(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "names" not in data and "encodings" not in data:
                return data
        except Exception:
            pass
    return {}


def save_encodings_db(db: dict) -> None:
    parent = os.path.dirname(ENCODINGS_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(db, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a face from webcam (ArcFace)")
    parser.add_argument("name", help="Person's name, e.g. 'Rahul Raj'")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES, help="Number of samples")
    args = parser.parse_args()

    name = args.name.strip()
    if not name:
        print("ERROR: name cannot be empty")
        sys.exit(1)

    person_dir = os.path.join(KNOWN_FACES_DIR, name.replace(" ", "_").lower())
    known_faces_path = Path(KNOWN_FACES_DIR)
    if known_faces_path.exists() and not known_faces_path.is_dir():
        known_faces_path.unlink()
    os.makedirs(person_dir, exist_ok=True)

    print("Loading insightface buffalo_l (first run downloads ~300MB)...")
    app = init_app()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {args.camera}")
        sys.exit(1)

    total_needed = args.samples
    prompt_schedule = []
    for text, count in PROMPTS:
        ratio = count / sum(c for _, c in PROMPTS)
        n = max(1, round(ratio * total_needed))
        prompt_schedule.append((text, n))
    overflow = sum(n for _, n in prompt_schedule) - total_needed
    if overflow > 0:
        prompt_schedule[-1] = (prompt_schedule[-1][0], prompt_schedule[-1][1] - overflow)

    embeddings = []
    sample_idx = 0
    current_prompt_idx = 0
    samples_in_prompt = 0
    countdown_start = 0.0
    waiting_for_countdown = True

    print(f"\nEnrolling: {name}")
    print(f"Will capture {total_needed} samples. Press 'q' to abort.\n")

    while sample_idx < total_needed:
        ret, frame = cap.read()
        if not ret:
            continue

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        display = frame.copy()
        h_frame, _ = display.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        prompt_text, prompt_count = prompt_schedule[current_prompt_idx]
        brightness_warn = check_brightness(gray)

        faces = app.get(frame)
        face = largest_face(faces)

        cv2.putText(display, f"Enrollment: {name}", (10, 28), _FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Sample {sample_idx}/{total_needed}", (10, 56), _FONT, 0.6, (200, 200, 200), 1)
        cv2.putText(display, prompt_text, (10, h_frame - 40), _FONT, 0.7, (0, 220, 255), 2)
        if brightness_warn:
            cv2.putText(display, brightness_warn, (10, h_frame - 12), _FONT, 0.5, (0, 100, 255), 1)

        if face is not None:
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if len(faces) > 1:
                cv2.putText(display, f"{len(faces)} faces — using largest",
                            (10, 82), _FONT, 0.5, (0, 180, 255), 1)

            now = time.monotonic()
            if waiting_for_countdown:
                countdown_start = now
                waiting_for_countdown = False

            elapsed = now - countdown_start
            remaining = max(0, CAPTURE_INTERVAL - elapsed)

            if remaining > 0:
                cv2.putText(display, f"Capturing in {remaining:.1f}s",
                            (x1, y1 - 10), _FONT, 0.5, (0, 255, 255), 1)
            else:
                embedding = np.asarray(face.normed_embedding, dtype=np.float32)
                embeddings.append(embedding)
                sample_path = os.path.join(person_dir, f"sample_{sample_idx:03d}.npy")
                np.save(sample_path, embedding)

                sample_idx += 1
                samples_in_prompt += 1
                print(f"  [{sample_idx}/{total_needed}] Captured — {prompt_text}")
                cv2.putText(display, "CAPTURED!", (x1, y1 - 10), _FONT, 0.6, (0, 255, 0), 2)

                if samples_in_prompt >= prompt_count and current_prompt_idx < len(prompt_schedule) - 1:
                    current_prompt_idx += 1
                    samples_in_prompt = 0

                waiting_for_countdown = True
        else:
            cv2.putText(display, "No face detected — adjust position",
                        (10, 82), _FONT, 0.5, (0, 0, 255), 1)
            waiting_for_countdown = True

        cv2.imshow("Face Enrollment (ArcFace)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\nAborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if not embeddings:
        print("\nNo face samples captured. Enrollment failed.")
        sys.exit(1)

    avg = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    db = load_encodings_db()
    db[name] = [avg]
    save_encodings_db(db)

    print(f"\nEnrolled {name} with {len(embeddings)} face samples (ArcFace 512-d).")
    print(f"  Samples saved to: {person_dir}/")
    print(f"  Encodings DB updated: {ENCODINGS_PATH}")


if __name__ == "__main__":
    main()
