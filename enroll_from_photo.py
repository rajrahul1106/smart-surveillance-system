"""Enroll a face from one or more photo files — uses insightface ArcFace."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ENCODINGS_PATH = "data/face_encodings.pkl"
KNOWN_FACES_DIR = "data/known_faces"
MODEL_ROOT = "data/model_artifacts"
BRIGHTNESS_LOW = 50.0
BRIGHTNESS_HIGH = 220.0


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


def process_photo(path: str, app, person_dir: str, idx: int) -> Optional[np.ndarray]:
    frame = cv2.imread(path)
    if frame is None:
        print(f"  [skip] Could not read image: {path}")
        return None

    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < BRIGHTNESS_LOW:
        print(f"  [warn] {path} — image may be too dark (brightness={mean_brightness:.0f})")
    elif mean_brightness > BRIGHTNESS_HIGH:
        print(f"  [warn] {path} — image may be too bright (brightness={mean_brightness:.0f})")

    faces = app.get(frame)
    face = largest_face(faces)
    if face is None:
        print(f"  [skip] No face detected in: {path}")
        return None

    if len(faces) > 1:
        print(f"  [info] {len(faces)} faces in {path} — using largest")

    embedding = np.asarray(face.normed_embedding, dtype=np.float32)
    sample_path = os.path.join(person_dir, f"sample_{idx:03d}.npy")
    np.save(sample_path, embedding)
    x1, y1, x2, y2 = face.bbox.astype(int).tolist()
    print(f"  [ok] {path} — face at ({x1},{y1})-({x2},{y2})")
    return embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a face from photo files (ArcFace)")
    parser.add_argument("name", help="Person's name, e.g. 'Prince Gupta'")
    parser.add_argument("photos", nargs="+", help="One or more image file paths")
    args = parser.parse_args()

    name = args.name.strip()
    if not name:
        print("ERROR: name cannot be empty")
        sys.exit(1)

    for photo in args.photos:
        if not os.path.isfile(photo):
            print(f"ERROR: file not found: {photo}")
            sys.exit(1)

    person_dir = os.path.join(KNOWN_FACES_DIR, name.replace(" ", "_").lower())
    known_faces_path = Path(KNOWN_FACES_DIR)
    if known_faces_path.exists() and not known_faces_path.is_dir():
        known_faces_path.unlink()
    os.makedirs(person_dir, exist_ok=True)

    print("Loading insightface buffalo_l (first run downloads ~300MB)...")
    app = init_app()

    print(f"\nEnrolling: {name}")
    print(f"Processing {len(args.photos)} photo(s)...\n")

    embeddings = []
    for idx, photo in enumerate(args.photos):
        emb = process_photo(photo, app, person_dir, idx)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        print("\nNo face samples extracted. Enrollment failed.")
        sys.exit(1)

    avg = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    db = load_encodings_db()
    db[name] = [avg]
    save_encodings_db(db)

    print(f"\nEnrolled {name} with {len(embeddings)} ArcFace samples (from {len(args.photos)} photos).")
    print(f"  Samples saved to: {person_dir}/")
    print(f"  Encodings DB updated: {ENCODINGS_PATH}")


if __name__ == "__main__":
    main()
