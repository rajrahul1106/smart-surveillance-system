"""
SUPERSEDED — kept for reference only.

The fire model in use is ``fire_yolo11s_480.onnx`` (YOLO11s, 480×480, fire +
smoke), trained in the D-Fire YOLO11s Colab notebook on D-Fire plus the two
Roboflow fire/smoke datasets.  This script trains the older Roboflow-only
model (archived as ``data/model_artifacts/archive/fire_yolo11n_416_v1.onnx``);
the file it exports is no longer loaded by the app.

One-time script to train YOLOv8n on a fire/smoke dataset and export to ONNX.

The resulting ONNX model was used at runtime by ``models/fire_model.py``
via onnxruntime (no ultralytics dependency required at inference time).

Usage
-----
    conda activate surveillance
    pip install ultralytics roboflow
    python scripts/train_fire_model.py

Environment variables
---------------------
    ROBOFLOW_API_KEY   Free API key from https://app.roboflow.com/settings/api
                       Required for automatic dataset download.
"""

from __future__ import annotations

import os
import shutil
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEST_DIR = os.path.join(_PROJECT_ROOT, "data", "model_artifacts", "models")
_DEST_PATH = os.path.join(_DEST_DIR, "fire_yolov8n.onnx")
_DATASET_DIR = os.path.join(_PROJECT_ROOT, "data", "fire_dataset")


def _ensure_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        print("Installing ultralytics ...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO  # noqa: F401


def _download_roboflow_dataset() -> str | None:
    """Download the fire/smoke dataset from Roboflow.  Returns the data.yaml path."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        return None

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow ...")
        os.system(f"{sys.executable} -m pip install roboflow")
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)

    # Primary: fire-detection-yolo (7896 images, classes: fire, smoke)
    try:
        project = rf.workspace("yolofire-wbkwv").project("fire-detection-yolo-1gn9t")
        version = project.version(1)
        dataset = version.download("yolov8", location=_DATASET_DIR)
        return os.path.join(_DATASET_DIR, "data.yaml")
    except Exception as exc:
        print(f"Primary dataset unavailable: {exc}")

    # Fallback: fire-and-smoke-detection (9860 images)
    try:
        project = rf.workspace("fire-and-smoke-detection-yolo").project(
            "fire-and-smoke-detection-o4uhv",
        )
        version = project.version(1)
        dataset = version.download("yolov8", location=_DATASET_DIR)
        return os.path.join(_DATASET_DIR, "data.yaml")
    except Exception as exc:
        print(f"Fallback dataset also unavailable: {exc}")

    return None


def _train_and_export(dataset_yaml: str) -> None:
    """Fine-tune YOLOv8n on *dataset_yaml* and export the best weights to ONNX."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # COCO-pretrained starting weights

    # Use MPS (Apple Silicon GPU) when available, otherwise CPU.
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        patience=10,
        workers=4,
        project=os.path.join(_PROJECT_ROOT, "runs", "fire"),
        name="yolov8n_fire",
        exist_ok=True,
    )

    best = os.path.join(_PROJECT_ROOT, "runs", "fire", "yolov8n_fire", "weights", "best.pt")
    if not os.path.exists(best):
        best = os.path.join(_PROJECT_ROOT, "runs", "fire", "yolov8n_fire", "weights", "last.pt")

    model = YOLO(best)
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True, opset=12)

    os.makedirs(_DEST_DIR, exist_ok=True)
    shutil.copy2(onnx_path, _DEST_PATH)

    size_mb = os.path.getsize(_DEST_PATH) / 1024 / 1024
    print()
    print("=" * 60)
    print(f"SUCCESS  Fire model exported to {_DEST_PATH}")
    print(f"         Size: {size_mb:.1f} MB")
    print("=" * 60)


def main() -> None:
    _ensure_ultralytics()

    # Allow a manual dataset path via CLI: --dataset path/to/data.yaml
    if len(sys.argv) > 2 and sys.argv[1] == "--dataset":
        dataset_yaml = sys.argv[2]
        if not os.path.isfile(dataset_yaml):
            sys.exit(f"Dataset file not found: {dataset_yaml}")
        _train_and_export(dataset_yaml)
        return

    dataset_yaml = _download_roboflow_dataset()
    if dataset_yaml is not None:
        _train_and_export(dataset_yaml)
        return

    # No API key and no --dataset flag — print instructions.
    print()
    print("=" * 60)
    print("ROBOFLOW API KEY REQUIRED")
    print("=" * 60)
    print()
    print("Option A  Set the environment variable and re-run:")
    print('  export ROBOFLOW_API_KEY="your_api_key_here"')
    print("  python scripts/train_fire_model.py")
    print()
    print("Option B  Download the dataset manually:")
    print("  1. https://universe.roboflow.com/yolofire-wbkwv/fire-detection-yolo-1gn9t")
    print("  2. Click 'Download Dataset' > YOLOv8 > Download zip")
    print(f"  3. Extract into {_DATASET_DIR}/")
    print("  4. python scripts/train_fire_model.py --dataset "
          f"{_DATASET_DIR}/data.yaml")
    print()
    print("A free Roboflow account is all you need:")
    print("  https://app.roboflow.com/settings/api")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
