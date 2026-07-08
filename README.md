# SENTINEL — Smart AI-Based Surveillance System

A real-time gesture-gated AI surveillance system with a 5-stage progressive activation pipeline and dual-factor (intent + identity) authentication.

**Patent Status:** Provisional patent filed for *"Gesture-Gated Progressive Activation Pipeline with Dual-Factor Authentication for AI Surveillance Systems"*

---

## How It Works

SENTINEL activates threat detection **only** after verifying both human intent and identity. The system remains dormant until a user performs an SOS gesture and passes face authentication.

```
IDLE → VERIFYING_GESTURE → VERIFYING_IDENTITY → ACTIVE_DETECTION → COOLDOWN → IDLE
```

| Stage | What Happens |
|-------|-------------|
| **IDLE** | Monitors for SOS gesture only. All detection models offline. Minimal CPU usage. |
| **VERIFYING_GESTURE** | Validates 4-step Palm→Fist→Palm→Fist sequence at 99% confidence per gesture. |
| **VERIFYING_IDENTITY** | ArcFace face authentication against enrolled identities. 3 attempt limit. |
| **ACTIVE_DETECTION** | Fire (YOLOv11n), injury (MediaPipe Pose), and activity (HOG+SVM) detection run in parallel. |
| **COOLDOWN** | Unloads all models, clears state, prepares for next cycle. |

---

## Features

- **Gesture-Gated Activation** — SOS gesture sequence (Palm→Fist→Palm→Fist) prevents accidental or unauthorized activation
- **Face Authentication** — InsightFace ArcFace (buffalo_l) with cosine similarity matching against enrolled identities
- **Fire Detection** — YOLOv11n ONNX model trained on 10,000+ fire/smoke images with leaky-accumulator temporal verification
- **Injury Detection** — MediaPipe Pose estimation for fallen/collapsed person detection
- **Activity Monitoring** — HOG+SVM for suspicious activity classification
- **Live Dashboard** — Cyberpunk-themed SENTINEL web dashboard with real-time MJPEG video, state telemetry, model status, and alert history via WebSocket
- **Long-Range Detection** — Gesture recognition at 2-3m using center-crop upscaling (1.67x digital zoom)
- **State-Driven Architecture** — Event-bus pattern with per-state model loading/unloading. No if/elif chains.
- **184 Passing Tests** — Comprehensive test coverage across all models and pipeline states

---

## System Metrics

| Metric | Value |
|--------|-------|
| Process FPS | 23–30 (CPU only) |
| Fire mAP@50 | 65.9% |
| Detection Range | 2–3 meters |
| Auth Latency | < 2 seconds |
| Test Suite | 184 passing |
| Fire Training Images | 10,000+ |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core** | Python 3.11, OpenCV, NumPy |
| **Gesture** | MediaPipe Tasks API (Hand Landmarker) |
| **Face Auth** | InsightFace ArcFace (buffalo_l), ONNX Runtime |
| **Fire Detection** | YOLOv11n, ONNX Runtime, trained on Roboflow datasets |
| **Injury** | MediaPipe Pose Estimation |
| **Activity** | HOG + SVM |
| **API** | FastAPI, WebSocket, MJPEG streaming |
| **Dashboard** | HTML/CSS/JS (cyberpunk theme) |
| **Database** | SQLite |
| **Testing** | pytest |

---

## Project Structure

```
smart_surveillance/
├── main.py                          # Application entry point
├── config.yaml                      # System configuration
├── core/
│   ├── pipeline.py                  # Frame processing pipeline
│   ├── state_machine.py             # State machine controller
│   ├── event_bus.py                 # Event-driven communication
│   ├── events.py                    # Event type definitions
│   ├── frame_annotator.py           # Video overlay rendering
│   ├── frame_manager.py             # Frame lifecycle management
│   ├── shared_frame.py              # Thread-safe frame sharing
│   ├── config.py                    # Config loader
│   └── states/
│       ├── idle_state.py            # IDLE state logic
│       ├── verifying_gesture_state.py
│       ├── verifying_identity_state.py
│       ├── active_detection_state.py
│       └── cooldown_state.py
├── models/
│   ├── base_model.py                # Base model with lifecycle hooks
│   ├── gesture_model.py             # MediaPipe hand gesture detection
│   ├── face_model.py                # InsightFace ArcFace recognition
│   ├── fire_model.py                # YOLOv11n ONNX fire/smoke detection
│   ├── injury_model.py              # MediaPipe pose injury detection
│   └── activity_model.py            # HOG+SVM activity monitoring
├── services/
│   ├── alert_service.py             # Alert dispatching (DRY RUN / live)
│   ├── storage_service.py           # SQLite persistence
│   ├── audit_logger.py              # Event audit logging
│   └── enrollment_service.py        # Face enrollment management
├── camera/
│   ├── cam_manager.py               # Camera capture management
│   └── cam_config.py                # Camera configuration
├── api/
│   └── app.py                       # FastAPI REST + WebSocket + MJPEG
├── dashboard/                       # SENTINEL web dashboard (HTML/CSS/JS)
├── data/
│   └── model_artifacts/
│       └── models/
│           ├── buffalo_l/           # InsightFace ArcFace models
│           └── fire_yolov8n.onnx    # YOLOv11n fire detection model
├── scripts/
│   └── train_fire_model.py          # One-time fire model training script
└── tests/
    └── test_models.py               # 184 tests
```

---

## Setup

### Prerequisites

- Python 3.11+
- macOS (Apple Silicon) or Linux
- Webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/rajrahul1106/smart-surveillance-system.git
cd smart-surveillance-system

# Create conda environment
conda create -n surveillance python=3.11 -y
conda activate surveillance

# Install dependencies
pip install -r requirements.txt
```

### Download Model Artifacts

**InsightFace (Face Authentication):**
```bash
# Downloads automatically on first run, or manually:
mkdir -p data/model_artifacts/models/buffalo_l
# Models are downloaded by InsightFace on first load
```

**Fire Detection (YOLOv11n):**
```bash
# Option A: Train from scratch (requires GPU, ~10 min on Colab T4)
pip install ultralytics roboflow
export ROBOFLOW_API_KEY="your_key"
python scripts/train_fire_model.py

# Option B: Use the pre-trained model included in the repo
# data/model_artifacts/models/fire_yolov8n.onnx is already included
```

### Face Enrollment

Enroll your face before first use:
```bash
python main.py
# Open http://localhost:8000/dashboard
# Use the Face Enrollment section to register your identity
```

---

## Usage

```bash
conda activate surveillance
python main.py
```

Open the dashboard: **http://localhost:8000/dashboard**

### Activating the System

1. Stand in front of the camera (works up to 2-3 meters)
2. Perform the SOS gesture: **Palm → Fist → Palm → Fist**
3. Face authentication runs automatically
4. System enters ACTIVE_DETECTION mode
5. Fire, injury, and activity monitoring activates for 30 seconds
6. System returns to IDLE through COOLDOWN

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /dashboard/` | Live surveillance dashboard |
| `GET /api/status` | System status JSON |
| `GET /api/video_feed` | MJPEG video stream |
| `WS /ws/telemetry` | Real-time WebSocket telemetry |
| `GET /api/events?type=FIRE_DETECTED` | Event history |
| `GET /api/transitions` | State transition log |
| `GET /api/enroll/list` | Enrolled identities |

---

## Testing

```bash
pytest tests/ -v
```

All 184 tests should pass.

---

## Configuration

Edit `config.yaml` to adjust:

```yaml
camera:
  index: 0                    # Camera device index
  width: 640
  height: 480

gesture:
  confidence_threshold: 0.90  # Gesture recognition confidence
  sequence_timeout_seconds: 5.0

fire:
  confidence_threshold: 0.35  # YOLO detection confidence
  input_size: 416             # Must match ONNX export size
  score_threshold: 3.0        # Leaky accumulator trigger

detection:
  active_timeout_seconds: 30  # ACTIVE_DETECTION duration
  cooldown_seconds: 5

alerts:
  dry_run: true               # Set to false for live alerts
```

---

## Architecture Highlights

**State-Driven Design:** Each pipeline state is a separate class with `on_enter()`, `on_frame()`, and `on_exit()` hooks. Models load/unload automatically per state, keeping memory usage minimal.

**Event Bus Pattern:** All inter-module communication happens through a publish-subscribe event bus. No direct coupling between models, states, or services.

**Leaky Accumulator:** Fire detection uses a score-based system instead of binary frame counting. A single frame dip doesn't reset detection — the score decays gradually, preventing flapping between detected/not-detected states.

**Per-State Model Loading:** Only the models needed for the current state are loaded in memory. IDLE loads only the gesture model (~50MB). ACTIVE_DETECTION loads fire + injury + activity (~20MB combined). Face model (~500MB) loads only during VERIFYING_IDENTITY and unloads immediately after.

---

## Author

**Rahul Raj**
- B.Tech ECE (AI & ML), MIT World Peace University, Pune (2023–2027)
- GitHub: [rajrahul1106](https://github.com/rajrahul1106)
- LinkedIn: [rahul-raj-5258aa290](https://www.linkedin.com/in/rahul-raj-5258aa290)

---

## License

This project is part of an academic capstone and patent application. All rights reserved.