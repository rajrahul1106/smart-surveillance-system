"""Tests for model classes — lifecycle, predict contracts, and memory."""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import shutil
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.base_model import BaseModel
from models.gesture_model import GestureModel
from models.face_model import FaceModel
from models.fire_model import FireModel
from models.injury_model import InjuryModel
from models.activity_model import ActivityModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mp_modules(landmarker_attr: str, instance):
    """Build a sys.modules dict for mocking the MediaPipe Tasks API.

    MediaPipe 0.10.14+ removed the legacy ``mp.solutions`` namespace; models
    now use ``mediapipe.tasks.python.vision.{Hand,Pose}Landmarker``.

    Returns a dict suitable for ``patch.dict("sys.modules", ...)`` plus a
    fake ``models._mediapipe_assets`` so no .task download is attempted.
    """
    mock_mp = MagicMock()
    mock_mp.Image = MagicMock(return_value=MagicMock())
    mock_mp.ImageFormat = MagicMock()
    mock_mp.ImageFormat.SRGB = "SRGB"

    mock_vision = MagicMock()
    mock_vision.RunningMode = MagicMock()
    mock_vision.RunningMode.IMAGE = "IMAGE"
    mock_landmarker_cls = MagicMock()
    mock_landmarker_cls.create_from_options.return_value = instance
    setattr(mock_vision, landmarker_attr, mock_landmarker_cls)

    mock_tasks_python = MagicMock()
    mock_tasks_python.BaseOptions = MagicMock()
    mock_tasks_python.vision = mock_vision

    mock_tasks = MagicMock()
    mock_tasks.python = mock_tasks_python

    mock_mp.tasks = mock_tasks

    mock_assets = MagicMock()
    mock_assets.hand_landmarker_path = MagicMock(return_value="/tmp/fake_hand.task")
    mock_assets.pose_landmarker_path = MagicMock(return_value="/tmp/fake_pose.task")

    return {
        "mediapipe": mock_mp,
        "mediapipe.tasks": mock_tasks,
        "mediapipe.tasks.python": mock_tasks_python,
        "mediapipe.tasks.python.vision": mock_vision,
        "models._mediapipe_assets": mock_assets,
    }


def _mock_mediapipe_hands():
    """Mediapipe Tasks API mock — HandLandmarker returns 21 landmarks at (0.5, 0.5)."""
    instance = MagicMock()
    instance.close = MagicMock()
    lm = MagicMock(); lm.x = 0.5; lm.y = 0.5; lm.z = 0.0
    result = MagicMock()
    result.hand_landmarks = [[lm] * 21]
    instance.detect.return_value = result
    return _mp_modules("HandLandmarker", instance)


def _mock_mediapipe_hands_no_detection():
    """Mediapipe Tasks API mock — HandLandmarker returns empty hand_landmarks."""
    instance = MagicMock()
    instance.close = MagicMock()
    result = MagicMock()
    result.hand_landmarks = []
    instance.detect.return_value = result
    return _mp_modules("HandLandmarker", instance)


def _mock_mediapipe_pose():
    """Mediapipe Tasks API mock — PoseLandmarker returns 33 standing-pose landmarks."""
    instance = MagicMock()
    instance.close = MagicMock()

    positions = [(0.5, 0.5, 0.0)] * 33
    positions[11] = (0.4, 0.3, 0.0)
    positions[12] = (0.6, 0.3, 0.0)
    positions[23] = (0.4, 0.55, 0.0)
    positions[24] = (0.6, 0.55, 0.0)
    positions[27] = (0.4, 0.9, 0.0)
    positions[28] = (0.6, 0.9, 0.0)

    landmarks = []
    for x, y, z in positions:
        lm = MagicMock(); lm.x = x; lm.y = y; lm.z = z; lm.visibility = 0.99
        landmarks.append(lm)

    result = MagicMock()
    result.pose_landmarks = [landmarks]
    instance.detect.return_value = result
    return _mp_modules("PoseLandmarker", instance)


# ---------------------------------------------------------------------------
# BaseModel contract
# ---------------------------------------------------------------------------

class TestBaseModelContract:
    """Every model must follow the load → predict → unload lifecycle."""

    @pytest.fixture(params=[
        ("gesture", GestureModel),
        ("fire", FireModel),
        ("activity", ActivityModel),
    ])
    def model_cls(self, request):
        return request.param

    def test_predict_raises_when_not_loaded(self, model_cls):
        _, cls = model_cls
        model = cls()
        with pytest.raises(RuntimeError, match="before load"):
            model.predict(_dummy_frame())

    def test_is_loaded_flag(self):
        model = FireModel()
        assert not model.is_loaded
        model.load()
        assert model.is_loaded
        model.unload()
        assert not model.is_loaded

    def test_load_time_tracked(self):
        model = FireModel()
        model.load()
        assert model.load_time_ms > 0
        model.unload()

    def test_double_load_is_noop(self):
        model = FireModel()
        model.load()
        t1 = model.load_time_ms
        model.load()
        assert model.load_time_ms == t1
        model.unload()

    def test_double_unload_is_noop(self):
        model = FireModel()
        model.load()
        model.unload()
        model.unload()  # should not raise
        assert not model.is_loaded


# ---------------------------------------------------------------------------
# GestureModel
# ---------------------------------------------------------------------------

class TestGestureModel:
    @patch.dict("sys.modules", _mock_mediapipe_hands())
    def test_load_unload_lifecycle(self):
        tmpdir = tempfile.mkdtemp()
        try:
            model = GestureModel(near_miss_dir=tmpdir)
            model.load()
            assert model.is_loaded
            model.unload()
            assert not model.is_loaded
        finally:
            shutil.rmtree(tmpdir)

    @patch.dict("sys.modules", _mock_mediapipe_hands())
    def test_predict_returns_required_keys(self):
        tmpdir = tempfile.mkdtemp()
        try:
            model = GestureModel(near_miss_dir=tmpdir)
            model.load()
            result = model.predict(_dummy_frame())
            assert "gesture" in result
            assert "confidence" in result
            assert "landmarks" in result
            assert result["gesture"] in ("palm", "fist", "none")
            assert isinstance(result["confidence"], float)
            model.unload()
        finally:
            shutil.rmtree(tmpdir)

    @patch.dict("sys.modules", _mock_mediapipe_hands_no_detection())
    def test_predict_no_hand_returns_none(self):
        tmpdir = tempfile.mkdtemp()
        try:
            model = GestureModel(near_miss_dir=tmpdir)
            model.load()
            result = model.predict(_dummy_frame())
            assert result["gesture"] == "none"
            assert result["confidence"] == 0.0
            assert result["landmarks"] is None
            model.unload()
        finally:
            shutil.rmtree(tmpdir)

    @patch.dict("sys.modules", _mock_mediapipe_hands())
    def test_near_miss_saves_frame(self):
        tmpdir = tempfile.mkdtemp()
        try:
            model = GestureModel(near_miss_dir=tmpdir)
            model.load()

            # Force a near-miss confidence by patching _classify_gesture
            original_classify = model._classify_gesture
            model._classify_gesture = lambda lm: ("palm", 0.45)

            model.predict(_dummy_frame())

            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0].startswith("near_miss_")
            model.unload()
        finally:
            shutil.rmtree(tmpdir)

    def test_predict_raises_when_not_loaded(self):
        model = GestureModel()
        with pytest.raises(RuntimeError, match="before load"):
            model.predict(_dummy_frame())


# ---------------------------------------------------------------------------
# FaceModel
# ---------------------------------------------------------------------------

def _mock_insightface(faces=None):
    """Build a sys.modules dict for mocking the insightface API.

    The real ``insightface.app.FaceAnalysis`` downloads ~300MB on first
    use and runs an ONNX model — far too heavy for unit tests.  Mock the
    chain ``insightface.app.FaceAnalysis(...).prepare(...) / .get(frame)``
    so it returns the requested fake face list.
    """
    mock_app_instance = MagicMock()
    mock_app_instance.prepare = MagicMock()
    mock_app_instance.get = MagicMock(return_value=list(faces or []))

    mock_face_analysis = MagicMock(return_value=mock_app_instance)
    mock_app_module = MagicMock()
    mock_app_module.FaceAnalysis = mock_face_analysis

    mock_insightface = MagicMock()
    mock_insightface.app = mock_app_module

    return {
        "insightface": mock_insightface,
        "insightface.app": mock_app_module,
    }


def _fake_face(bbox=(100, 100, 200, 200), embedding=None):
    """Build a fake insightface Face object with .bbox and .normed_embedding."""
    f = MagicMock()
    f.bbox = np.asarray(bbox, dtype=np.float32)
    if embedding is None:
        embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
    f.normed_embedding = np.asarray(embedding, dtype=np.float32)
    return f


class TestFaceModel:
    @patch.dict("sys.modules", _mock_insightface())
    def test_load_unload_lifecycle(self):
        model = FaceModel()
        model.load()
        assert model.is_loaded
        assert model._app is not None
        model.unload()
        assert not model.is_loaded
        assert model._app is None

    @patch.dict("sys.modules", _mock_insightface(faces=[_fake_face()]))
    def test_predict_returns_required_keys(self):
        model = FaceModel()
        model.load()
        result = model.predict(_dummy_frame())
        for key in ("user_id", "confidence", "is_live", "is_authorized", "authorized", "bbox"):
            assert key in result, f"Missing key: {key}"
        model.unload()

    @patch.dict("sys.modules", _mock_insightface(faces=[]))
    def test_predict_no_face_in_black_frame(self):
        model = FaceModel()
        model.load()
        result = model.predict(_dummy_frame())
        assert result["authorized"] is False
        assert result["is_authorized"] is False
        assert result["bbox"] is None
        assert result["user_id"] is None
        model.unload()

    @patch.dict("sys.modules", _mock_insightface(faces=[_fake_face()]))
    def test_predict_matches_enrolled_face(self):
        model = FaceModel()
        model.load()

        # Add a known encoding identical to what the mocked face returns,
        # so cosine similarity == 1.0 → above the 0.4 threshold.
        known = np.ones(512, dtype=np.float32) / np.sqrt(512)
        model.add_encoding("Rahul", known)

        result = model.predict(_dummy_frame())
        assert result["is_authorized"] is True
        assert result["user_id"] == "Rahul"
        assert result["confidence"] > 0.99
        assert result["bbox"] == (166, 142, 70, 70)
        model.unload()

    @patch.dict("sys.modules", _mock_insightface(faces=[_fake_face(
        embedding=np.array([1.0] + [0.0] * 511, dtype=np.float32),
    )]))
    def test_predict_below_threshold_returns_unauthorized(self):
        model = FaceModel()
        model.load()
        # Known encoding is orthogonal → cosine similarity 0.0
        model.add_encoding("Other", np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32))

        result = model.predict(_dummy_frame())
        assert result["is_authorized"] is False
        assert result["user_id"] is None
        model.unload()

    def test_predict_raises_when_not_loaded(self):
        model = FaceModel()
        with pytest.raises(RuntimeError):
            model.predict(_dummy_frame())

    @patch.dict("sys.modules", _mock_insightface())
    def test_unload_clears_encodings(self):
        # Use a tmp encodings_path so any real on-disk identities don't bleed
        # into this test.
        with tempfile.TemporaryDirectory() as tmp:
            model = FaceModel(encodings_path=os.path.join(tmp, "enc.pkl"))
            model.load()
            model.add_encoding("user_a", np.zeros(512, dtype=np.float32))
            model.add_encoding("user_b", np.zeros(512, dtype=np.float32))
            assert len(model.get_enrolled_names()) == 2
            model.unload()
            assert model._known_encodings == {}
            assert model._app is None


_ENROLLED = np.ones(512, dtype=np.float32) / np.sqrt(512)
# Orthogonal to _ENROLLED (cosine similarity 0): a stranger.
_STRANGER = np.array([1.0, -1.0] * 256, dtype=np.float32) / np.sqrt(512)


def _face_model_with(faces=(), **kwargs):
    """A loaded FaceModel whose mocked insightface app returns *faces*.

    Uses a missing encodings file so real enrollments never leak in.
    """
    with patch.dict("sys.modules", _mock_insightface(faces=list(faces))):
        model = FaceModel(encodings_path="/nonexistent/enc.pkl", **kwargs)
        model.load()
    return model


class TestFaceModelMultiFace:
    def test_enrolled_face_authorized_and_stranger_unknown(self):
        model = _face_model_with([
            _fake_face(bbox=(40, 60, 140, 180), embedding=_ENROLLED),
            _fake_face(bbox=(400, 60, 500, 180), embedding=_STRANGER),
        ])
        model.set_mode("presence")
        model.add_encoding("Rahul Raj", _ENROLLED)

        first = {f["bbox"]: f for f in model.predict(_dummy_frame())["faces"]}
        rahul, stranger = first[(40, 60, 100, 120)], first[(400, 60, 100, 120)]
        assert rahul["status"] == "AUTHORIZED"
        assert rahul["identity"] == "Rahul Raj"
        assert stranger["status"] == "UNCERTAIN"
        assert stranger["identity"] is None

        for _ in range(FaceModel.UNKNOWN_CONFIRM_FRAMES - 1):
            result = model.predict(_dummy_frame())
        faces = {f["track_id"]: f for f in result["faces"]}
        assert faces[rahul["track_id"]]["status"] == "AUTHORIZED"
        assert faces[rahul["track_id"]]["identity"] == "Rahul Raj"
        assert faces[rahul["track_id"]]["confidence"] == pytest.approx(100.0, abs=0.1)
        assert faces[stranger["track_id"]]["status"] == "UNKNOWN"
        assert faces[stranger["track_id"]]["identity"] is None
        model.unload()

    def test_presence_counts_with_zero_one_and_three_faces(self):
        model = _face_model_with([])
        model.set_mode("presence")
        model.add_encoding("Rahul Raj", _ENROLLED)

        assert model.predict(_dummy_frame())["presence"] == {
            "authorized": [], "unknown_count": 0, "uncertain_count": 0, "total": 0,
        }

        model._app.get.return_value = [_fake_face(bbox=(40, 60, 140, 180), embedding=_ENROLLED)]
        assert model.predict(_dummy_frame())["presence"] == {
            "authorized": ["Rahul Raj"], "unknown_count": 0, "uncertain_count": 0, "total": 1,
        }

        model._app.get.return_value = [
            _fake_face(bbox=(40, 60, 140, 180), embedding=_ENROLLED),
            _fake_face(bbox=(250, 60, 350, 180), embedding=_STRANGER),
            _fake_face(bbox=(450, 60, 550, 180), embedding=_STRANGER),
        ]
        assert model.predict(_dummy_frame())["presence"] == {
            "authorized": ["Rahul Raj"], "unknown_count": 0, "uncertain_count": 2, "total": 3,
        }
        for _ in range(FaceModel.UNKNOWN_CONFIRM_FRAMES - 1):
            presence = model.predict(_dummy_frame())["presence"]
        assert presence == {
            "authorized": ["Rahul Raj"], "unknown_count": 2, "uncertain_count": 0, "total": 3,
        }
        model.unload()

    def test_top_level_keys_describe_the_authorized_face(self):
        # The stranger is the larger face; the legacy keys must still
        # describe the AUTHORIZED one.
        model = _face_model_with([
            _fake_face(bbox=(300, 50, 600, 400), embedding=_STRANGER),
            _fake_face(bbox=(40, 60, 140, 180), embedding=_ENROLLED),
        ])
        model.set_mode("presence")
        model.add_encoding("Rahul Raj", _ENROLLED)

        result = model.predict(_dummy_frame())
        for key in ("user_id", "name", "label", "confidence", "is_live",
                    "authorized", "is_authorized", "bbox", "frame_size"):
            assert key in result, f"Missing key: {key}"
        assert result["user_id"] == "Rahul Raj"
        assert result["name"] == "Rahul Raj"
        assert result["label"] == "Rahul Raj"
        assert result["authorized"] is True
        assert result["is_authorized"] is True
        assert result["is_live"] is True
        assert result["confidence"] == pytest.approx(1.0, abs=1e-3)
        assert result["bbox"] == (40, 60, 100, 120)
        assert result["frame_size"] == (640, 480)
        model.unload()

    def test_top_level_keys_fall_back_to_the_largest_face(self):
        model = _face_model_with([
            _fake_face(bbox=(40, 60, 140, 180), embedding=_STRANGER),
            _fake_face(bbox=(300, 50, 600, 400), embedding=_STRANGER),
        ])
        model.set_mode("presence")
        model.add_encoding("Rahul Raj", _ENROLLED)

        result = model.predict(_dummy_frame())
        assert result["is_authorized"] is False
        assert result["user_id"] is None
        assert result["label"] == "UNKNOWN"
        assert result["bbox"] == (300, 50, 300, 350)
        model.unload()

    def test_person_score_is_the_max_over_their_samples(self):
        model = _face_model_with([_fake_face(embedding=_ENROLLED)])
        model.add_encoding("Rahul Raj", _STRANGER)   # a poor sample
        model.add_encoding("Rahul Raj", _ENROLLED)   # a good sample

        face = model.predict(_dummy_frame())["faces"][0]
        assert face["status"] == "AUTHORIZED"
        assert face["identity"] == "Rahul Raj"
        assert face["confidence"] == pytest.approx(100.0, abs=0.1)
        model.unload()

    def test_presence_mode_uses_full_frame_and_verify_mode_crops(self):
        model = _face_model_with([_fake_face(bbox=(100, 100, 200, 200))])
        frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)

        assert model.mode == "verify"
        assert model.predict(frame)["bbox"] == (166, 142, 70, 70)  # remapped out of the crop
        assert not np.array_equal(model._app.get.call_args[0][0], frame)

        model.set_mode("presence")
        assert model.predict(frame)["bbox"] == (100, 100, 100, 100)  # full frame, no remap
        np.testing.assert_array_equal(model._app.get.call_args[0][0], frame)
        model.unload()

    def test_caps_at_max_faces_keeping_the_largest(self):
        # Eight non-overlapping faces, widths 20, 25, ..., 55 px.
        faces = [
            _fake_face(bbox=(i * 75, 10, i * 75 + 20 + 5 * i, 30 + 5 * i))
            for i in range(8)
        ]
        model = _face_model_with(faces, max_faces=6)
        model.set_mode("presence")

        result = model.predict(_dummy_frame())
        assert len(result["faces"]) == 6
        assert sorted(f["bbox"][2] for f in result["faces"]) == [30, 35, 40, 45, 50, 55]
        model.unload()

    def test_loads_only_needed_modules_and_sets_det_thresh_per_mode(self):
        modules = _mock_insightface()
        with patch.dict("sys.modules", modules):
            model = FaceModel(encodings_path="/nonexistent/enc.pkl")
            model.load()

        kwargs = modules["insightface.app"].FaceAnalysis.call_args.kwargs
        assert kwargs["allowed_modules"] == ["detection", "recognition"]
        prepare = model._app.prepare.call_args.kwargs
        assert prepare["det_size"] == (960, 960)
        assert prepare["det_thresh"] == FaceModel.VERIFY_DET_THRESH

        model.set_mode("presence")
        assert model._app.det_model.det_thresh == FaceModel.PRESENCE_DET_THRESH
        model.set_mode("verify")
        assert model._app.det_model.det_thresh == FaceModel.VERIFY_DET_THRESH
        with pytest.raises(ValueError):
            model.set_mode("crowd")
        model.unload()

    def test_unload_resets_tracks_and_mode(self):
        model = _face_model_with([_fake_face()])
        model.set_mode("presence")
        model.predict(_dummy_frame())
        assert model._tracker.tracks

        model.unload()
        assert model._tracker.tracks == []
        assert model.mode == "verify"

    def test_thread_cap_applies_in_presence_mode_only(self):
        # insightface only forwards providers to its sessions, so the cap is
        # applied by rebuilding them; verify mode keeps the runtime default.
        model = _face_model_with([], intra_op_threads=1)
        det = MagicMock(model_file="det_10g.onnx")
        rec = MagicMock(model_file="w600k_r50.onnx")
        model._app.models = {"detection": det, "recognition": rec}

        def fake_session(path, sess_options, providers):
            return (path, sess_options.intra_op_num_threads)

        with patch("onnxruntime.InferenceSession", side_effect=fake_session) as ctor:
            model.set_mode("presence")
            assert det.session == ("det_10g.onnx", 1)
            assert rec.session == ("w600k_r50.onnx", 1)
            model.set_mode("presence")  # already capped: no rebuild
            assert ctor.call_count == 2
            model.set_mode("verify")    # back to the runtime default
            assert det.session == ("det_10g.onnx", 0)
            assert rec.session == ("w600k_r50.onnx", 0)
        model.unload()

    def test_no_thread_cap_leaves_insightface_sessions_alone(self):
        model = _face_model_with([])
        model._app.models = {"detection": MagicMock(model_file="det_10g.onnx")}
        with patch("onnxruntime.InferenceSession") as ctor:
            model.set_mode("presence")
        ctor.assert_not_called()
        model.unload()


# ---------------------------------------------------------------------------
# FireModel (YOLO11s ONNX)
# ---------------------------------------------------------------------------

_REAL_FIRE_MODEL = os.path.join(
    _project_root, "data", "model_artifacts", "models", "fire_yolo11s_480.onnx",
)
requires_real_fire_model = pytest.mark.skipif(
    not os.path.isfile(_REAL_FIRE_MODEL), reason="fire_yolo11s_480.onnx not present",
)


def _yolo_fire_output(
    cx: float = 240, cy: float = 240,
    w: float = 240, h: float = 240,
    fire_conf: float = 0.9, smoke_conf: float = 0.01,
) -> np.ndarray:
    """Create a synthetic YOLO output tensor with one detection.

    The box is in 480×480 model-input pixels (centred).  Shape: ``(1, 6, 1)``
    — batch=1, channels=4(box)+2(classes), preds=1.
    """
    data = np.zeros((1, 6, 1), dtype=np.float32)
    data[0, 0, 0] = cx
    data[0, 1, 0] = cy
    data[0, 2, 0] = w
    data[0, 3, 0] = h
    data[0, 4, 0] = fire_conf
    data[0, 5, 0] = smoke_conf
    return data


def _attach_mock_session(model, run_return):
    """Wire a MagicMock ONNX session onto *model* so inference works."""
    mock_session = MagicMock()
    mock_session.run.return_value = [run_return]
    model._session = mock_session
    model._input_name = "images"
    model._class_names = {0: "fire", 1: "smoke"}


class TestFireModel:
    def test_load_unload_lifecycle(self):
        model = FireModel()
        model.load()
        assert model.is_loaded
        model.unload()
        assert not model.is_loaded

    def test_predict_returns_required_keys(self):
        model = FireModel()
        model.load()
        result = model.predict(_dummy_frame())
        for key in ("detected", "fire_detected", "severity_level", "confidence", "bbox"):
            assert key in result, f"Missing key: {key}"
        model.unload()

    def test_no_fire_in_black_frame(self):
        model = FireModel()
        model.load()
        result = model.predict(_dummy_frame())
        assert result["fire_detected"] is False
        assert result["severity_level"] == "none"
        model.unload()

    def test_fire_detected_with_leaky_score(self):
        # ONNX detections feed the leaky accumulator.  score_threshold=3.0
        # with BOOST_RATE=1.0 means fire is confirmed on the 3rd frame.
        model = FireModel(score_threshold=3.0)
        model.load()

        fire_output = _yolo_fire_output(fire_conf=0.9)
        _attach_mock_session(model, fire_output)

        frame = _dummy_frame(100, 100)

        # Frames 1-2: score accumulates but hasn't reached threshold.
        for i in range(2):
            r = model.predict(frame)
            assert r["fire_detected"] is False, f"frame {i+1} triggered too early"

        # Frame 3: score reaches threshold — fire confirmed.
        result = model.predict(frame)
        assert result["fire_detected"] is True
        assert result["severity_level"] != "none"
        assert result["confidence"] > 0
        assert result["frame_size"] == (100, 100)
        assert result["bbox"] is not None
        model.unload()

    def test_fire_score_decays_without_detections(self):
        model = FireModel(score_threshold=3.0)
        model.load()

        fire_output = _yolo_fire_output(fire_conf=0.9)
        no_fire = _yolo_fire_output(fire_conf=0.01)
        frame = _dummy_frame(100, 100)

        # Build up score with 2 fire frames.
        _attach_mock_session(model, fire_output)
        model.predict(frame)
        model.predict(frame)
        assert model._fire_score == 2.0

        # Switch to empty detections — score should decay.
        _attach_mock_session(model, no_fire)
        result = model.predict(frame)
        assert result["fire_detected"] is False
        assert model._fire_score < 2.0
        model.unload()

    def test_no_crash_without_onnx_model(self):
        # When the ONNX file is absent, every frame returns not-detected.
        model = FireModel(model_path="/nonexistent/fire_yolo11s_480.onnx")
        model.load()
        white = np.full((100, 100, 3), 240, dtype=np.uint8)
        for _ in range(10):
            r = model.predict(white)
        assert r["fire_detected"] is False
        model.unload()

    def test_severity_levels(self):
        model = FireModel()
        assert model._classify_severity(0.35) == "uncontrollable"
        assert model._classify_severity(0.20) == "hazardous"
        assert model._classify_severity(0.08) == "controllable"
        assert model._classify_severity(0.025) == "small"

    def test_defaults_point_at_yolo11s_480(self):
        assert FireModel.MODEL_PATH.endswith("fire_yolo11s_480.onnx")
        assert FireModel.LABELS_PATH.endswith("fire_yolo11s_labels.json")
        assert FireModel.INPUT_SIZE == 480
        assert FireModel.CONFIDENCE_THRESHOLD == 0.35

    def test_load_trusts_onnx_over_config_and_labels_file(self, tmp_path, caplog):
        model_file = tmp_path / "fire.onnx"
        model_file.write_bytes(b"")  # only has to exist; the session is mocked
        labels = tmp_path / "labels.json"
        labels.write_text('{"classes": ["smoke", "fire"]}')  # disagrees with ONNX

        model_input = MagicMock(shape=[1, 3, 480, 480])
        model_input.name = "images"
        session = MagicMock()
        session.get_inputs.return_value = [model_input]
        session.get_modelmeta.return_value.custom_metadata_map = {
            "description": "Ultralytics YOLO11s model trained on fire.yaml",
            "names": "{0: 'fire', 1: 'smoke'}",
        }

        model = FireModel(model_path=str(model_file), labels_path=str(labels), input_size=416)
        caplog.set_level("INFO", logger="models.fire_model")
        with patch("onnxruntime.InferenceSession", return_value=session):
            model.load()

        assert model._input_size == 480                          # ONNX shape wins
        assert model._class_names == {0: "fire", 1: "smoke"}      # ONNX names win
        assert "using the ONNX metadata" in caplog.text           # mismatch warned
        assert ("FireModel loaded - YOLO11s ONNX, input=[1,3,480,480], "
                "classes=['fire','smoke'], conf=0.35, iou=0.45, trigger=3.0") in caplog.text
        model.unload()

    def test_session_uses_configured_thread_cap(self, tmp_path):
        model_file = tmp_path / "fire.onnx"
        model_file.write_bytes(b"")  # only has to exist; the session is mocked
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(shape=[1, 3, 480, 480])]
        session.get_modelmeta.return_value.custom_metadata_map = {}

        model = FireModel(model_path=str(model_file), intra_op_threads=4)
        with patch("onnxruntime.InferenceSession", return_value=session) as ctor:
            model.load()
        assert ctor.call_args.kwargs["sess_options"].intra_op_num_threads == 4
        model.unload()


def test_session_options_caps_intra_op_threads():
    from models.ort_options import session_options

    assert session_options(0).intra_op_num_threads == 0  # runtime default
    assert session_options(3).intra_op_num_threads == 3


@requires_real_fire_model
class TestFireModelRealOnnx:
    """Runs the real fire_yolo11s_480.onnx; skipped when the file isn't present."""

    def test_signature(self):
        model = FireModel(model_path=_REAL_FIRE_MODEL)
        model.load()
        session = model._session
        assert session.get_inputs()[0].shape == [1, 3, 480, 480]
        blob = np.zeros((1, 3, 480, 480), dtype=np.float32)
        output = session.run(None, {model._input_name: blob})[0]
        assert output.ndim == 3 and output.shape[:2] == (1, 6) and output.shape[2] > 0
        assert [model._class_names[i] for i in sorted(model._class_names)] == ["fire", "smoke"]
        assert model._input_size == 480
        model.unload()

    @pytest.mark.parametrize("value", [0, 255], ids=["black", "bright_white"])
    def test_blank_frames_never_trigger(self, value):
        # Basic guard against bright-light false positives.
        model = FireModel(model_path=_REAL_FIRE_MODEL)
        model.load()
        frame = np.full((480, 640, 3), value, dtype=np.uint8)
        results = [model.predict(frame) for _ in range(10)]
        assert not any(r["detected"] for r in results)
        assert results[-1]["fire_score"] == 0.0  # not even one raw detection
        model.unload()


# ---------------------------------------------------------------------------
# InjuryModel
# ---------------------------------------------------------------------------

class TestInjuryModel:
    @patch.dict("sys.modules", _mock_mediapipe_pose())
    def test_load_unload_lifecycle(self):
        model = InjuryModel()
        model.load()
        assert model.is_loaded
        model.unload()
        assert not model.is_loaded

    @patch.dict("sys.modules", _mock_mediapipe_pose())
    def test_predict_returns_required_keys(self):
        model = InjuryModel()
        model.load()
        result = model.predict(_dummy_frame())
        for key in ("detected", "injury_detected", "posture_type", "confidence", "pose_landmarks"):
            assert key in result, f"Missing key: {key}"
        assert result["posture_type"] in ("standing", "sitting", "lying", "collapsed", "none")
        model.unload()

    @patch.dict("sys.modules", _mock_mediapipe_pose())
    def test_standing_posture_not_injury(self):
        model = InjuryModel()
        model.load()
        result = model.predict(_dummy_frame())
        assert result["posture_type"] == "standing"
        assert result["injury_detected"] is False
        model.unload()

    def test_predict_raises_when_not_loaded(self):
        model = InjuryModel()
        with pytest.raises(RuntimeError):
            model.predict(_dummy_frame())


# ---------------------------------------------------------------------------
# ActivityModel (MobileNetV3-Small ONNX)
# ---------------------------------------------------------------------------

# Raw logits — the ONNX model does NOT apply softmax.  Class order is
# [normal, robbery, violence]; comments give softmax(logits) in percent.
_NORMAL_LOGITS = [[3.7, -3.9, 1.4]]        # normal 90.85 / robbery 0.05 / violence 9.11
_VIOLENCE_LOGITS = [[-2.0, -1.0, 4.0]]     # violence 99.09
_WEAK_ROBBERY_LOGITS = [[0.0, 0.3, -0.5]]  # robbery on top at 45.66 (< 60 % threshold)


def _offline_activity_model(**kwargs) -> ActivityModel:
    """ActivityModel pointed at non-existent artefacts, so tests never depend
    on the real ONNX file — inference goes through a mock session instead."""
    return ActivityModel(
        model_path="/nonexistent/sentinel_activity_mnv3.onnx",
        label_map_path="/nonexistent/activity_label_map.json",
        **kwargs,
    )


def _attach_mock_activity_session(model, logits):
    """Wire a MagicMock ONNX session onto *model* that returns *logits*."""
    mock_session = MagicMock()
    mock_session.run.return_value = [np.asarray(logits, dtype=np.float32)]
    model._session = mock_session
    return mock_session


class TestActivityModel:
    def test_load_unload_lifecycle(self, tmp_path):
        model_file = tmp_path / "activity.onnx"
        model_file.write_bytes(b"")  # only has to exist; the session is mocked
        label_map = tmp_path / "labels.json"
        label_map.write_text(
            '{"classes": ["normal", "robbery", "violence"], "img_size": 224}'
        )
        fake_input = MagicMock()
        fake_input.name = "input"
        fake_input.shape = ["batch", 3, 224, 224]
        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [fake_input]

        model = ActivityModel(model_path=str(model_file), label_map_path=str(label_map))
        with patch("onnxruntime.InferenceSession", return_value=fake_session):
            model.load()
        assert model.is_loaded
        assert model._session is fake_session
        assert model._input_name == "input"
        assert model._classes == ["normal", "robbery", "violence"]
        assert model._img_size == 224

        model.unload()
        assert not model.is_loaded
        assert model._session is None

    def test_predict_returns_required_keys(self):
        model = _offline_activity_model()
        model.load()
        _attach_mock_activity_session(model, _NORMAL_LOGITS)
        result = model.predict(_dummy_frame())
        for key in (
            # Contract read by the annotator / pipeline / alert service
            "detected", "suspicious", "activity_type", "confidence", "bbox",
            "frame_size",
            # Additive keys from the ONNX classifier
            "activity_detected", "severity_level", "confidence_pct",
            "activity_score", "class_probabilities",
        ):
            assert key in result, f"Missing key: {key}"
        assert result["bbox"] is None  # whole-frame classifier
        assert result["frame_size"] == (640, 480)
        model.unload()

    def test_preprocess_matches_model_signature(self):
        # The ONNX input is "input", float32 NCHW [1, 3, 224, 224], RGB with
        # ImageNet normalisation.  A pure-blue BGR frame checks channel order.
        model = _offline_activity_model()
        model.load()
        session = _attach_mock_activity_session(model, _NORMAL_LOGITS)
        blue = np.zeros((240, 320, 3), dtype=np.uint8)
        blue[:, :, 0] = 255  # BGR blue
        model.predict(blue)

        blob = session.run.call_args[0][1]["input"]
        assert blob.shape == (1, 3, 224, 224)
        assert blob.dtype == np.float32
        # After the BGR→RGB flip: R=0, G=0, B=1.0, then (x - mean) / std
        np.testing.assert_allclose(
            blob[0, :, 0, 0], [-2.1179, -2.0357, 2.64], atol=1e-3,
        )
        model.unload()

    def test_softmax_applied_to_raw_logits(self):
        model = _offline_activity_model()
        model.load()
        _attach_mock_activity_session(model, _NORMAL_LOGITS)
        result = model.predict(_dummy_frame())
        # softmax([3.7, -3.9, 1.4]) → normal ≈ 90.85 %, not the raw logit 3.7
        assert result["confidence"] == pytest.approx(90.85, abs=0.01)
        probs = result["class_probabilities"]
        assert probs["normal"] == pytest.approx(90.85, abs=0.01)
        assert sum(probs.values()) == pytest.approx(100.0, abs=0.05)
        assert result["detected"] is False
        assert result["activity_type"] == "none"
        model.unload()

    def test_leaky_accumulator_confirms_after_three_suspicious_frames(self):
        model = _offline_activity_model()
        model.load()
        _attach_mock_activity_session(model, _VIOLENCE_LOGITS)
        frame = _dummy_frame(240, 320)

        # Frames 1-2: score climbs 1.0 → 2.0 but hasn't reached the 3.0 trigger.
        for i in range(2):
            r = model.predict(frame)
            assert r["detected"] is False, f"frame {i+1} triggered too early"

        # Frame 3: score reaches 3.0 — activity confirmed.
        r = model.predict(frame)
        assert r["detected"] is True
        assert r["suspicious"] is True
        assert r["activity_type"] == "violence"
        assert r["severity_level"] == "violence"
        assert r["confidence"] == 99.0  # softmax 99.09 %, capped at 99 like FireModel
        assert r["frame_size"] == (320, 240)

        # One normal frame decays the score (3.0 → 2.7) instead of zeroing it...
        _attach_mock_activity_session(model, _NORMAL_LOGITS)
        r = model.predict(frame)
        assert r["detected"] is False
        assert r["activity_score"] == pytest.approx(2.7)

        # ...so a single further suspicious frame re-confirms (2.7 → 3.7).
        _attach_mock_activity_session(model, _VIOLENCE_LOGITS)
        assert model.predict(frame)["detected"] is True
        model.unload()

    def test_confirmed_alert_reports_trigger_class_while_decaying(self):
        model = _offline_activity_model()
        model.load()
        frame = _dummy_frame(240, 320)
        _attach_mock_activity_session(model, _VIOLENCE_LOGITS)
        for _ in range(4):
            model.predict(frame)  # score 4.0

        # Normal frame: score 3.7 is still above the trigger, so the alert
        # holds — and must keep naming the class that raised it.
        _attach_mock_activity_session(model, _NORMAL_LOGITS)
        r = model.predict(frame)
        assert r["detected"] is True
        assert r["activity_type"] == "violence"
        assert r["confidence"] == 99.0  # softmax 99.09 %, capped at 99 like FireModel
        assert r["class_probabilities"]["normal"] == pytest.approx(90.85, abs=0.01)
        model.unload()

    def test_partial_boost_below_confidence_threshold(self):
        # Suspicious class on top but under 60 % → +0.4 per frame, not +1.0.
        model = _offline_activity_model()
        model.load()
        _attach_mock_activity_session(model, _WEAK_ROBBERY_LOGITS)
        frame = _dummy_frame(240, 320)

        r = model.predict(frame)
        assert r["activity_score"] == pytest.approx(0.4)
        assert r["detected"] is False
        for _ in range(6):
            r = model.predict(frame)
        assert r["detected"] is False  # 7 × 0.4 = 2.8
        r = model.predict(frame)
        assert r["detected"] is True   # 8 × 0.4 = 3.2
        assert r["activity_type"] == "robbery"
        model.unload()

    def test_no_crash_without_onnx_model(self, caplog):
        # A missing ONNX file logs an error and every frame reports not-detected.
        model = _offline_activity_model()
        model.load()
        assert model.is_loaded
        assert model._session is None
        assert any(
            rec.levelname == "ERROR" and "not found" in rec.getMessage()
            for rec in caplog.records
        )
        for _ in range(5):
            r = model.predict(_dummy_frame())
        assert r["detected"] is False
        assert r["activity_type"] == "none"
        model.unload()

    def test_predict_raises_when_not_loaded(self):
        model = ActivityModel()
        with pytest.raises(RuntimeError):
            model.predict(_dummy_frame())

    def test_session_uses_configured_thread_cap(self, tmp_path):
        model_file = tmp_path / "activity.onnx"
        model_file.write_bytes(b"")  # only has to exist; the session is mocked
        fake_input = MagicMock(shape=["batch", 3, 224, 224])
        fake_input.name = "input"
        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [fake_input]

        model = ActivityModel(
            model_path=str(model_file),
            label_map_path=str(tmp_path / "missing.json"),
            intra_op_threads=1,
        )
        with patch("onnxruntime.InferenceSession", return_value=fake_session) as ctor:
            model.load()
        assert ctor.call_args.kwargs["sess_options"].intra_op_num_threads == 1
        model.unload()


# ---------------------------------------------------------------------------
# Memory: unload reduces footprint
# ---------------------------------------------------------------------------

class TestMemoryCleanup:
    def test_unload_triggers_gc(self):
        """Verify that unload() calls gc.collect (tested indirectly via
        the is_loaded flag and attribute cleanup)."""
        model = _offline_activity_model()
        model.load()
        _attach_mock_activity_session(model, _VIOLENCE_LOGITS)
        model.predict(_dummy_frame())
        assert model.is_loaded
        model.unload()
        assert not model.is_loaded
        assert model._session is None
        assert model._activity_score == 0.0

    @patch.dict("sys.modules", _mock_insightface())
    def test_unload_reduces_memory(self):
        """Load a model, measure RSS, unload, verify RSS doesn't grow."""
        import psutil, os

        process = psutil.Process(os.getpid())

        model = FaceModel()
        model.load()
        # Inflate encodings to make memory impact measurable
        # (5000 ArcFace 512-d float32 vectors ≈ 10 MB)
        model._known_encodings = {
            f"user_{i}": [np.random.rand(512).astype(np.float32)] for i in range(5000)
        }

        rss_loaded = process.memory_info().rss

        model.unload()
        gc.collect()

        rss_unloaded = process.memory_info().rss

        # After unloading 5000 encodings, RSS should not have grown
        # (allow small variance from OS memory management)
        assert rss_unloaded <= rss_loaded + 2 * 1024 * 1024  # 2 MB tolerance
