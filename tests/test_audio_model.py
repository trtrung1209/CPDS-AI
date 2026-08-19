import json
import numpy as np
import pytest
import wave
from types import ModuleType

from src.inference.verify_audio import infer_audio, load_labels, preprocess_audio, verify_audio_model


def test_default_audio_label_order_is_explicit():
    assert load_labels() == ["noise", "cry"]


def test_invalid_audio_labels_are_rejected(tmp_path):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text('["noise", "alarm"]', encoding="utf-8")

    with pytest.raises(ValueError, match="'cry'"):
        load_labels(labels_path)


def test_missing_audio_file_fails_before_importing_ml_dependency(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        preprocess_audio(tmp_path / "missing.wav")


def test_preprocess_audio_has_stable_onnx_shape_and_normalization(monkeypatch, tmp_path):
    fake_librosa = ModuleType("librosa")
    fake_librosa.load = lambda path, sr, duration: (np.array([0.5, -0.5], dtype=np.float32), sr)
    fake_librosa.feature = type(
        "Feature", (), {"melspectrogram": staticmethod(lambda **_kwargs: np.array([[2.0, 4.0], [6.0, 8.0]]))}
    )()
    fake_librosa.power_to_db = lambda spectrogram, ref: spectrogram
    monkeypatch.setitem(__import__("sys").modules, "librosa", fake_librosa)
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

    result = preprocess_audio(audio_path, sr=4, duration=1.0)

    assert result.shape == (1, 1, 2, 2)
    assert result.dtype == np.float32
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_invalid_label_file_is_rejected(tmp_path):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError):
        load_labels(labels_path)


def test_audio_inference_uses_softmax_and_explicit_label_order(monkeypatch, tmp_path):
    class FakeSession:
        def __init__(self, _path):
            pass

        def get_inputs(self):
            return [type("Input", (), {"name": "audio"})()]

        def get_outputs(self):
            return [type("Output", (), {"name": "scores"})()]

        def run(self, output_names, inputs):
            assert output_names == ["scores"]
            assert list(inputs) == ["audio"]
            return [np.array([[0.0, 2.0]], dtype=np.float32)]

    fake_runtime = ModuleType("onnxruntime")
    fake_runtime.InferenceSession = FakeSession
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_runtime)
    monkeypatch.setattr(
        "src.inference.verify_audio.preprocess_audio",
        lambda _path: np.zeros((1, 1, 128, 63), dtype=np.float32),
    )
    model_path = tmp_path / "audio.onnx"
    model_path.touch()

    result = infer_audio(model_path, "sample.wav")

    assert result["is_crying"] is True
    assert result["confidence"] == pytest.approx(0.880797, abs=1e-6)
    assert result["probabilities"] == pytest.approx({"noise": 0.119203, "cry": 0.880797}, abs=1e-6)


def test_missing_audio_model_is_rejected_without_starting_onnx(tmp_path):
    with pytest.raises(FileNotFoundError, match="Audio model does not exist"):
        infer_audio(tmp_path / "missing.onnx", "sample.wav")


def test_audio_inference_rejects_unexpected_model_output(monkeypatch, tmp_path):
    class FakeSession:
        def __init__(self, _path):
            pass

        def get_inputs(self):
            return [type("Input", (), {"name": "audio"})()]

        def get_outputs(self):
            return [type("Output", (), {"name": "scores"})()]

        def run(self, _output_names, _inputs):
            return [np.zeros((1, 3), dtype=np.float32)]

    fake_runtime = ModuleType("onnxruntime")
    fake_runtime.InferenceSession = FakeSession
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_runtime)
    monkeypatch.setattr("src.inference.verify_audio.preprocess_audio", lambda _path: np.zeros((1, 1, 128, 63)))
    model_path = tmp_path / "audio.onnx"
    model_path.touch()

    with pytest.raises(ValueError, match="Expected 2 audio logits"):
        infer_audio(model_path, "sample.wav")


def test_verify_audio_writes_result_atomically_after_success(monkeypatch, tmp_path):
    expected = {"is_crying": True, "confidence": 0.9}
    monkeypatch.setattr("src.inference.verify_audio.infer_audio", lambda *_args: expected)
    monkeypatch.setattr("src.inference.verify_audio.get_next_run_dir", lambda: tmp_path)

    assert verify_audio_model("model.onnx", "audio.wav") == expected
    assert json.loads((tmp_path / "audio_verified.json").read_text(encoding="utf-8")) == expected

# MOCK: Bỏ qua test nếu file ONNX chưa được tải về máy
@pytest.fixture
def dummy_audio_file(tmp_path):
    # Tạo một file âm thanh rác (.wav) 2 giây
    file_path = tmp_path / "dummy_audio.wav"
    sr = 16000
    audio_data = (np.random.uniform(-1, 1, int(sr * 2.0)) * 32767).astype(np.int16)
    with wave.open(str(file_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sr)
        output.writeframes(audio_data.tobytes())
    return str(file_path)

def test_audio_preprocessing_shape(dummy_audio_file):
    pytest.importorskip("librosa")
    input_data = preprocess_audio(dummy_audio_file)
    
    # Kích thước phải là (1, 1, 128, T)
    assert len(input_data.shape) == 4
    assert input_data.shape[0] == 1
    assert input_data.shape[1] == 1
    assert input_data.shape[2] == 128
    assert input_data.shape[3] == 63
    assert input_data.dtype == np.float32
    assert np.isfinite(input_data).all()
    assert 0.0 <= input_data.min() <= input_data.max() <= 1.0
