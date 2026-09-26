import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.daemon.config import Config
from src.daemon.backend import ONNXBackend

@pytest.fixture
def mock_config():
    config = Config()
    config.vision_model_path = "dummy_vision.onnx"
    config.audio_model_path = "dummy_audio.onnx"
    config.audio_labels_path = ""
    return config

@patch("ultralytics.YOLO")
@patch("onnxruntime.InferenceSession")
@patch("src.inference.verify_audio.load_labels")
def test_backend_init(mock_load, mock_ort, mock_yolo, mock_config):
    mock_load.return_value = ["noise", "cry"]
    backend = ONNXBackend(mock_config)
    mock_yolo.assert_called_once_with("dummy_vision.onnx", task="detect")
    mock_ort.assert_called_once_with("dummy_audio.onnx")
    assert backend.audio_labels == ["noise", "cry"]

@patch("ultralytics.YOLO")
@patch("onnxruntime.InferenceSession")
@patch("src.inference.verify_audio.load_labels")
@patch("cv2.imdecode")
@patch("src.inference.verify_vision.summarize_detections")
def test_predict_vision(mock_summarize, mock_imdecode, mock_load, mock_ort, mock_yolo, mock_config):
    mock_load.return_value = ["noise", "cry"]
    mock_yolo_instance = MagicMock()
    mock_yolo.return_value = mock_yolo_instance
    
    # Mock YOLO result
    mock_result = MagicMock()
    mock_result.boxes = [MagicMock()]
    mock_result.names = {0: "child"}
    mock_yolo_instance.return_value = [mock_result]
    
    mock_imdecode.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_summarize.return_value = {"child_detected": True}
    
    backend = ONNXBackend(mock_config)
    res = backend.predict_vision(b"dummy")
    assert res["child_detected"] is True

@patch("ultralytics.YOLO")
@patch("onnxruntime.InferenceSession")
@patch("src.inference.verify_audio.load_labels")
@patch("src.inference.verify_audio.preprocess_audio")
def test_predict_audio(mock_preprocess, mock_load, mock_ort, mock_yolo, mock_config):
    mock_load.return_value = ["noise", "cry"]
    mock_ort_instance = MagicMock()
    mock_ort.return_value = mock_ort_instance
    
    # Mock output name
    mock_out = MagicMock()
    mock_out.name = "output"
    mock_ort_instance.get_outputs.return_value = [mock_out]
    
    # Mock input name
    mock_in = MagicMock()
    mock_in.name = "input"
    mock_ort_instance.get_inputs.return_value = [mock_in]
    
    # Mock inference result
    mock_ort_instance.run.return_value = [np.array([[0.1, 0.9]])] # Logits
    
    backend = ONNXBackend(mock_config)
    
    pcm_data = np.zeros(16000, dtype=np.float32)
    res = backend.predict_audio(pcm_data.tobytes(), 16000)
    
    assert res["is_crying"] is True
