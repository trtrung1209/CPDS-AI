"""Unit tests for the new evaluation suite, using mocks to avoid heavy inference."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_prep.evaluate_fused_model import evaluate_fused_model
from src.data_prep.evaluate_vision_model import evaluate_vision_model


@pytest.fixture
def mock_vision_test_dir(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    
    (tmp_path / "data.yaml").write_text("path: .\nnames: ['adult', 'child']\n")
    
    # 1 Child image
    (images_dir / "child1.jpg").touch()
    (labels_dir / "child1.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    
    # 1 Adult image
    (images_dir / "adult1.jpg").touch()
    (labels_dir / "adult1.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    
    return tmp_path


@patch("ultralytics.YOLO")
def test_evaluate_vision_model_calculates_metrics_correctly(mock_yolo_class, mock_vision_test_dir):
    # Mock Ultralytics YOLO model
    mock_model_instance = MagicMock()
    mock_yolo_class.return_value = mock_model_instance
    
    # Mock val() metrics
    mock_val_result = MagicMock()
    mock_val_result.box.map50 = 0.95
    mock_val_result.box.map = 0.85
    mock_model_instance.val.return_value = mock_val_result
    
    # Mock inference: Miss the child
    mock_pred_result = MagicMock()
    mock_pred_result.boxes = []
    mock_model_instance.return_value = [mock_pred_result]
    
    dummy_model = mock_vision_test_dir / "dummy.onnx"
    dummy_model.touch()
    
    report = evaluate_vision_model(dummy_model, mock_vision_test_dir)
    
    assert report["mAP50"] == 0.95
    assert report["child_samples"] == 1
    assert report["child_recall"] == 0.0  # Missed the child
    assert len(report["missed_children"]) == 1
    assert report["missed_children"][0]["file"] == "child1.jpg"
    assert report["missed_children"][0]["reason"] == "Small bounding box (far away or highly occluded)"


@pytest.fixture
def mock_audio_test_dir(tmp_path):
    cry_dir = tmp_path / "cry"
    noise_dir = tmp_path / "noise"
    cry_dir.mkdir()
    noise_dir.mkdir()
    
    (cry_dir / "cry1.wav").touch()
    (noise_dir / "noise1.wav").touch()
    
    return tmp_path


@patch("src.data_prep.evaluate_fused_model.run_dual")
def test_evaluate_fused_model_calculates_fnr(mock_run_dual, mock_vision_test_dir, mock_audio_test_dir):
    # Mock run_dual to always return no alarm (triggering False Negative for the Child+Cry case)
    def side_effect(image_path, audio_path, vision_model, audio_model):
        return {
            "alarm_triggered": False,
            "vision": {"child_detected": True, "confidence": 0.8},
            "audio": {"is_crying": False, "confidence": 0.2}
        }
    mock_run_dual.side_effect = side_effect
    
    report = evaluate_fused_model(Path("v.onnx"), Path("a.onnx"), mock_vision_test_dir, mock_audio_test_dir)
    
    assert report["total_scenarios"] > 0
    assert report["false_negative_rate"] == 1.0  # Failed the only Child+Cry case
    assert len(report["failures"]) > 0
    
    fn_failure = [f for f in report["failures"] if f["expected_alarm"] is True][0]
    assert fn_failure["actual_alarm"] is False
