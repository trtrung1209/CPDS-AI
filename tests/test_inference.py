from src.inference.run_inference import DEFAULT_CRY_THRESHOLD, DEFAULT_VISION_THRESHOLD, infer_audio, infer_vision, run
import json
import pytest

def test_run_combines_real_inference_results(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.inference.run_inference.infer_vision",
        lambda *_args: ({"child_detected": True, "child_confidence": 0.95, "adult_detected": False, "adult_confidence": 0.0}, object()),
    )
    monkeypatch.setattr(
        "src.inference.run_inference.infer_audio",
        lambda *_args: {"is_crying": True, "confidence": 0.88, "probabilities": {"noise": 0.12, "cry": 0.88}},
    )
    monkeypatch.setattr("src.inference.run_inference.get_next_run_dir", lambda: tmp_path)

    result = run("image.jpg", "audio.wav", "vision.onnx", "audio.onnx")

    assert result["alarm_triggered"] is True
    saved_result = json.loads((tmp_path / "inference_log.json").read_text(encoding="utf-8"))
    assert saved_result == result
    assert set(result) == {"vision", "audio", "alarm_triggered", "thresholds"}
    assert result["thresholds"] == {"vision": DEFAULT_VISION_THRESHOLD, "cry": DEFAULT_CRY_THRESHOLD}


def test_run_does_not_trigger_alarm_without_child(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.inference.run_inference.infer_vision",
        lambda *_args: ({"child_detected": False, "child_confidence": 0.0, "adult_detected": True, "adult_confidence": 0.95}, object()),
    )
    monkeypatch.setattr("src.inference.run_inference.infer_audio", lambda *_args: {"is_crying": True, "confidence": 0.95})
    monkeypatch.setattr("src.inference.run_inference.get_next_run_dir", lambda: tmp_path)

    assert run("image.jpg", "audio.wav", "vision.onnx", "audio.onnx")["alarm_triggered"] is False


@pytest.mark.parametrize(
    ("child_detected", "is_crying", "expected"),
    [(False, False, False), (False, True, True), (True, False, True), (True, True, True)],
)
def test_alarm_triggers_on_either_signal(monkeypatch, tmp_path, child_detected, is_crying, expected):
    monkeypatch.setattr(
        "src.inference.run_inference.infer_vision",
        lambda *_args: ({"child_detected": child_detected, "child_confidence": 0.95, "adult_detected": False, "adult_confidence": 0.0}, object()),
    )
    monkeypatch.setattr("src.inference.run_inference.infer_audio", lambda *_args: {"is_crying": is_crying, "confidence": 0.95})
    monkeypatch.setattr("src.inference.run_inference.get_next_run_dir", lambda: tmp_path)

    assert run("image.jpg", "audio.wav", "vision.onnx", "audio.onnx")["alarm_triggered"] is expected


def test_alarm_requires_confidences_to_meet_thresholds(monkeypatch, tmp_path):
    # Both are present but under threshold -> Alarm False
    monkeypatch.setattr(
        "src.inference.run_inference.infer_vision",
        lambda *_args: ({"child_detected": True, "child_confidence": 0.59, "adult_detected": False, "adult_confidence": 0.0}, object()),
    )
    monkeypatch.setattr("src.inference.run_inference.infer_audio", lambda *_args: {"is_crying": True, "confidence": 0.69})
    monkeypatch.setattr("src.inference.run_inference.get_next_run_dir", lambda: tmp_path)

    assert run("image.jpg", "audio.wav", "vision.onnx", "audio.onnx")["alarm_triggered"] is False


def test_invalid_confidence_threshold_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.inference.run_inference.infer_vision",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Inference must not start for invalid thresholds")),
    )
    monkeypatch.setattr(
        "src.inference.run_inference.infer_audio",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Inference must not start for invalid thresholds")),
    )

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        run("image.jpg", "audio.wav", "vision.onnx", "audio.onnx", vision_threshold=1.1)


def test_failed_vision_inference_does_not_write_partial_result(monkeypatch, tmp_path):
    monkeypatch.setattr("src.inference.run_inference.infer_vision", lambda *_args: (_ for _ in ()).throw(RuntimeError("bad model")))
    monkeypatch.setattr("src.inference.run_inference.get_next_run_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="bad model"):
        run("image.jpg", "audio.wav", "vision.onnx", "audio.onnx")
    assert not (tmp_path / "inference_log.json").exists()


def test_lazy_inference_wrappers_delegate_to_implementations(monkeypatch):
    monkeypatch.setattr("src.inference.verify_vision.infer_vision", lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr("src.inference.verify_audio.infer_audio", lambda *args, **kwargs: (args, kwargs))

    assert infer_vision("model", image="image") == (("model",), {"image": "image"})
    assert infer_audio("model", audio="audio") == (("model",), {"audio": "audio"})
