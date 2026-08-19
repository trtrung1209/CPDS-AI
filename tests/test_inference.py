from src.inference.run_inference import run

def test_run_combines_real_inference_results(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.inference.run_inference.infer_vision",
        lambda *_args: ({"class": "Child", "confidence": 0.95, "child_detected": True}, object()),
    )
    monkeypatch.setattr(
        "src.inference.run_inference.infer_audio",
        lambda *_args: {"is_crying": True, "confidence": 0.88, "probabilities": {"noise": 0.12, "cry": 0.88}},
    )
    monkeypatch.setattr("src.inference.run_inference.get_next_run_dir", lambda: tmp_path)

    result = run("image.jpg", "audio.wav", "vision.onnx", "audio.onnx")

    assert result["alarm_triggered"] is True
    assert (tmp_path / "inference_log.json").exists()


def test_run_does_not_trigger_alarm_without_child(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.inference.run_inference.infer_vision",
        lambda *_args: ({"class": "Adult", "confidence": 0.95, "child_detected": False}, object()),
    )
    monkeypatch.setattr("src.inference.run_inference.infer_audio", lambda *_args: {"is_crying": True})
    monkeypatch.setattr("src.inference.run_inference.get_next_run_dir", lambda: tmp_path)

    assert run("image.jpg", "audio.wav", "vision.onnx", "audio.onnx")["alarm_triggered"] is False
