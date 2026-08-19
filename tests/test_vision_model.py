from types import ModuleType, SimpleNamespace
import time

import pytest

from src.inference.verify_vision import infer_vision, summarize_detections, verify_vision_model


def box(class_id, confidence):
    return SimpleNamespace(
        cls=SimpleNamespace(item=lambda: class_id),
        conf=SimpleNamespace(item=lambda: confidence),
    )


def test_child_detection_is_prioritized():
    result = summarize_detections({0: "Adult", 1: "Child"}, [box(0, 0.99), box(1, 0.75)])

    assert result == {"class": "Child", "confidence": 0.75, "child_detected": True}


def test_no_detections_returns_safe_default():
    assert summarize_detections({0: "Adult"}, []) == {
        "class": "None", "confidence": 0.0, "child_detected": False
    }


def test_most_confident_child_is_selected_case_insensitively():
    result = summarize_detections(
        {0: "Adult", 1: "CHILD", 2: "child"},
        [box(1, 0.60), box(0, 0.99), box(2, 0.80)],
    )

    assert result == {"class": "child", "confidence": 0.80, "child_detected": True}


def test_best_non_child_is_returned_when_no_child_exists():
    result = summarize_detections({0: "Adult", 1: "Driver"}, [box(0, 0.60), box(1, 0.80)])

    assert result == {"class": "Driver", "confidence": 0.80, "child_detected": False}


def test_missing_model_fails_before_loading_ultralytics(tmp_path):
    with pytest.raises(FileNotFoundError, match="Vision model does not exist"):
        infer_vision(tmp_path / "missing.onnx", tmp_path / "missing.jpg")


def test_missing_image_is_rejected_before_loading_ultralytics(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        infer_vision(model_path, tmp_path / "missing.jpg")


def test_vision_inference_uses_yolo_result_and_returns_child(monkeypatch, tmp_path):
    captured = {}
    result = SimpleNamespace(names={0: "Adult", 1: "Child"}, boxes=[box(0, 0.95), box(1, 0.80)])

    class FakeYOLO:
        def __init__(self, path, task):
            captured.update(path=path, task=task)

        def __call__(self, image_path, verbose):
            captured.update(image_path=image_path, verbose=verbose)
            return [result]

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_ultralytics)
    model_path, image_path = tmp_path / "model.onnx", tmp_path / "image.jpg"
    model_path.touch(); image_path.touch()

    summary, returned_result = infer_vision(model_path, image_path)

    assert summary == {"class": "Child", "confidence": 0.80, "child_detected": True}
    assert returned_result is result
    assert captured == {"path": str(model_path), "task": "detect", "image_path": str(image_path), "verbose": False}


def test_verify_vision_checks_image_write_result(monkeypatch, tmp_path):
    result = SimpleNamespace(plot=lambda: "annotated-image")
    monkeypatch.setattr(
        "src.inference.verify_vision.infer_vision",
        lambda *_args: ({"class": "Child", "confidence": 0.8, "child_detected": True}, result),
    )
    monkeypatch.setattr("src.inference.verify_vision.get_next_run_dir", lambda: tmp_path)
    fake_cv2 = ModuleType("cv2")
    fake_cv2.imwrite = lambda path, image: path.endswith("verified_output.jpg") and image == "annotated-image"
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)

    assert verify_vision_model("model.onnx", "image.jpg")["child_detected"] is True


def test_verify_vision_raises_when_image_cannot_be_written(monkeypatch, tmp_path):
    monkeypatch.setattr("src.inference.verify_vision.infer_vision", lambda *_args: ({}, SimpleNamespace(plot=lambda: "image")))
    monkeypatch.setattr("src.inference.verify_vision.get_next_run_dir", lambda: tmp_path)
    fake_cv2 = ModuleType("cv2")
    fake_cv2.imwrite = lambda *_args: False
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)

    with pytest.raises(OSError, match="Could not save"):
        verify_vision_model("model.onnx", "image.jpg")


@pytest.mark.performance
def test_detection_postprocessing_performance_regression_guard():
    boxes = [box(index % 3, 0.5 + (index % 50) / 100) for index in range(1000)]
    start = time.perf_counter()
    for _ in range(10):
        summarize_detections({0: "Adult", 1: "Child", 2: "Driver"}, boxes)

    assert time.perf_counter() - start < 1.0
