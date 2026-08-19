from types import SimpleNamespace

from src.inference.verify_vision import summarize_detections


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
