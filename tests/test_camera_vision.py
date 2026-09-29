from types import ModuleType, SimpleNamespace

import pytest

from src.inference.camera_vision import run_camera


def test_camera_rejects_missing_model_before_loading_ml_dependencies(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        run_camera(tmp_path / "missing.onnx")


def test_camera_releases_device_when_it_cannot_be_opened(monkeypatch, tmp_path):
    released = []

    class ClosedCamera:
        def isOpened(self):
            return False

        def release(self):
            released.append(True)

        def set(self, prop, value):
            return True

    fake_cv2 = ModuleType("cv2")
    fake_cv2.VideoCapture = lambda _index: ClosedCamera()
    class FakeYOLO:
        def __init__(self, _path, task):
            assert task == "detect"

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_ultralytics)
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    with pytest.raises(RuntimeError, match="Could not open source 3"):
        run_camera(model_path, source="3")
    assert released == [True]


def test_camera_releases_device_after_user_exits(monkeypatch, tmp_path):
    events = []

    class OpenCamera:
        def isOpened(self):
            return True

        def read(self):
            import numpy as np
            return True, np.zeros((10, 10, 3), dtype=np.uint8)

        def release(self):
            events.append("release")

        def set(self, prop, value):
            return True

    class FakeYOLO:
        def __init__(self, path, task):
            assert path.endswith("model.onnx")
            assert task == "detect"

        def __call__(self, frame, verbose):
            assert verbose is False
            import numpy as np
            return [SimpleNamespace(plot=lambda: np.zeros((10, 10, 3), dtype=np.uint8))]

    fake_cv2 = ModuleType("cv2")
    fake_cv2.CAP_PROP_FRAME_WIDTH = 3
    fake_cv2.CAP_PROP_FRAME_HEIGHT = 4
    fake_cv2.FONT_HERSHEY_SIMPLEX = 0
    fake_cv2.LINE_AA = 16
    fake_cv2.rectangle = lambda *args, **kwargs: None
    fake_cv2.putText = lambda *args, **kwargs: None
    fake_cv2.addWeighted = lambda *args, **kwargs: None
    fake_cv2.imwrite = lambda *args, **kwargs: True
    fake_cv2.VideoCapture = lambda _index: OpenCamera()
    fake_cv2.imshow = lambda title, image: events.append((title, image))
    fake_cv2.waitKey = lambda _delay: ord("q")
    fake_cv2.destroyAllWindows = lambda: events.append("destroy")
    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_ultralytics)
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    run_camera(model_path)

    assert len(events) == 3
    assert events[0][0] == "CPDS-AI: YOLOv8 Live Inference"
    assert events[1] == "release"
    assert events[2] == "destroy"


def test_camera_releases_device_when_a_frame_cannot_be_read(monkeypatch, tmp_path):
    events = []

    class OpenCamera:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            events.append("release")

        def set(self, prop, value):
            return True

    class FakeYOLO:
        def __init__(self, _path, task):
            assert task == "detect"

    fake_cv2 = ModuleType("cv2")
    fake_cv2.CAP_PROP_FRAME_WIDTH = 3
    fake_cv2.CAP_PROP_FRAME_HEIGHT = 4
    fake_cv2.VideoCapture = lambda _index: OpenCamera()
    fake_cv2.destroyAllWindows = lambda: events.append("destroy")
    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_ultralytics)
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    with pytest.raises(RuntimeError, match="Could not read"):
        run_camera(model_path)
    assert events == ["release", "destroy"]
