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

    with pytest.raises(RuntimeError, match="camera index 3"):
        run_camera(model_path, camera_index=3)
    assert released == [True]


def test_camera_releases_device_after_user_exits(monkeypatch, tmp_path):
    events = []

    class OpenCamera:
        def isOpened(self):
            return True

        def read(self):
            return True, "frame"

        def release(self):
            events.append("release")

    class FakeYOLO:
        def __init__(self, path, task):
            assert path.endswith("model.onnx")
            assert task == "detect"

        def __call__(self, frame, verbose):
            assert frame == "frame"
            assert verbose is False
            return [SimpleNamespace(plot=lambda: "annotated")]

    fake_cv2 = ModuleType("cv2")
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

    assert events == [("CPDS-AI: YOLOv8 Live Inference", "annotated"), "release", "destroy"]


def test_camera_releases_device_when_a_frame_cannot_be_read(monkeypatch, tmp_path):
    events = []

    class OpenCamera:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            events.append("release")

    class FakeYOLO:
        def __init__(self, _path, task):
            assert task == "detect"

    fake_cv2 = ModuleType("cv2")
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
