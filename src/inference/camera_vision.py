import argparse
import sys
from pathlib import Path

# Allow both `python -m src.inference.camera_vision` and direct script execution.
if __package__ in {None, ""}:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def run_camera(model_path, camera_index=0):
    """Run live YOLO ONNX inference until the user presses q."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    import cv2
    from ultralytics import YOLO

    print(f"Loading YOLOv8 ONNX model from: {model_path} ...")
    model = YOLO(str(model_path), task="detect")
    print("Opening camera. Press 'q' in the preview window to exit.")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            result = model(frame, verbose=False)[0]
            cv2.imshow("CPDS-AI: YOLOv8 Live Inference", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run live camera inference with a YOLOv8 ONNX model.")
    parser.add_argument("--model", default="data/models/yolov8n-adult-child.onnx", help="Path to vision ONNX model")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")

    args = parser.parse_args()
    try:
        run_camera(args.model, args.camera_index)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
