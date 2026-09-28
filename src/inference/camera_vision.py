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
    import time
    from ultralytics import YOLO

    print(f"Loading YOLOv8 ONNX model from: {model_path} ...")
    model = YOLO(str(model_path), task="detect")
    print("Opening camera. Press 'q' in the preview window to exit.")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    # Đặt độ phân giải mặc định nhẹ nhàng để test
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        prev_time = time.time()
        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            result = model(frame, verbose=False)[0]
            annotated_frame = result.plot()
            
            # Thêm lớp nền đen mờ (opacity) cho text dễ đọc
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (5, 5), (320, 75), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            # In thông số FPS và Độ phân giải
            h, w = frame.shape[:2]
            cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Res: {w}x{h}", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # In thông số thời gian Inference (ms)
            if hasattr(result, 'speed') and 'inference' in result.speed:
                speed_ms = result.speed['inference']
                cv2.putText(annotated_frame, f"Inference: {speed_ms:.1f}ms", (15, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("CPDS-AI: YOLOv8 Live Inference", annotated_frame)
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
