import argparse
from pathlib import Path

from src.utils import get_next_run_dir


def summarize_detections(names, boxes):
    """Select the most confident child, or the most confident detection."""
    best = {"class": "None", "confidence": 0.0, "child_detected": False}
    for box in boxes:
        class_id = int(box.cls.item())
        class_name = str(names[class_id])
        confidence = float(box.conf.item())
        is_child = class_name.strip().lower() in {"child", "children", "kid"}
        if is_child and (not best["child_detected"] or confidence > best["confidence"]):
            best = {"class": class_name, "confidence": confidence, "child_detected": True}
        elif not best["child_detected"] and confidence > best["confidence"]:
            best = {"class": class_name, "confidence": confidence, "child_detected": False}
    return best


def infer_vision(model_path, image_path):
    """Run YOLO ONNX inference and return the best child-related detection."""
    model_path = Path(model_path)
    image_path = Path(image_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Vision model does not exist: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    results = model(str(image_path), verbose=False)
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return summarize_detections(result.names, []), result
    return summarize_detections(result.names, result.boxes), result


def verify_vision_model(model_path, image_path):
    """
    Tải file ONNX YOLOv8 và chạy trên 1 bức ảnh.
    Lưu ảnh kết quả có vẽ Bounding Box.
    """
    import cv2

    vision_result, result = infer_vision(model_path, image_path)
    run_dir = get_next_run_dir()
    output_image_path = run_dir / "verified_output.jpg"
    if not cv2.imwrite(str(output_image_path), result.plot()):
        raise OSError(f"Could not save annotated image: {output_image_path}")
    print(f"Verification successful! Output image saved at: {output_image_path}")
    return vision_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify YOLOv8 ONNX Model")
    parser.add_argument("--model", type=str, required=True, help="Path to best.onnx")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    
    args = parser.parse_args()
    try:
        verify_vision_model(args.model, args.image)
    except (OSError, ValueError) as error:
        parser.error(str(error))
