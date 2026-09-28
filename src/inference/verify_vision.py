import argparse
import sys
from pathlib import Path

# Support direct script execution in addition to module execution.
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_next_run_dir


def summarize_detections(names, boxes):
    """Trích xuất riêng rẽ mức độ tự tin cao nhất của Child và Adult."""
    child_conf = 0.0
    adult_conf = 0.0
    
    for box in boxes:
        class_id = int(box.cls.item())
        class_name = str(names[class_id]).strip().lower()
        confidence = float(box.conf.item())
        
        if class_name in {"child", "children", "kid"}:
            child_conf = max(child_conf, confidence)
        else:
            adult_conf = max(adult_conf, confidence)
            
    return {
        "child_detected": child_conf > 0,
        "child_confidence": child_conf,
        "adult_detected": adult_conf > 0,
        "adult_confidence": adult_conf,
        # Giữ lại các key cũ để tương thích với evaluate script
        "confidence": child_conf if child_conf > 0 else adult_conf,
        "class": "child" if child_conf > 0 else ("adult" if adult_conf > 0 else "None")
    }


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
    Load a YOLOv8 ONNX file, run it on one image, and save annotated output.
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
    parser.add_argument("--model", type=str, required=True, help="Path to vision ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    
    args = parser.parse_args()
    try:
        verify_vision_model(args.model, args.image)
    except (OSError, ValueError) as error:
        parser.error(str(error))
