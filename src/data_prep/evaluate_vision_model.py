"""Evaluate the YOLOv8 vision model and provide detailed failure analysis for safety-critical metrics."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_yolo_label(label_path: Path) -> list:
    """Parse YOLO TXT label format: class x_center y_center width height."""
    boxes = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append({
                    "class_id": int(parts[0]),
                    "area_pct": float(parts[3]) * float(parts[4]) * 100.0,
                })
    return boxes


def evaluate_vision_model(model_path: Path, test_dir: Path, output_dir: Path = None) -> Dict[str, Any]:
    """Run validation and extract detailed False Negative (Miss) analysis for the child class."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Vision model not found: {model_path}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test dataset not found: {test_dir}")
        
    data_yaml = test_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml is required in the test dataset: {data_yaml}")

    from ultralytics import YOLO
    model = YOLO(str(model_path), task="detect")
    
    print("[INFO] Running Ultralytics validation for mAP metrics...")
    if output_dir:
        val_metrics = model.val(data=str(data_yaml), device="cpu", verbose=False, project=str(output_dir), name="vision_details", exist_ok=True)
    else:
        val_metrics = model.val(data=str(data_yaml), device="cpu", verbose=False)
    
    # We must explicitly find all child instances and check if the model missed them.
    # We assume class 1 is 'child' based on our data.yaml.
    child_class_id = 1
    
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"
    
    child_samples = 0
    missed_children = []
    
    print("[INFO] Running manual inference for detailed False Negative analysis...")
    for img_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = parse_yolo_label(label_path)
        
        gt_children = [b for b in gt_boxes if b["class_id"] == child_class_id]
        if not gt_children:
            continue  # No child in this ground truth image
            
        child_samples += len(gt_children)
        
        # Inference
        result = model(str(img_path), verbose=False)[0]
        
        # Did the model predict a child with high confidence?
        pred_child_confs = []
        has_adult = False
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id == child_class_id:
                    pred_child_confs.append(conf)
                else:
                    has_adult = True
                    
        # YOLO default conf threshold is usually 0.25, but we check if any valid child was detected
        best_child_conf = max(pred_child_confs) if pred_child_confs else 0.0
        if best_child_conf < 0.25:
            # MISS! Analyze why.
            for gt_child in gt_children:
                reason = "Unknown"
                if gt_child["area_pct"] < 2.0:
                    reason = "Small bounding box (far away or highly occluded)"
                elif 0.10 <= best_child_conf < 0.25:
                    reason = f"Confidence {best_child_conf:.2f} is below threshold (0.25)"
                elif has_adult:
                    reason = "Misclassified as adult or suppressed by adult detection"
                
                missed_children.append({
                    "file": img_path.name,
                    "gt_area_pct": round(gt_child["area_pct"], 2),
                    "best_pred_conf": round(best_child_conf, 3),
                    "reason": reason
                })

    recall_child = 1.0 if child_samples == 0 else (child_samples - len(missed_children)) / child_samples
    
    report = {
        "mAP50": float(val_metrics.box.map50),
        "mAP50_95": float(val_metrics.box.map),
        "child_samples": child_samples,
        "child_recall": float(recall_child),
        "missed_children": missed_children,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO vision model for CPDS-AI.")
    parser.add_argument("--model", type=Path, required=True, help="Path to vision ONNX model")
    parser.add_argument("--test-dir", type=Path, required=True, help="Path to test dataset directory")
    args = parser.parse_args()
    
    try:
        report = evaluate_vision_model(args.model, args.test-dir)
        print(json.dumps(report, indent=2))
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
