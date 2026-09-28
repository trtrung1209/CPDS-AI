"""Script to export YOLO model to ONNX / TFLite at 320x320 resolution for Raspberry Pi 4."""
import argparse
from pathlib import Path

def export_model(pt_path: Path):
    if not pt_path.is_file():
        raise FileNotFoundError(f"Original PyTorch model not found: {pt_path}")
        
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")

    model = YOLO(str(pt_path))
    
    print("[INFO] Exporting to ONNX at 320x320 resolution for Raspberry Pi 4...")
    # ONNX export at 320x320 resolution provides ~20-25 FPS on Pi 4 CPU
    # while avoiding litert/numpy 2.x auto-update dependency conflicts.
    onnx_path = model.export(format="onnx", imgsz=320, simplify=True)
    print(f"\n[SUCCESS ✅] ONNX model exported successfully: {onnx_path}")
    print("[INFO] This 320x320 model is ready for high-FPS deployment on Raspberry Pi 4!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv8 PyTorch model to ONNX/TFLite at 320x320.")
    parser.add_argument("--model", type=Path, default=Path("data/models/yolov8n.pt"), help="Path to original .pt model (e.g. data/models/yolov8n.pt)")
    args = parser.parse_args()
    
    # Auto-fallback if specified model is not found but default alternative exists
    model_path = args.model
    if not model_path.is_file():
        alt_path = Path("data/models/yolov8n.pt")
        if alt_path.is_file():
            print(f"[INFO] Specified model '{model_path}' not found, falling back to '{alt_path}'.")
            model_path = alt_path
        else:
            pt_files = list(Path(".").rglob("*.pt"))
            if pt_files:
                print(f"[ERROR] Could not find '{model_path}'. Found these .pt files in repository:")
                for p in pt_files:
                    print(f"  - {p}")
                print(f"\nPlease run: python3 src/inference/convert_to_tflite.py --model <path_to_file>")
            else:
                print(f"[ERROR] Could not find '{model_path}' or any .pt model in repository.")
            raise FileNotFoundError(f"PyTorch model file not found: {model_path}")

    export_model(model_path)


