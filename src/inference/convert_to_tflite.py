"""Script to export YOLO model to TFLite INT8 for Raspberry Pi 4 / Orange Pi 5."""
import argparse
from pathlib import Path

def export_tflite(pt_path: Path):
    if not pt_path.is_file():
        raise FileNotFoundError(f"Original PyTorch model not found: {pt_path}")
        
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")

    model = YOLO(str(pt_path))
    
    print("[INFO] Exporting to TFLite INT8 at 320x320 resolution...")
    # int8=True enables quantization for maximum speed on ARM CPUs
    # imgsz=320 reduces the compute load by 4x compared to 640x640
    model.export(format="tflite", imgsz=320, int8=True, optimize=True)
    
    print(f"[SUCCESS] TFLite model exported successfully in the same directory as {pt_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv8 PyTorch model to TFLite INT8.")
    parser.add_argument("--model", type=Path, default=Path("data/models/best.pt"), help="Path to original .pt model (e.g. best.pt)")
    args = parser.parse_args()
    export_tflite(args.model)
