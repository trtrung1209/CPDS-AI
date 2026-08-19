import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import get_next_run_dir, save_result

def mock_vision_inference(image_path):
    """
    Simulates YOLOv8 Vision inference.
    """
    return {"class": "Child", "confidence": 0.95}

def mock_audio_inference(audio_path):
    """
    Simulates CNN Audio inference.
    """
    return {"is_crying": True, "confidence": 0.88, "noise_detected": "car_engine"}

def run(image_path, audio_path):
    print("Starting CPDS-AI Inference...")
    
    # 1. Run inferences
    vision_result = mock_vision_inference(image_path)
    audio_result = mock_audio_inference(audio_path)
    
    # 2. Combine results
    final_result = {
        "vision": vision_result,
        "audio": audio_result,
        "alarm_triggered": vision_result["class"] == "Child" and audio_result["is_crying"]
    }
    
    # 3. Save to next run directory
    run_dir = get_next_run_dir()
    print(f"Saving results to {run_dir}...")
    
    save_result(run_dir, "inference_log.json", json.dumps(final_result, indent=4))
    print("Inference completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPDS-AI Inference")
    parser.add_argument("--image", type=str, default="sample.jpg", help="Path to input image")
    parser.add_argument("--audio", type=str, default="sample.wav", help="Path to input audio")
    
    args = parser.parse_args()
    run(args.image, args.audio)
