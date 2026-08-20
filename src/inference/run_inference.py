import argparse
import json
import sys
from pathlib import Path

# Support direct script execution in addition to module execution.
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_next_run_dir, save_result


DEFAULT_VISION_THRESHOLD = 0.60
DEFAULT_CRY_THRESHOLD = 0.70


def infer_vision(*args, **kwargs):
    """Lazy import keeps CLI help and unit tests free from ML dependencies."""
    from src.inference.verify_vision import infer_vision as implementation
    return implementation(*args, **kwargs)


def infer_audio(*args, **kwargs):
    """Lazy import keeps CLI help and unit tests free from ML dependencies."""
    from src.inference.verify_audio import infer_audio as implementation
    return implementation(*args, **kwargs)

def should_trigger_alarm(vision_result, audio_result, vision_threshold, cry_threshold):
    """Return true only when both detection confidences meet their safety thresholds."""
    return (
        vision_result["child_detected"]
        and vision_result["confidence"] >= vision_threshold
        and audio_result["is_crying"]
        and audio_result["confidence"] >= cry_threshold
    )


def run(
    image_path,
    audio_path,
    vision_model,
    audio_model,
    audio_labels=None,
    vision_threshold=DEFAULT_VISION_THRESHOLD,
    cry_threshold=DEFAULT_CRY_THRESHOLD,
):
    """Run both real ONNX models and persist a single alarm decision."""
    if not 0.0 <= vision_threshold <= 1.0 or not 0.0 <= cry_threshold <= 1.0:
        raise ValueError("Confidence thresholds must be between 0.0 and 1.0.")

    print("Starting CPDS-AI Inference...")

    vision_result, _ = infer_vision(vision_model, image_path)
    audio_result = infer_audio(audio_model, audio_path, audio_labels)
    final_result = {
        "vision": vision_result,
        "audio": audio_result,
        "alarm_triggered": should_trigger_alarm(vision_result, audio_result, vision_threshold, cry_threshold),
        "thresholds": {"vision": vision_threshold, "cry": cry_threshold},
    }

    run_dir = get_next_run_dir()
    print(f"Saving results to {run_dir}...")
    save_result(run_dir, "inference_log.json", json.dumps(final_result, indent=4))
    print("Inference completed successfully!")
    return final_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPDS-AI Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--audio", required=True, help="Path to input audio")
    parser.add_argument("--vision-model", default="data/models/yolov8n-adult-child.onnx", help="Path to vision ONNX model")
    parser.add_argument("--audio-model", default="data/models/audio_model.onnx", help="Path to audio ONNX model")
    parser.add_argument("--audio-labels", help="Optional path to audio_labels.json exported by training")
    parser.add_argument("--vision-threshold", type=float, default=DEFAULT_VISION_THRESHOLD, help="Minimum child confidence")
    parser.add_argument("--cry-threshold", type=float, default=DEFAULT_CRY_THRESHOLD, help="Minimum cry confidence")
    
    args = parser.parse_args()
    try:
        run(
            args.image, args.audio, args.vision_model, args.audio_model, args.audio_labels,
            args.vision_threshold, args.cry_threshold,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
