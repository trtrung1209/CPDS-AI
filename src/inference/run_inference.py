import argparse
import json

from src.utils import get_next_run_dir, save_result


def infer_vision(*args, **kwargs):
    """Lazy import keeps CLI help and unit tests free from ML dependencies."""
    from src.inference.verify_vision import infer_vision as implementation
    return implementation(*args, **kwargs)


def infer_audio(*args, **kwargs):
    """Lazy import keeps CLI help and unit tests free from ML dependencies."""
    from src.inference.verify_audio import infer_audio as implementation
    return implementation(*args, **kwargs)

def run(image_path, audio_path, vision_model, audio_model, audio_labels=None):
    """Run both real ONNX models and persist a single alarm decision."""
    print("Starting CPDS-AI Inference...")

    vision_result, _ = infer_vision(vision_model, image_path)
    audio_result = infer_audio(audio_model, audio_path, audio_labels)
    final_result = {
        "vision": vision_result,
        "audio": audio_result,
        "alarm_triggered": vision_result["child_detected"] and audio_result["is_crying"],
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
    parser.add_argument("--vision-model", default="data/models/best.onnx", help="Path to vision ONNX model")
    parser.add_argument("--audio-model", default="data/models/audio_model.onnx", help="Path to audio ONNX model")
    parser.add_argument("--audio-labels", help="Optional path to audio_labels.json exported by training")
    
    args = parser.parse_args()
    try:
        run(args.image, args.audio, args.vision_model, args.audio_model, args.audio_labels)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
