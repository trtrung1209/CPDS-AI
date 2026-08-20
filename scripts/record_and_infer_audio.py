"""Record a short microphone sample and run exported audio ONNX inference."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sound_device
from scipy.io.wavfile import write as write_wav

from src.inference.verify_audio import infer_audio


def record_and_infer(model_path: Path, labels_path: Path | None, duration: float, sample_rate: int) -> dict:
    """Record one mono sample, infer it, and remove the temporary WAV file."""
    if duration <= 0 or sample_rate <= 0:
        raise ValueError("Duration and sample rate must be positive.")

    recording = sound_device.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sound_device.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
        audio_path = Path(temporary_file.name)
    try:
        write_wav(audio_path, sample_rate, np.asarray(recording))
        return infer_audio(model_path, audio_path, labels_path)
    finally:
        audio_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record microphone audio and run CPDS-AI ONNX inference.")
    parser.add_argument("--model", type=Path, default=Path("data/models/audio_model.onnx"))
    parser.add_argument("--labels", type=Path, help="Optional audio_labels.json exported with the model.")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    result = record_and_infer(args.model, args.labels, args.duration, args.sample_rate)
    label = "cry" if result["is_crying"] else "noise"
    print(f"Prediction: {label} (cry confidence: {result['confidence']:.3f})")


if __name__ == "__main__":
    main()
