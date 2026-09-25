"""Record a short microphone sample and run exported audio ONNX inference."""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import sounddevice as sound_device
from scipy.io.wavfile import write as write_wav

from src.inference.verify_audio import infer_audio


def record_and_infer(model_path: Path, labels_path: Path | None, duration: float, sample_rate: int) -> dict:
    """Record one mono sample, apply peak normalization, infer it, and remove temp WAV."""
    if duration <= 0 or sample_rate <= 0:
        raise ValueError("Duration and sample rate must be positive.")

    recording = sound_device.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sound_device.wait()
    recording = recording.flatten()

    # Peak normalization: boost low microphone volume to standard 0.95 peak amplitude
    max_val = np.max(np.abs(recording))
    if max_val > 1e-4:
        recording = (recording / max_val) * 0.95

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
        audio_path = Path(temporary_file.name)
    try:
        write_wav(audio_path, sample_rate, recording.astype(np.float32))
        return infer_audio(model_path, audio_path, labels_path)
    finally:
        audio_path.unlink(missing_ok=True)


# Rationale for periodic listening:
# 1. Baby cries are sustained sounds (seconds to minutes), so missing a fraction of a second is acceptable.
# 2. Gateway box has limited battery (~30 mins) when the vehicle is off. Continuous listening (100% duty cycle)
#    drains battery too fast, reducing the overall protection window. A dead battery is worse than a short delay.
# 3. Reducing false negatives (misses) is prioritized over false positives, so we alarm immediately on the first positive cycle.
def listen_periodically(model_path, labels_path, capture_duration, idle_seconds, sample_rate,
                         max_cycles=None, on_result=None, sleep_fn=time.sleep):
    """Loop: record `capture_duration`s -> infer -> `on_result(result)` -> idle `idle_seconds` -> repeat.

    Runs unattended (no keypress needed) so it can actually run on a headless Gateway box.
    `max_cycles=None` loops forever (stop with Ctrl+C); a finite value is mainly for tests/smoke-checks.
    """
    if capture_duration <= 0 or sample_rate <= 0:
        raise ValueError("Duration and sample rate must be positive.")
    if idle_seconds < 0:
        raise ValueError("idle_seconds must not be negative.")

    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        result = record_and_infer(model_path, labels_path, capture_duration, sample_rate)
        if on_result is not None:
            on_result(result)
        cycle += 1
        if idle_seconds > 0 and (max_cycles is None or cycle < max_cycles):
            sleep_fn(idle_seconds)
    return cycle


def _print_prediction(result: dict) -> None:
    label = "cry" if result["is_crying"] else "noise"
    print(f"Prediction: {label} (cry confidence: {result['confidence']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record microphone audio and run CPDS-AI ONNX inference.")
    parser.add_argument("--model", type=Path, default=Path("data/models/audio_model.onnx"))
    parser.add_argument("--labels", type=Path, help="Optional audio_labels.json exported with the model.")
    parser.add_argument("--duration", type=float, default=2.0, help="Recording length per cycle, in seconds.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--loop", action="store_true",
                         help="Listen periodically and unattended instead of one single recording.")
    parser.add_argument("--idle-seconds", type=float, default=3.0,
                         help="Idle time between recordings when --loop is set "
                              "(periodic listening, not continuous). Use 0 to approximate continuous listening.")
    parser.add_argument("--max-cycles", type=int, default=None,
                         help="Stop --loop after N cycles (mainly for testing/smoke-checks).")
    args = parser.parse_args()

    if args.loop:
        print(f"Listening periodically: {args.duration:.1f}s record + {args.idle_seconds:.1f}s idle per cycle. "
              f"Press Ctrl+C to stop.")
        try:
            listen_periodically(args.model, args.labels, args.duration, args.idle_seconds, args.sample_rate,
                                 args.max_cycles, on_result=_print_prediction)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        result = record_and_infer(args.model, args.labels, args.duration, args.sample_rate)
        _print_prediction(result)


if __name__ == "__main__":
    main()
