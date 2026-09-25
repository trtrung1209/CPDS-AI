#!/usr/bin/env python3
"""CPDS-AI: Unified entry point for inference, evaluation, and deployment testing.

This script delegates to the well-tested modules under src/ and scripts/.
It auto-detects whether a virtual environment is active and prints setup
guidance when required dependencies are missing.

Usage:
    python main.py                                         # Check system readiness
    python main.py --mode prepare                          # Download test audio data
    python main.py --mode evaluate                         # Run unit tests + dataset evaluation + report
    python main.py --mode audio --audio Y.wav              # Single audio file inference
    python main.py --mode file --image X.jpg --audio Y.wav # Dual-modal inference
    python main.py --mode camera                           # Live webcam detection
    python main.py --mode mic                              # Periodic microphone detection (unattended)
    python main.py --mode mic --idle-seconds 0             # ~Continuous listening (no idle gap), for comparison
"""

import argparse
import json
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap: ensure root is always importable regardless of cwd.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VISION_MODEL = PROJECT_ROOT / "data" / "models" / "yolov8n-adult-child.onnx"
AUDIO_MODEL = PROJECT_ROOT / "data" / "models" / "audio_model.onnx"

BANNER = r"""
------------------------------------------------------------
CPDS-AI: Child Protection & Distress Detection System
------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_models() -> dict[str, bool]:
    """Return a mapping of model name -> exists on disk."""
    status = {
        "Vision (YOLOv8)": VISION_MODEL.is_file(),
        "Audio  (AudioCNN)": AUDIO_MODEL.is_file(),
    }
    for name, ready in status.items():
        state = "[OK]" if ready else "[MISSING]"
        print(f"  {state:<10} {name}")
    return status


def _require_audio_deps() -> None:
    """Fail fast with a helpful message when audio libraries are missing."""
    try:
        from src.inference.verify_audio import validate_audio_runtime
        validate_audio_runtime()
    except Exception:
        print("\n[ERROR] Audio dependencies are missing.")
        print("Fix: bash setup_environment.sh --audio")
        print("Then: .venv/bin/python main.py <your command>")
        sys.exit(1)


def _require_vision_deps() -> None:
    """Fail fast with a helpful message when vision libraries are missing."""
    try:
        import cv2
        import ultralytics
    except ImportError:
        print("\n[ERROR] Vision dependencies (opencv-python, ultralytics) are missing.")
        print("Fix: bash setup_environment.sh --full")
        print("Then: .venv/bin/python main.py --mode camera")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------
def mode_check() -> None:
    print("[INFO] Checking pre-trained models:")
    status = _check_models()
    if all(status.values()):
        print("\n[OK] System is ready for deployment.\n")
        print("Available commands:")
        print("  python main.py --mode prepare                          # Download test data")
        print("  python main.py --mode evaluate                         # Unit tests + Metrics report")
        print("  python main.py --mode audio --audio file.wav           # Single audio test")
        print("  python main.py --mode file --image X.jpg --audio Y.wav # Dual inference")
        print("  python main.py --mode camera                           # Live webcam")
        print("  python main.py --mode mic                              # Live microphone")
    else:
        print("\n[WARN] Place missing .onnx models into data/models/ first.")


def mode_prepare() -> None:
    """Delegate to scripts/prepare_audio_evaluation_data.py."""
    from src.data_prep.prepare_audio_evaluation_data import prepare_audio_evaluation_data

    output_dir = PROJECT_ROOT / "data" / "test_audio"
    cache_dir = PROJECT_ROOT / ".cache" / "cpds-ai-audio"
    print("[INFO] Downloading and preparing test audio data...")
    manifest = prepare_audio_evaluation_data(
        output_dir=output_dir,
        cache_dir=cache_dir,
        samples_per_class=20,
        seed=42,
        overwrite=True,
    )
    print(f"[OK] Done. {manifest['cry_count']} cry + {manifest['noise_count']} noise samples.")
    print(f"Location: {output_dir}")


def mode_evaluate() -> None:
    """Run Pytest unit tests first, then run dataset evaluation and export reports."""
    _require_audio_deps()

    print("\n--- STEP 1: UNIT TESTS (PYTEST) ---")
    try:
        import pytest
        pytest_args = [str(PROJECT_ROOT / "tests"), "-v", "--no-cov", "-q"]
        exit_code = pytest.main(pytest_args)
        if exit_code == 0:
            print("[OK] All unit tests passed.")
        else:
            print("[WARN] Some unit tests failed. Proceeding with dataset evaluation...")
    except Exception as error:
        print(f"[WARN] Pytest execution skipped: {error}")

    print("\n--- STEP 2: EVALUATING MODEL ON TEST DATASET ---")
    from src.data_prep.evaluate_audio_model import evaluate_model

    test_dir = PROJECT_ROOT / "data" / "test_audio"
    if not test_dir.is_dir():
        print("[FAIL] Test data not found. Running auto-prepare first...")
        mode_prepare()

    report = evaluate_model(AUDIO_MODEL, test_dir)

    print("\n--- EVALUATION METRICS REPORT ---")
    print(f"Overall Accuracy : {report['accuracy'] * 100:.2f}%")
    print(f"Samples evaluated: {report['evaluated_samples']}")
    if report["failed_samples"]:
        print(f"[WARN] Failed samples: {len(report['failed_samples'])}")

    cr = report["classification_report"]
    print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 42)
    for cls in ["noise", "cry"]:
        if cls in cr:
            print(f"{cls:<10} {cr[cls]['precision']:>10.2f} {cr[cls]['recall']:>10.2f} {cr[cls]['f1-score']:>10.2f}")

    cm = report["confusion_matrix"]
    print("\nConfusion Matrix:")
    print("                 Predicted NOISE   Predicted CRY")
    print(f"  Actual NOISE    {cm[0][0]:<17} {cm[0][1]}")
    print(f"  Actual CRY      {cm[1][0]:<17} {cm[1][1]}")
    print("-" * 50)

    report_dir = PROJECT_ROOT / "test_reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "audio_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Full JSON report saved: {report_path}")


def mode_audio(audio_path: str) -> None:
    """Run audio inference on a single audio file."""
    _require_audio_deps()

    from src.inference.verify_audio import infer_audio

    result = infer_audio(AUDIO_MODEL, Path(audio_path))
    print("\n--- AUDIO INFERENCE RESULT ---")
    print(f"File        : {audio_path}")
    print(f"Is Crying   : {result['is_crying']}")
    print(f"Confidence  : {result['confidence']:.4f}")
    
    probs = result['probabilities']
    print(f"Noise Prob  : {probs['noise']:.4f}")
    print(f"Cry Prob    : {probs['cry']:.4f}")
    
    label = "ALARM: CRY DETECTED" if result['is_crying'] else "NORMAL: NOISE"
    print(f"\nResult: {label}")
    print("-" * 30)


def mode_file(image_path: str, audio_path: str) -> None:
    """Run dual-modal inference on one image + one audio file."""
    _require_vision_deps()
    _require_audio_deps()

    from src.inference.run_inference import run as run_dual

    result = run_dual(
        image_path=image_path,
        audio_path=audio_path,
        vision_model=str(VISION_MODEL),
        audio_model=str(AUDIO_MODEL),
    )

    print("\n--- DUAL-MODAL INFERENCE RESULT ---")
    v = result["vision"]
    a = result["audio"]
    print(f"Child detected : {v['child_detected']} (confidence: {v['confidence']:.2f})")
    print(f"Baby crying    : {a['is_crying']} (confidence: {a['confidence']:.2f})")
    
    alarm = result["alarm_triggered"]
    if alarm:
        print("\nSTATUS: ALARM TRIGGERED - Child is crying!")
    else:
        print("\nSTATUS: NORMAL - No alarm triggered.")
    print("-" * 35)


def mode_camera() -> None:
    """Launch live webcam inference."""
    _require_vision_deps()

    from src.inference.camera_vision import run_camera
    print("[INFO] Starting live camera inference...")
    run_camera(model_path=str(VISION_MODEL))


def mode_mic(idle_seconds: float = 3.0) -> None:
    """Launch live microphone inference with auto peak gain normalization."""
    _require_audio_deps()

    from src.cli.record_and_infer_audio import listen_periodically

    print("[INFO] Starting periodic microphone inference (Auto Gain Boost Enabled)...")
    print(f"[INFO] Cycle: 2.0s record + {idle_seconds:.1f}s idle. Press Ctrl+C to stop.\n")

    def _report(result: dict) -> None:
        label = "ALARM: CRY DETECTED" if result["is_crying"] else "NORMAL: Noise"
        cry_prob = result["probabilities"]["cry"]
        noise_prob = result["probabilities"]["noise"]
        print(f"[{label}] Cry: {cry_prob * 100:.1f}% | Noise: {noise_prob * 100:.1f}%")

    try:
        listen_periodically(
            model_path=AUDIO_MODEL,
            labels_path=None,
            capture_duration=2.0,
            idle_seconds=idle_seconds,
            sample_rate=16000,
            on_result=_report,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Exited microphone test.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPDS-AI: Unified deployment interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["check", "prepare", "evaluate", "audio", "file", "camera", "mic"],
        default="check",
        help="Operation mode (default: check)",
    )
    parser.add_argument("--image", help="Image path (required for --mode file)")
    parser.add_argument("--audio", help="Audio path (required for --mode audio/file)")
    parser.add_argument("--idle-seconds", type=float, default=3.0,
                         help="--mode mic only: idle time between recordings")
    args = parser.parse_args()

    print(BANNER)

    if args.mode == "check":
        mode_check()
    elif args.mode == "prepare":
        mode_prepare()
    elif args.mode == "evaluate":
        mode_evaluate()
    elif args.mode == "audio":
        if not args.audio:
            parser.error("--mode audio requires --audio <path_to_wav>.")
        mode_audio(args.audio)
    elif args.mode == "file":
        if not args.image or not args.audio:
            parser.error("--mode file requires both --image and --audio paths.")
        mode_file(args.image, args.audio)
    elif args.mode == "camera":
        mode_camera()
    elif args.mode == "mic":
        mode_mic(args.idle_seconds)


if __name__ == "__main__":
    main()
