#!/usr/bin/env python3
"""CPDS-AI: Unified entry point for inference, evaluation, and deployment testing.

This script delegates to the well-tested modules under src/ and scripts/.
It auto-detects whether a virtual environment is active and prints setup
guidance when required dependencies are missing.

Usage:
    python main.py                                         # Check system readiness
    python main.py --mode prepare                          # Download test audio data
    python main.py --mode evaluate                         # Run unit tests + dataset evaluation + report
    python main.py --mode file --image X.jpg --audio Y.wav # Dual-modal inference
    python main.py --mode camera                           # Live webcam detection
    python main.py --mode mic                              # Live microphone detection
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
============================================================
   CPDS-AI: Child Protection & Distress Detection System
============================================================
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_models() -> dict[str, bool]:
    """Return a mapping of model name → exists on disk."""
    status = {
        "Vision (YOLOv8)": VISION_MODEL.is_file(),
        "Audio  (ResNet18)": AUDIO_MODEL.is_file(),
    }
    for name, ready in status.items():
        icon = "✅" if ready else "❌"
        print(f"  {icon} {name}")
    return status


def _require_audio_deps() -> None:
    """Fail fast with a helpful message when audio libraries are missing."""
    try:
        from src.inference.verify_audio import validate_audio_runtime
        validate_audio_runtime()
    except Exception:
        print("\n❌ Audio dependencies are missing.")
        print("   Fix: bash setup_environment.sh --audio")
        print("   Then: .venv/bin/python main.py <your command>")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------
def mode_check() -> None:
    print("\n🔍 Pre-trained models:")
    status = _check_models()
    if all(status.values()):
        print("\n✅ System is ready for deployment!")
        print("\nAvailable commands:")
        print("  python main.py --mode prepare                          # Download test data")
        print("  python main.py --mode evaluate                         # Unit tests + Metrics report")
        print("  python main.py --mode file --image X.jpg --audio Y.wav # Dual inference")
        print("  python main.py --mode camera                           # Live webcam")
        print("  python main.py --mode mic                              # Live microphone")
    else:
        print("\n⚠️  Place missing .onnx models into data/models/ first.")


def mode_prepare() -> None:
    """Delegate to scripts/prepare_audio_evaluation_data.py."""
    from scripts.prepare_audio_evaluation_data import prepare_audio_evaluation_data

    output_dir = PROJECT_ROOT / "data" / "test_audio"
    cache_dir = PROJECT_ROOT / ".cache" / "cpds-ai-audio"
    print("🚀 Downloading & preparing test audio data ...")
    manifest = prepare_audio_evaluation_data(
        output_dir=output_dir,
        cache_dir=cache_dir,
        samples_per_class=20,
        seed=42,
        overwrite=True,
    )
    print(f"\n✅ Done! {manifest['cry_count']} cry + {manifest['noise_count']} noise samples")
    print(f"   Location: {output_dir}")


def mode_evaluate() -> None:
    """Run Pytest unit tests first (with green PASSED output), then run dataset evaluation and export reports."""
    _require_audio_deps()

    print("\n" + "=" * 50)
    print("🧪 STEP 1: RUNNING UNIT TESTS (PYTEST)")
    print("=" * 50)
    
    try:
        import pytest
        # Run pytest programmatically on tests/
        pytest_args = [str(PROJECT_ROOT / "tests"), "-v", "--no-cov"]
        exit_code = pytest.main(pytest_args)
        if exit_code == 0:
            print("✅ All unit tests PASSED successfully!")
        else:
            print("⚠️ Some unit tests failed. Proceeding with dataset evaluation...")
    except Exception as error:
        print(f"⚠️ Pytest execution skipped: {error}")

    print("\n" + "=" * 50)
    print("📊 STEP 2: EVALUATING MODEL ON TEST DATASET")
    print("=" * 50)

    from scripts.evaluate_audio_model import evaluate_model

    test_dir = PROJECT_ROOT / "data" / "test_audio"
    if not test_dir.is_dir():
        print("❌ Test data not found. Running auto-prepare first...")
        mode_prepare()

    report = evaluate_model(AUDIO_MODEL, test_dir)

    print("\n" + "=" * 50)
    print("📊 EVALUATION METRICS REPORT")
    print("=" * 50)
    print(f"🎯 Overall Accuracy: {report['accuracy'] * 100:.2f}%")
    print(f"   Samples evaluated: {report['evaluated_samples']}")
    if report["failed_samples"]:
        print(f"   ⚠️  Failed samples: {len(report['failed_samples'])}")

    cr = report["classification_report"]
    print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 42)
    for cls in ["noise", "cry"]:
        if cls in cr:
            print(f"{cls:<10} {cr[cls]['precision']:>10.2f} {cr[cls]['recall']:>10.2f} {cr[cls]['f1-score']:>10.2f}")

    cm = report["confusion_matrix"]
    print(f"\n🧩 Confusion Matrix:")
    print(f"                 Predicted NOISE   Predicted CRY")
    print(f"  Actual NOISE    {cm[0][0]:<17} {cm[0][1]}")
    print(f"  Actual CRY      {cm[1][0]:<17} {cm[1][1]}")
    print("=" * 50)

    # Save reports
    report_dir = PROJECT_ROOT / "test_reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "audio_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n📄 Full JSON report saved: {report_path}")


def mode_file(image_path: str, audio_path: str) -> None:
    """Run dual-modal inference on one image + one audio file."""
    _require_audio_deps()

    from src.inference.run_inference import run as run_dual

    result = run_dual(
        image_path=image_path,
        audio_path=audio_path,
        vision_model=str(VISION_MODEL),
        audio_model=str(AUDIO_MODEL),
    )

    print("\n" + "=" * 50)
    print("📊 DUAL-MODAL INFERENCE RESULT")
    print("=" * 50)
    v = result["vision"]
    a = result["audio"]
    print(f"  👁️  Child detected : {v['child_detected']}  (confidence: {v['confidence']:.2f})")
    print(f"  🔊 Baby crying     : {a['is_crying']}  (confidence: {a['confidence']:.2f})")
    alarm = result["alarm_triggered"]
    if alarm:
        print(f"\n  🚨 ALARM TRIGGERED — Child is crying!")
    else:
        print(f"\n  💤 No alarm — situation is normal.")
    print("=" * 50)


def mode_camera() -> None:
    """Launch live webcam inference."""
    from src.inference.camera_vision import run_camera
    print("📷 Starting live camera inference ...")
    run_camera(model_path=str(VISION_MODEL))


def mode_mic() -> None:
    """Launch live microphone inference."""
    _require_audio_deps()

    from scripts.record_and_infer_audio import record_and_infer

    print("🎤 Starting live microphone inference ...")
    print("   Press Ctrl+C to stop.\n")
    try:
        while True:
            result = record_and_infer(
                model_path=AUDIO_MODEL,
                labels_path=None,
                duration=2.0,
                sample_rate=16000,
            )
            label = "🚨 CRY DETECTED" if result["is_crying"] else "💤 Noise (normal)"
            print(f"  → {label}  (cry confidence: {result['confidence']:.3f})")
            input("  Press Enter to record again ...")
    except KeyboardInterrupt:
        print("\n👋 Exited microphone test.")


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
        choices=["check", "prepare", "evaluate", "file", "camera", "mic"],
        default="check",
        help="Operation mode (default: check)",
    )
    parser.add_argument("--image", help="Image path (required for --mode file)")
    parser.add_argument("--audio", help="Audio path (required for --mode file)")
    args = parser.parse_args()

    print(BANNER)

    if args.mode == "check":
        mode_check()
    elif args.mode == "prepare":
        mode_prepare()
    elif args.mode == "evaluate":
        mode_evaluate()
    elif args.mode == "file":
        if not args.image or not args.audio:
            parser.error("--mode file requires both --image and --audio paths.")
        mode_file(args.image, args.audio)
    elif args.mode == "camera":
        mode_camera()
    elif args.mode == "mic":
        mode_mic()


if __name__ == "__main__":
    main()
