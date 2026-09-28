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
import sys
import yaml
import datetime
import hashlib
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
    """Fail fast with a helpful message when vision libraries are missing or broken."""
    try:
        import cv2
        import ultralytics
        
        # Test if OpenCV is the correct GUI-enabled version (headless versions often lack 'imshow')
        if not hasattr(cv2, 'imshow'):
            raise AttributeError("module 'cv2' has no attribute 'imshow'")
            
    except (ImportError, AttributeError) as e:
        print(f"\n[ERROR] Vision dependencies are missing or broken: {e}")
        print("This typically occurs if packages are missing or the headless OpenCV version is installed.")
        print("-> FIX: Run the following setup script to clean and reinstall the correct dependencies:")
        print("\n   bash setup_environment.sh --full\n")
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


def get_model_hash(path: Path) -> str:
    """Return MD5 hash prefix of the model for report tracking."""
    if not path.is_file(): return "N/A"
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]

def mode_prepare() -> None:
    """Delegate to data prep scripts."""
    from src.data_prep.prepare_audio_evaluation_data import prepare_audio_evaluation_data
    from src.data_prep.prepare_vision_evaluation_data import prepare_vision_evaluation_data
    
    audio_dir = PROJECT_ROOT / "data" / "test_audio"
    cache_dir = PROJECT_ROOT / ".cache" / "cpds-ai-audio"
    print("[INFO] Preparing audio test data (100 samples per class)...")
    try:
        manifest_audio = prepare_audio_evaluation_data(audio_dir, cache_dir, samples_per_class=100, seed=42, overwrite=True)
        print(f"  [OK] Audio: {manifest_audio['cry_count']} cry + {manifest_audio['noise_count']} noise")
    except Exception as e:
        print(f"  [WARN] Audio prep failed: {e}")
        
    vision_dir = PROJECT_ROOT / "data" / "test_vision"
    print("[INFO] Preparing vision test data...")
    try:
        manifest_vision = prepare_vision_evaluation_data(vision_dir)
        print(f"  [OK] Vision: {manifest_vision['images_count']} test images")
    except Exception as e:
        print(f"  [WARN] Vision prep failed: {e}")

def mode_evaluate() -> None:
    """Run Pytest, then evaluate Vision, Audio, and Fused pipelines against defined safety criteria."""
    _require_audio_deps()
    _require_vision_deps()
    
    config_path = PROJECT_ROOT / "eval_config.yaml"
    if not config_path.is_file():
        print("[ERROR] eval_config.yaml not found.")
        sys.exit(1)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)["thresholds"]
        
    # Prepare Report Directory (grouped into runs, e.g., test_reports/run1)
    from src.utils import get_next_run_dir
    import shutil
    
    report_dir = get_next_run_dir(PROJECT_ROOT / "test_reports")
    report_path = report_dir / "evaluation_report.md"
    
    class TeeLogger:
        def __init__(self, path):
            self.terminal = sys.stdout
            self.log = open(path, "a", encoding="utf-8")
        def write(self, msg):
            self.terminal.write(msg)
            self.log.write(msg)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        def isatty(self):
            if hasattr(self.terminal, 'isatty'):
                return self.terminal.isatty()
            return False

    console_log_path = report_dir / "console.log"
    sys.stdout = TeeLogger(console_log_path)
    
    print(f"\n[INFO] Starting Evaluation Run: {report_dir.name}")
    print("\n--- STEP 1: UNIT TESTS (PYTEST) ---")
    try:
        import pytest
        exit_code = pytest.main([str(PROJECT_ROOT / "tests"), "-v", "--no-cov", "-q"])
        if exit_code == 0:
            print("[OK] All unit tests passed.")
        else:
            print("[WARN] Some unit tests failed. Proceeding with dataset evaluation...")
    except Exception as error:
        print(f"[WARN] Pytest execution skipped: {error}")

    print("\n--- STEP 2: EVALUATING VISION MODEL (YOLOv8) ---")
    vision_test_dir = PROJECT_ROOT / "data" / "test_vision"
    if not vision_test_dir.is_dir():
        print("[FAIL] Vision test data not found. Please run: ROBOFLOW_API_KEY=xxx python3 main.py --mode prepare")
        sys.exit(1)
    
    from src.data_prep.evaluate_vision_model import evaluate_vision_model
    vision_report = evaluate_vision_model(VISION_MODEL, vision_test_dir, report_dir)
    print(f"  mAP@50: {vision_report['mAP50']:.4f} | Child Recall: {vision_report['child_recall']:.4f}")
    
    # Vision plots are now neatly generated inside report_dir / "vision_details"
    vision_plots = []
    vision_details_dir = report_dir / "vision_details"
    if vision_details_dir.is_dir():
        for plot_name in [
            "confusion_matrix.png", "F1_curve.png", "PR_curve.png",
            "P_curve.png", "R_curve.png", 
            "val_batch0_labels.jpg", "val_batch0_pred.jpg", 
            "results.png"
        ]:
            plot_file = vision_details_dir / plot_name
            if plot_file.is_file():
                vision_plots.append(f"vision_details/{plot_name}")
    
    print("\n--- STEP 3: EVALUATING AUDIO MODEL (AudioCNN) ---")
    audio_test_dir = PROJECT_ROOT / "data" / "test_audio"
    if not audio_test_dir.is_dir():
        print("[FAIL] Audio test data not found. Running auto-prepare...")
        mode_prepare()
        
    from src.data_prep.evaluate_audio_model import evaluate_model as evaluate_audio_model, plot_audio_metrics
    audio_report = evaluate_audio_model(AUDIO_MODEL, audio_test_dir)
    audio_plot_paths = plot_audio_metrics(audio_report, report_dir)
    print(f"  Accuracy: {audio_report['accuracy']:.4f} | Cry Recall: {audio_report['cry_recall']:.4f}")
    
    print("\n--- STEP 4: EVALUATING FUSED PIPELINE ---")
    from src.data_prep.evaluate_fused_model import evaluate_fused_model
    fused_report = evaluate_fused_model(VISION_MODEL, AUDIO_MODEL, vision_test_dir, audio_test_dir)
    print(f"  Scenarios: {fused_report['total_scenarios']} | FNR: {fused_report['false_negative_rate']:.4f}")
    
    # Check Verdict
    is_ready = True
    reasons = []
    
    if vision_report["child_samples"] < config["vision"]["min_test_samples"]:
        is_ready = False
        reasons.append(f"- Vision: Not enough child samples ({vision_report['child_samples']} < {config['vision']['min_test_samples']}). Need to augment test dataset.")
    elif vision_report["child_recall"] < config["vision"]["child_recall"]:
        is_ready = False
        reasons.append(f"- Vision: Child Recall {vision_report['child_recall']:.2f} < {config['vision']['child_recall']}. Missed {len(vision_report['missed_children'])} images.")
        
    if audio_report["cry_samples"] < config["audio"]["min_test_samples"]:
        is_ready = False
        reasons.append(f"- Audio: Not enough cry samples ({audio_report['cry_samples']} < {config['audio']['min_test_samples']}). Need to augment test dataset.")
    elif audio_report["cry_recall"] < config["audio"]["cry_recall"]:
        is_ready = False
        reasons.append(f"- Audio: Cry Recall {audio_report['cry_recall']:.2f} < {config['audio']['cry_recall']}. Missed {len(audio_report['missed_cries'])} audios.")
        
    if fused_report["false_negative_rate"] > config["fused"]["false_negative_rate"]:
        is_ready = False
        reasons.append(f"- Fused: False Negative Rate {fused_report['false_negative_rate']:.2f} > {config['fused']['false_negative_rate']}. The system failed to trigger alarm on real distress cases.")
        
    # Generate Markdown Report (Paths defined at the top of the function)
    
    lines = [
        f"# CPDS-AI Safety & Evaluation Report ({report_dir.name})",
        f"**Date:** {datetime.datetime.now().isoformat()}",
        f"**Vision Model Hash:** {get_model_hash(VISION_MODEL)}",
        f"**Audio Model Hash:** {get_model_hash(AUDIO_MODEL)}",
        "",
        "## VERDICT",
        "**PRODUCTION READY: YES**" if is_ready else "**PRODUCTION READY: NO**",
    ]
    
    if not is_ready:
        lines.append("\n### Action Items / Failure Reasons:")
        lines.extend(reasons)
        lines.append("\n*Recommendation: Do NOT blindly retrain. Analyze the detailed failures below, identify patterns (e.g., all missed cries are < 1s, all missed children are highly occluded), and augment the training data specifically for those edge cases before retraining.*")
        
    lines.extend([
        "\n## 1. Vision Model (YOLOv8)",
        f"- mAP@50: {vision_report['mAP50']:.4f}",
        f"- mAP@50-95: {vision_report['mAP50_95']:.4f}",
        f"- Child Recall: {vision_report['child_recall']:.4f} (Target: {config['vision']['child_recall']})",
    ])
    if vision_plots:
        lines.append("\n### Vision Performance Plots:")
        for p in vision_plots:
            lines.append(f"![{p.split('/')[-1]}]({p})")
            
    lines.append("\n### Vision False Negatives (Missed Children):")
    if not vision_report["missed_children"]:
        lines.append("None. Perfect recall.")
    else:
        for m in vision_report["missed_children"]:
            lines.append(f"- `{m['file']}`: Conf={m['best_pred_conf']}, Area={m['gt_area_pct']}%. Reason: {m['reason']}")
            
    audio_cry = audio_report["classification_report"].get("cry", {})
    lines.extend([
        "\n## 2. Audio Model (AudioCNN)",
        f"- Accuracy: {audio_report['accuracy']:.4f}",
        f"- ROC-AUC: {audio_report['roc_auc']:.4f}",
        f"- Cry Precision: {audio_cry.get('precision', 0):.4f}",
        f"- Cry Recall: {audio_report['cry_recall']:.4f} (Target: {config['audio']['cry_recall']})",
        f"- Cry F1-Score: {audio_cry.get('f1-score', 0):.4f}",
    ])
    if audio_plot_paths:
        lines.append("\n### Audio Performance Plots:")
        for name, p in audio_plot_paths.items():
            lines.append(f"![{name}]({p.name})")
            
    lines.append("\n### Audio False Negatives (Missed Cries):")
    if not audio_report["missed_cries"]:
        lines.append("None. Perfect recall.")
    else:
        for m in audio_report["missed_cries"]:
            lines.append(f"- `{m['file']}`: Prob={m['cry_prob']}, RMS={m['rms_volume']}, Duration={m['duration_sec']}s. Reason: {m['reason']}")
            
    lines.extend([
        "\n## 3. Fused Pipeline",
        f"- Scenarios Evaluated: {fused_report['total_scenarios']}",
        f"- Correct Alarms/Silences: {fused_report['correct']}",
        f"- False Negative Rate: {fused_report['false_negative_rate']:.4f} (Target: {config['fused']['false_negative_rate']})",
        f"- False Positive Rate: {fused_report['false_positive_rate']:.4f}",
        "\n### Fused Pipeline Failures:"
    ])
    if not fused_report["failures"]:
        lines.append("None. Perfect logic.")
    else:
        for m in fused_report["failures"]:
            lines.append(f"- Scenario `{m['scenario']}`: {m['image']} + {m['audio']} | Expected: {m['expected_alarm']}, Got: {m['actual_alarm']} (VisionConf: {m['vision_child_conf']}, AudioConf: {m['audio_cry_conf']})")
            
    report_path.write_text("\n".join(lines), encoding="utf-8")
    
    print("\n--- FINAL VERDICT ---")
    if is_ready:
        print("[SUCCESS] PRODUCTION READY: YES")
    else:
        print("[FAIL] PRODUCTION READY: NO")
        for r in reasons:
            print(f"  {r}")
            
    print(f"\nDetailed markdown report written to: {report_path}")
    print(f"Full console log saved to: {console_log_path}")
    if not is_ready:
        sys.exit(1)


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

    # Detect if we are running outside a virtual environment
    if sys.prefix == sys.base_prefix:
        print("\033[93m[WARN] You are not running inside a virtual environment!")
        print("       It is highly recommended to use one to avoid package conflicts.")
        print("       To create and activate one, run:")
        print("         python3 -m venv .venv")
        print("         source .venv/bin/activate")
        print("       Then run setup: bash setup_environment.sh --full\033[0m\n")

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
