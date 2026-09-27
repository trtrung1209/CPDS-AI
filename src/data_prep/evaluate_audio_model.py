"""Evaluate an exported audio ONNX model against a labelled directory tree."""

import argparse
import json
import numpy as np
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

from src.inference.verify_audio import infer_audio, validate_audio_runtime


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
CLASS_NAMES = ["noise", "cry"]


def collect_audio_files(directory: Path) -> list[Path]:
    """Return supported audio files in deterministic order."""
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS)


def analyze_audio_file(audio_path: Path):
    """Return RMS and duration of an audio file."""
    import librosa
    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(str(audio_path), sr=None)
        
    rms = float(np.sqrt(np.mean(y**2)))
    duration = float(len(y) / sr)
    return rms, duration


def evaluate_model(model_path: Path, test_dir: Path, labels_path: Path | None = None) -> dict:
    """Run inference for each labelled sample and return serialisable metrics."""
    model_path, test_dir = Path(model_path), Path(test_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Audio model does not exist: {model_path}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory does not exist: {test_dir}")

    librosa = validate_audio_runtime()

    y_true, y_pred, y_prob, failures = [], [], [], []
    missed_cries = []
    
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_directory = test_dir / class_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Expected class directory does not exist: {class_directory}")
        for audio_path in collect_audio_files(class_directory):
            try:
                result = infer_audio(model_path, audio_path, labels_path)
            except (OSError, RuntimeError, ValueError) as error:
                print(f"Error evaluating {audio_path}: {repr(error)}")
                failures.append({"file": str(audio_path), "error": str(error)})
                continue
            
            y_true.append(class_index)
            y_pred.append(int(result["is_crying"]))
            prob_cry = result["probabilities"]["cry"]
            y_prob.append(prob_cry)
            
            # Analyze False Negatives for 'cry' (class_index == 1)
            if class_index == 1 and not result["is_crying"]:
                rms, duration = analyze_audio_file(audio_path)
                reason = "Unknown"
                if rms < 0.02:
                    reason = "Extremely low volume (RMS < 0.02), likely suppressed by Noise Gate"
                elif duration < 1.0:
                    reason = "Short duration (< 1.0s), not enough features"
                else:
                    reason = "Model missed clearly audible cry"
                    
                missed_cries.append({
                    "file": audio_path.name,
                    "rms_volume": round(rms, 4),
                    "duration_sec": round(duration, 2),
                    "cry_prob": round(prob_cry, 3),
                    "reason": reason
                })

    if not y_true:
        raise ValueError("No audio files could be evaluated.")

    roc_auc = 0.0
    thresholds_sweep = []
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        for f, t, th in zip(fpr, tpr, thresh):
            thresholds_sweep.append({"threshold": float(th), "fpr": float(f), "tpr": float(t)})
    except ValueError:
        pass  # Only one class present

    cry_samples = sum(1 for y in y_true if y == 1)
    recall_cry = 1.0 if cry_samples == 0 else sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1) / cry_samples

    return {
        "model": str(model_path),
        "test_directory": str(test_dir),
        "evaluated_samples": len(y_true),
        "cry_samples": cry_samples,
        "failed_samples": failures,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": roc_auc,
        "cry_recall": float(recall_cry),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "missed_cries": missed_cries,
        "thresholds_sweep": thresholds_sweep,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an audio ONNX model on labelled audio files.")
    parser.add_argument("--model", type=Path, default=Path("data/models/audio_model.onnx"))
    parser.add_argument("--test-dir", type=Path, required=True, help="Directory containing cry/ and noise/ subdirectories.")
    parser.add_argument("--labels", type=Path, help="Optional audio_labels.json exported with the model.")
    parser.add_argument("--report", type=Path, default=Path("audio_evaluation_report.json"))
    args = parser.parse_args()

    report = evaluate_model(args.model, args.test_dir, args.labels)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Accuracy: {report['accuracy']:.3f} | ROC-AUC: {report['roc_auc']:.3f} ({report['evaluated_samples']} samples)")
    print(f"JSON report: {args.report}")


if __name__ == "__main__":
    main()
