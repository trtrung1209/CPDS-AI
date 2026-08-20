"""Evaluate an exported audio ONNX model against a labelled directory tree."""

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.inference.verify_audio import infer_audio


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
CLASS_NAMES = ["noise", "cry"]


def collect_audio_files(directory: Path) -> list[Path]:
    """Return supported audio files in deterministic order."""
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS)


def evaluate_model(model_path: Path, test_dir: Path, labels_path: Path | None = None) -> dict:
    """Run inference for each labelled sample and return serialisable metrics."""
    model_path, test_dir = Path(model_path), Path(test_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Audio model does not exist: {model_path}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory does not exist: {test_dir}")

    y_true, y_pred, failures = [], [], []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_directory = test_dir / class_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Expected class directory does not exist: {class_directory}")
        for audio_path in collect_audio_files(class_directory):
            try:
                result = infer_audio(model_path, audio_path, labels_path)
            except (OSError, RuntimeError, ValueError) as error:
                failures.append({"file": str(audio_path), "error": str(error)})
                continue
            y_true.append(class_index)
            y_pred.append(int(result["is_crying"]))

    if not y_true:
        raise ValueError("No audio files could be evaluated.")

    return {
        "model": str(model_path),
        "test_directory": str(test_dir),
        "evaluated_samples": len(y_true),
        "failed_samples": failures,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
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
    print(f"Accuracy: {report['accuracy']:.3f} ({report['evaluated_samples']} samples)")
    print(f"JSON report: {args.report}")


if __name__ == "__main__":
    main()
