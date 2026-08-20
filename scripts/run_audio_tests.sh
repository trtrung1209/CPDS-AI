#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/shell_helpers.sh"
require_project_venv
cd "$PROJECT_ROOT"

MODEL_PATH="data/models/audio_model.onnx"
LABELS_PATH="data/models/audio_labels.json"
AUDIO_PATH=""
EVALUATION_DIR=""
REPORT_DIR="$(next_report_dir)"

usage() {
    echo "Usage: bash run_audio_tests.sh [--audio PATH] [--evaluate DIR] [--model PATH] [--labels PATH]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --audio) AUDIO_PATH="${2:?--audio requires a path}"; shift 2 ;;
        --evaluate) EVALUATION_DIR="${2:?--evaluate requires a directory}"; shift 2 ;;
        --model) MODEL_PATH="${2:?--model requires a path}"; shift 2 ;;
        --labels) LABELS_PATH="${2:?--labels requires a path}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

echo "======================================"
echo " Running CPDS-AI Audio Tests"
echo "======================================"
echo "[1] Running audio unit tests..."
set +e
"$PYTHON_BIN" -m pytest --no-cov tests/test_audio_model.py --junitxml="$REPORT_DIR/results.xml" 2>&1 | tee "$REPORT_DIR/test_output.txt"
TEST_STATUS=${PIPESTATUS[0]}
set -e
write_pytest_report "$REPORT_DIR" "CPDS-AI Audio Test Report" ".venv/bin/python -m pytest --no-cov tests/test_audio_model.py"
[[ "$TEST_STATUS" -eq 0 ]] || exit "$TEST_STATUS"

if [[ -n "$AUDIO_PATH" ]]; then
    echo "[2] Running audio ONNX smoke test: $AUDIO_PATH"
    "$PYTHON_BIN" -c "import onnxruntime; from src.inference.verify_audio import validate_audio_runtime; validate_audio_runtime()" >/dev/null 2>&1 || {
        echo "Audio ONNX smoke tests require the audio environment. Run: bash setup_environment.sh --audio --recreate" >&2
        exit 1
    }
    "$PYTHON_BIN" -m src.inference.verify_audio --model "$MODEL_PATH" --audio "$AUDIO_PATH" --labels "$LABELS_PATH"
fi

if [[ -n "$EVALUATION_DIR" ]]; then
    echo "[3] Evaluating model on: $EVALUATION_DIR"
    "$PYTHON_BIN" -c "import onnxruntime, sklearn; from src.inference.verify_audio import validate_audio_runtime; validate_audio_runtime()" >/dev/null 2>&1 || {
        echo "Audio evaluation requires the audio environment. Run: bash setup_environment.sh --audio --recreate" >&2
        exit 1
    }
    "$PYTHON_BIN" scripts/evaluate_audio_model.py --model "$MODEL_PATH" --labels "$LABELS_PATH" --test-dir "$EVALUATION_DIR"
fi

echo "Audio checks completed successfully."
