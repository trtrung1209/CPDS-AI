#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/shell_helpers.sh"
require_project_venv
cd "$PROJECT_ROOT"

MODEL_PATH="data/models/yolov8n-adult-child.onnx"
RUN_CAMERA=false
IMAGE_PATH=""
REPORT_DIR="$(next_report_dir)"

usage() {
    echo "Usage: bash run_vision_tests.sh [--image PATH] [--camera] [MODEL_PATH]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            [[ $# -ge 2 ]] || { echo "--image requires a file path." >&2; exit 2; }
            IMAGE_PATH="$2"
            shift 2
            ;;
        --camera)
            RUN_CAMERA=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            MODEL_PATH="$1"
            shift
            ;;
    esac
done

echo "======================================"
echo " Running CPDS-AI Vision Tests"
echo " Model: $MODEL_PATH"
echo "======================================"

echo ""
echo "[1] Running vision unit tests..."
# The focused suite should not be measured against global project coverage.
set +e
"$PYTHON_BIN" -m pytest --no-cov tests/test_vision_model.py --junitxml="$REPORT_DIR/results.xml" 2>&1 | tee "$REPORT_DIR/test_output.txt"
TEST_STATUS=${PIPESTATUS[0]}
set -e
write_pytest_report "$REPORT_DIR" "CPDS-AI Vision Test Report" ".venv/bin/python -m pytest --no-cov tests/test_vision_model.py"
if [[ "$TEST_STATUS" -ne 0 ]]; then
    exit "$TEST_STATUS"
fi
echo "Vision unit tests passed."

if [[ -n "$IMAGE_PATH" ]]; then
    echo ""
    echo "[2] Running ONNX smoke test on: $IMAGE_PATH"
    "$PYTHON_BIN" -c "import cv2, ultralytics" >/dev/null 2>&1 || {
        echo "Vision smoke tests require the full environment. Run: bash setup_environment.sh --full --recreate" >&2
        exit 1
    }
    "$PYTHON_BIN" -m src.inference.verify_vision --model "$MODEL_PATH" --image "$IMAGE_PATH"
elif [[ "$RUN_CAMERA" == true ]]; then
    echo ""
    echo "[2] Launching live camera inference..."
    "$PYTHON_BIN" -c "import cv2, ultralytics" >/dev/null 2>&1 || {
        echo "Camera inference requires the full environment. Run: bash setup_environment.sh --full --recreate" >&2
        exit 1
    }
    "$PYTHON_BIN" -m src.inference.camera_vision --model "$MODEL_PATH"
else
    echo ""
    echo "[2] Model smoke test skipped. Use --image PATH or --camera to run $MODEL_PATH."
fi

echo ""
echo "======================================"
echo " Vision checks completed successfully."
echo "======================================"
