#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="data/models/best.onnx"
RUN_CAMERA=false
IMAGE_PATH=""
REPORT_ROOT="test_reports"
REPORT_NUMBER=1
while [[ -e "$REPORT_ROOT/report$REPORT_NUMBER" ]]; do
    ((REPORT_NUMBER += 1))
done
REPORT_DIR="$REPORT_ROOT/report$REPORT_NUMBER"
mkdir -p "$REPORT_DIR"

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
python3 -m pytest --no-cov tests/test_vision_model.py --junitxml="$REPORT_DIR/results.xml" 2>&1 | tee "$REPORT_DIR/test_output.txt"
TEST_STATUS=${PIPESTATUS[0]}
set -e
python3 scripts/generate_test_report.py \
    --xml "$REPORT_DIR/results.xml" \
    --output "$REPORT_DIR/test_report.md" \
    --title "CPDS-AI Vision Test Report" \
    --command "python3 -m pytest --no-cov tests/test_vision_model.py" \
    --log "$REPORT_DIR/test_output.txt"
echo "Markdown report: $REPORT_DIR/test_report.md"
if [[ "$TEST_STATUS" -ne 0 ]]; then
    exit "$TEST_STATUS"
fi
echo "Vision unit tests passed."

if [[ -n "$IMAGE_PATH" ]]; then
    echo ""
    echo "[2] Running ONNX smoke test on: $IMAGE_PATH"
    python3 -m src.inference.verify_vision --model "$MODEL_PATH" --image "$IMAGE_PATH"
elif [[ "$RUN_CAMERA" == true ]]; then
    echo ""
    echo "[2] Launching live camera inference..."
    python3 -m src.inference.camera_vision --model "$MODEL_PATH"
else
    echo ""
    echo "[2] Model smoke test skipped. Use --image PATH or --camera to run $MODEL_PATH."
fi

echo ""
echo "======================================"
echo " Vision checks completed successfully."
echo "======================================"
