#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
INSTALL_MICROPHONE=false
RECREATE=false
INSTALL_FULL=false
INSTALL_AUDIO=false

usage() {
    echo "Usage: bash setup_environment.sh [--audio] [--full] [--microphone] [--recreate]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --microphone) INSTALL_MICROPHONE=true ;;
        --audio) INSTALL_AUDIO=true ;;
        --full) INSTALL_FULL=true; INSTALL_MICROPHONE=true ;;
        --recreate) RECREATE=true ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [[ "$RECREATE" == true && -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
REQUIREMENTS_FILE="requirements-test.txt"
if [[ "$INSTALL_AUDIO" == true ]]; then
    REQUIREMENTS_FILE="requirements-audio.txt"
fi
if [[ "$INSTALL_FULL" == true ]]; then
    REQUIREMENTS_FILE="requirements.txt"
fi
"$VENV_DIR/bin/python" -m pip install --upgrade --force-reinstall -r "$PROJECT_ROOT/$REQUIREMENTS_FILE"
if [[ "$INSTALL_MICROPHONE" == true ]]; then
    "$VENV_DIR/bin/python" -m pip install --upgrade --force-reinstall -r "$PROJECT_ROOT/requirements-microphone.txt"
fi

echo "Environment ready: $VENV_DIR ($REQUIREMENTS_FILE)"
if [[ "$INSTALL_FULL" == false ]]; then
    echo "Use --audio for audio ONNX smoke tests/evaluation, or --full for real vision ONNX smoke tests/webcam inference."
fi
echo "Run tests with: bash run_tests.sh"
