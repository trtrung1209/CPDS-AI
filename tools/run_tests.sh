#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/shell_helpers.sh"
require_project_venv
cd "$PROJECT_ROOT"

REPORT_DIR="$(next_report_dir)"
TEST_COMMAND=".venv/bin/python -m pytest"
echo "Running the full CPDS-AI test suite."
echo "Report directory: $REPORT_DIR"

set +e
"$PYTHON_BIN" -m pytest --junitxml="$REPORT_DIR/results.xml" 2>&1 | tee "$REPORT_DIR/test_output.txt"
TEST_STATUS=${PIPESTATUS[0]}
set -e

write_pytest_report "$REPORT_DIR" "CPDS-AI Full Test Report" "$TEST_COMMAND"
exit "$TEST_STATUS"
