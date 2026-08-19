#!/usr/bin/env bash
set -uo pipefail

REPORT_ROOT="test_reports"
REPORT_NUMBER=1
while [[ -e "$REPORT_ROOT/report$REPORT_NUMBER" ]]; do
    ((REPORT_NUMBER += 1))
done
REPORT_DIR="$REPORT_ROOT/report$REPORT_NUMBER"
mkdir -p "$REPORT_DIR"

TEST_COMMAND="python3 -m pytest"
echo "Running the full CPDS-AI test suite."
echo "Report directory: $REPORT_DIR"

set +e
python3 -m pytest --junitxml="$REPORT_DIR/results.xml" 2>&1 | tee "$REPORT_DIR/test_output.txt"
TEST_STATUS=${PIPESTATUS[0]}
set -e

python3 scripts/generate_test_report.py \
    --xml "$REPORT_DIR/results.xml" \
    --output "$REPORT_DIR/test_report.md" \
    --title "CPDS-AI Full Test Report" \
    --command "$TEST_COMMAND" \
    --log "$REPORT_DIR/test_output.txt"

echo "Markdown report: $REPORT_DIR/test_report.md"
exit "$TEST_STATUS"
