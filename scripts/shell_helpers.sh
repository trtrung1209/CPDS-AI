#!/usr/bin/env bash
# Shared helpers for reproducible local CPDS-AI shell workflows.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

require_project_venv() {
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "Project virtual environment is missing: $PROJECT_ROOT/.venv" >&2
        echo "Run: bash setup_environment.sh" >&2
        exit 1
    fi
    if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
        echo "Project virtual environment is incomplete: pytest is not installed." >&2
        echo "Run: bash setup_environment.sh --recreate" >&2
        exit 1
    fi
}

next_report_dir() {
    local report_root="$PROJECT_ROOT/test_reports"
    local report_number=1
    while [[ -e "$report_root/report$report_number" ]]; do
        ((report_number += 1))
    done
    local report_dir="$report_root/report$report_number"
    mkdir -p "$report_dir"
    printf '%s\n' "$report_dir"
}

write_pytest_report() {
    local report_dir="$1"
    local title="$2"
    local test_command="$3"
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/generate_test_report.py" \
        --xml "$report_dir/results.xml" \
        --output "$report_dir/test_report.md" \
        --title "$title" \
        --command "$test_command" \
        --log "$report_dir/test_output.txt"
    echo "Markdown report: $report_dir/test_report.md"
}
