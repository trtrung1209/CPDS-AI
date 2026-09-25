#!/usr/bin/env python3
"""Convert a pytest JUnit XML result into a concise Markdown test report."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import textwrap
import xml.etree.ElementTree as ET


def clean_text(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def test_status(case: ET.Element) -> tuple[str, str]:
    if case.find("failure") is not None:
        return "FAILED", clean_text(case.find("failure").text or case.find("failure").get("message"))
    if case.find("error") is not None:
        return "ERROR", clean_text(case.find("error").text or case.find("error").get("message"))
    if case.find("skipped") is not None:
        return "SKIPPED", clean_text(case.find("skipped").get("message") or case.find("skipped").text)
    return "PASSED", ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Markdown report from pytest JUnit XML.")
    parser.add_argument("--xml", required=True, type=Path, help="Path to pytest JUnit XML output")
    parser.add_argument("--output", required=True, type=Path, help="Output Markdown report path")
    parser.add_argument("--title", default="CPDS-AI Test Report", help="Report title")
    parser.add_argument("--command", required=True, help="Command used to run the tests")
    parser.add_argument("--log", type=Path, help="Optional pytest terminal output for coverage extraction")
    args = parser.parse_args()

    root = ET.parse(args.xml).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    cases = [case for suite in suites for case in suite.findall("testcase")]
    total = len(cases)
    failed = sum(1 for case in cases if test_status(case)[0] in {"FAILED", "ERROR"})
    skipped = sum(1 for case in cases if test_status(case)[0] == "SKIPPED")
    passed = total - failed - skipped
    duration = sum(float(case.get("time", "0")) for case in cases)
    outcome = "PASSED" if failed == 0 else "FAILED"
    coverage = None
    if args.log and args.log.is_file():
        import re

        match = re.search(r"Total coverage:\s*([0-9.]+)%", args.log.read_text(encoding="utf-8", errors="replace"))
        coverage = match.group(1) if match else None

    lines = [
        f"# {args.title}",
        "",
        f"**Result:** {outcome}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Generated (UTC) | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| Test command | `{args.command}` |",
        f"| Total | {total} |",
        f"| Passed | {passed} |",
        f"| Failed / errors | {failed} |",
        f"| Skipped | {skipped} |",
        f"| Test duration | {duration:.3f} s |",
    ]
    if coverage is not None:
        lines.append(f"| Coverage | {coverage}% |")
    lines.extend(["", "## Test Cases", "", "| Status | Test | Duration |", "| --- | --- | ---: |"])

    for case in cases:
        status, _ = test_status(case)
        name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
        lines.append(f"| {status} | `{name}` | {float(case.get('time', '0')):.3f} s |")

    problems = [(case, *test_status(case)) for case in cases if test_status(case)[0] in {"FAILED", "ERROR"}]
    if problems:
        lines.extend(["", "## Failures", ""])
        for case, status, detail in problems:
            name = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
            lines.extend([f"### {status}: `{name}`", "", textwrap.fill(detail or "No failure details provided.", width=100), ""])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
