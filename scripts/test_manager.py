import argparse
import sys
import os
import pytest
import xml.etree.ElementTree as ET
from tabulate import tabulate
from datetime import datetime

# Add project root to sys.path so tests can import from 'agent.py'
sys.path.append(os.getcwd())

import subprocess

# Configuration
REPORT_DIR = "test_reports"
BASELINE_FILE = os.path.join(REPORT_DIR, "baseline.xml")
CURRENT_FILE = os.path.join(REPORT_DIR, "current.xml")
BASELINE_TXT = os.path.join(REPORT_DIR, "baseline.txt")
CURRENT_TXT = os.path.join(REPORT_DIR, "current.txt")
COMPARISON_TXT = os.path.join(REPORT_DIR, "comparison_report.txt")

def run_tests(xml_file, txt_file):
    """Runs pytest and outputs to the specified XML and TXT files."""
    print(f"Running tests...")
    print(f"  - XML report: {xml_file}")
    print(f"  - Text output: {txt_file}")
    
    # cmd = [sys.executable, "-m", "pytest", "tests/", f"--junitxml={xml_file}"]
    # Using 'pytest' directly or via sys.executable if installed in env
    cmd = [sys.executable, "-m", "pytest", "tests/", f"--junitxml={xml_file}"]

    with open(txt_file, "w") as f:
        # Capture both stdout and stderr to the file
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    
    print(f"Tests completed with exit code: {result.returncode}")
    return result.returncode

def parse_junit_xml(xml_file):
    """Parses JUnit XML to a dictionary {test_name: status}."""
    if not os.path.exists(xml_file):
        return {}

    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    results = {}
    
    for testcase in root.iter("testcase"):
        # Construct a unique name: classname + name
        name = f"{testcase.get('classname')}::{testcase.get('name')}"
        
        status = "PASS"
        if testcase.find("failure") is not None:
            status = "FAIL"
        elif testcase.find("error") is not None:
            status = "ERROR"
        elif testcase.find("skipped") is not None:
            status = "SKIP"
            
        results[name] = status
        
    return results

def compare_results(baseline_results, current_results, report_file=None):
    """Compares baseline and current results, prints a table, and optionally saves to file."""
    all_tests = set(baseline_results.keys()) | set(current_results.keys())
    
    table_data = []
    headers = ["Test Name", "Baseline", "Current", "Status"]
    
    regressions = 0
    fixes = 0
    
    for test in sorted(all_tests):
        base = baseline_results.get(test, "N/A")
        curr = current_results.get(test, "N/A")
        
        status = "STABLE"
        
        if base == "PASS" and curr in ["FAIL", "ERROR"]:
            status = "REGRESSION"
            regressions += 1
        elif base in ["FAIL", "ERROR"] and curr == "PASS":
            status = "FIXED"
            fixes += 1
        elif base == "N/A":
            status = "NEW"
        elif curr == "N/A":
            status = "MISSING"
        elif base != curr:
            status = "CHANGED"
            
        table_data.append([test, base, curr, status])

    # Build the report string
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"TEST COMPARISON REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*60)
    report_lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))
    report_lines.append("\nSummary:")
    report_lines.append(f"Total Tests: {len(all_tests)}")
    report_lines.append(f"Regressions: {regressions}")
    report_lines.append(f"Fixes:       {fixes}")
    
    if regressions > 0:
        report_lines.append("\n*** WARNING: REGRESSIONS DETECTED ***")
    else:
        report_lines.append("\nTest run clean (no regressions against baseline).")
        
    full_report = "\n".join(report_lines)
    
    # Print to console
    print("\n" + full_report)
    
    # Save to file if requested
    if report_file:
        with open(report_file, "w") as f:
            f.write(full_report)
        print(f"\nComparison report saved to {report_file}")

    if regressions > 0:
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Manage test baselines and runs.")
    parser.add_argument("--baseline", action="store_true", help="Run tests and update the baseline.")
    args = parser.parse_args()

    # Ensure report directory exists
    os.makedirs(REPORT_DIR, exist_ok=True)

    if args.baseline:
        print("Generating new baseline...")
        run_tests(BASELINE_FILE, BASELINE_TXT)
        print(f"Baseline saved to {BASELINE_FILE} and {BASELINE_TXT}")
    else:
        # Run current tests
        run_tests(CURRENT_FILE, CURRENT_TXT)
        
        # Load and compare
        if not os.path.exists(BASELINE_FILE):
            print("No baseline found. Please run with --baseline first to establish a baseline.")
            return

        base_res = parse_junit_xml(BASELINE_FILE)
        curr_res = parse_junit_xml(CURRENT_FILE)
        
        compare_results(base_res, curr_res, report_file=COMPARISON_TXT)

if __name__ == "__main__":
    main()
