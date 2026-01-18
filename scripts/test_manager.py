import argparse
import sys
import os
import pytest
import xml.etree.ElementTree as ET
from tabulate import tabulate
from datetime import datetime

# Add project root to sys.path so tests can import from 'agent.py'
sys.path.append(os.getcwd())

# Configuration
REPORT_DIR = "test_reports"
BASELINE_FILE = os.path.join(REPORT_DIR, "baseline.xml")
CURRENT_FILE = os.path.join(REPORT_DIR, "current.xml")

def run_tests(output_file):
    """Runs pytest and outputs to the specified XML file."""
    print(f"Running tests and saving report to {output_file}...")
    # Using sys.executable to ensure we use the same python environment
    retcode = pytest.main(["tests/", f"--junitxml={output_file}"])
    return retcode

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

def compare_results(baseline_results, current_results):
    """Compares baseline and current results and prints a table."""
    all_tests = set(baseline_results.keys()) | set(current_results.keys())
    
    table_data = []
    headers = ["Test Name", "Baseline", "Current", "Status"]
    
    regressions = 0
    fixes = 0
    
    for test in sorted(all_tests):
        base = baseline_results.get(test, "N/A")
        curr = current_results.get(test, "N/A")
        
        status = "STABLE"
        color = "" # Plain text for now
        
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

    print("\n" + "="*60)
    print(f"TEST COMPARISON REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\nSummary:")
    print(f"Total Tests: {len(all_tests)}")
    print(f"Regressions: {regressions}")
    print(f"Fixes:       {fixes}")
    
    if regressions > 0:
        print("\n*** WARNING: REGRESSIONS DETECTED ***")
        sys.exit(1)
    else:
        print("\nTest run clean (no regressions against baseline).")

def main():
    parser = argparse.ArgumentParser(description="Manage test baselines and runs.")
    parser.add_argument("--baseline", action="store_true", help="Run tests and update the baseline.")
    args = parser.parse_args()

    # Ensure report directory exists
    os.makedirs(REPORT_DIR, exist_ok=True)

    if args.baseline:
        print("Generating new baseline...")
        run_tests(BASELINE_FILE)
        print(f"Baseline saved to {BASELINE_FILE}")
    else:
        # Run current tests
        run_tests(CURRENT_FILE)
        
        # Load and compare
        if not os.path.exists(BASELINE_FILE):
            print("No baseline found. Please run with --baseline first to establish a baseline.")
            return

        base_res = parse_junit_xml(BASELINE_FILE)
        curr_res = parse_junit_xml(CURRENT_FILE)
        
        compare_results(base_res, curr_res)

if __name__ == "__main__":
    main()
