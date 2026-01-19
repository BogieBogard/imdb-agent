# imdb-agent

## Testing Workflow

This project includes a test manager script to facilitate regression testing and output comparison.

### 1. Establish Baseline
Run this command to generate the baseline test reports. This represents the "known good" state of your code.
```bash
python scripts/test_manager.py --baseline
```
-   Generates `test_reports/baseline.xml` (Status)
-   Generates `test_reports/baseline.txt` (Full Output)

### 2. Run Tests & Compare
Run this command to execute current tests and compare them against the baseline.
```bash
python scripts/test_manager.py
```
-   Generates `test_reports/current.xml` (Status)
-   Generates `test_reports/current.txt` (Full Output)
-   Generates `test_reports/comparison_report.txt` (Comparison Table)
-   Prints the comparison report to the console.

### 3. Manual Output Comparison
To see exactly what changed in the test output (stdout), use `diff`:
```bash
diff test_reports/baseline.txt test_reports/current.txt
```