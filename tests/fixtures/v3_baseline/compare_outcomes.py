"""Compare the current CPU test outcomes with the recorded v2 baseline.

    PYTHONPATH=src python tests/fixtures/v3_baseline/compare_outcomes.py

Tests are matched by file name and test id, not by folder, so moving a test file is not a change.
Runs the suite, then reports tests that passed in the baseline but no longer pass (regressions or
removed tests) and tests that fail now but were not failing in the baseline. Exit code 1 on either,
unless every missing test is listed in ``REMOVED`` or after ``--removed`` (file-name prefixes deleted on purpose).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
# Baseline tests deleted on purpose during the restructure (file-name prefixes), with the phase that removed them.
REMOVED = (
    "test_plan.py", "test_extensions.py", "test_urdf_planning.py", "test_check.py",       # Phase 1: CLI
    "test_sweep.py::test_gap", "test_stack_bowls.py::test_terminal_stability",            # Phase 1: tools/
)


def key(nodeid):
    path, _, rest = nodeid.partition("::")
    return f"{Path(path).name}::{rest}" if rest else Path(path).name


def run():
    out = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-rA",
                          "--continue-on-collection-errors"], cwd=ROOT, capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        match = re.match(r"^(PASSED|FAILED|ERROR)\s+(\S+)", line)
        if match:
            found[key(match.group(2))] = match.group(1).lower()
    return found, out.strip().splitlines()[-1]


def main(argv):
    removed = REMOVED + (tuple(argv[argv.index("--removed") + 1:]) if "--removed" in argv else ())
    baseline = {key(k): v for k, v in json.loads((HERE / "local_cpu_outcomes.json").read_text())["outcomes"].items()
                if v in ("passed", "failed", "error")}
    now, summary = run()
    lost = sorted(k for k, v in baseline.items() if v == "passed" and now.get(k) != "passed"
                  and not k.startswith(removed))  # --removed takes file-name prefixes
    broken = sorted(k for k, v in now.items() if v != "passed" and baseline.get(k) not in ("failed", "error"))
    print(summary)
    print(f"baseline passed: {sum(v == 'passed' for v in baseline.values())}; now passed: "
          f"{sum(v == 'passed' for v in now.values())}")
    for title, names in (("No longer passing", lost), ("Newly failing", broken)):
        print(f"{title}: {len(names)}")
        for name in names:
            print("  ", name)
    return 1 if lost or broken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
