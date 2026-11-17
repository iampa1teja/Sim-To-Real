"""Audit a local dataset and persist the report."""

import argparse as _argparse

from core.datasets.audit import audit_dataset as _audit_dataset, write_audit_report as _write_audit_report


def main(argv=None):
    parser = _argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--source_root")
    parser.add_argument("--no_manifest", action="store_true")
    args = parser.parse_args(argv)
    report = _audit_dataset(args.root, source_root=args.source_root, check_manifest=not args.no_manifest)
    _write_audit_report(args.root, report)
    for check in report.checks:
        print(f"PASS {check['name']}" if check["passed"] else f"FAIL {check['name']}: {'; '.join(check['details'])}")
    print(f"Audit {'passed' if report.passed else 'failed'}: {report.episodes} episodes")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
