#!/usr/bin/env python
"""
TerraSpectra Environment Diagnostic Script.
Runs hardware, runtime, and dependency checks and prints an audit report.
"""

import sys
from pathlib import Path

# Add src to pythonpath
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from terraspectra.core.diagnostics import run_environment_diagnostics, format_diagnostics_report


def main():
    report = run_environment_diagnostics()
    formatted = format_diagnostics_report(report)
    print(formatted)


if __name__ == "__main__":
    main()
