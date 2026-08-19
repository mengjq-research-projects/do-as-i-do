#!/usr/bin/env python3
"""Validate a final UR3e + Sharpa + object package without a GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from final_package import validate_final_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package_dir",
        type=Path,
        help="Directory containing deployment_manifest.json.",
    )
    args = parser.parse_args()
    report = validate_final_package(args.package_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
