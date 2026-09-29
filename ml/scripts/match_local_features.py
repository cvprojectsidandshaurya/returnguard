#!/usr/bin/env python3
"""Inspect local visual evidence between packing and rider image sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.local_features import best_local_match  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packing", required=True, nargs="+", type=Path, help="one or more seller packing image paths")
    parser.add_argument("--rider", required=True, nargs="+", type=Path, help="one or more rider image paths")
    parser.add_argument("--ratio-threshold", type=float, default=0.75)
    args = parser.parse_args()

    result = best_local_match(args.packing, args.rider, args.ratio_threshold)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
