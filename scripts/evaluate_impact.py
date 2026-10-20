#!/usr/bin/env python3
"""Evaluate ChangeRadar impact reports against a small labeled fixture set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from changeradar.evaluation import DEFAULT_LABELS, load_impact_labels, run_impact_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("lexical", "semantic", "hybrid"),
        default="lexical",
        help="retrieval method; lexical is offline and does not load a model",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="path to the labeled impact cases JSON file",
    )
    args = parser.parse_args()
    labels = load_impact_labels(args.labels)
    result = run_impact_evaluation(method=args.method, cases=labels)
    print("Impact report evaluation (computed from current deterministic reports):")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
