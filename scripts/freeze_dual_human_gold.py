#!/usr/bin/env python3
"""Freeze dual-human consensus while preserving uncertain cases for review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jurisdrive.gold_consensus import (  # noqa: E402
    PROTOCOL_VERSION,
    freeze_dual_human_consensus,
    sha256_file,
)


def optional_count(value: str) -> int | None:
    return None if value.lower() == "any" else int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=optional_count, default=900)
    parser.add_argument("--expected-consensus", type=optional_count, default=743)
    parser.add_argument("--expected-review", type=optional_count, default=157)
    parser.add_argument("--semantic-review-sample-size", type=int, default=100)
    parser.add_argument("--semantic-sample-seed", type=int, default=20260823)
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION)
    parser.add_argument(
        "--protocol-statement",
        type=Path,
        help="Optional author-approved protocol statement bound into the freeze manifest.",
    )
    parser.add_argument(
        "--protocol-statement-status",
        default="author-approved-not-interaction-log-verified",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace artifacts in output-dir; source files are never modified.",
    )
    args = parser.parse_args()
    protocol_statement = None
    if args.protocol_statement is not None:
        statement = args.protocol_statement.resolve()
        if not statement.is_file():
            raise FileNotFoundError(f"protocol statement not found: {statement}")
        protocol_statement = {
            "path": str(statement),
            "sha256": sha256_file(statement),
            "status": args.protocol_statement_status,
        }
    manifest = freeze_dual_human_consensus(
        tasks_path=args.tasks.resolve(),
        annotator_a_path=args.annotator_a.resolve(),
        annotator_b_path=args.annotator_b.resolve(),
        output_dir=args.output_dir.resolve(),
        expected_total=args.expected_total,
        expected_consensus=args.expected_consensus,
        expected_review=args.expected_review,
        semantic_review_sample_size=args.semantic_review_sample_size,
        semantic_sample_seed=args.semantic_sample_seed,
        protocol_version=args.protocol_version,
        protocol_statement=protocol_statement,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
