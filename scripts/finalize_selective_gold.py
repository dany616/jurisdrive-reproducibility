#!/usr/bin/env python3
"""Apply the 157-case additional review without forcing unresolved cases binary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jurisdrive.gold_consensus import apply_additional_review  # noqa: E402


def optional_count(value: str) -> int | None:
    return None if value.lower() == "any" else int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-reference", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-review", type=optional_count, default=157)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    manifest = apply_additional_review(
        full_reference_path=args.full_reference.resolve(),
        adjudication_path=args.adjudication.resolve(),
        output_path=output_dir / "full_adjudicated_gold.jsonl",
        manifest_path=output_dir / "adjudication_manifest.json",
        expected_review=args.expected_review,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
