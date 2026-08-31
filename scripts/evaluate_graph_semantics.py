#!/usr/bin/env python3
"""Evaluate evidence-graph semantics separately from exact-span integrity."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jurisdrive.gold_consensus import (  # noqa: E402
    evaluate_graph_semantics,
    sha256_file,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    reference = args.semantic_reference.resolve()
    predictions = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "graph_semantic_metrics.json"
    manifest_path = output_dir / "graph_semantic_manifest.json"
    if not args.overwrite and (metrics_path.exists() or manifest_path.exists()):
        raise FileExistsError("refusing to overwrite existing graph semantic evaluation")
    metrics = evaluate_graph_semantics(reference, predictions)
    write_json(metrics_path, metrics, overwrite=args.overwrite)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "claim_boundary": (
            "This evaluation measures semantic agreement. Exact source offsets remain a separate provenance-integrity check."
        ),
        "inputs": {
            "semantic_reference": {"path": str(reference), "sha256": sha256_file(reference)},
            "predictions": {"path": str(predictions), "sha256": sha256_file(predictions)},
        },
        "outputs": {"metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)}},
    }
    write_json(manifest_path, manifest, overwrite=args.overwrite)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
