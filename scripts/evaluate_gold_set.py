#!/usr/bin/env python3
"""Validate the gold workspace and emit publication-ready accuracy tables when adjudication is complete."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jurisdrive.gold import benchmark_gold, gold_status  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["candidate_id"])] = row
    return rows


def verify_event_chain(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "events": 0, "valid": True}
    previous = None
    events = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            claimed = event.pop("event_sha256", None)
            if event.get("previous_event_sha256") != previous:
                return {"exists": True, "events": events, "valid": False, "error": f"previous hash mismatch at line {line_number}"}
            calculated = hashlib.sha256(
                ((previous or "") + json.dumps(event, ensure_ascii=False, sort_keys=True)).encode("utf-8")
            ).hexdigest()
            if claimed != calculated:
                return {"exists": True, "events": events, "valid": False, "error": f"event hash mismatch at line {line_number}"}
            previous = claimed
            events += 1
    return {"exists": True, "events": events, "valid": True, "last_event_sha256": previous}


def write_metric_csv(path: Path, metrics: dict) -> None:
    rows = []
    for method, result in metrics["methods"].items():
        unweighted = result.get("binary_metrics_on_covered") or {}
        weighted_container = result.get("population_weighted") or {}
        weighted = weighted_container.get("binary_metrics_on_covered") or {}
        rows.append(
            {
                "method": method,
                "status": result.get("status"),
                "sample_coverage": result.get("coverage"),
                "sample_selective_risk": result.get("risk"),
                "sample_precision": unweighted.get("precision"),
                "sample_recall": unweighted.get("recall"),
                "sample_f1": unweighted.get("f1"),
                "sample_mcc": unweighted.get("mcc"),
                "sample_false_acceptance_rate": unweighted.get("false_acceptance_rate"),
                "population_weighted_coverage": weighted_container.get("coverage"),
                "population_weighted_selective_risk": weighted_container.get("risk"),
                "population_weighted_precision": weighted.get("precision"),
                "population_weighted_recall": weighted.get("recall"),
                "population_weighted_f1": weighted.get("f1"),
                "population_weighted_mcc": weighted.get("mcc"),
                "population_weighted_false_acceptance_rate": weighted.get("false_acceptance_rate"),
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_disagreements(gold_dir: Path, path: Path) -> int:
    tasks = read_jsonl(gold_dir / "annotation_tasks.jsonl")
    left = read_jsonl(gold_dir / "annotator_a.jsonl")
    right = read_jsonl(gold_dir / "annotator_b.jsonl")
    rows = []
    for candidate_id in sorted(tasks):
        if left[candidate_id].get("label") != right[candidate_id].get("label"):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "annotator_a": left[candidate_id],
                    "annotator_b": right[candidate_id],
                    "adjudication_required": True,
                }
            )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    gold_dir = args.gold_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    status = gold_status(gold_dir)
    write_json(output_dir / "gold_status.json", status)
    event_audit = {
        role: verify_event_chain(gold_dir / f"annotation_events_{role}.jsonl")
        for role in ("annotator_a", "annotator_b", "adjudicated")
    }
    required_files = [
        "sampling_summary.json",
        "annotation_tasks.jsonl",
        "annotator_a.jsonl",
        "annotator_b.jsonl",
        "adjudicated.jsonl",
        "predictions_rule_only.jsonl",
        "predictions_qwen_only.jsonl",
        "predictions_hybrid.jsonl",
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "gold_dir": str(gold_dir),
        "files": {
            name: {"sha256": sha256_file(gold_dir / name), "bytes": (gold_dir / name).stat().st_size}
            for name in required_files
        },
        "annotation_event_chains": event_audit,
        "metrics_generated": False,
    }
    if not all(audit["valid"] for audit in event_audit.values()):
        write_json(output_dir / "evaluation_manifest.json", manifest)
        raise SystemExit("Annotation event chain validation failed")

    if not status["metrics_ready"]:
        manifest["blocked_reason"] = "Adjudicated binary labels are incomplete; accuracy metrics were intentionally withheld."
        write_json(output_dir / "evaluation_manifest.json", manifest)
        print(json.dumps({"status": status, "manifest": manifest}, ensure_ascii=False, indent=2))
        return 2 if args.require_complete else 0

    metrics = benchmark_gold(gold_dir, output_dir / "metrics.json")
    write_metric_csv(output_dir / "metrics_table.csv", metrics)
    disagreement_count = write_disagreements(gold_dir, output_dir / "disagreements.jsonl")
    manifest["metrics_generated"] = True
    manifest["disagreement_count"] = disagreement_count
    manifest["outputs"] = {
        name: sha256_file(output_dir / name)
        for name in ("gold_status.json", "metrics.json", "metrics_table.csv", "disagreements.jsonl")
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
