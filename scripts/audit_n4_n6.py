#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jurisdrive.evidence import validate_evidence_spans
from jurisdrive.io import iter_jsonl, load_candidate, read_json, sha256_file, write_json
from jurisdrive.models import EvidenceGraphV1, ScenarioContractV1, SimulationResultV1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/audit/final_car_to_car_manifest.jsonl"),
    )
    parser.add_argument("--baseline", type=Path, default=Path("results/n0_n3_summary.json"))
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = {int(row["candidate_id"]): row for row in iter_jsonl(args.manifest)}
    graph_counts: Counter[str] = Counter()
    span_errors: list[str] = []
    for path in sorted(args.graph_dir.glob("jurisdrive_*.json")):
        graph = EvidenceGraphV1.model_validate(read_json(path))
        graph_counts["total"] += 1
        record = load_candidate(manifest[graph.candidate_id])
        errors = validate_evidence_spans(graph, str(record.get("source_text") or ""))
        if errors:
            span_errors.extend(f"{graph.scenario_id}: {error}" for error in errors)
        else:
            graph_counts["exact_evidence_span"] += 1

    contract_counts: Counter[str] = Counter()
    tier_c_promotions: list[int] = []
    for path in sorted(args.contract_dir.glob("jurisdrive_*.json")):
        contract = ScenarioContractV1.model_validate(read_json(path))
        contract_counts["total"] += 1
        contract_counts[contract.status.value] += 1
        if contract.readiness_tier.startswith("C_"):
            contract_counts["tier_c"] += 1
            if contract.status.value in {"ready", "needs_defaults"}:
                tier_c_promotions.append(contract.candidate_id)

    bundle_counts: Counter[str] = Counter()
    checksum_errors: list[str] = []
    execution_errors: list[str] = []
    for bundle in sorted(path for path in args.bundle_dir.glob("jurisdrive_*") if path.is_dir()):
        bundle_counts["total"] += 1
        result = SimulationResultV1.model_validate(read_json(bundle / "dry_run_report.json"))
        if (
            not result.executed
            and result.status.value == "not_executed"
            and result.actor_states is None
            and result.collisions is None
            and result.minimum_ttc_seconds is None
            and result.keyframes is None
        ):
            bundle_counts["strict_not_executed"] += 1
        else:
            execution_errors.append(result.scenario_id)
        for line in (bundle / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            if sha256_file(bundle / name) != expected:
                checksum_errors.append(f"{bundle.name}/{name}")
    baseline = read_json(args.baseline)
    counts = baseline["counts"]
    baseline_invariant = (
        counts["final_car_to_car"]
        + counts["final_not_car_to_car"]
        + counts["final_unresolved"]
        == counts["zeroshot_records"]
        == 76291
    )
    gold_summary = read_json(args.gold_dir / "sampling_summary.json")
    payload: dict[str, Any] = {
        "baseline": {
            "total": counts["zeroshot_records"],
            "final_car_to_car": counts["final_car_to_car"],
            "final_not_car_to_car": counts["final_not_car_to_car"],
            "final_unresolved": counts["final_unresolved"],
            "invariant_2902_plus_72653_plus_736": baseline_invariant,
        },
        "gold_kit": {
            "total": gold_summary["total"],
            "seed": gold_summary["seed"],
            "tasks_sha256": gold_summary["tasks_sha256"],
            "human_labels_complete": gold_summary["human_labels_complete"],
            "metrics_generated": (args.gold_dir / "metrics.json").exists(),
        },
        "evidence_graph": {
            **graph_counts,
            "span_error_count": len(span_errors),
            "span_errors": span_errors,
        },
        "scenario_contract": {
            **contract_counts,
            "tier_c_auto_promotion_count": len(tier_c_promotions),
            "tier_c_auto_promotions": tier_c_promotions,
        },
        "dry_run": {
            **bundle_counts,
            "checksum_error_count": len(checksum_errors),
            "checksum_errors": checksum_errors,
            "execution_state_error_count": len(execution_errors),
            "execution_state_errors": execution_errors,
        },
    }
    payload["acceptance"] = {
        "baseline_invariant": baseline_invariant,
        "graph_400_crash_zero_and_schema_pass": graph_counts["total"] == 400,
        "observed_evidence_span_100_percent": (
            graph_counts["exact_evidence_span"] == graph_counts["total"]
        ),
        "contract_400_schema_pass": contract_counts["total"] == 400,
        "tier_c_not_auto_promoted": not tier_c_promotions,
        "dry_run_200_bundled": bundle_counts["total"] == 200,
        "dry_run_200_not_executed": bundle_counts["strict_not_executed"] == 200,
        "bundle_checksums_valid": not checksum_errors,
        "gold_metrics_withheld_until_labels": not (args.gold_dir / "metrics.json").exists(),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(payload["acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
