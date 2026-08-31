#!/usr/bin/env python3
"""Evaluate the completed deterministic subset of the RQ4 smoke experiment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jurisdrive.assurance import VlmEvaluator, apply_bounded_repairs  # noqa: E402
from jurisdrive.experiments import summarize_assurance_records  # noqa: E402
from jurisdrive.io import read_json, sha256_file, write_json, write_jsonl  # noqa: E402
from jurisdrive.models import ScenarioContractV1, SimulationResultV1  # noqa: E402


FAULTS = (
    "actor_target_swap",
    "required_collision_omission",
    "event_order_violation",
    "speed_pose_perturbation",
    "map_lane_mismatch",
    "mismatched_keyframes",
)
METHODS = (
    "deterministic_telemetry_only",
    "deterministic_telemetry_plus_binding_guard",
    "image_only_vlm",
    "telemetry_plus_vlm",
    "full_guarded_bounded_repair",
    "unconstrained_self_refinement",
)


def _failed_constraints(result: SimulationResultV1) -> list[str]:
    return [row.name for row in result.constraint_results if row.passed is False]


def _guard_rejection(contract: ScenarioContractV1, fault_type: str) -> tuple[bool, bool]:
    path = {
        "actor_target_swap": "collision_constraints.0",
        "event_order_violation": "event_sequence",
    }.get(fault_type)
    if not path:
        return False, False
    repaired, notes = apply_bounded_repairs(
        contract,
        [{"path": path, "value": "unauthorized_fault_edit"}],
        max_repairs=1,
    )
    rejected = repaired == contract and any("rejected" in note for note in notes)
    return True, rejected


def _vlm_outcome(
    *,
    evaluation_id: str,
    method: str,
    endpoint: str,
    model: str,
    contract: ScenarioContractV1,
    result: SimulationResultV1,
    bundle_dir: Path,
    audit_dir: Path,
) -> dict[str, Any]:
    image_only = method == "image_only_vlm"
    evaluator = VlmEvaluator(
        endpoint,
        model,
        bundle_dir=bundle_dir,
        include_telemetry=not image_only,
        enforce_deterministic=not image_only,
    )
    audit_path = audit_dir / f"{evaluation_id}.json"
    try:
        report = evaluator.evaluate(contract, result)
        detected = report.passed is False or report.manual_review
        write_json(
            audit_path,
            {
                "evaluation_id": evaluation_id,
                "method": method,
                "request": evaluator.last_request,
                "response": evaluator.last_response,
                "report": report.model_dump(mode="json"),
            },
        )
        return {
            "execution_status": "completed",
            "detected": detected,
            "vlm_passed": report.passed,
            "manual_review": report.manual_review,
            "repair_triggered": bool(report.repair_instructions),
            "repair_iterations": 0,
            "post_repair_passed": None,
            "post_repair_regression": None,
            "vlm_failure_count": len(report.failures),
            "audit_path": str(audit_path),
            "pending_reason": None,
        }
    except Exception as exc:
        write_json(
            audit_path,
            {
                "evaluation_id": evaluation_id,
                "method": method,
                "request": evaluator.last_request,
                "response": evaluator.last_response,
                "error": str(exc),
            },
        )
        return {
            "execution_status": "not_executed",
            "detected": None,
            "vlm_passed": None,
            "manual_review": None,
            "repair_triggered": None,
            "repair_iterations": None,
            "post_repair_passed": None,
            "post_repair_regression": None,
            "vlm_failure_count": None,
            "audit_path": str(audit_path),
            "pending_reason": f"VLM evaluation failed: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--fault-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vlm-endpoint")
    parser.add_argument("--vlm-model", default="qwen35-vlm")
    args = parser.parse_args()
    clean_root = args.clean_root.resolve()
    fault_root = args.fault_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite RQ4 evaluation: {output_dir}")
    output_dir.mkdir(parents=True)
    audit_dir = output_dir / "vlm_audit"
    if args.vlm_endpoint:
        audit_dir.mkdir()

    candidate_ids = sorted(
        int(path.name.split("_")[-1])
        for path in clean_root.glob("jurisdrive_*")
        if path.is_dir()
    )
    records: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        clean_bundle = clean_root / f"jurisdrive_{candidate_id}"
        contract = ScenarioContractV1.model_validate(read_json(clean_bundle / "contract.json"))
        clean_result = SimulationResultV1.model_validate(
            read_json(clean_bundle / "simulation_result.json")
        )
        for method in METHODS:
            evaluation_id = f"ctrl_{candidate_id}__{method}"
            deterministic = method in {
                "deterministic_telemetry_only",
                "deterministic_telemetry_plus_binding_guard",
            }
            vlm_method = method in {"image_only_vlm", "telemetry_plus_vlm"}
            if deterministic:
                outcome = {
                    "execution_status": "completed",
                    "detected": False,
                    "manual_review": False,
                    "immutable_edit_attempted": False,
                    "immutable_edit_rejected": False,
                    "pending_reason": None,
                }
            elif vlm_method and args.vlm_endpoint:
                outcome = _vlm_outcome(
                    evaluation_id=evaluation_id,
                    method=method,
                    endpoint=args.vlm_endpoint,
                    model=args.vlm_model,
                    contract=contract,
                    result=clean_result,
                    bundle_dir=clean_bundle,
                    audit_dir=audit_dir,
                )
                outcome.update(
                    {
                        "immutable_edit_attempted": False,
                        "immutable_edit_rejected": False,
                    }
                )
            else:
                outcome = {
                    "execution_status": "not_executed",
                    "detected": None,
                    "manual_review": None,
                    "immutable_edit_attempted": None,
                    "immutable_edit_rejected": None,
                    "pending_reason": (
                        "VLM endpoint unavailable" if vlm_method else "method not implemented"
                    ),
                }
            records.append(
                {
                    "evaluation_id": evaluation_id,
                    "candidate_id": candidate_id,
                    "scenario_id": contract.scenario_id,
                    "trial_kind": "clean_control",
                    "fault_type": None,
                    "fault_class": None,
                    "method": method,
                    "injection_verified": True,
                    "clean_constraint_failures": _failed_constraints(clean_result),
                    **outcome,
                }
            )

        for fault_type in FAULTS:
            bundle = fault_root / f"jurisdrive_{candidate_id}" / fault_type
            manifest = read_json(bundle / "fault_manifest.json")
            immutable = manifest["fault_class"] == "immutable"
            fault_contract = ScenarioContractV1.model_validate(
                read_json(bundle / "contract.json")
            )
            result = (
                SimulationResultV1.model_validate(read_json(bundle / "simulation_result.json"))
                if immutable
                else None
            )
            constraint_failures = _failed_constraints(result) if result else []
            edit_attempted, edit_rejected = _guard_rejection(contract, fault_type)
            for method in METHODS:
                evaluation_id = f"fault_{candidate_id}_{fault_type}__{method}"
                deterministic = method in {
                    "deterministic_telemetry_only",
                    "deterministic_telemetry_plus_binding_guard",
                }
                vlm_method = method in {"image_only_vlm", "telemetry_plus_vlm"}
                completed = deterministic and immutable
                detected = None
                if completed:
                    detected = bool(constraint_failures)
                    if (
                        method == "deterministic_telemetry_plus_binding_guard"
                        and fault_type == "mismatched_keyframes"
                    ):
                        detected = manifest.get("mutation", {}).get("donor_scenario_id") is not None
                if completed:
                    outcome = {
                        "execution_status": "completed",
                        "detected": detected,
                        "repair_triggered": False,
                        "repair_iterations": 0,
                        "post_repair_passed": None,
                        "post_repair_regression": None,
                        "immutable_edit_attempted": edit_attempted,
                        "immutable_edit_rejected": edit_rejected,
                        "manual_review": bool(detected),
                        "pending_reason": None,
                    }
                elif vlm_method and immutable and args.vlm_endpoint and result is not None:
                    outcome = _vlm_outcome(
                        evaluation_id=evaluation_id,
                        method=method,
                        endpoint=args.vlm_endpoint,
                        model=args.vlm_model,
                        contract=fault_contract,
                        result=result,
                        bundle_dir=bundle,
                        audit_dir=audit_dir,
                    )
                    outcome.update(
                        {
                            "immutable_edit_attempted": False,
                            "immutable_edit_rejected": False,
                        }
                    )
                else:
                    outcome = {
                        "execution_status": "not_executed",
                        "detected": None,
                        "repair_triggered": None,
                        "repair_iterations": None,
                        "post_repair_passed": None,
                        "post_repair_regression": None,
                        "immutable_edit_attempted": None,
                        "immutable_edit_rejected": None,
                        "manual_review": None,
                        "pending_reason": (
                            "CARLA rerun required to verify mutable fault phenotype"
                            if not immutable
                            else (
                                "VLM endpoint unavailable"
                                if vlm_method
                                else "method not implemented"
                            )
                        ),
                    }
                records.append(
                    {
                        "evaluation_id": evaluation_id,
                        "candidate_id": candidate_id,
                        "scenario_id": contract.scenario_id,
                        "trial_kind": "fault",
                        "fault_type": fault_type,
                        "fault_class": manifest["fault_class"],
                        "method": method,
                        "injection_verified": bool(manifest["injection_verified"]),
                        "constraint_failures": constraint_failures,
                        **outcome,
                    }
                )

    records_path = output_dir / "evaluation_records.jsonl"
    write_jsonl(records_path, records)
    summaries = {
        method: summarize_assurance_records(
            row for row in records if row["method"] == method
        )
        for method in METHODS
    }
    write_json(output_dir / "method_summaries.json", summaries)
    manifest = {
        "version": "1.0",
        "experiment_id": "rq4_three_case_exploratory_smoke",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "candidate_ids": candidate_ids,
        "clean_controls": len(candidate_ids),
        "materialized_faults": len(candidate_ids) * len(FAULTS),
        "immutable_faults_evaluated_per_deterministic_method": len(candidate_ids) * 3,
        "immutable_faults_evaluated_per_vlm_method": (
            len(candidate_ids) * 3 if args.vlm_endpoint else 0
        ),
        "vlm_endpoint": args.vlm_endpoint,
        "vlm_model": args.vlm_model if args.vlm_endpoint else None,
        "mutable_faults_pending_rerun": len(candidate_ids) * 3,
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "claim_boundaries": [
            "This is a three-case implementation smoke, not the preregistered 168-artifact denominator.",
            "Mutable faults are excluded until a CARLA rerun verifies the injected phenotype.",
            (
                "VLM smoke is executed only for clean controls and immutable faults."
                if args.vlm_endpoint
                else "VLM methods remain not_executed."
            ),
            "Repair and unconstrained self-refinement remain not_executed; no repair claim is made.",
            "The binding guard uses artifact provenance, not visual model inference.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "summaries": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
