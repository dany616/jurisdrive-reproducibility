#!/usr/bin/env python3
"""Run the publication-scale deterministic and VLM observations for RQ4.

The script is resumable and never counts ``manual_review`` as a detected fault.
It evaluates the same 24 clean controls and 144 verified fault artifacts with
the deterministic telemetry baseline, image-only VLM, and telemetry+VLM.  The
guarded and self-refinement policies are derived later from these frozen
observations so the two policies receive identical multimodal evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jurisdrive.assurance import VlmEvaluator  # noqa: E402
from jurisdrive.experiments import read_jsonl  # noqa: E402
from jurisdrive.io import read_json, sha256_file, write_json, write_jsonl  # noqa: E402
from jurisdrive.models import EvaluationReport, ScenarioContractV1, SimulationResultV1  # noqa: E402


VLM_METHODS = (
    "image_only_vlm",
    "telemetry_plus_vlm_no_repair",
    "unconstrained_self_refinement",
)


def _failed_constraints(result: SimulationResultV1) -> list[str]:
    return [row.name for row in result.constraint_results if row.passed is False]


def _bundle_keyframe_hashes(bundle: Path, result: SimulationResultV1) -> list[str]:
    hashes: list[str] = []
    for value in result.keyframes or []:
        path = Path(value)
        if not path.is_absolute():
            path = bundle / path
        hashes.append(sha256_file(path))
    return hashes


def load_trials(materialization_records: Path, mutable_records: Path) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in read_jsonl(materialization_records)]
    mutable = {row["trial_id"]: dict(row) for row in read_jsonl(mutable_records)}
    if len(materialized) != 168:
        raise ValueError(f"expected 168 materialized trials, found {len(materialized)}")
    if len(mutable) != 72:
        raise ValueError(f"expected 72 mutable reruns, found {len(mutable)}")
    trials: list[dict[str, Any]] = []
    for row in materialized:
        contract_bundle = Path(row["bundle_path"])
        result_bundle = contract_bundle
        injection_verified = bool(row.get("injection_verified"))
        if row.get("fault_class") == "mutable":
            rerun = mutable.get(row["trial_id"])
            if not rerun:
                raise ValueError(f"missing mutable rerun: {row['trial_id']}")
            result_bundle = Path(rerun["rerun_bundle_path"])
            injection_verified = bool(rerun.get("injection_verified"))
        if not injection_verified:
            raise ValueError(f"unverified injected phenotype: {row['trial_id']}")
        contract = ScenarioContractV1.model_validate(read_json(contract_bundle / "contract.json"))
        result = SimulationResultV1.model_validate(read_json(result_bundle / "simulation_result.json"))
        clean_bundle = Path(row["clean_bundle_path"])
        clean_contract_path = clean_bundle / "contract.json"
        clean_result = SimulationResultV1.model_validate(read_json(clean_bundle / "simulation_result.json"))
        contract_mismatch = sha256_file(contract_bundle / "contract.json") != sha256_file(clean_contract_path)
        keyframe_mismatch = _bundle_keyframe_hashes(result_bundle, result) != _bundle_keyframe_hashes(
            clean_bundle, clean_result
        )
        trials.append(
            {
                **row,
                "contract_bundle": str(contract_bundle),
                "result_bundle": str(result_bundle),
                "contract": contract,
                "result": result,
                "clean_contract_sha256": sha256_file(clean_contract_path),
                "trial_contract_sha256": sha256_file(contract_bundle / "contract.json"),
                "contract_binding_mismatch": contract_mismatch,
                "keyframe_binding_mismatch": keyframe_mismatch,
                "binding_guard_detected": contract_mismatch or keyframe_mismatch,
                "deterministic_failures": _failed_constraints(result),
                "deterministic_detected": bool(_failed_constraints(result)),
            }
        )
    controls = sum(row["trial_kind"] == "clean_control" for row in trials)
    faults = sum(row["trial_kind"] == "fault" for row in trials)
    if (controls, faults) != (24, 144):
        raise ValueError(f"unexpected denominator controls={controls}, faults={faults}")
    return sorted(trials, key=lambda row: str(row["trial_id"]))


def _raw_report(evaluator: VlmEvaluator) -> dict[str, Any] | None:
    try:
        content = evaluator.last_response["choices"][0]["message"]["content"]  # type: ignore[index]
        return EvaluationReport.model_validate(json.loads(content)).model_dump(mode="json")
    except Exception:
        return None


def evaluate_one(
    trial: dict[str, Any],
    method: str,
    *,
    endpoint: str,
    model: str,
    audit_dir: Path,
    retries: int,
) -> dict[str, Any]:
    evaluation_id = f"{trial['trial_id']}__{method}"
    audit_path = audit_dir / f"{evaluation_id}.json"
    last_error: str | None = None
    started = time.perf_counter()
    for attempt in range(1, retries + 2):
        evaluator = VlmEvaluator(
            endpoint,
            model,
            timeout=240.0,
            bundle_dir=Path(trial["result_bundle"]),
            include_telemetry=method != "image_only_vlm",
            enforce_deterministic=method != "image_only_vlm",
            enforce_provenance_guard=method != "unconstrained_self_refinement",
        )
        try:
            report = evaluator.evaluate(trial["contract"], trial["result"])
            raw_report = _raw_report(evaluator)
            audit = {
                "evaluation_id": evaluation_id,
                "trial_id": trial["trial_id"],
                "method": method,
                "attempt": attempt,
                "request": evaluator.last_request,
                "response": evaluator.last_response,
                "raw_report": raw_report,
                "guarded_report": report.model_dump(mode="json"),
            }
            write_json(audit_path, audit)
            return {
                "evaluation_id": evaluation_id,
                "trial_id": trial["trial_id"],
                "candidate_id": trial["candidate_id"],
                "scenario_id": trial["scenario_id"],
                "topology": trial["topology"],
                "source_stage": trial["source_stage"],
                "trial_kind": trial["trial_kind"],
                "fault_type": trial.get("fault_type"),
                "fault_class": trial.get("fault_class"),
                "method": method,
                "execution_status": "completed",
                "injection_verified": True,
                # Deliberately do not promote abstention/manual review to detection.
                "detected": report.passed is False,
                "passed": report.passed,
                "manual_review": bool(
                    report.manual_review or (raw_report or {}).get("manual_review")
                ),
                "failure_count": len(report.failures),
                "accepted_repair_instructions": [
                    item.model_dump(mode="json") for item in report.repair_instructions
                ],
                "raw_repair_instructions": (raw_report or {}).get("repair_instructions", []),
                "attempts": attempt,
                "wall_seconds": time.perf_counter() - started,
                "audit_path": str(audit_path),
                "audit_sha256": sha256_file(audit_path),
                "error": None,
            }
        except Exception as exc:  # retry transport/schema failures, never fabricate a verdict
            last_error = f"{type(exc).__name__}: {exc}"
            write_json(
                audit_path,
                {
                    "evaluation_id": evaluation_id,
                    "trial_id": trial["trial_id"],
                    "method": method,
                    "attempt": attempt,
                    "request": evaluator.last_request,
                    "response": evaluator.last_response,
                    "error": last_error,
                },
            )
    return {
        "evaluation_id": evaluation_id,
        "trial_id": trial["trial_id"],
        "candidate_id": trial["candidate_id"],
        "scenario_id": trial["scenario_id"],
        "topology": trial["topology"],
        "source_stage": trial["source_stage"],
        "trial_kind": trial["trial_kind"],
        "fault_type": trial.get("fault_type"),
        "fault_class": trial.get("fault_class"),
        "method": method,
        "execution_status": "failed",
        "injection_verified": True,
        "detected": None,
        "passed": None,
        "manual_review": None,
        "failure_count": None,
        "accepted_repair_instructions": [],
        "raw_repair_instructions": [],
        "attempts": retries + 1,
        "wall_seconds": time.perf_counter() - started,
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-records", type=Path, required=True)
    parser.add_argument("--mutable-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite RQ4 VLM evaluation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "vlm_audit"
    audit_dir.mkdir(exist_ok=True)
    records_path = output_dir / "vlm_observations.jsonl"
    trials = load_trials(args.materialization_records.resolve(), args.mutable_records.resolve())
    if args.limit is not None:
        trials = trials[: args.limit]
    previous = {
        row["evaluation_id"]: row
        for row in (read_jsonl(records_path) if records_path.exists() else [])
    }
    jobs = []
    for trial in trials:
        for method in VLM_METHODS:
            evaluation_id = f"{trial['trial_id']}__{method}"
            if previous.get(evaluation_id, {}).get("execution_status") == "completed":
                continue
            jobs.append((trial, method))
    lock = threading.Lock()
    completed_now = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                trial,
                method,
                endpoint=args.endpoint,
                model=args.model,
                audit_dir=audit_dir,
                retries=args.retries,
            ): (trial["trial_id"], method)
            for trial, method in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            with lock:
                previous[row["evaluation_id"]] = row
                ordered = [previous[key] for key in sorted(previous)]
                write_jsonl(records_path, ordered)
                completed_now += 1
                print(
                    f"[{completed_now}/{len(jobs)}] {row['evaluation_id']} "
                    f"status={row['execution_status']} passed={row['passed']} "
                    f"review={row['manual_review']} wall={row['wall_seconds']:.2f}s",
                    flush=True,
                )
    rows = [previous[key] for key in sorted(previous)]
    expected = len(trials) * len(VLM_METHODS)
    completed = sum(row.get("execution_status") == "completed" for row in rows)
    manifest = {
        "version": "1.0",
        "experiment_id": "rq4_publication_vlm_observations",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_trials": len(trials),
        "expected_observations": expected,
        "completed_observations": completed,
        "failed_observations": expected - completed,
        "methods": list(VLM_METHODS),
        "endpoint": args.endpoint,
        "model": args.model,
        "concurrency": args.concurrency,
        "retries": args.retries,
        "detection_rule": "passed is false; manual_review/abstention is reported separately",
        "inputs": {
            "materialization_records": {
                "path": str(args.materialization_records.resolve()),
                "sha256": sha256_file(args.materialization_records.resolve()),
            },
            "mutable_records": {
                "path": str(args.mutable_records.resolve()),
                "sha256": sha256_file(args.mutable_records.resolve()),
            },
        },
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if completed == expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
