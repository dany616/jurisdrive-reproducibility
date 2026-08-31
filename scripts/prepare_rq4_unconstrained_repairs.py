#!/usr/bin/env python3
"""Materialize schema-valid unconstrained self-refinement proposals for CARLA rerun."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jurisdrive.assurance import _field_is_mutable  # noqa: E402
from jurisdrive.experiments import read_jsonl  # noqa: E402
from jurisdrive.io import read_json, sha256_file, write_json, write_jsonl  # noqa: E402
from jurisdrive.models import ScenarioContractV1  # noqa: E402


def _assign(document: Any, path: str, value: Any) -> tuple[bool, str]:
    parts = path.split(".")
    current = document
    try:
        for part in parts[:-1]:
            current = current[int(part)] if isinstance(current, list) else current[part]
        leaf = parts[-1]
        if isinstance(current, list):
            index = int(leaf)
            if index < 0 or index >= len(current):
                return False, "list index out of range"
            target = current[index]
            if isinstance(target, dict) and "value" in target and not isinstance(value, dict):
                target["value"] = value
            else:
                current[index] = value
        elif isinstance(current, dict) and leaf in current:
            target = current[leaf]
            if isinstance(target, dict) and "value" in target and not isinstance(value, dict):
                target["value"] = value
            else:
                current[leaf] = value
        else:
            return False, "path does not exist"
        return True, "assigned"
    except (KeyError, IndexError, TypeError, ValueError):
        return False, "path does not exist"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-records", type=Path, required=True)
    parser.add_argument("--vlm-observations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite unconstrained repair plan: {output_dir}")
    output_dir.mkdir(parents=True)
    contracts_dir = output_dir / "contracts"
    contracts_dir.mkdir()
    base_rows = {row["trial_id"]: dict(row) for row in read_jsonl(args.materialization_records.resolve())}
    observations = [
        dict(row)
        for row in read_jsonl(args.vlm_observations.resolve())
        if row.get("method") == "unconstrained_self_refinement"
    ]
    if len(base_rows) != 168 or len(observations) != 168:
        raise ValueError(f"expected 168 base and unconstrained rows; got {len(base_rows)}, {len(observations)}")
    rows: list[dict[str, Any]] = []
    for observation in sorted(observations, key=lambda row: row["trial_id"]):
        base = base_rows[observation["trial_id"]]
        bundle = Path(base["bundle_path"])
        contract = ScenarioContractV1.model_validate(read_json(bundle / "contract.json"))
        original = contract.model_dump(mode="json")
        updated = copy.deepcopy(original)
        instructions = list(observation.get("raw_repair_instructions") or [])[:3]
        attempts: list[dict[str, Any]] = []
        applied = 0
        unsafe = 0
        for instruction in instructions:
            path = str(instruction.get("path") or "")
            mutable = path not in contract.immutable_paths and _field_is_mutable(original, path)
            success, note = _assign(updated, path, instruction.get("value"))
            attempts.append(
                {
                    "path": path,
                    "value": instruction.get("value"),
                    "reason": instruction.get("reason"),
                    "normally_mutable": mutable,
                    "unsafe_immutable_edit": not mutable,
                    "assignment_applied": success,
                    "note": note,
                }
            )
            if success:
                applied += 1
                unsafe += int(not mutable)
        repaired_contract: ScenarioContractV1 | None = None
        validation_error: str | None = None
        if applied:
            try:
                repaired_contract = ScenarioContractV1.model_validate(updated)
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
        contract_path: Path | None = None
        if repaired_contract is not None and repaired_contract != contract:
            contract_dir = contracts_dir / observation["trial_id"]
            contract_dir.mkdir()
            contract_path = contract_dir / "contract.json"
            write_json(contract_path, repaired_contract.model_dump(mode="json"))
        rows.append(
            {
                **base,
                "method": "unconstrained_self_refinement",
                "vlm_evaluation_id": observation["evaluation_id"],
                "detected": observation.get("detected"),
                "manual_review": observation.get("manual_review"),
                "repair_instruction_count": len(instructions),
                "repair_attempts": attempts,
                "repair_edits_applied": applied,
                "unsafe_immutable_edits_applied": unsafe,
                "schema_validation_error": validation_error,
                "repair_triggered": bool(instructions),
                "prepared_for_carla": contract_path is not None,
                "unconstrained_contract_path": str(contract_path) if contract_path else None,
                "unconstrained_contract_sha256": sha256_file(contract_path) if contract_path else None,
                "execution_status": "prepared" if contract_path else "no_executable_repair",
                "post_repair_passed": None,
                "post_repair_regression": None,
            }
        )
    records_path = output_dir / "unconstrained_repair_plan.jsonl"
    write_jsonl(records_path, rows)
    manifest = {
        "version": "1.0",
        "experiment_id": "rq4_unconstrained_self_refinement",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_trials": len(rows),
        "detected": sum(bool(row["detected"]) for row in rows),
        "repair_triggered": sum(bool(row["repair_triggered"]) for row in rows),
        "schema_valid_changed_contracts": sum(bool(row["prepared_for_carla"]) for row in rows),
        "trials_with_unsafe_immutable_edits": sum(row["unsafe_immutable_edits_applied"] > 0 for row in rows),
        "total_unsafe_immutable_edits": sum(row["unsafe_immutable_edits_applied"] for row in rows),
        "schema_rollbacks_or_invalid_paths": sum(
            bool(row["repair_triggered"]) and not bool(row["prepared_for_carla"]) for row in rows
        ),
        "max_repairs_per_trial": 3,
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "inputs": {
            "materialization": sha256_file(args.materialization_records.resolve()),
            "vlm_observations": sha256_file(args.vlm_observations.resolve()),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
