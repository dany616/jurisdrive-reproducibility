#!/usr/bin/env python3
"""Prepare the 72 provenance-guarded RQ4 repair contracts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jurisdrive.assurance import apply_bounded_repairs  # noqa: E402
from jurisdrive.experiments import read_jsonl  # noqa: E402
from jurisdrive.io import read_json, sha256_file, write_json, write_jsonl  # noqa: E402
from jurisdrive.models import ScenarioContractV1  # noqa: E402


def _path_value(document: dict[str, Any], path: str) -> Any:
    current: Any = document
    for part in path.split("."):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutable-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite guarded repair plan: {output_dir}")
    output_dir.mkdir(parents=True)
    contracts_dir = output_dir / "contracts"
    contracts_dir.mkdir()
    rows = [dict(row) for row in read_jsonl(args.mutable_records.resolve())]
    if len(rows) != 72 or any(row.get("fault_class") != "mutable" for row in rows):
        raise ValueError("guarded repair input must contain exactly 72 mutable faults")
    repair_rows: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    for row in rows:
        trial_id = row["trial_id"]
        fault_bundle = Path(row["bundle_path"])
        clean_bundle = Path(row["clean_bundle_path"])
        fault_manifest = read_json(fault_bundle / "fault_manifest.json")
        mutation = fault_manifest["mutation"]
        path = str(mutation["path"])
        oracle_value = mutation["oracle_value"]
        fault_contract = ScenarioContractV1.model_validate(read_json(fault_bundle / "contract.json"))
        clean_contract = ScenarioContractV1.model_validate(read_json(clean_bundle / "contract.json"))
        clean_target = _path_value(clean_contract.model_dump(mode="json"), path)
        if not isinstance(clean_target, dict) or clean_target.get("value") != oracle_value:
            blockers.append({"trial_id": trial_id, "reason": "manifest oracle differs from frozen clean contract"})
            continue
        repaired, notes = apply_bounded_repairs(
            fault_contract,
            [{"path": path, "value": oracle_value, "reason": "restore frozen evidence-bound value"}],
            max_repairs=1,
        )
        repaired_data = repaired.model_dump(mode="json")
        exact_clean = repaired_data == clean_contract.model_dump(mode="json")
        if repaired == fault_contract or not any(note.startswith("applied repair") for note in notes):
            blockers.append({"trial_id": trial_id, "reason": f"provenance guard did not apply repair: {notes}"})
            continue
        if not exact_clean:
            blockers.append({"trial_id": trial_id, "reason": "bounded repair does not exactly restore frozen clean contract"})
            continue
        contract_dir = contracts_dir / trial_id
        contract_dir.mkdir()
        contract_path = contract_dir / "contract.json"
        write_json(contract_path, repaired_data)
        repair_rows.append(
            {
                **row,
                "repair_method": "guarded_bounded_repair",
                "repair_triggered": True,
                "repair_iteration": 1,
                "max_repair_iterations": 3,
                "repair_path": path,
                "repair_value": oracle_value,
                "repair_notes": notes,
                "fault_contract_sha256": sha256_file(fault_bundle / "contract.json"),
                "clean_contract_sha256": sha256_file(clean_bundle / "contract.json"),
                "repaired_contract_path": str(contract_path),
                "repaired_contract_sha256": sha256_file(contract_path),
                "exact_frozen_contract_restoration": True,
                "execution_status": "prepared",
                "post_repair_passed": None,
            }
        )
    records_path = output_dir / "guarded_repair_plan.jsonl"
    write_jsonl(records_path, repair_rows)
    manifest = {
        "version": "1.0",
        "experiment_id": "rq4_guarded_bounded_repair72",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_mutable_faults": len(rows),
        "prepared_repairs": len(repair_rows),
        "blockers": blockers,
        "max_repair_iterations": 3,
        "actual_attribute_repairs_per_trial": 1,
        "guard_rule": "only inferred/defaulted value wrappers; exact frozen clean restoration required",
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "input": {"path": str(args.mutable_records.resolve()), "sha256": sha256_file(args.mutable_records.resolve())},
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if len(repair_rows) == 72 and not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
