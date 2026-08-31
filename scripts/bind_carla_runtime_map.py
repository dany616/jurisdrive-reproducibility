#!/usr/bin/env python3
"""Bind defaulted CARLA map fields to an already-running server map without load_world."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jurisdrive.experiments import load_preregistration, write_experiment_plan  # noqa: E402
from jurisdrive.io import read_json, sha256_file, write_json  # noqa: E402
from jurisdrive.models import ScenarioContractV1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--runtime-map", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_config = args.frozen_config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite runtime map binding: {output_dir}")
    contracts_dir = output_dir / "contracts"
    contracts_dir.mkdir(parents=True)
    config = read_json(source_config)
    audit = []
    for case in config["cases"]:
        source_contract = Path(case["contract_path"])
        payload = read_json(source_contract)
        map_field = payload["map_binding"]["carla_map"]
        if map_field.get("provenance") != "defaulted":
            raise ValueError(
                f"{case['slot_id']}: observed/inferred map binding cannot be changed"
            )
        original_map = str(map_field.get("value"))
        map_field["value"] = args.runtime_map
        map_field["provenance"] = "defaulted"
        map_field["confidence"] = min(float(map_field.get("confidence", 0.5)), 0.5)
        contract = ScenarioContractV1.model_validate(payload)
        target = contracts_dir / f"{contract.scenario_id}.json"
        write_json(target, contract)
        case["contract_path"] = str(target)
        case["notes"] = (
            f"runtime map fallback {original_map}->{args.runtime_map}; "
            "defaulted map field only; load_world disabled"
        )
        audit.append(
            {
                "slot_id": case["slot_id"],
                "scenario_id": contract.scenario_id,
                "original_contract": str(source_contract),
                "original_contract_sha256": sha256_file(source_contract),
                "runtime_contract": str(target),
                "runtime_contract_sha256": sha256_file(target),
                "original_map": original_map,
                "runtime_map": args.runtime_map,
                "edited_provenance": "defaulted",
                "observed_fields_changed": False,
            }
        )
    config.setdefault("notes", []).append(
        f"Runtime environment binding: all defaulted map fields use {args.runtime_map}; load_world is prohibited."
    )
    runtime_config = output_dir / "carla_assurance_24_runtime_map.json"
    write_json(runtime_config, config)
    load_preregistration(runtime_config)
    write_json(output_dir / "runtime_map_audit.json", audit)
    plan = write_experiment_plan(runtime_config, output_dir / "experiment_plan")
    manifest = {
        "version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "parent_frozen_config": {"path": str(source_config), "sha256": sha256_file(source_config)},
        "runtime_config": {"path": str(runtime_config), "sha256": sha256_file(runtime_config)},
        "runtime_map": args.runtime_map,
        "cases_bound": len(audit),
        "observed_fields_changed": False,
        "load_world_allowed": False,
        "plan": plan,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
