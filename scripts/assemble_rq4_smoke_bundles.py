#!/usr/bin/env python3
"""Assemble immutable RQ4 smoke bundles from executed runs and evolved N4 contracts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jurisdrive.io import read_json, sha256_file, write_json  # noqa: E402
from jurisdrive.models import ScenarioContractV1, SimulationResultV1  # noqa: E402


def _parse_run(value: str) -> tuple[int, Path]:
    candidate, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("run bundle must be CANDIDATE_ID=PATH")
    return int(candidate), Path(path)


def _checksums(root: Path) -> None:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    lines = [
        f"{sha256_file(path)}  {str(path.relative_to(root)).replace(chr(92), '/')}"
        for path in paths
        if path.name != "checksums.sha256"
    ]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--run-bundle", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite bundle root: {output_dir}")
    output_dir.mkdir(parents=True)
    rows = []
    for candidate_id, run_bundle_value in args.run_bundle:
        run_bundle = run_bundle_value.resolve()
        old_contract_path = run_bundle / "contract.json"
        result_path = run_bundle / "simulation_result.json"
        new_contract_path = args.contract_root.resolve() / f"jurisdrive_{candidate_id}.json"
        graph_path = args.graph_root.resolve() / f"jurisdrive_{candidate_id}.json"
        old_contract = ScenarioContractV1.model_validate(read_json(old_contract_path))
        new_contract = ScenarioContractV1.model_validate(read_json(new_contract_path))
        result = SimulationResultV1.model_validate(read_json(result_path))
        if not result.executed or result.status.value != "passed":
            raise ValueError(f"candidate {candidate_id}: clean CARLA result is not passed")
        if old_contract.scenario_id != new_contract.scenario_id or result.scenario_id != new_contract.scenario_id:
            raise ValueError(f"candidate {candidate_id}: scenario ID drift")
        old_pairs = [
            (row.actor_id, row.target_id, row.required)
            for row in old_contract.collision_constraints
        ]
        new_pairs = [
            (row.actor_id, row.target_id, row.required)
            for row in new_contract.collision_constraints
        ]
        if old_pairs != new_pairs:
            raise ValueError(f"candidate {candidate_id}: collision oracle drift")
        if old_contract.topology.value != new_contract.topology.value:
            raise ValueError(f"candidate {candidate_id}: topology drift")
        if old_contract.map_binding.carla_map.value != new_contract.map_binding.carla_map.value:
            raise ValueError(f"candidate {candidate_id}: executed map drift")

        destination = output_dir / new_contract.scenario_id
        shutil.copytree(run_bundle, destination)
        shutil.copy2(old_contract_path, destination / "executed_contract.json")
        shutil.copy2(new_contract_path, destination / "contract.json")
        shutil.copy2(graph_path, destination / "evidence_graph.json")
        provenance = {
            "version": "1.0",
            "candidate_id": candidate_id,
            "scenario_id": new_contract.scenario_id,
            "assembled_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "clean_run_bundle": str(run_bundle),
            "executed_contract_sha256": sha256_file(old_contract_path),
            "event_enriched_contract_sha256": sha256_file(new_contract_path),
            "simulation_result_sha256": sha256_file(result_path),
            "collision_oracle_unchanged": True,
            "topology_and_map_unchanged": True,
            "new_information": "observed pre-collision event sequence only",
            "claim_boundary": (
                "The CARLA result was not rerun after event-sequence enrichment; only immutable "
                "source-grounded event descriptions were added, while physical fields and collision oracle stayed fixed."
            ),
        }
        write_json(destination / "assembly_provenance.json", provenance)
        _checksums(destination)
        rows.append(provenance)
    write_json(
        output_dir / "manifest.json",
        {
            "version": "1.0",
            "bundle_count": len(rows),
            "bundles": rows,
            "scope": "RQ4 exploratory smoke only; not the preregistered 24-case denominator",
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "bundle_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
