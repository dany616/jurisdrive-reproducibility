#!/usr/bin/env python3
"""Freeze one successful fidelity bundle per case as the 24 RQ4 clean controls."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jurisdrive.experiments import load_preregistration, read_jsonl, write_experiment_plan  # noqa: E402
from jurisdrive.io import read_json, sha256_file, write_json  # noqa: E402


def checksums(bundle: Path) -> None:
    rows = []
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        rows.append(
            f"{sha256_file(path)}  {str(path.relative_to(bundle)).replace(chr(92), '/')}"
        )
    (bundle / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--fidelity-records", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite clean controls: {output_dir}")
    controls_dir = output_dir / "clean_controls"
    controls_dir.mkdir(parents=True)
    config = read_json(args.runtime_config.resolve())
    records = read_jsonl(args.fidelity_records.resolve())
    selected = {
        row["slot_id"]: row
        for row in records
        if row.get("seed_index") == 1
        and row.get("repeat_index") == 1
        and row.get("execution_status") == "completed"
        and row.get("hard_constraint_pass") is True
    }
    if len(selected) != 24:
        raise ValueError(f"expected 24 clean s1/r1 controls, found {len(selected)}")

    audit = []
    graph_dir = args.graph_dir.resolve()
    for case in config["cases"]:
        row = selected[case["slot_id"]]
        source = Path(row["bundle_path"])
        target = controls_dir / case["scenario_id"]
        shutil.copytree(source, target)
        graph = graph_dir / f"{case['scenario_id']}.json"
        if not graph.is_file():
            raise FileNotFoundError(f"evidence graph unavailable: {graph}")
        shutil.copy2(graph, target / "evidence_graph.json")
        provenance = {
            "version": "1.0",
            "trial_kind": "clean_control",
            "selection_rule": "frozen fidelity seed_index=1 repeat_index=1",
            "slot_id": case["slot_id"],
            "scenario_id": case["scenario_id"],
            "source_bundle": str(source),
            "source_run_record_sha256": row["run_record_sha256"],
            "source_telemetry_sha256": row["telemetry_sha256"],
            "hard_constraint_pass": True,
        }
        write_json(target / "assembly_provenance.json", provenance)
        checksums(target)
        case["clean_bundle_path"] = str(target)
        audit.append(
            {
                **provenance,
                "clean_bundle": str(target),
                "contract_sha256": sha256_file(target / "contract.json"),
                "simulation_result_sha256": sha256_file(target / "simulation_result.json"),
                "checksums_sha256": sha256_file(target / "checksums.sha256"),
            }
        )

    config.setdefault("notes", []).append(
        "RQ4 clean controls are the preregistered seed-1/repeat-1 successful fidelity bundles."
    )
    config_path = output_dir / "carla_assurance_24_with_clean_controls.json"
    write_json(config_path, config)
    load_preregistration(config_path)
    write_json(output_dir / "clean_control_audit.json", audit)
    plan = write_experiment_plan(config_path, output_dir / "experiment_plan")
    manifest = {
        "version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime_config": {"path": str(args.runtime_config.resolve()), "sha256": sha256_file(args.runtime_config.resolve())},
        "fidelity_records": {"path": str(args.fidelity_records.resolve()), "sha256": sha256_file(args.fidelity_records.resolve())},
        "clean_controls": 24,
        "all_hard_constraint_pass": True,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "plan": plan,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
