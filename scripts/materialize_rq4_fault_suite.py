#!/usr/bin/env python3
"""Materialize the frozen 24-control/144-fault RQ4 suite incrementally."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jurisdrive.experiments import materialize_fault_bundle, read_jsonl  # noqa: E402
from jurisdrive.io import sha256_file, write_json, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite RQ4 materialization: {output_dir}")
    bundles_dir = output_dir / "fault_bundles"
    bundles_dir.mkdir(parents=True)
    plan = read_jsonl(args.fault_plan.resolve())
    if len(plan) != 168:
        raise ValueError(f"fault plan must contain 168 rows, got {len(plan)}")

    records = []
    for index, row in enumerate(plan, start=1):
        record = dict(row)
        if row["trial_kind"] == "clean_control":
            bundle = Path(row["clean_bundle_path"])
            record.update(
                {
                    "bundle_path": str(bundle),
                    "execution_status": "materialized",
                    "injection_verified": True,
                    "materialization_manifest_sha256": None,
                }
            )
        else:
            bundle = bundles_dir / row["trial_id"]
            manifest = materialize_fault_bundle(
                Path(row["clean_bundle_path"]),
                bundle,
                row["fault_type"],
                donor_bundle=(Path(row["donor_bundle_path"]) if row.get("donor_bundle_path") else None),
                variant=row.get("variant"),
            )
            record.update(
                {
                    "bundle_path": str(bundle),
                    "execution_status": "materialized",
                    "injection_verified": bool(manifest["injection_verified"]),
                    "materialization_manifest_sha256": sha256_file(bundle / "fault_manifest.json"),
                }
            )
        records.append(record)
        if index % 24 == 0:
            write_jsonl(output_dir / "materialization_records.jsonl", records)
            print(f"[{index}/168] materialized", flush=True)
    records_path = output_dir / "materialization_records.jsonl"
    write_jsonl(records_path, records)
    counts = Counter(
        "control" if row["trial_kind"] == "clean_control" else row["fault_class"]
        for row in records
    )
    manifest = {
        "version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "fault_plan": {"path": str(args.fault_plan.resolve()), "sha256": sha256_file(args.fault_plan.resolve())},
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "counts": dict(counts),
        "total_artifacts": len(records),
        "fault_bundles": sum(row["trial_kind"] == "fault" for row in records),
        "injection_verified_at_materialization": sum(bool(row["injection_verified"]) for row in records),
        "mutable_awaiting_carla_rerun": sum(
            row.get("fault_class") == "mutable" and not row["injection_verified"]
            for row in records
        ),
        "claim_boundary": "Mutable faults are not verified until their CARLA reruns complete.",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
