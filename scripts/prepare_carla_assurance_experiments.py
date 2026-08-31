#!/usr/bin/env python3
"""Prepare and summarize preregistered RQ3/RQ4 experiments safely."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jurisdrive.experiments import (
    FAULT_DEFINITIONS,
    load_preregistration,
    materialize_fault_bundle,
    read_jsonl,
    summarize_assurance_records,
    summarize_fidelity_records,
    write_experiment_plan,
    write_summary_tables,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a 24-case preregistration")
    validate.add_argument("--config", type=Path, required=True)

    plan = commands.add_parser("plan", help="write the 96-run and 168-trial plans")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--allow-pending", action="store_true")

    inject = commands.add_parser("inject", help="materialize one copied fault bundle")
    inject.add_argument("--source-bundle", type=Path, required=True)
    inject.add_argument("--output-dir", type=Path, required=True)
    inject.add_argument("--fault-type", choices=tuple(FAULT_DEFINITIONS), required=True)
    inject.add_argument("--donor-bundle", type=Path)
    inject.add_argument("--variant", choices=("speed", "pose"))

    for name in ("fidelity", "assurance"):
        summary = commands.add_parser(f"summarize-{name}")
        summary.add_argument("--records", type=Path, required=True)
        summary.add_argument("--output-dir", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "validate":
        config = load_preregistration(args.config)
        result = {
            "valid": True,
            "experiment_id": config.experiment_id,
            "selection_frozen": config.selection_frozen,
            "selected_cases": sum(case.candidate_id is not None for case in config.cases),
            "total_slots": len(config.cases),
        }
    elif args.command == "plan":
        result = write_experiment_plan(
            args.config, args.output_dir, allow_pending=args.allow_pending
        )
    elif args.command == "inject":
        result = materialize_fault_bundle(
            args.source_bundle,
            args.output_dir,
            args.fault_type,
            donor_bundle=args.donor_bundle,
            variant=args.variant,
        )
    else:
        rows = read_jsonl(args.records)
        if args.command == "summarize-fidelity":
            result = summarize_fidelity_records(rows)
            name = "fidelity"
        else:
            result = summarize_assurance_records(rows)
            name = "assurance"
        write_summary_tables(args.output_dir, result, name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
