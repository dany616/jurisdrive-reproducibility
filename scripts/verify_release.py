#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COUNTS = {
    "raw_records": 76_291,
    "zeroshot_records": 76_291,
    "rule_car_to_car": 2_471,
    "rule_not_car_to_car": 71_296,
    "routed_to_llm": 2_524,
    "llm_car_to_car": 431,
    "llm_not_car_to_car": 1_357,
    "llm_unresolved": 736,
    "final_car_to_car": 2_902,
    "final_not_car_to_car": 72_653,
    "final_unresolved": 736,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def verify_frozen_summaries() -> dict[str, Any]:
    audit = read_json(REPO_ROOT / "results" / "n0_n3_summary.json")
    counts = audit.get("counts")
    require(counts == EXPECTED_COUNTS, f"Frozen N0-N3 counts changed: {counts}")

    checks = audit.get("integrity_checks")
    require(isinstance(checks, dict) and checks, "N0-N3 integrity checks are missing")
    require(all(checks.values()), f"N0-N3 integrity check failed: {checks}")
    require(
        counts["final_car_to_car"]
        + counts["final_not_car_to_car"]
        + counts["final_unresolved"]
        == counts["raw_records"],
        "Final N0-N3 branch identity failed",
    )

    private_verification = audit.get("private_artifact_verification")
    require(isinstance(private_verification, dict), "Private artifact verification is missing")
    manifest = private_verification.get("candidate_manifest")
    require(isinstance(manifest, dict), "Candidate-manifest verification is missing")
    manifest_rows = manifest.get("rows")
    require(
        manifest_rows == counts["final_car_to_car"],
        f"Candidate manifest rows changed: {manifest_rows}",
    )
    manifest_sha256 = str(manifest.get("sha256") or "")
    require(
        len(manifest_sha256) == 64
        and all(character in "0123456789abcdef" for character in manifest_sha256),
        "Candidate-manifest SHA-256 is invalid",
    )

    validation = read_json(REPO_ROOT / "results" / "n4_n6_validation_summary.json")
    acceptance = validation.get("acceptance")
    require(
        isinstance(acceptance, dict) and all(acceptance.values()),
        f"Frozen N4-N6 acceptance check failed: {acceptance}",
    )
    require(validation["evidence_graph"]["total"] == 400, "Graph denominator changed")
    require(validation["scenario_contract"]["total"] == 400, "Contract denominator changed")
    require(validation["dry_run"]["total"] == 200, "Dry-run denominator changed")

    return {
        "n0_n3": counts,
        "candidate_manifest": {"rows": manifest_rows, "sha256": manifest_sha256},
        "n4_graphs": validation["evidence_graph"]["total"],
        "n5_contracts": validation["scenario_contract"]["total"],
        "dry_runs": validation["dry_run"]["total"],
    }


def verify_synthetic_pipeline(temporary_root: Path) -> dict[str, Any]:
    manifest = REPO_ROOT / "examples" / "sample_manifest.jsonl"
    full_run = REPO_ROOT / "examples" / "full_run"
    graphs = temporary_root / "graphs"
    contracts = temporary_root / "contracts"
    bundles = temporary_root / "bundles"

    common = [
        "--manifest",
        str(manifest),
        "--full-run-dir",
        str(full_run),
        "--limit",
        "1",
    ]
    run(
        [
            sys.executable,
            "-m",
            "jurisdrive",
            "build-graph",
            *common,
            "--output-dir",
            str(graphs),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "jurisdrive",
            "build-contract",
            *common,
            "--graph-dir",
            str(graphs),
            "--output-dir",
            str(contracts),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "jurisdrive",
            "compile",
            "--manifest",
            str(manifest),
            "--limit",
            "1",
            "--graph-dir",
            str(graphs),
            "--contract-dir",
            str(contracts),
            "--output-dir",
            str(bundles),
        ]
    )

    graph_summary = read_json(graphs / "summary.json")
    contract_summary = read_json(contracts / "summary.json")
    bundle_summary = read_json(bundles / "summary.json")
    require(graph_summary.get("schema_valid") == 1, f"Graph smoke failed: {graph_summary}")
    require(graph_summary.get("crashes") == 0, f"Graph smoke crashed: {graph_summary}")
    require(
        contract_summary.get("schema_valid") == 1,
        f"Contract smoke failed: {contract_summary}",
    )
    require(
        contract_summary.get("crashes") == 0,
        f"Contract smoke crashed: {contract_summary}",
    )
    require(bundle_summary.get("bundled") == 1, f"Bundle smoke failed: {bundle_summary}")
    require(bundle_summary.get("crashes") == 0, f"Bundle smoke crashed: {bundle_summary}")
    require(
        bundle_summary.get("unexpected_executed", 0) == 0,
        f"Dry-run unexpectedly executed: {bundle_summary}",
    )
    require(
        bundle_summary.get("unexpected_simulation_metrics", 0) == 0,
        f"Dry-run fabricated metrics: {bundle_summary}",
    )
    return {
        "graph": graph_summary,
        "contract": contract_summary,
        "dry_run": bundle_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the clean-clone JurisDrive package.")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the unit-test subprocess; all other checks still run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary: dict[str, Any] = {"frozen": verify_frozen_summaries()}

    with tempfile.TemporaryDirectory(prefix="jurisdrive_verify_") as temporary:
        temporary_root = Path(temporary)
        compile_env = os.environ.copy()
        compile_env["PYTHONPYCACHEPREFIX"] = str(temporary_root / "pycache")
        run(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "jurisdrive",
                "src",
                "scripts",
                "tests",
            ],
            env=compile_env,
        )
        run([sys.executable, "-m", "jurisdrive", "--help"])
        if not args.skip_tests:
            run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
            summary["tests"] = "pass"
        else:
            summary["tests"] = "skipped"
        summary["synthetic_pipeline"] = verify_synthetic_pipeline(temporary_root)

    summary["status"] = "pass"
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
