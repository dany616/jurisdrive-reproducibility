#!/usr/bin/env python3
"""Execute the schema-valid unconstrained self-refinement contracts in CARLA."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jurisdrive.experiments import read_jsonl  # noqa: E402
from jurisdrive.io import read_json, sha256_file, write_json, write_jsonl  # noqa: E402


def server_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-plan", type=Path, required=True)
    parser.add_argument("--python37", type=Path, required=True)
    parser.add_argument("--carla-api", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--frame-limit", type=int, default=80)
    parser.add_argument("--run-timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite unconstrained repair execution: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "repair_runs"
    runs_dir.mkdir(exist_ok=True)
    records_path = output_dir / "unconstrained_repair_records.jsonl"
    plan = [dict(row) for row in read_jsonl(args.repair_plan.resolve())]
    if len(plan) != 168:
        raise ValueError(f"expected 168 method rows, found {len(plan)}")
    previous = {
        row["trial_id"]: row for row in (read_jsonl(records_path) if records_path.exists() else [])
    }
    rows = [previous.get(row["trial_id"], row) for row in plan]
    executable = [row for row in rows if row.get("prepared_for_carla")]
    if len(executable) != 49:
        raise ValueError(f"expected frozen executable denominator 49, found {len(executable)}")
    position = 0
    for row in rows:
        if not row.get("prepared_for_carla"):
            row["repair_execution_status"] = "not_executable"
            continue
        if row.get("repair_execution_status") == "completed":
            continue
        position += 1
        if not server_ready(args.host, args.port):
            row.update(
                {
                    "repair_execution_status": "infrastructure_failed",
                    "post_repair_passed": False,
                    "failure_reason": "CARLA RPC server unavailable",
                }
            )
            write_jsonl(records_path, rows)
            break
        clean_bundle = Path(row["clean_bundle_path"])
        clean_record = read_json(clean_bundle / "run_record.json")
        run_dir = runs_dir / row["trial_id"]
        attempt = 1
        while run_dir.exists():
            attempt += 1
            run_dir = runs_dir / f"{row['trial_id']}_attempt{attempt}"
        command = [
            str(args.python37.resolve()), str(args.runner.resolve()),
            "--carla-api", str(args.carla_api.resolve()),
            "--contract", str(Path(row["unconstrained_contract_path"]).resolve()),
            "--output-dir", str(run_dir),
            "--host", args.host, "--port", str(args.port),
            "--frame-limit", str(args.frame_limit), "--post-collision-frames", "20",
            "--seed", str(clean_record["seed"]), "--reuse-only",
        ]
        if clean_record.get("map_fallback_from"):
            command.extend(["--map-fallback-from", str(clean_record["map_fallback_from"])])
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=args.run_timeout, check=False
            )
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = (error.stderr or "") + "\nrun timeout"
            returncode = 124
        (output_dir / f"{row['trial_id']}.stdout.log").write_text(stdout, encoding="utf-8")
        (output_dir / f"{row['trial_id']}.stderr.log").write_text(stderr, encoding="utf-8")
        record_path = run_dir / "run_record.json"
        result_path = run_dir / "simulation_result.json"
        if record_path.is_file() and result_path.is_file():
            run = read_json(record_path)
            result = read_json(result_path)
            failed = [item["name"] for item in result.get("constraint_results", []) if item.get("passed") is False]
            post_pass = bool(
                run.get("execution_status") == "completed"
                and run.get("hard_constraint_pass") is True
                and run.get("actor_target_correct") is True
                and run.get("lane_topology_valid") is True
                and run.get("event_order_valid") is True
                and not failed
            )
            clean_result = read_json(clean_bundle / "simulation_result.json")
            clean_passing = {item["name"] for item in clean_result.get("constraint_results", []) if item.get("passed") is True}
            repaired_passing = {item["name"] for item in result.get("constraint_results", []) if item.get("passed") is True}
            regressions = sorted(clean_passing - repaired_passing)
            row.update(
                {
                    "repair_execution_status": "completed",
                    "post_repair_passed": post_pass and not regressions,
                    "post_repair_regression": bool(regressions),
                    "regressed_constraints": regressions,
                    "failed_constraints": failed,
                    "repair_run_bundle_path": str(run_dir),
                    "repair_run_record_sha256": sha256_file(record_path),
                    "runner_returncode": returncode,
                    "repair_wall_seconds": time.perf_counter() - started,
                    "failure_reason": run.get("failure_reason"),
                }
            )
        else:
            precondition_failed = "topology contract requires a collision constraint" in stderr
            row.update(
                {
                    "repair_execution_status": (
                        "precondition_failed" if precondition_failed else "runner_failed"
                    ),
                    "post_repair_passed": False,
                    "post_repair_regression": True if precondition_failed else None,
                    "repair_run_bundle_path": str(run_dir),
                    "runner_returncode": returncode,
                    "repair_wall_seconds": time.perf_counter() - started,
                    "failure_reason": (
                        "unconstrained edit removed the required collision execution precondition"
                        if precondition_failed
                        else f"runner exit {returncode} without complete bundle"
                    ),
                }
            )
        write_jsonl(records_path, rows)
        print(
            f"[{position}/49] {row['trial_id']} post_pass={row.get('post_repair_passed')} "
            f"unsafe_edits={row.get('unsafe_immutable_edits_applied')} "
            f"wall={row['repair_wall_seconds']:.2f}s",
            flush=True,
        )
    write_jsonl(records_path, rows)
    completed = [row for row in rows if row.get("repair_execution_status") == "completed"]
    terminal = [
        row
        for row in rows
        if row.get("repair_execution_status") in {"completed", "precondition_failed"}
    ]
    manifest = {
        "version": "1.0",
        "experiment_id": "rq4_unconstrained_self_refinement",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method_rows": 168,
        "repair_triggered": sum(bool(row.get("repair_triggered")) for row in rows),
        "schema_valid_executable_repairs": len(executable),
        "completed_repair_runs": len(completed),
        "precondition_failed": sum(
            row.get("repair_execution_status") == "precondition_failed" for row in rows
        ),
        "terminal_repair_attempts": len(terminal),
        "post_repair_passed": sum(bool(row.get("post_repair_passed")) for row in completed),
        "post_repair_regressions": sum(bool(row.get("post_repair_regression")) for row in completed),
        "unsafe_edit_trials_executed": sum(
            row.get("unsafe_immutable_edits_applied", 0) > 0 for row in completed
        ),
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "input": {"path": str(args.repair_plan.resolve()), "sha256": sha256_file(args.repair_plan.resolve())},
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(terminal) == len(executable) else 2


if __name__ == "__main__":
    raise SystemExit(main())
