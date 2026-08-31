#!/usr/bin/env python3
"""Execute and verify the 72 guarded bounded repairs in CARLA."""

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
        raise FileExistsError(f"refusing to overwrite guarded repair execution: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "repair_runs"
    runs_dir.mkdir(exist_ok=True)
    records_path = output_dir / "guarded_repair_records.jsonl"
    plan = [dict(row) for row in read_jsonl(args.repair_plan.resolve())]
    if len(plan) != 72:
        raise ValueError(f"expected 72 guarded repairs, found {len(plan)}")
    previous = {
        row["trial_id"]: row for row in (read_jsonl(records_path) if records_path.exists() else [])
    }
    rows = [previous.get(row["trial_id"], row) for row in plan]
    for index, row in enumerate(rows, start=1):
        if row.get("execution_status") == "completed" and row.get("post_repair_passed"):
            continue
        if not server_ready(args.host, args.port):
            row.update(
                {
                    "execution_status": "infrastructure_failed",
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
            str(args.python37.resolve()),
            str(args.runner.resolve()),
            "--carla-api", str(args.carla_api.resolve()),
            "--contract", str(Path(row["repaired_contract_path"]).resolve()),
            "--output-dir", str(run_dir),
            "--host", args.host,
            "--port", str(args.port),
            "--frame-limit", str(args.frame_limit),
            "--post-collision-frames", "20",
            "--seed", str(clean_record["seed"]),
            "--reuse-only",
        ]
        if clean_record.get("map_fallback_from"):
            command.extend(["--map-fallback-from", str(clean_record["map_fallback_from"])])
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.run_timeout,
                check=False,
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
            failed_constraints = [
                item["name"] for item in result.get("constraint_results", []) if item.get("passed") is False
            ]
            post_pass = bool(
                run.get("execution_status") == "completed"
                and run.get("hard_constraint_pass") is True
                and run.get("actor_target_correct") is True
                and run.get("lane_topology_valid") is True
                and run.get("event_order_valid") is True
                and not failed_constraints
            )
            clean_result = read_json(clean_bundle / "simulation_result.json")
            clean_passing = {
                item["name"] for item in clean_result.get("constraint_results", []) if item.get("passed") is True
            }
            repaired_passing = {
                item["name"] for item in result.get("constraint_results", []) if item.get("passed") is True
            }
            regressions = sorted(clean_passing - repaired_passing)
            row.update(
                {
                    "execution_status": "completed",
                    "post_repair_passed": post_pass and not regressions,
                    "post_repair_regression": bool(regressions),
                    "regressed_constraints": regressions,
                    "failed_constraints": failed_constraints,
                    "repair_run_bundle_path": str(run_dir),
                    "repair_run_record_sha256": sha256_file(record_path),
                    "repair_result_sha256": sha256_file(result_path),
                    "runner_returncode": returncode,
                    "wall_seconds": time.perf_counter() - started,
                    "failure_reason": run.get("failure_reason"),
                }
            )
        else:
            row.update(
                {
                    "execution_status": "runner_failed",
                    "post_repair_passed": False,
                    "post_repair_regression": None,
                    "repair_run_bundle_path": str(run_dir),
                    "runner_returncode": returncode,
                    "wall_seconds": time.perf_counter() - started,
                    "failure_reason": f"runner exit {returncode} without complete bundle",
                }
            )
        write_jsonl(records_path, rows)
        print(
            f"[{index}/72] {row['trial_id']} post_pass={row.get('post_repair_passed')} "
            f"regression={row.get('post_repair_regression')} wall={row['wall_seconds']:.2f}s",
            flush=True,
        )
    completed_count = sum(row.get("execution_status") == "completed" for row in rows)
    passed_count = sum(bool(row.get("post_repair_passed")) for row in rows)
    regressions = sum(bool(row.get("post_repair_regression")) for row in rows)
    manifest = {
        "version": "1.0",
        "experiment_id": "rq4_guarded_bounded_repair72",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "planned_repairs": 72,
        "completed_repair_runs": completed_count,
        "post_repair_passed": passed_count,
        "post_repair_regressions": regressions,
        "repair_iteration": 1,
        "max_repair_iterations": 3,
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "input": {"path": str(args.repair_plan.resolve()), "sha256": sha256_file(args.repair_plan.resolve())},
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if completed_count == 72 and passed_count == 72 and regressions == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
