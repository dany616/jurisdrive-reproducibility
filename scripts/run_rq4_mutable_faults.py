#!/usr/bin/env python3
"""Rerun and phenotype-verify the 72 mutable RQ4 fault contracts in CARLA."""

from __future__ import annotations

import argparse
import json
import re
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


def verify_phenotype(row: dict, rerun_dir: Path) -> tuple[bool, dict]:
    run = read_json(rerun_dir / "run_record.json")
    fault_bundle = Path(row["bundle_path"])
    fault_manifest = read_json(fault_bundle / "fault_manifest.json")
    mutation = fault_manifest["mutation"]
    details = {"fault_type": row["fault_type"], "mutation": mutation}
    if row["fault_type"] == "required_collision_omission":
        verified = run.get("actor_target_correct") is False
        details["actor_target_correct"] = run.get("actor_target_correct")
    elif row["fault_type"] == "map_lane_mismatch":
        verified = run.get("lane_topology_valid") is False
        details["lane_topology_valid"] = run.get("lane_topology_valid")
    elif row["fault_type"] == "speed_pose_perturbation" and row.get("variant") == "pose":
        verified = run.get("lane_topology_valid") is False
        details["lane_topology_valid"] = run.get("lane_topology_valid")
    elif row["fault_type"] == "speed_pose_perturbation":
        contract = read_json(fault_bundle / "contract.json")
        actor_index = int(str(mutation["path"]).split(".")[1])
        actor_id = contract["actors"][actor_index]["id"]
        first = json.loads((rerun_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()[0])
        observed = next(item["speed_mps"] for item in first["actors"] if item["actor_id"] == actor_id)
        oracle = float(mutation["oracle_value"])
        injected = float(mutation["injected_value"])
        verified = abs(observed - injected) < abs(observed - oracle)
        details.update(
            {
                "actor_id": actor_id,
                "oracle_speed_mps": oracle,
                "injected_speed_mps": injected,
                "observed_first_speed_mps": observed,
            }
        )
    else:
        raise ValueError(f"unsupported mutable fault row: {row['trial_id']}")
    details["injection_verified"] = verified
    return verified, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-records", type=Path, required=True)
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
        raise FileExistsError(f"refusing to overwrite mutable reruns: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reruns_dir = output_dir / "reruns"
    reruns_dir.mkdir(exist_ok=True)
    records_path = output_dir / "mutable_rerun_records.jsonl"
    rows = [
        dict(row)
        for row in read_jsonl(args.materialization_records.resolve())
        if row.get("fault_class") == "mutable"
    ]
    if len(rows) != 72:
        raise ValueError(f"expected 72 mutable faults, found {len(rows)}")
    previous = {
        row["trial_id"]: row for row in (read_jsonl(records_path) if records_path.exists() else [])
    }
    rows = [previous.get(row["trial_id"], row) for row in rows]

    for index, row in enumerate(rows, start=1):
        if row.get("execution_status") == "completed" and row.get("injection_verified"):
            continue
        if not server_ready(args.host, args.port):
            row["execution_status"] = "infrastructure_failed"
            row["injection_verified"] = False
            row["failure_reason"] = "CARLA RPC server unavailable"
            write_jsonl(records_path, rows)
            break
        fault_bundle = Path(row["bundle_path"])
        clean_manifest = read_json(Path(row["clean_bundle_path"]) / "run_manifest.json")
        original_map = clean_manifest.get("map_fallback_from")
        rerun_dir = reruns_dir / row["trial_id"]
        attempt = 1
        while rerun_dir.exists():
            attempt += 1
            rerun_dir = reruns_dir / f"{row['trial_id']}_attempt{attempt}"
        command = [
            str(args.python37.resolve()),
            str(args.runner.resolve()),
            "--carla-api", str(args.carla_api.resolve()),
            "--contract", str((fault_bundle / "contract.json").resolve()),
            "--output-dir", str(rerun_dir),
            "--host", args.host,
            "--port", str(args.port),
            "--frame-limit", str(args.frame_limit),
            "--post-collision-frames", "20",
            "--seed", str(20260823 + index),
            "--reuse-only",
        ]
        if original_map:
            command.extend(["--map-fallback-from", str(original_map)])
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
        record_path = rerun_dir / "run_record.json"
        if record_path.is_file():
            verified, details = verify_phenotype(row, rerun_dir)
            run = read_json(record_path)
            row.update(
                {
                    "execution_status": "completed",
                    "injection_verified": verified,
                    "phenotype": details,
                    "rerun_bundle_path": str(rerun_dir),
                    "rerun_record_sha256": sha256_file(record_path),
                    "runner_returncode": returncode,
                    "wall_seconds": time.perf_counter() - started,
                    "observed_hard_constraint_pass": run.get("hard_constraint_pass"),
                    "observed_failure_reason": run.get("failure_reason"),
                }
            )
        else:
            row.update(
                {
                    "execution_status": "runner_failed",
                    "injection_verified": False,
                    "failure_reason": f"runner exit {returncode} without run_record.json",
                    "rerun_bundle_path": str(rerun_dir),
                    "runner_returncode": returncode,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        write_jsonl(records_path, rows)
        print(
            f"[{index}/72] {row['trial_id']} verified={row.get('injection_verified')} "
            f"status={row.get('execution_status')} wall={row['wall_seconds']:.2f}s",
            flush=True,
        )

    manifest = {
        "version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "materialization_records": {"path": str(args.materialization_records.resolve()), "sha256": sha256_file(args.materialization_records.resolve())},
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "planned_mutable_faults": 72,
        "completed_reruns": sum(row.get("execution_status") == "completed" for row in rows),
        "injection_verified": sum(bool(row.get("injection_verified")) for row in rows),
        "verification_failed": [row["trial_id"] for row in rows if not row.get("injection_verified")],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if manifest["injection_verified"] == 72 else 2


if __name__ == "__main__":
    raise SystemExit(main())
