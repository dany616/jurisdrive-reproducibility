#!/usr/bin/env python3
"""Execute the frozen 96-run CARLA fidelity schedule with incremental records."""

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

from jurisdrive.experiments import (  # noqa: E402
    read_jsonl,
    summarize_fidelity_records,
    write_summary_tables,
)
from jurisdrive.io import read_json, sha256_file, write_json, write_jsonl  # noqa: E402


def server_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--python37", type=Path, required=True)
    parser.add_argument("--carla-api", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--frame-limit", type=int, default=400)
    parser.add_argument("--post-collision-frames", type=int, default=20)
    parser.add_argument("--run-timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite fidelity output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    records_path = output_dir / "fidelity_records.jsonl"

    schedule = read_jsonl(args.schedule.resolve())
    if len(schedule) != 96:
        raise ValueError(f"fidelity schedule must contain 96 rows, got {len(schedule)}")
    runtime_config = read_json(args.runtime_config.resolve())
    case_by_slot = {case["slot_id"]: case for case in runtime_config["cases"]}
    previous = {
        row["run_id"]: row for row in (read_jsonl(records_path) if records_path.exists() else [])
    }
    records = []
    for planned in schedule:
        records.append(previous.get(planned["run_id"], dict(planned)))

    started = time.perf_counter()
    for index, row in enumerate(records, start=1):
        if row.get("execution_status") == "completed":
            continue
        if not row.get("ready_for_execution"):
            row["execution_status"] = "blocked"
            row["failure_reason"] = "; ".join(row.get("blockers") or ["not ready"])
            write_jsonl(records_path, records)
            continue
        if not server_ready(args.host, args.port):
            row["execution_status"] = "infrastructure_failed"
            row["failure_reason"] = "CARLA RPC server is unavailable"
            write_jsonl(records_path, records)
            print(f"[{index}/96] {row['run_id']} infrastructure_failed", flush=True)
            break

        case = case_by_slot[row["slot_id"]]
        note = str(case.get("notes") or "")
        match = re.search(r"runtime map fallback ([^\-]+)->", note)
        original_map = match.group(1) if match else None
        attempt = 1
        run_dir = runs_dir / row["run_id"]
        while run_dir.exists():
            attempt += 1
            run_dir = runs_dir / f"{row['run_id']}_attempt{attempt}"
        command = [
            str(args.python37.resolve()),
            str(args.runner.resolve()),
            "--carla-api",
            str(args.carla_api.resolve()),
            "--contract",
            str(Path(row["contract_path"]).resolve()),
            "--output-dir",
            str(run_dir),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--frame-limit",
            str(args.frame_limit),
            "--post-collision-frames",
            str(args.post_collision_frames),
            "--seed",
            str(row["seed"]),
            "--reuse-only",
        ]
        if original_map:
            command.extend(["--map-fallback-from", original_map])
        run_started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.run_timeout,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = (error.stderr or "") + "\nrun timeout"
            returncode = 124
        (output_dir / f"{row['run_id']}.stdout.log").write_text(stdout, encoding="utf-8")
        (output_dir / f"{row['run_id']}.stderr.log").write_text(stderr, encoding="utf-8")
        wall_seconds = time.perf_counter() - run_started
        record_path = run_dir / "run_record.json"
        if record_path.is_file():
            observed = read_json(record_path)
            row.update(observed)
            row["slot_id"] = case["slot_id"]
            row["seed_index"] = schedule[index - 1]["seed_index"]
            row["repeat_index"] = schedule[index - 1]["repeat_index"]
            row["source_stage"] = case["source_stage"]
            row["bundle_path"] = str(run_dir)
            row["wall_seconds"] = wall_seconds
            row["runner_returncode"] = returncode
            row["run_record_sha256"] = sha256_file(record_path)
        else:
            row["execution_status"] = "runner_failed"
            row["failure_reason"] = f"runner exit {returncode} without run_record.json"
            row["bundle_path"] = str(run_dir)
            row["wall_seconds"] = wall_seconds
            row["runner_returncode"] = returncode
        write_jsonl(records_path, records)
        completed_count = sum(item.get("execution_status") == "completed" for item in records)
        progress = {
            "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "planned_runs": 96,
            "completed_runs": completed_count,
            "hard_pass_runs": sum(item.get("hard_constraint_pass") is True for item in records),
            "failed_or_blocked_runs": sum(
                item.get("execution_status") not in ("completed", "not_executed")
                for item in records
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "last_run_id": row["run_id"],
        }
        write_json(output_dir / "progress.json", progress)
        print(
            f"[{index}/96] {row['run_id']} status={row.get('execution_status')} "
            f"hard={row.get('hard_constraint_pass')} wall={wall_seconds:.2f}s",
            flush=True,
        )

    summary = summarize_fidelity_records(records)
    if summary["completed_runs"] == 96:
        summary_dir = output_dir / "summary"
        if summary_dir.exists():
            raise FileExistsError(f"summary already exists: {summary_dir}")
        write_summary_tables(summary_dir, summary, "fidelity")
    else:
        write_json(output_dir / "interim_summary.json", summary)
    manifest = {
        "version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "schedule": {"path": str(args.schedule.resolve()), "sha256": sha256_file(args.schedule.resolve())},
        "runtime_config": {"path": str(args.runtime_config.resolve()), "sha256": sha256_file(args.runtime_config.resolve())},
        "runner": {"path": str(args.runner.resolve()), "sha256": sha256_file(args.runner.resolve())},
        "records": {"path": str(records_path), "sha256": sha256_file(records_path)},
        "summary": summary,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["completed_runs"] == 96 else 2


if __name__ == "__main__":
    raise SystemExit(main())
