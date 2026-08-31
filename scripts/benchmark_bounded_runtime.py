#!/usr/bin/env python3
"""Measure bounded CARLA and N6 VLM runtime without overwriting prior evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCENARIOS = (25, 71, 99, 367, 460, 692)
LONG_SCENARIOS = {25, 71}
SCENARIO_WORKERS = {
    2: {
        25: (0,),
        71: (1,),
        99: (0,),
        367: (1,),
        460: (0,),
        692: (1,),
    },
    6: {scenario_id: (index,) for index, scenario_id in enumerate(SCENARIOS)},
    12: {
        scenario_id: (index, index + len(SCENARIOS))
        for index, scenario_id in enumerate(SCENARIOS)
    },
}
INPUT_FILES = (
    "contract.json",
    "run_config.yaml",
    "scenario.scenic",
    "telemetry_schema.json",
    "checksums.sha256",
    "dry_run_report.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def append_jsonl(path: Path, payload: Dict[str, Any], lock: Optional[threading.Lock] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def append() -> None:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    if lock is None:
        append()
    else:
        with lock:
            append()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_plan(args: argparse.Namespace) -> int:
    if args.workers not in SCENARIO_WORKERS:
        raise SystemExit("the validated scenario-sharded design requires 2, 6, or 12 workers")
    scenario_workers = SCENARIO_WORKERS[args.workers]
    base_count, remainder = divmod(args.total, len(SCENARIOS))
    requested_counts = {
        scenario_id: base_count + (1 if offset < remainder else 0)
        for offset, scenario_id in enumerate(SCENARIOS)
    }
    rows = []
    run_index = 0
    for scenario_id in SCENARIOS:
        assigned_workers = scenario_workers[scenario_id]
        for replicate_index in range(1, requested_counts[scenario_id] + 1):
            run_index += 1
            rows.append(
                {
                    "run_index": run_index,
                    "scenario_id": scenario_id,
                    "frame_limit": 120 if scenario_id in LONG_SCENARIOS else 80,
                    "worker_index": assigned_workers[
                        (replicate_index - 1) % len(assigned_workers)
                    ],
                    "replicate_index": replicate_index,
                }
            )
    counts = {
        str(scenario_id): sum(row["scenario_id"] == scenario_id for row in rows)
        for scenario_id in SCENARIOS
    }
    payload = {
        "version": "1.0",
        "created_at_utc": utc_now(),
        "design": "balanced repeated-measures bounded CARLA/N6 runtime benchmark",
        "measured_runs": args.total,
        "unmeasured_warmups_per_map_phase": args.warmups_per_map,
        "workers": args.workers,
        "worker_scenario_shards": {
            str(worker_index): [
                scenario_id
                for scenario_id, worker_indices in scenario_workers.items()
                if worker_index in worker_indices
            ]
            for worker_index in range(args.workers)
        },
        "scenario_counts": counts,
        "frame_policy": {"25": 120, "71": 120, "99": 80, "367": 80, "460": 80, "692": 80},
        "fixed_delta_seconds": 0.05,
        "camera_capture": True,
        "execution_profile": "contract_collision",
        "world_policy": (
            "map-sharded workers; one unmeasured warm-up at each map phase, then reuse "
            "only when the already-loaded CARLA map matches the contract"
        ),
        "rows": rows,
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output.resolve()), "scenario_counts": counts}, indent=2))
    return 0


def copy_bundle_inputs(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("refusing to overwrite existing run directory: %s" % destination)
    destination.mkdir(parents=True)
    copied = 0
    for name in INPUT_FILES:
        source_path = source / name
        if source_path.exists():
            shutil.copy2(str(source_path), str(destination / name))
            copied += 1
    if not (destination / "contract.json").is_file():
        raise FileNotFoundError("source bundle has no contract.json: %s" % source)
    if copied == 0:
        raise FileNotFoundError("no bundle inputs copied from %s" % source)


def parse_result_log(logs: Iterable[str]) -> Dict[str, str]:
    values = {}
    for item in logs:
        if "=" in item:
            key, value = item.split("=", 1)
            values[key] = value
    return values


def execute_carla_once(
    source_bundle_root: Path,
    output_dir: Path,
    scenario_id: int,
    frame_limit: int,
    host: str,
    port: int,
    reuse_world: bool,
    measured: bool,
    run_index: int,
    worker_index: int,
) -> Dict[str, Any]:
    from jurisdrive.io import write_json as write_model_json
    from jurisdrive.models import ScenarioContractV1
    from jurisdrive.simulator import CarlaBackend

    source_bundle = source_bundle_root / ("jurisdrive_%s" % scenario_id)
    copy_bundle_inputs(source_bundle, output_dir)
    contract = ScenarioContractV1.model_validate(read_json(output_dir / "contract.json"))
    os.environ["JURISDRIVE_EXECUTION_PROFILE"] = "contract_collision"
    os.environ["JURISDRIVE_CAPTURE_CAMERA"] = "1"
    os.environ["JURISDRIVE_FRAME_LIMIT"] = str(frame_limit)
    os.environ["JURISDRIVE_REUSE_LOADED_WORLD"] = "1" if reuse_world else "0"
    backend = CarlaBackend(output_dir, host=host, port=port)
    compiled = backend.compile(contract)
    compile_errors = backend.validate(compiled)
    if compile_errors:
        raise ValueError("contract validation failed: %s" % compile_errors)

    started_utc = utc_now()
    wall_started = time.perf_counter()
    result = backend.run(compiled)
    simulation_wall_seconds = time.perf_counter() - wall_started
    serialization_started = time.perf_counter()
    result_path = output_dir / "simulation_result.json"
    write_model_json(result_path, result)
    serialization_seconds = time.perf_counter() - serialization_started
    total_wall_seconds = time.perf_counter() - wall_started
    frame_count = len({state.frame for state in result.actor_states or []})
    keyframes = [output_dir / value for value in result.keyframes or []]
    log_fields = parse_result_log(result.logs)
    record = {
        "run_index": run_index,
        "worker_index": worker_index,
        "scenario_id": scenario_id,
        "seed": contract.seed,
        "frame_limit": frame_limit,
        "fixed_delta_seconds": contract.fixed_delta_seconds,
        "simulated_seconds": frame_count * contract.fixed_delta_seconds,
        "measured": measured,
        "started_at_utc": started_utc,
        "finished_at_utc": utc_now(),
        "status": "success",
        "simulation_status": result.status.value,
        "constraints_passed": all(item.passed for item in result.constraint_results),
        "simulation_wall_seconds": round(simulation_wall_seconds, 6),
        "serialization_seconds": round(serialization_seconds, 6),
        "total_wall_seconds": round(total_wall_seconds, 6),
        "world_acquire_mode": log_fields.get("world_acquire_mode"),
        "world_acquire_seconds": float(log_fields.get("world_acquire_seconds", "nan")),
        "frame_count": frame_count,
        "collision_events": len(result.collisions or []),
        "keyframe_count": len(keyframes),
        "keyframe_bytes": sum(path.stat().st_size for path in keyframes if path.is_file()),
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "error": None,
    }
    return record


def run_carla_worker(args: argparse.Namespace) -> int:
    plan = read_json(args.plan)
    rows = [row for row in plan["rows"] if int(row["worker_index"]) == args.worker_index]
    record_path = args.output_root / "records" / ("carla_worker_%d.jsonl" % args.worker_index)
    completed = {
        int(row["run_index"])
        for existing_path in sorted((args.output_root / "records").glob("carla_worker_*.jsonl"))
        for row in read_jsonl(existing_path)
        if row.get("status") == "success" and row.get("measured")
    }
    failures = 0
    current_phase_map = None
    phase_index = 0
    for ordinal, row in enumerate(rows, 1):
        run_index = int(row["run_index"])
        if run_index in completed:
            continue
        scenario_id = int(row["scenario_id"])
        contract_payload = read_json(
            args.source_bundle_root / ("jurisdrive_%s" % scenario_id) / "contract.json"
        )
        requested_map = str(contract_payload["map_binding"]["carla_map"]["value"])
        if requested_map != current_phase_map:
            phase_index += 1
            current_phase_map = requested_map
            for warmup_index in range(1, args.warmups + 1):
                warmup_dir = args.output_root / "warmups" / (
                    "worker_%d_phase_%02d_%s_warmup_%02d_jurisdrive_%d"
                    % (
                        args.worker_index,
                        phase_index,
                        requested_map,
                        warmup_index,
                        scenario_id,
                    )
                )
                if warmup_dir.exists():
                    warmup_dir = warmup_dir.with_name(
                        warmup_dir.name + "_resume_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                    )
                record = execute_carla_once(
                    args.source_bundle_root,
                    warmup_dir,
                    scenario_id,
                    int(row["frame_limit"]),
                    args.host,
                    args.port,
                    args.reuse_world,
                    False,
                    -phase_index * 100 - warmup_index,
                    args.worker_index,
                )
                append_jsonl(record_path, record)
                print(
                    json.dumps(
                        {
                            "map_warmup": warmup_index,
                            "map_phase": phase_index,
                            "map": requested_map,
                            "worker": args.worker_index,
                            "seconds": record["total_wall_seconds"],
                        }
                    ),
                    flush=True,
                )
        run_dir = args.output_root / "carla_runs" / (
            "run_%04d_jurisdrive_%d" % (run_index, scenario_id)
        )
        if run_dir.exists():
            run_dir = run_dir.with_name(
                run_dir.name
                + "_resume_w%d_" % args.worker_index
                + datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        try:
            record = execute_carla_once(
                args.source_bundle_root,
                run_dir,
                scenario_id,
                int(row["frame_limit"]),
                args.host,
                args.port,
                args.reuse_world,
                True,
                run_index,
                args.worker_index,
            )
        except Exception as exc:
            failures += 1
            record = {
                "run_index": run_index,
                "worker_index": args.worker_index,
                "scenario_id": int(row["scenario_id"]),
                "frame_limit": int(row["frame_limit"]),
                "measured": True,
                "started_at_utc": utc_now(),
                "finished_at_utc": utc_now(),
                "status": "failed",
                "error": "%s: %s" % (exc.__class__.__name__, exc),
                "result_path": None,
            }
        append_jsonl(record_path, record)
        print(
            json.dumps(
                {
                    "worker": args.worker_index,
                    "progress": "%d/%d" % (ordinal, len(rows)),
                    "run_index": run_index,
                    "scenario": row["scenario_id"],
                    "status": record["status"],
                    "seconds": record.get("total_wall_seconds"),
                }
            ),
            flush=True,
        )
        if failures >= args.max_failures:
            print("worker failure gate reached", file=sys.stderr)
            return 2
    return 0 if failures == 0 else 1


def collect_carla_records(output_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((output_root / "records").glob("carla_worker_*.jsonl")):
        rows.extend(read_jsonl(path))
    return sorted(
        [row for row in rows if row.get("measured")], key=lambda row: int(row["run_index"])
    )


def run_vlm(args: argparse.Namespace) -> int:
    from jurisdrive.assurance import VlmEvaluator
    from jurisdrive.io import write_json as write_model_json
    from jurisdrive.models import ScenarioContractV1, SimulationResultV1

    carla_records = [row for row in collect_carla_records(args.output_root) if row.get("status") == "success"]
    if len(carla_records) != args.expected_runs:
        raise SystemExit(
            "expected %d successful CARLA records, found %d" % (args.expected_runs, len(carla_records))
        )
    if args.record_label and not all(
        character.isalnum() or character in {"-", "_"}
        for character in args.record_label
    ):
        raise SystemExit("record label may contain only letters, digits, '-' and '_'")
    label_suffix = "_" + args.record_label if args.record_label else ""
    record_path = args.output_root / "records" / ("vlm_records%s.jsonl" % label_suffix)
    scenario_groups = {
        scenario_id: [
            row for row in carla_records if int(row["scenario_id"]) == scenario_id
        ]
        for scenario_id in SCENARIOS
    }
    balanced_records = []
    for replicate_offset in range(max(len(rows) for rows in scenario_groups.values())):
        for scenario_id in SCENARIOS:
            if replicate_offset < len(scenario_groups[scenario_id]):
                balanced_records.append(scenario_groups[scenario_id][replicate_offset])
    completed = {
        int(row["run_index"])
        for row in read_jsonl(record_path)
        if row.get("status") == "success" and row.get("measured")
    }
    record_lock = threading.Lock()

    def evaluate_one(carla_record: Dict[str, Any], mode: str, measured: bool) -> Dict[str, Any]:
        run_index = int(carla_record["run_index"])
        run_dir = Path(carla_record["result_path"]).parent
        contract = ScenarioContractV1.model_validate(read_json(run_dir / "contract.json"))
        result = SimulationResultV1.model_validate(read_json(Path(carla_record["result_path"])))
        evaluator = VlmEvaluator(
            args.endpoint,
            args.model,
            timeout=args.timeout,
            bundle_dir=run_dir,
        )
        started_utc = utc_now()
        started = time.perf_counter()
        try:
            report = evaluator.evaluate(contract, result)
            inference_seconds = time.perf_counter() - started
            write_started = time.perf_counter()
            if measured:
                write_model_json(run_dir / ("evaluation_vlm%s.json" % label_suffix), report)
                write_json(run_dir / ("evaluation_vlm%s_request.json" % label_suffix), evaluator.last_request)
                write_json(run_dir / ("evaluation_vlm%s_raw_response.json" % label_suffix), evaluator.last_response)
            write_seconds = time.perf_counter() - write_started
            usage = (evaluator.last_response or {}).get("usage") or {}
            return {
                "run_index": run_index,
                "scenario_id": carla_record["scenario_id"],
                "mode": mode,
                "measured": measured,
                "status": "success",
                "passed": report.passed,
                "manual_review": report.manual_review,
                "failure_count": len(report.failures),
                "repair_count": len(report.repair_instructions),
                "started_at_utc": started_utc,
                "finished_at_utc": utc_now(),
                "inference_seconds": round(inference_seconds, 6),
                "write_seconds": round(write_seconds, 6),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "image_count": min(3, len(result.keyframes or [])),
                "error": None,
            }
        except Exception as exc:
            return {
                "run_index": run_index,
                "scenario_id": carla_record["scenario_id"],
                "mode": mode,
                "measured": measured,
                "status": "failed",
                "passed": False,
                "started_at_utc": started_utc,
                "finished_at_utc": utc_now(),
                "inference_seconds": round(time.perf_counter() - started, 6),
                "error": "%s: %s" % (exc.__class__.__name__, exc),
            }

    if not completed:
        for warmup_record in balanced_records[: args.warmups]:
            warmup_result = evaluate_one(warmup_record, "warmup", False)
            append_jsonl(record_path, warmup_result, record_lock)
            print(json.dumps({"vlm_warmup": warmup_result["run_index"], "seconds": warmup_result.get("inference_seconds")}), flush=True)
            if warmup_result["status"] != "success":
                return 2

    pending = [row for row in balanced_records if int(row["run_index"]) not in completed]
    sequential = pending[: args.sequential_count]
    concurrent_rows = pending[args.sequential_count :]
    phase_started = time.perf_counter()
    failures = 0
    for ordinal, carla_record in enumerate(sequential, 1):
        record = evaluate_one(carla_record, "c1", True)
        append_jsonl(record_path, record, record_lock)
        failures += record["status"] != "success"
        print(json.dumps({"vlm_c1": "%d/%d" % (ordinal, len(sequential)), "run_index": record["run_index"], "status": record["status"], "seconds": record.get("inference_seconds")}), flush=True)
    sequential_wall_seconds = time.perf_counter() - phase_started

    concurrent_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(evaluate_one, carla_record, "c%d" % args.concurrency, True): carla_record
            for carla_record in concurrent_rows
        }
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            append_jsonl(record_path, record, record_lock)
            completed_count += 1
            failures += record["status"] != "success"
            print(json.dumps({"vlm_concurrent": "%d/%d" % (completed_count, len(concurrent_rows)), "run_index": record["run_index"], "status": record["status"], "seconds": record.get("inference_seconds")}), flush=True)
    concurrent_wall_seconds = time.perf_counter() - concurrent_started
    phase_summary = {
        "generated_at_utc": utc_now(),
        "endpoint": args.endpoint,
        "model": args.model,
        "warmups": args.warmups,
        "selection_order": "scenario-balanced round-robin",
        "sequential_count": len(sequential),
        "sequential_wall_seconds": round(sequential_wall_seconds, 6),
        "concurrency": args.concurrency,
        "concurrent_count": len(concurrent_rows),
        "concurrent_wall_seconds": round(concurrent_wall_seconds, 6),
        "concurrent_throughput_requests_per_second": (
            round(len(concurrent_rows) / concurrent_wall_seconds, 6) if concurrent_rows else None
        ),
        "failures": int(failures),
    }
    write_json(
        args.output_root
        / "records"
        / ("vlm_phase_summary%s_raw.json" % label_suffix),
        phase_summary,
    )
    return 0 if failures == 0 else 1


def metric_summary(values: List[float]) -> Dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}

    def percentile(fraction: float) -> float:
        index = max(0, min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1))
        return ordered[index]

    return {
        "n": len(ordered),
        "total": round(sum(ordered), 6),
        "mean": round(statistics.mean(ordered), 6),
        "stdev": round(statistics.stdev(ordered), 6) if len(ordered) > 1 else 0.0,
        "min": round(ordered[0], 6),
        "p50": round(statistics.median(ordered), 6),
        "p95": round(percentile(0.95), 6),
        "max": round(ordered[-1], 6),
    }


def phase_wall_seconds(rows: List[Dict[str, Any]]) -> Optional[float]:
    if not rows:
        return None
    starts = [datetime.fromisoformat(str(row["started_at_utc"])) for row in rows]
    finishes = [datetime.fromisoformat(str(row["finished_at_utc"])) for row in rows]
    return round((max(finishes) - min(starts)).total_seconds(), 6)


def summarize_resource_records(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "samples": len(rows),
        "host_cpu_percent": metric_summary(
            [float(row["host_cpu_percent"]) for row in rows if row.get("host_cpu_percent") is not None]
        ),
        "carla_cpu_percent": metric_summary(
            [float(row["carla_cpu_percent"]) for row in rows if row.get("carla_cpu_percent") is not None]
        ),
        "carla_rss_mib": metric_summary(
            [float(row["carla_rss_mib"]) for row in rows if row.get("carla_rss_mib") is not None]
        ),
        "host_memory_used_mib": metric_summary(
            [float(row["host_memory_used_mib"]) for row in rows if row.get("host_memory_used_mib") is not None]
        ),
        "gpus": {},
    }
    gpu_indices = sorted(
        {int(gpu["index"]) for row in rows for gpu in row.get("gpus", [])}
    )
    for gpu_index in gpu_indices:
        gpu_rows = [
            gpu
            for row in rows
            for gpu in row.get("gpus", [])
            if int(gpu["index"]) == gpu_index
        ]
        summary["gpus"][str(gpu_index)] = {
            "name": gpu_rows[0].get("name") if gpu_rows else None,
            "utilization_percent": metric_summary(
                [float(gpu["gpu_utilization_percent"]) for gpu in gpu_rows if gpu.get("gpu_utilization_percent") is not None]
            ),
            "memory_used_mib": metric_summary(
                [float(gpu["memory_used_mib"]) for gpu in gpu_rows if gpu.get("memory_used_mib") is not None]
            ),
            "power_watts": metric_summary(
                [float(gpu["power_watts"]) for gpu in gpu_rows if gpu.get("power_watts") is not None]
            ),
        }
    return summary


def aggregate(args: argparse.Namespace) -> int:
    carla = collect_carla_records(args.output_root)
    label_suffix = "_" + args.vlm_label if args.vlm_label else ""
    vlm = [
        row
        for row in read_jsonl(
            args.output_root / "records" / ("vlm_records%s.jsonl" % label_suffix)
        )
        if row.get("measured")
    ]
    successful_carla = [row for row in carla if row.get("status") == "success"]
    successful_vlm = [row for row in vlm if row.get("status") == "success"]
    resource_rows = read_jsonl(args.output_root / "records" / "resource_samples.jsonl")
    vlm_resource_rows = read_jsonl(
        args.output_root
        / "records"
        / ("vlm_resource_samples%s.jsonl" % label_suffix)
    )
    if successful_vlm:
        vlm_started = min(
            datetime.fromisoformat(str(row["started_at_utc"])) for row in successful_vlm
        )
        vlm_finished = max(
            datetime.fromisoformat(str(row["finished_at_utc"])) for row in successful_vlm
        )
        vlm_resource_rows = [
            row
            for row in vlm_resource_rows
            if vlm_started
            <= datetime.fromisoformat(str(row["timestamp_utc"]))
            <= vlm_finished
        ]
    phase = read_json(
        args.output_root
        / "records"
        / ("vlm_phase_summary%s_raw.json" % label_suffix)
    )
    carla_phase_wall = phase_wall_seconds(successful_carla)
    vlm_phase_wall = phase_wall_seconds(successful_vlm)
    prompt_tokens_total = sum(int(row.get("prompt_tokens") or 0) for row in successful_vlm)
    completion_tokens_total = sum(
        int(row.get("completion_tokens") or 0) for row in successful_vlm
    )
    total_tokens_total = sum(int(row.get("total_tokens") or 0) for row in successful_vlm)
    concurrent_tokens_total = sum(
        int(row.get("total_tokens") or 0)
        for row in successful_vlm
        if row.get("mode") != "c1"
    )
    summary = {
        "generated_at_utc": utc_now(),
        "output_root": str(args.output_root.resolve()),
        "carla": {
            "attempted": len(carla),
            "success": len(successful_carla),
            "failed": len(carla) - len(successful_carla),
            "constraint_pass": sum(bool(row.get("constraints_passed")) for row in successful_carla),
            "total_wall_seconds": metric_summary([float(row["total_wall_seconds"]) for row in successful_carla]),
            "world_acquire_seconds": metric_summary([float(row["world_acquire_seconds"]) for row in successful_carla]),
            "simulated_seconds_total": round(sum(float(row["simulated_seconds"]) for row in successful_carla), 6),
            "phase_wall_seconds": carla_phase_wall,
            "throughput_runs_per_second": (
                round(len(successful_carla) / carla_phase_wall, 6)
                if carla_phase_wall and successful_carla
                else None
            ),
            "reused_world_runs": sum(
                row.get("world_acquire_mode") == "reused" for row in successful_carla
            ),
        },
        "vlm": {
            "attempted": len(vlm),
            "success": len(successful_vlm),
            "failed": len(vlm) - len(successful_vlm),
            "passed": sum(bool(row.get("passed")) for row in successful_vlm),
            "manual_review": sum(bool(row.get("manual_review")) for row in successful_vlm),
            "inference_all": metric_summary([float(row["inference_seconds"]) for row in successful_vlm]),
            "inference_c1": metric_summary([float(row["inference_seconds"]) for row in successful_vlm if row.get("mode") == "c1"]),
            "inference_concurrent": metric_summary([float(row["inference_seconds"]) for row in successful_vlm if row.get("mode") != "c1"]),
            "phase_wall_seconds": vlm_phase_wall,
            "token_usage": {
                "prompt_tokens_total": prompt_tokens_total,
                "completion_tokens_total": completion_tokens_total,
                "total_tokens": total_tokens_total,
                "mean_total_tokens_per_request": round(
                    total_tokens_total / len(successful_vlm), 6
                )
                if successful_vlm
                else None,
                "concurrent_total_tokens": concurrent_tokens_total,
                "concurrent_tokens_per_second": round(
                    concurrent_tokens_total / float(phase["concurrent_wall_seconds"]),
                    6,
                )
                if phase.get("concurrent_wall_seconds")
                else None,
            },
            "phase": phase,
        },
        "resources": summarize_resource_records(resource_rows),
        "vlm_resources": summarize_resource_records(vlm_resource_rows),
        "by_scenario": {},
    }
    for scenario_id in SCENARIOS:
        carla_rows = [row for row in successful_carla if int(row["scenario_id"]) == scenario_id]
        vlm_rows = [row for row in successful_vlm if int(row["scenario_id"]) == scenario_id]
        summary["by_scenario"][str(scenario_id)] = {
            "carla_runs": len(carla_rows),
            "carla_wall_seconds": metric_summary([float(row["total_wall_seconds"]) for row in carla_rows]),
            "vlm_runs": len(vlm_rows),
            "vlm_inference_seconds": metric_summary([float(row["inference_seconds"]) for row in vlm_rows]),
        }

    table_dir = args.output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    write_json(table_dir / "runtime_summary.json", summary)
    with (table_dir / "runtime_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("component", "mode", "n", "mean_s", "p50_s", "p95_s", "min_s", "max_s", "failures"))
        writer.writeheader()
        for component, mode, metric, failures in (
            ("CARLA", "warm_reused_world", summary["carla"]["total_wall_seconds"], summary["carla"]["failed"]),
            ("VLM", "c1", summary["vlm"]["inference_c1"], summary["vlm"]["failed"]),
            ("VLM", "c%d" % phase["concurrency"], summary["vlm"]["inference_concurrent"], summary["vlm"]["failed"]),
        ):
            writer.writerow({
                "component": component,
                "mode": mode,
                "n": metric.get("n"),
                "mean_s": metric.get("mean"),
                "p50_s": metric.get("p50"),
                "p95_s": metric.get("p95"),
                "min_s": metric.get("min"),
                "max_s": metric.get("max"),
                "failures": failures,
            })
    with (table_dir / "runtime_by_scenario.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("scenario_id", "carla_n", "carla_mean_s", "carla_p95_s", "vlm_n", "vlm_mean_s", "vlm_p95_s"))
        writer.writeheader()
        for scenario_id, row in summary["by_scenario"].items():
            writer.writerow({
                "scenario_id": scenario_id,
                "carla_n": row["carla_runs"],
                "carla_mean_s": row["carla_wall_seconds"].get("mean"),
                "carla_p95_s": row["carla_wall_seconds"].get("p95"),
                "vlm_n": row["vlm_runs"],
                "vlm_mean_s": row["vlm_inference_seconds"].get("mean"),
                "vlm_p95_s": row["vlm_inference_seconds"].get("p95"),
            })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def read_host_cpu_times() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def read_carla_processes() -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
            if "CarlaUE4-Linux-Shipping" not in command:
                continue
            stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
            stat_fields = stat_text.rsplit(") ", 1)[1].split()
            cpu_ticks = float(int(stat_fields[11]) + int(stat_fields[12]))
            rss_kib = 0.0
            for line in (proc_dir / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_kib = float(line.split()[1])
                    break
            rows[int(proc_dir.name)] = {"cpu_ticks": cpu_ticks, "rss_kib": rss_kib}
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return rows


def read_gpu_rows() -> List[Dict[str, Any]]:
    nvidia_smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    completed = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 6:
            continue

        def number(value: str) -> Optional[float]:
            try:
                return float(value)
            except ValueError:
                return None

        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "gpu_utilization_percent": number(fields[2]),
                "memory_utilization_percent": number(fields[3]),
                "memory_used_mib": number(fields[4]),
                "power_watts": number(fields[5]),
            }
        )
    return rows


def sample_resources(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous_time = started
    previous_total, previous_idle = read_host_cpu_times()
    previous_processes = read_carla_processes()
    clock_ticks = float(os.sysconf("SC_CLK_TCK"))
    while args.duration <= 0 or time.monotonic() - started < args.duration:
        time.sleep(args.interval)
        now = time.monotonic()
        current_total, current_idle = read_host_cpu_times()
        total_delta = current_total - previous_total
        idle_delta = current_idle - previous_idle
        host_cpu_percent = (
            100.0 * (total_delta - idle_delta) / total_delta if total_delta > 0 else None
        )
        current_processes = read_carla_processes()
        process_ticks_delta = sum(
            max(0.0, values["cpu_ticks"] - previous_processes.get(pid, values)["cpu_ticks"])
            for pid, values in current_processes.items()
        )
        elapsed = max(now - previous_time, 1e-9)
        memory_fields = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory_fields[key] = float(value.split()[0])
        record = {
            "timestamp_utc": utc_now(),
            "elapsed_seconds": round(now - started, 6),
            "host_cpu_percent": round(host_cpu_percent, 6) if host_cpu_percent is not None else None,
            "carla_process_count": len(current_processes),
            "carla_cpu_percent": round(process_ticks_delta / clock_ticks / elapsed * 100.0, 6),
            "carla_rss_mib": round(
                sum(values["rss_kib"] for values in current_processes.values()) / 1024.0,
                6,
            ),
            "host_memory_used_mib": round(
                (memory_fields.get("MemTotal", 0.0) - memory_fields.get("MemAvailable", 0.0))
                / 1024.0,
                6,
            ),
            "gpus": read_gpu_rows(),
        }
        append_jsonl(args.output, record)
        print(json.dumps(record), flush=True)
        previous_time = now
        previous_total, previous_idle = current_total, current_idle
        previous_processes = current_processes
    return 0


def command_capture(arguments: List[str]) -> Dict[str, Any]:
    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return {
        "command": arguments,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def capture_manifest(args: argparse.Namespace) -> int:
    plan = read_json(args.output_root / "benchmark_plan.json")
    cpu_model = None
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    memory_total_kib = None
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            memory_total_kib = int(line.split()[1])
            break
    os_release = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip().strip('"')

    image_query = command_capture(
        ["docker", "image", "inspect", args.client_image]
    )
    image_details: Dict[str, Any] = {}
    if image_query["returncode"] == 0:
        raw_image = json.loads(image_query["stdout"])[0]
        image_details = {
            "reference": args.client_image,
            "id": raw_image.get("Id"),
            "repo_digests": raw_image.get("RepoDigests"),
            "created": raw_image.get("Created"),
        }

    vlm_query = command_capture(["docker", "inspect", args.vlm_container])
    vlm_details: Dict[str, Any] = {}
    if vlm_query["returncode"] == 0:
        raw_vlm = json.loads(vlm_query["stdout"])[0]
        environment = {}
        for item in raw_vlm.get("Config", {}).get("Env", []):
            if "=" in item:
                key, value = item.split("=", 1)
                if "TOKEN" not in key.upper():
                    environment[key] = value
        container_started_at = raw_vlm.get("State", {}).get("StartedAt")
        ready_at = None
        startup_seconds = None
        log_query = command_capture(
            ["docker", "logs", "--timestamps", args.vlm_container]
        )
        for line in (log_query["stdout"] + "\n" + log_query["stderr"]).splitlines():
            if "Application startup complete." in line:
                ready_at = line.split(" ", 1)[0]
        if container_started_at and ready_at:
            startup_seconds = round(
                (
                    datetime.fromisoformat(ready_at.replace("Z", "+00:00"))
                    - datetime.fromisoformat(container_started_at.replace("Z", "+00:00"))
                ).total_seconds(),
                6,
            )
        vlm_details = {
            "container": args.vlm_container,
            "image": raw_vlm.get("Config", {}).get("Image"),
            "image_id": raw_vlm.get("Image"),
            "command": raw_vlm.get("Config", {}).get("Cmd"),
            "environment": environment,
            "port_bindings": raw_vlm.get("HostConfig", {}).get("PortBindings"),
            "network_mode": raw_vlm.get("HostConfig", {}).get("NetworkMode"),
            "last_started_at_utc": container_started_at,
            "last_ready_at_utc": ready_at,
            "last_startup_seconds": startup_seconds,
        }

    source_hashes = {}
    for scenario_id in SCENARIOS:
        contract_path = args.source_bundle_root / ("jurisdrive_%s" % scenario_id) / "contract.json"
        source_hashes[str(scenario_id)] = sha256_file(contract_path)
    server_units = {}
    for worker_index in range(int(plan["workers"])):
        server_units[str(worker_index)] = command_capture(
            [
                "systemctl",
                "--user",
                "show",
                "--property=Id,MainPID,ActiveState,SubState,ExecMainStartTimestamp,ExecStart",
                "jurisdrive-carla-bench%d.service" % worker_index,
            ]
        )["stdout"]

    manifest = {
        "version": "1.0",
        "captured_at_utc": utc_now(),
        "output_root": str(args.output_root.resolve()),
        "benchmark_plan_sha256": sha256_file(args.output_root / "benchmark_plan.json"),
        "hardware": {
            "cpu_model": cpu_model,
            "logical_cpus": os.cpu_count(),
            "memory_total_gib": (
                round(memory_total_kib / 1024.0 / 1024.0, 3)
                if memory_total_kib is not None
                else None
            ),
            "gpus": read_gpu_rows(),
        },
        "software": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "os_release": os_release,
            "docker_version": command_capture(["docker", "version", "--format", "{{json .}}"]),
            "git_head": command_capture(["git", "rev-parse", "HEAD"]),
            "git_status": command_capture(["git", "status", "--porcelain=v1"]),
            "client_image": image_details,
            "vlm": vlm_details,
            "files": {
                "benchmark_script": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "simulator": {
                    "path": str((PROJECT_ROOT / "jurisdrive" / "simulator.py").resolve()),
                    "sha256": sha256_file(PROJECT_ROOT / "jurisdrive" / "simulator.py"),
                },
                "carla_launcher": {
                    "path": str((args.carla_root / "CarlaUE4.sh").resolve()),
                    "sha256": sha256_file(args.carla_root / "CarlaUE4.sh"),
                },
            },
        },
        "inputs": {
            "source_bundle_root": str(args.source_bundle_root.resolve()),
            "contract_sha256_by_scenario": source_hashes,
        },
        "carla_conditions": {
            "workers": plan["workers"],
            "worker_scenario_shards": plan["worker_scenario_shards"],
            "measured_runs": plan["measured_runs"],
            "warmups_per_map_phase": plan["unmeasured_warmups_per_map_phase"],
            "frame_policy": plan["frame_policy"],
            "fixed_delta_seconds": plan["fixed_delta_seconds"],
            "camera_capture": plan["camera_capture"],
            "camera_resolution": [800, 450],
            "camera_sensor_tick_seconds": 0.5,
            "execution_profile": plan["execution_profile"],
            "world_policy": plan["world_policy"],
            "ports": [2000 + 100 * index for index in range(int(plan["workers"]))],
            "server_flags": (
                "-prefernvidia -RenderOffScreen -nosound -quality-level=Low "
                "-ResX=320 -ResY=180 -NoVSync -graphicsadapter={0|1} "
                "-carla-port={port} -log -stdout -FullStdOutLogOutput"
            ),
            "server_units": server_units,
        },
        "vlm_conditions": {
            "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
            "model": vlm_details.get("environment", {}).get("SERVED_MODEL_NAME"),
            "model_id": vlm_details.get("environment", {}).get("MODEL_ID"),
            "model_revision": vlm_details.get("environment", {}).get("MODEL_REVISION"),
            "warmups": 6,
            "warmup_sampling": "one run per scenario",
            "sequential_latency_runs": 30,
            "sequential_sampling": "five runs per scenario in round-robin order",
            "throughput_runs": 170,
            "concurrency": 8,
            "images_per_request": 3,
            "temperature": 0,
            "max_output_tokens": 1024,
        },
    }
    write_json(args.output_root / "benchmark_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def write_paper_outputs(args: argparse.Namespace) -> int:
    summary = read_json(args.output_root / "tables" / "runtime_summary.json")
    manifest = read_json(args.output_root / "benchmark_manifest.json")
    carla = summary["carla"]
    vlm = summary["vlm"]
    phase = vlm["phase"]
    resources = summary["resources"]
    vlm_resources = summary["vlm_resources"]

    def value(metric: Dict[str, Any], key: str, digits: int = 2) -> str:
        raw = metric.get(key)
        return "N/A" if raw is None else ("%.*f" % (digits, float(raw)))

    carla_throughput_per_minute = (
        float(carla["throughput_runs_per_second"]) * 60.0
        if carla.get("throughput_runs_per_second") is not None
        else None
    )
    concurrent_throughput = phase.get("concurrent_throughput_requests_per_second")
    vlm_startup_seconds = manifest["software"]["vlm"].get("last_startup_seconds")
    markdown_lines = [
        "# Bounded CARLA and VLM Runtime Benchmark (200 Measured Repetitions)",
        "",
        "## Experimental protocol",
        "",
        (
            "We measured 200 successful bounded CARLA executions and then evaluated the "
            "same 200 run artifacts with the N6 VLM. The six executable scenarios were "
            "balanced at 33--34 repetitions each. CARLA used %d scenario-isolated server "
            "processes, synchronous stepping at 0.05 s, the contract-collision execution "
            "profile, 800 x 450 RGB capture, 120 frames for scenarios 25 and 71, and 80 "
            "frames for scenarios 99, 367, 460, and 692. One unmeasured warm-up preceded "
            "each server/map stream. Server startup and warm-up time are excluded from "
            "per-run latency but retained in the raw logs."
            % manifest["carla_conditions"]["workers"]
        ),
        "",
        (
            "For N6, Qwen3.5-35B-A3B-FP8 (revision `%s`) was served by vLLM with tensor "
            "parallelism 2, a 32,768-token context window, and `max_num_seqs=8`. Six VLM "
            "warm-ups (one per scenario) were excluded. We used 30 sequential requests "
            "(five per scenario, round-robin) to estimate single-request latency and 170 "
            "requests at concurrency 8 to estimate throughput. "
            "Each request used up to three collision-centered keyframes, temperature 0, "
            "a fixed contract seed, and schema-constrained JSON output. The cached service "
            "restart took %.2f s and was excluded from request latency."
            % (
                manifest["vlm_conditions"].get("model_revision"),
                float(vlm_startup_seconds),
            )
        ),
        "",
        "## Main results",
        "",
        "| Component | Condition | n | Success | Mean (s) | P50 (s) | P95 (s) | Throughput |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| CARLA | %d parallel scenario-isolated workers | %d | %d | %s | %s | %s | %s runs/min |"
        % (
            manifest["carla_conditions"]["workers"],
            carla["attempted"],
            carla["success"],
            value(carla["total_wall_seconds"], "mean"),
            value(carla["total_wall_seconds"], "p50"),
            value(carla["total_wall_seconds"], "p95"),
            "N/A" if carla_throughput_per_minute is None else "%.3f" % carla_throughput_per_minute,
        ),
        "| N6 VLM | sequential (`c=1`) | %d | %d | %s | %s | %s | %s req/s |"
        % (
            vlm["inference_c1"]["n"],
            vlm["inference_c1"]["n"],
            value(vlm["inference_c1"], "mean"),
            value(vlm["inference_c1"], "p50"),
            value(vlm["inference_c1"], "p95"),
            (
                "%.4f" % (vlm["inference_c1"]["n"] / phase["sequential_wall_seconds"])
                if phase.get("sequential_wall_seconds")
                else "N/A"
            ),
        ),
        "| N6 VLM | asynchronous (`c=%d`) | %d | %d | %s | %s | %s | %s req/s |"
        % (
            phase["concurrency"],
            vlm["inference_concurrent"]["n"],
            vlm["inference_concurrent"]["n"],
            value(vlm["inference_concurrent"], "mean"),
            value(vlm["inference_concurrent"], "p50"),
            value(vlm["inference_concurrent"], "p95"),
            "N/A" if concurrent_throughput is None else "%.4f" % concurrent_throughput,
        ),
        "",
        (
            "All %d CARLA runs passed the simulation and hard constraints; %d/%d N6 "
            "evaluations passed, with %d manual-review flags. The CARLA measured phase "
            "took %.2f min wall-clock and represented %.1f s of simulated time. Across "
            "the 200 measured VLM requests, the server processed %d tokens in total; the "
            "c=8 phase achieved %.1f total tokens/s."
            % (
                carla["constraint_pass"],
                vlm["passed"],
                vlm["success"],
                vlm["manual_review"],
                float(carla["phase_wall_seconds"]) / 60.0,
                float(carla["simulated_seconds_total"]),
                int(vlm["token_usage"]["total_tokens"]),
                float(vlm["token_usage"]["concurrent_tokens_per_second"]),
            )
        ),
        "",
        "## Scenario-level CARLA latency",
        "",
        "| Scenario | Frames | n | Mean (s) | P50 (s) | P95 (s) | VLM mean (s) | VLM P95 (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario_id in SCENARIOS:
        row = summary["by_scenario"][str(scenario_id)]
        markdown_lines.append(
            "| %d | %d | %d | %s | %s | %s | %s | %s |"
            % (
                scenario_id,
                120 if scenario_id in LONG_SCENARIOS else 80,
                row["carla_runs"],
                value(row["carla_wall_seconds"], "mean"),
                value(row["carla_wall_seconds"], "p50"),
                value(row["carla_wall_seconds"], "p95"),
                value(row["vlm_inference_seconds"], "mean"),
                value(row["vlm_inference_seconds"], "p95"),
            )
        )
    markdown_lines.extend(
        [
            "",
            "## Resource envelope during CARLA",
            "",
            "| Metric | Mean | P95 | Maximum |",
            "|---|---:|---:|---:|",
            "| Host CPU (%%) | %s | %s | %s |"
            % (
                value(resources["host_cpu_percent"], "mean"),
                value(resources["host_cpu_percent"], "p95"),
                value(resources["host_cpu_percent"], "max"),
            ),
            "| CARLA aggregate CPU (%% of one logical core) | %s | %s | %s |"
            % (
                value(resources["carla_cpu_percent"], "mean"),
                value(resources["carla_cpu_percent"], "p95"),
                value(resources["carla_cpu_percent"], "max"),
            ),
            "| CARLA RSS (MiB) | %s | %s | %s |"
            % (
                value(resources["carla_rss_mib"], "mean"),
                value(resources["carla_rss_mib"], "p95"),
                value(resources["carla_rss_mib"], "max"),
            ),
            "| Host memory used (MiB) | %s | %s | %s |"
            % (
                value(resources["host_memory_used_mib"], "mean"),
                value(resources["host_memory_used_mib"], "p95"),
                value(resources["host_memory_used_mib"], "max"),
            ),
            "",
            "## Resource envelope during measured VLM inference",
            "",
            "| Metric | Mean | P95 | Maximum |",
            "|---|---:|---:|---:|",
            "| Host CPU (%%) | %s | %s | %s |"
            % (
                value(vlm_resources["host_cpu_percent"], "mean"),
                value(vlm_resources["host_cpu_percent"], "p95"),
                value(vlm_resources["host_cpu_percent"], "max"),
            ),
            "| Host memory used (MiB) | %s | %s | %s |"
            % (
                value(vlm_resources["host_memory_used_mib"], "mean"),
                value(vlm_resources["host_memory_used_mib"], "p95"),
                value(vlm_resources["host_memory_used_mib"], "max"),
            ),
            "| GPU 0 utilization (%%) | %s | %s | %s |"
            % (
                value(vlm_resources["gpus"]["0"]["utilization_percent"], "mean"),
                value(vlm_resources["gpus"]["0"]["utilization_percent"], "p95"),
                value(vlm_resources["gpus"]["0"]["utilization_percent"], "max"),
            ),
            "| GPU 1 utilization (%%) | %s | %s | %s |"
            % (
                value(vlm_resources["gpus"]["1"]["utilization_percent"], "mean"),
                value(vlm_resources["gpus"]["1"]["utilization_percent"], "p95"),
                value(vlm_resources["gpus"]["1"]["utilization_percent"], "max"),
            ),
            "| GPU 0 memory used (MiB) | %s | %s | %s |"
            % (
                value(vlm_resources["gpus"]["0"]["memory_used_mib"], "mean"),
                value(vlm_resources["gpus"]["0"]["memory_used_mib"], "p95"),
                value(vlm_resources["gpus"]["0"]["memory_used_mib"], "max"),
            ),
            "| GPU 1 memory used (MiB) | %s | %s | %s |"
            % (
                value(vlm_resources["gpus"]["1"]["memory_used_mib"], "mean"),
                value(vlm_resources["gpus"]["1"]["memory_used_mib"], "p95"),
                value(vlm_resources["gpus"]["1"]["memory_used_mib"], "max"),
            ),
            "",
            "## Interpretation and reporting boundaries",
            "",
            "The 200-run CARLA figure is a batch-throughput experiment, not a cold-start "
            "latency measurement. A previously recorded independent-run reference of eight "
            "80-frame executions required 1,818.6 s in total (3.79 min/run). The present "
            "study excludes server startup and warm-ups from per-run latency, preserves "
            "them in the logs, and reports both individual-run latency and %d-worker wall-"
            "clock throughput. VLM `c=1` latency and `c=8` throughput answer different "
            "questions and should not be merged into one mean."
            % manifest["carla_conditions"]["workers"],
            "",
            "The benchmark covers N5 CARLA execution and N6 multimodal assurance for six "
            "accepted executable contracts. It does not by itself re-run N0--N4 over 200 "
            "new judgments; those upstream extraction/compilation experiments require a "
            "separate end-to-end benchmark label.",
            "",
            "Raw records, requests, responses, resource samples, hashes, and exact commands "
            "are preserved beside this report for auditability.",
            "",
        ]
    )
    paper_dir = args.output_root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "runtime_benchmark_results.md").write_text(
        "\n".join(markdown_lines), encoding="utf-8"
    )

    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Bounded CARLA and VLM runtime over 200 measured repetitions.}",
        "\\label{tab:bounded-runtime-200}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Component & Condition & $n$ & Mean (s) & P50 (s) & P95 (s) & Throughput \\\\",
        "\\midrule",
        "CARLA & %d workers & %d & %s & %s & %s & %s runs/min \\\\"
        % (
            manifest["carla_conditions"]["workers"],
            carla["success"],
            value(carla["total_wall_seconds"], "mean"),
            value(carla["total_wall_seconds"], "p50"),
            value(carla["total_wall_seconds"], "p95"),
            "N/A" if carla_throughput_per_minute is None else "%.3f" % carla_throughput_per_minute,
        ),
        "N6 VLM & $c=1$ & %d & %s & %s & %s & %s req/s \\\\"
        % (
            vlm["inference_c1"]["n"],
            value(vlm["inference_c1"], "mean"),
            value(vlm["inference_c1"], "p50"),
            value(vlm["inference_c1"], "p95"),
            (
                "%.4f" % (vlm["inference_c1"]["n"] / phase["sequential_wall_seconds"])
                if phase.get("sequential_wall_seconds")
                else "N/A"
            ),
        ),
        "N6 VLM & $c=%d$ & %d & %s & %s & %s & %s req/s \\\\"
        % (
            phase["concurrency"],
            vlm["inference_concurrent"]["n"],
            value(vlm["inference_concurrent"], "mean"),
            value(vlm["inference_concurrent"], "p50"),
            value(vlm["inference_concurrent"], "p95"),
            "N/A" if concurrent_throughput is None else "%.4f" % concurrent_throughput,
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    (args.output_root / "tables" / "runtime_summary.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )
    print(str((paper_dir / "runtime_benchmark_results.md").resolve()))
    print(str((args.output_root / "tables" / "runtime_summary.tex").resolve()))
    return 0


def write_final_acceptance(args: argparse.Namespace) -> int:
    label_suffix = "_" + args.vlm_label if args.vlm_label else ""
    carla = collect_carla_records(args.output_root)
    vlm_all = read_jsonl(
        args.output_root / "records" / ("vlm_records%s.jsonl" % label_suffix)
    )
    vlm = [row for row in vlm_all if row.get("measured")]
    scenario_counts = {
        str(scenario_id): sum(int(row["scenario_id"]) == scenario_id for row in carla)
        for scenario_id in SCENARIOS
    }
    c1_counts = {
        str(scenario_id): sum(
            int(row["scenario_id"]) == scenario_id and row.get("mode") == "c1"
            for row in vlm
        )
        for scenario_id in SCENARIOS
    }
    expected_scenario_counts = {
        "25": 34,
        "71": 34,
        "99": 33,
        "367": 33,
        "460": 33,
        "692": 33,
    }
    file_counts = {
        "simulation_result": len(list((args.output_root / "carla_runs").rglob("simulation_result.json"))),
        "telemetry": len(list((args.output_root / "carla_runs").rglob("telemetry.jsonl"))),
        "vlm_evaluation": len(
            list(
                (args.output_root / "carla_runs").rglob(
                    "evaluation_vlm%s.json" % label_suffix
                )
            )
        ),
        "vlm_request": len(
            list(
                (args.output_root / "carla_runs").rglob(
                    "evaluation_vlm%s_request.json" % label_suffix
                )
            )
        ),
        "vlm_raw_response": len(
            list(
                (args.output_root / "carla_runs").rglob(
                    "evaluation_vlm%s_raw_response.json" % label_suffix
                )
            )
        ),
    }
    checks = {
        "carla_measured_200": len(carla) == 200,
        "carla_unique_run_indices_200": len({int(row["run_index"]) for row in carla}) == 200,
        "carla_scenario_balance": scenario_counts == expected_scenario_counts,
        "carla_success_200": sum(row.get("status") == "success" for row in carla) == 200,
        "carla_constraint_pass_200": sum(bool(row.get("constraints_passed")) for row in carla) == 200,
        "carla_collision_runs_200": sum(int(row.get("collision_events") or 0) > 0 for row in carla) == 200,
        "carla_keyframe_runs_200": sum(int(row.get("keyframe_count") or 0) > 0 for row in carla) == 200,
        "carla_world_reused_200": sum(row.get("world_acquire_mode") == "reused" for row in carla) == 200,
        "vlm_warmups_6": sum(not row.get("measured") for row in vlm_all) == 6,
        "vlm_measured_200": len(vlm) == 200,
        "vlm_unique_run_indices_200": len({int(row["run_index"]) for row in vlm}) == 200,
        "vlm_success_200": sum(row.get("status") == "success" for row in vlm) == 200,
        "vlm_pass_200": sum(bool(row.get("passed")) for row in vlm) == 200,
        "vlm_manual_review_0": sum(bool(row.get("manual_review")) for row in vlm) == 0,
        "vlm_three_images_200": sum(int(row.get("image_count") or 0) == 3 for row in vlm) == 200,
        "vlm_c1_30": sum(row.get("mode") == "c1" for row in vlm) == 30,
        "vlm_c1_five_per_scenario": all(count == 5 for count in c1_counts.values()),
        "vlm_c8_170": sum(row.get("mode") == "c8" for row in vlm) == 170,
        "expected_output_file_counts": all(value == 200 for value in file_counts.values()),
    }
    key_paths = (
        args.output_root / "benchmark_plan.json",
        args.output_root / "benchmark_manifest.json",
        args.output_root / "tables" / "runtime_summary.json",
        args.output_root / "tables" / "runtime_summary.csv",
        args.output_root / "tables" / "runtime_by_scenario.csv",
        args.output_root / "tables" / "runtime_summary.tex",
        args.output_root / "paper" / "runtime_benchmark_results.md",
    )
    payload = {
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "vlm_label": args.vlm_label,
        "checks": checks,
        "scenario_counts": scenario_counts,
        "c1_counts_by_scenario": c1_counts,
        "file_counts": file_counts,
        "key_artifact_sha256": {
            str(path.relative_to(args.output_root)): sha256_file(path)
            for path in key_paths
            if path.is_file()
        },
    }
    write_json(args.output_root / "final_acceptance.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--total", type=int, default=200)
    plan.add_argument("--workers", type=int, default=2)
    plan.add_argument("--warmups-per-map", type=int, default=1)

    worker = subparsers.add_parser("carla-worker")
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--source-bundle-root", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--host", required=True)
    worker.add_argument("--port", type=int, required=True)
    worker.add_argument("--warmups", type=int, default=1)
    worker.add_argument("--reuse-world", action="store_true")
    worker.add_argument("--max-failures", type=int, default=3)

    vlm = subparsers.add_parser("vlm")
    vlm.add_argument("--output-root", type=Path, required=True)
    vlm.add_argument("--endpoint", required=True)
    vlm.add_argument("--model", required=True)
    vlm.add_argument("--expected-runs", type=int, default=200)
    vlm.add_argument("--warmups", type=int, default=6)
    vlm.add_argument("--sequential-count", type=int, default=30)
    vlm.add_argument("--concurrency", type=int, default=8)
    vlm.add_argument("--timeout", type=float, default=600.0)
    vlm.add_argument("--record-label", default="")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-root", type=Path, required=True)
    aggregate_parser.add_argument("--vlm-label", default="")
    sampler = subparsers.add_parser("sample-resources")
    sampler.add_argument("--output", type=Path, required=True)
    sampler.add_argument("--interval", type=float, default=5.0)
    sampler.add_argument("--duration", type=float, default=0.0)
    manifest = subparsers.add_parser("capture-manifest")
    manifest.add_argument("--output-root", type=Path, required=True)
    manifest.add_argument("--source-bundle-root", type=Path, required=True)
    manifest.add_argument("--carla-root", type=Path, required=True)
    manifest.add_argument(
        "--client-image", default="jurisdrive/chatscene:0.9.13-client"
    )
    manifest.add_argument("--vlm-container", default="qwen35-vlm-server")
    paper = subparsers.add_parser("write-paper")
    paper.add_argument("--output-root", type=Path, required=True)
    acceptance = subparsers.add_parser("accept")
    acceptance.add_argument("--output-root", type=Path, required=True)
    acceptance.add_argument("--vlm-label", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "plan":
        return make_plan(args)
    if args.command == "carla-worker":
        return run_carla_worker(args)
    if args.command == "vlm":
        return run_vlm(args)
    if args.command == "aggregate":
        return aggregate(args)
    if args.command == "sample-resources":
        return sample_resources(args)
    if args.command == "capture-manifest":
        return capture_manifest(args)
    if args.command == "write-paper":
        return write_paper_outputs(args)
    if args.command == "accept":
        return write_final_acceptance(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
