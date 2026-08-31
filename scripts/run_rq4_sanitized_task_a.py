#!/usr/bin/env python3
"""Freeze, execute, summarize, and audit sanitized blind RQ4 Task A.

The command is deliberately append-only.  It never writes to the historical
RQ4 v4 tree or to the approved first-gate materialization.  Private labels are
loaded only by the summarize/audit phases, after blind inference has finished.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import platform
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jurisdrive.rq4_sanitized import (  # noqa: E402
    RESPONSE_SCHEMA,
    recursive_forbidden_hits,
    render_request,
)
from jurisdrive.rq4_task_a import (  # noqa: E402
    canonical_json,
    conservative_repeat_consensus,
    judgment_cluster_bootstrap,
    paired_common_coverage_test,
    parse_verdict_response,
    per_fault_summary,
    read_json,
    read_jsonl,
    repeat_agreement,
    sha256_bytes,
    sha256_file,
    task_a_metrics,
    write_json,
    write_jsonl,
)

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "migration_runs" / "20260825_1500" / "rq4_sanitized_v1"
EXPECTED_FIRST_GATE = {
    "judgments": 24,
    "artifacts": 168,
    "request_templates_verified": 1344,
    "rendered_samples_verified": 48,
    "model_visible_file_hashes_verified": 2353,
    "forbidden_hits": 0,
    "revealing_path_hits": 0,
    "clean_reference_field_hits": 0,
    "oracle_field_hits": 0,
    "model_calls": 0,
}


def vllm_compatible_response_schema() -> dict[str, Any]:
    """Remove the one unsupported grammar keyword without weakening validation.

    vLLM 0.23.0 rejects JSON schemas containing ``uniqueItems``.  Duplicate
    issue codes are still rejected by ``validate_verdict`` after generation.
    """

    schema = copy.deepcopy(RESPONSE_SCHEMA)
    schema["properties"]["issue_codes"].pop("uniqueItems", None)
    return schema


def materialize_execution_payload(
    template: Mapping[str, Any], model_visible_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    payload = render_request(template, model_visible_root)
    payload["model"] = config["served_model"]["served_name"]
    payload["seed"] = config["decoding"]["seed"]
    payload["chat_template_kwargs"] = config["decoding"]["chat_template_kwargs"]
    payload["response_format"]["json_schema"]["schema"] = copy.deepcopy(
        config["decoding"]["response_schema_sent"]
    )
    return payload


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")


def api_json(url: str, *, payload: Mapping[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None if payload is None else canonical_json(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object from {url}")
    return value


def command_output(command: list[str], *, timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def verify_first_gate(root: Path) -> dict[str, Any]:
    manifest_path = root / "audit" / "first_gate_validation_manifest.json"
    manifest = read_json(manifest_path)
    verification = manifest.get("verification") or {}
    failures = {
        key: {"expected": expected, "actual": verification.get(key)}
        for key, expected in EXPECTED_FIRST_GATE.items()
        if verification.get(key) != expected
    }
    hash_failures: list[dict[str, str]] = []
    for section in ("code_hashes", "output_hashes"):
        for relative, expected in (manifest.get(section) or {}).items():
            path = REPO_ROOT / relative if section == "code_hashes" else root / relative
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                hash_failures.append(
                    {"section": section, "path": str(path), "expected": str(expected), "actual": actual}
                )
    if failures or hash_failures:
        raise RuntimeError(f"first gate no longer passes: counts={failures}, hashes={hash_failures}")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "verification": verification,
        "verified_code_and_output_hashes": sum(
            len(manifest.get(section) or {}) for section in ("code_hashes", "output_hashes")
        ),
        "hash_failures": 0,
    }


def request_index(root: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(root / "task_a" / "model_visible" / "request_index.jsonl")
    if len(rows) != 1344 or len({row["request_id"] for row in rows}) != 1344:
        raise RuntimeError("Task-A request index is not the approved 1,344-template cohort")
    for row in rows:
        path = root / "task_a" / "model_visible" / str(row["request_path"])
        if sha256_file(path) != row["request_sha256"]:
            raise RuntimeError(f"request template hash mismatch: {row['request_id']}")
    return rows


def ordered_ids(rows: Iterable[Mapping[str, Any]], seed: int) -> list[str]:
    ids = sorted(str(row["request_id"]) for row in rows)
    random.Random(seed).shuffle(ids)
    return ids


def freeze(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    execution_root = args.execution_root.resolve()
    if execution_root.exists():
        raise FileExistsError(f"refusing to overwrite execution root: {execution_root}")
    gate = verify_first_gate(root)
    rows = request_index(root)
    models = api_json(args.endpoint.rstrip("/") + "/models", timeout=30)
    served = [str(row.get("id")) for row in models.get("data", []) if isinstance(row, Mapping)]
    if args.served_model not in served:
        raise RuntimeError(f"served model {args.served_model!r} not in endpoint inventory {served}")

    execution_root.mkdir(parents=True, exist_ok=False)
    (execution_root / "orders").mkdir()
    (execution_root / "calls").mkdir()
    for seed in args.order_seeds:
        write_json(execution_root / "orders" / f"order_{seed}.json", {"seed": seed, "request_ids": ordered_ids(rows, seed)})

    docker_inspect = command_output(
        ["docker", "inspect", args.container, "--format", "{{json .}}"], timeout=30
    )
    container_public: dict[str, Any] = {}
    if docker_inspect.get("returncode") == 0 and docker_inspect.get("stdout"):
        raw = json.loads(str(docker_inspect["stdout"]))
        allowed_env = {
            "MODEL_ID",
            "MODEL_REVISION",
            "SERVED_MODEL_NAME",
            "TENSOR_PARALLEL_SIZE",
            "MAX_MODEL_LEN",
            "MAX_NUM_SEQS",
            "GPU_MEMORY_UTILIZATION",
            "MAX_IMAGES_PER_PROMPT",
            "MM_PROCESSOR_CACHE_GB",
            "NCCL_P2P_DISABLE",
            "NCCL_CUMEM_ENABLE",
            "NCCL_CUMEM_HOST_ENABLE",
        }
        env = {}
        for item in ((raw.get("Config") or {}).get("Env") or []):
            key, _, value = str(item).partition("=")
            if key in allowed_env:
                env[key] = value
        container_public = {
            "id": str(raw.get("Id") or ""),
            "created": raw.get("Created"),
            "image_reference": (raw.get("Config") or {}).get("Image"),
            "image_id": raw.get("Image"),
            "started_at": (raw.get("State") or {}).get("StartedAt"),
            "environment_allowlist": env,
        }

    runtime_versions = command_output(
        ["docker", "exec", args.container, "python3", "-c", "import torch,vllm; print(vllm.__version__); print(torch.__version__); print(torch.version.cuda)"],
        timeout=30,
    )
    gpu_inventory = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,pstate",
            "--format=csv,noheader",
        ],
        timeout=30,
    )
    schema_hash = sha256_bytes(canonical_json(RESPONSE_SCHEMA).encode("utf-8"))
    sent_schema = vllm_compatible_response_schema()
    sent_schema_hash = sha256_bytes(canonical_json(sent_schema).encode("utf-8"))
    config = {
        "version": "1.0",
        "experiment_id": "rq4_sanitized_blind_v1_task_a",
        "status": "FROZEN_NO_MODEL_CALLS",
        "frozen_at_kst": now_kst(),
        "source_root": str(root),
        "execution_root": str(execution_root),
        "first_gate": gate,
        "lineage": (
            {
                "predecessor_failure_manifest": str(args.predecessor_failure_manifest.resolve()),
                "predecessor_failure_manifest_sha256": sha256_file(args.predecessor_failure_manifest.resolve()),
            }
            if args.predecessor_failure_manifest
            else None
        ),
        "frozen_code_hashes": {
            "jurisdrive/rq4_sanitized.py": sha256_file(REPO_ROOT / "jurisdrive" / "rq4_sanitized.py"),
            "jurisdrive/rq4_task_a.py": sha256_file(REPO_ROOT / "jurisdrive" / "rq4_task_a.py"),
            "scripts/run_rq4_sanitized_task_a.py": sha256_file(Path(__file__).resolve()),
            "tests/test_rq4_sanitized_execution.py": sha256_file(REPO_ROOT / "tests" / "test_rq4_sanitized_execution.py"),
            "server/docker-compose.yml": sha256_file(args.server_compose.resolve()),
        },
        "served_model": {
            "model_id": args.model_id,
            "checkpoint_revision": args.model_revision,
            "served_name": args.served_model,
            "endpoint": args.endpoint.rstrip("/"),
            "endpoint_inventory": models,
        },
        "runtime": {
            "container": args.container,
            "tensor_parallelism": 2,
            "vllm_version": "0.23.0",
            "docker_image_digest": args.image_digest,
            "container_public_inventory": container_public,
            "reported_versions": runtime_versions,
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "gpu_inventory": gpu_inventory,
        },
        "decoding": {
            "temperature": 0,
            "max_tokens": 512,
            "seed": args.model_seed,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_schema_sha256": schema_hash,
            "response_schema": RESPONSE_SCHEMA,
            "response_schema_sent_sha256": sent_schema_hash,
            "response_schema_sent": sent_schema,
            "compatibility_transform": "removed unsupported JSON-Schema uniqueItems keyword for vLLM 0.23.0; duplicate issue codes remain rejected by the post-response validator",
        },
        "execution_policy": {
            "request_order_seeds": args.order_seeds,
            "concurrency": args.concurrency,
            "request_timeout_seconds": args.timeout,
            "maximum_transport_retries": args.retries,
            "retry_backoff_seconds": [2, 5],
            "retry_scope": "transport errors, timeouts, HTTP 408/429/5xx only; parse/schema failures are retained without retry",
            "repeat_consensus": "three identical valid statuses required; otherwise MANUAL_REVIEW",
        },
        "statistics": {
            "judgment_cluster_bootstrap_samples": 10000,
            "bootstrap_seed": 20260825,
            "paired_cluster_sign_flip_samples": 100000,
            "paired_seed": 20260825,
            "paired_scope": "valid common decisive coverage only",
        },
        "denominators": {"judgments": 24, "artifacts": 168, "methods": 8, "requests_per_repeat": 1344, "repeats": 3, "planned_model_calls": 4032},
        "information_boundary": "sanitized blind Task A only; Task B and Task C are not materialized or executed",
        "approved_request_index_sha256": sha256_file(root / "task_a" / "model_visible" / "request_index.jsonl"),
        "order_file_sha256": {
            str(seed): sha256_file(execution_root / "orders" / f"order_{seed}.json") for seed in args.order_seeds
        },
        "model_calls_at_freeze": 0,
    }
    config["configuration_sha256"] = sha256_bytes(canonical_json(config).encode("utf-8"))
    write_json(execution_root / "execution_freeze.json", config)
    print(canonical_json({"status": "FROZEN", "execution_root": str(execution_root), "configuration_sha256": config["configuration_sha256"]}))


def should_retry_http(code: int) -> bool:
    return code in {408, 429} or 500 <= code <= 599


def one_call(
    *,
    index_row: Mapping[str, Any],
    position: int,
    order_seed: int,
    root: Path,
    freeze_config: Mapping[str, Any],
) -> dict[str, Any]:
    request_id = str(index_row["request_id"])
    template_path = root / "task_a" / "model_visible" / str(index_row["request_path"])
    source_hash = sha256_file(template_path)
    if source_hash != index_row["request_sha256"]:
        raise RuntimeError(f"template changed before execution: {request_id}")
    template = read_json(template_path)
    payload = materialize_execution_payload(
        template, root / "task_a" / "model_visible", freeze_config
    )
    hits = recursive_forbidden_hits(payload, origin=f"executed-request:{request_id}")
    if hits:
        raise RuntimeError(f"recursive leakage audit failed before call for {request_id}: {hits[:3]}")
    rendered_hash = sha256_bytes(canonical_json(payload).encode("utf-8"))
    policy = freeze_config["execution_policy"]
    endpoint = freeze_config["served_model"]["endpoint"].rstrip("/") + "/chat/completions"
    attempts: list[dict[str, Any]] = []
    response: dict[str, Any] | None = None
    start = time.time()
    max_attempts = int(policy["maximum_transport_retries"]) + 1
    for attempt_number in range(1, max_attempts + 1):
        attempt_start = time.time()
        try:
            response = api_json(endpoint, payload=payload, timeout=float(policy["request_timeout_seconds"]))
            attempts.append({"attempt": attempt_number, "started_unix": attempt_start, "elapsed_seconds": time.time() - attempt_start, "outcome": "response"})
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retry = should_retry_http(exc.code) and attempt_number < max_attempts
            attempts.append({"attempt": attempt_number, "started_unix": attempt_start, "elapsed_seconds": time.time() - attempt_start, "outcome": "http_error", "http_status": exc.code, "response_body": body, "retry": retry})
            if not retry:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            retry = attempt_number < max_attempts
            attempts.append({"attempt": attempt_number, "started_unix": attempt_start, "elapsed_seconds": time.time() - attempt_start, "outcome": "transport_error", "error_type": type(exc).__name__, "error": str(exc), "retry": retry})
            if not retry:
                break
        if attempt_number < max_attempts:
            backoffs = list(policy["retry_backoff_seconds"])
            time.sleep(float(backoffs[min(attempt_number - 1, len(backoffs) - 1)]))
    verdict = None
    parse_error = "no API response"
    raw_content = ""
    if response is not None:
        verdict, parse_error, raw_content = parse_verdict_response(response)
    return {
        "request_id": request_id,
        "method_code": str(index_row["method_code"]),
        "opaque_artifact_id": str(index_row["opaque_artifact_id"]),
        "order_seed": order_seed,
        "order_position": position,
        "source_template_path": str(index_row["request_path"]),
        "source_template_sha256": source_hash,
        "rendered_request_sha256": rendered_hash,
        "configuration_sha256": freeze_config["configuration_sha256"],
        "started_unix": start,
        "finished_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "attempts": attempts,
        "api_response": response,
        "raw_content": raw_content,
        "verdict": verdict,
        "parse_or_schema_error": parse_error,
        "status": verdict["status"] if verdict else None,
        "manual_review_outcome": "explicit_model_abstention" if verdict and verdict["status"] == "MANUAL_REVIEW" else "invalid_or_failed_response_routed_to_review" if verdict is None else None,
        "recorded_at_kst": now_kst(),
    }


def execute(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    execution_root = args.execution_root.resolve()
    freeze_path = execution_root / "execution_freeze.json"
    config = read_json(freeze_path)
    if config.get("status") != "FROZEN_NO_MODEL_CALLS":
        raise RuntimeError("execution manifest is not a pre-inference freeze")
    verify_first_gate(root)
    rows = request_index(root)
    by_id = {str(row["request_id"]): row for row in rows}
    if args.order_seed not in config["execution_policy"]["request_order_seeds"]:
        raise RuntimeError("order seed is not frozen")
    order_path = execution_root / "orders" / f"order_{args.order_seed}.json"
    if sha256_file(order_path) != config["order_file_sha256"][str(args.order_seed)]:
        raise RuntimeError("request order file changed after freeze")
    order = read_json(order_path)["request_ids"]
    output_path = execution_root / "calls" / f"calls_{args.order_seed}.jsonl"
    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {str(row["request_id"]) for row in existing}
    pending = [(position, request_id) for position, request_id in enumerate(order) if request_id not in done]
    print(f"seed={args.order_seed} existing={len(done)} pending={len(pending)}", flush=True)
    completed = len(done)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        with ThreadPoolExecutor(max_workers=int(config["execution_policy"]["concurrency"])) as pool:
            future_map = {
                pool.submit(one_call, index_row=by_id[request_id], position=position, order_seed=args.order_seed, root=root, freeze_config=config): request_id
                for position, request_id in pending
            }
            for future in as_completed(future_map):
                request_id = future_map[future]
                try:
                    record = future.result()
                except Exception as exc:  # Preserve even local preprocessing failures.
                    record = {
                        "request_id": request_id,
                        "method_code": str(by_id[request_id]["method_code"]),
                        "opaque_artifact_id": str(by_id[request_id]["opaque_artifact_id"]),
                        "order_seed": args.order_seed,
                        "status": None,
                        "verdict": None,
                        "api_response": None,
                        "attempts": [],
                        "parse_or_schema_error": f"runner_exception:{type(exc).__name__}:{exc}",
                        "manual_review_outcome": "runner_failure_routed_to_review",
                        "recorded_at_kst": now_kst(),
                    }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                completed += 1
                if completed % 25 == 0 or completed == 1344:
                    print(f"seed={args.order_seed} completed={completed}/1344", flush=True)
    final_rows = read_jsonl(output_path)
    if len(final_rows) != 1344 or len({row["request_id"] for row in final_rows}) != 1344:
        raise RuntimeError(f"seed {args.order_seed} did not produce exactly 1,344 unique records")
    write_json(
        execution_root / "calls" / f"seed_{args.order_seed}_completion.json",
        {
            "order_seed": args.order_seed,
            "completed_at_kst": now_kst(),
            "records": len(final_rows),
            "valid_verdicts": sum(row.get("status") in {"PASS", "FAIL", "MANUAL_REVIEW"} for row in final_rows),
            "invalid_or_failed": sum(row.get("status") not in {"PASS", "FAIL", "MANUAL_REVIEW"} for row in final_rows),
            "api_attempts": sum(len(row.get("attempts") or []) for row in final_rows),
            "retries": sum(max(0, len(row.get("attempts") or []) - 1) for row in final_rows),
            "records_sha256": sha256_file(output_path),
        },
    )


def labeled_seed_rows(root: Path, execution_root: Path) -> list[dict[str, Any]]:
    config = read_json(execution_root / "execution_freeze.json")
    labels = {str(row["opaque_artifact_id"]): row for row in read_jsonl(root / "private" / "opaque_id_mapping.jsonl")}
    rows: list[dict[str, Any]] = []
    for seed in config["execution_policy"]["request_order_seeds"]:
        for call in read_jsonl(execution_root / "calls" / f"calls_{seed}.jsonl"):
            private = labels[str(call["opaque_artifact_id"])]
            rows.append(
                {
                    "request_id": call["request_id"],
                    "method_code": call["method_code"],
                    "opaque_artifact_id": call["opaque_artifact_id"],
                    "order_seed": seed,
                    "judgment_slot": private["judgment_slot"],
                    "is_fault": private["trial_kind"] == "fault",
                    "fault_type": private.get("fault_type"),
                    "fault_class": private.get("fault_class"),
                    "status": call.get("status") if call.get("status") in {"PASS", "FAIL", "MANUAL_REVIEW"} else "MANUAL_REVIEW",
                    "source_outcome": "valid_model_verdict" if call.get("status") in {"PASS", "FAIL", "MANUAL_REVIEW"} else "invalid_or_failed_response",
                }
            )
    return rows


def summarize(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    execution_root = args.execution_root.resolve()
    seed_rows = labeled_seed_rows(root, execution_root)
    if len(seed_rows) != 4032:
        raise RuntimeError(f"expected all 4,032 seed-level records, found {len(seed_rows)}")
    write_jsonl(execution_root / "results" / "task_a_seed_scoring_records.jsonl", seed_rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["request_id"]), []).append(row)
    consensus_rows: list[dict[str, Any]] = []
    for request_id, group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["order_seed"]))
        base = dict(group[0])
        base["repeat_statuses"] = [row["status"] for row in group]
        base["repeat_source_outcomes"] = [row["source_outcome"] for row in group]
        base["status"] = conservative_repeat_consensus([row["status"] for row in group])
        base.pop("order_seed", None)
        base.pop("source_outcome", None)
        consensus_rows.append(base)
    if len(consensus_rows) != 1344:
        raise RuntimeError(f"expected 1,344 consensus rows, found {len(consensus_rows)}")
    results_dir = execution_root / "results"
    write_jsonl(results_dir / "task_a_blind_detection_records.jsonl", consensus_rows)

    method_mapping_document = read_json(root / "private" / "method_code_mapping.json")
    method_names = {
        str(code): str(name)
        for name, code in (method_mapping_document.get("mapping") or {}).items()
    }
    summary: dict[str, Any] = {
        "experiment_id": "rq4_sanitized_blind_v1_task_a",
        "generated_at_kst": now_kst(),
        "primary_result": "conservative three-repeat consensus",
        "denominators": {"judgments": 24, "artifacts": 168, "faults": 144, "clean_controls": 24, "methods": 8, "requests_per_repeat": 1344, "repeats": 3, "model_calls": 4032},
        "methods": {},
        "seed_level_methods": {},
        "repeat_agreement": repeat_agreement(seed_rows),
        "claim_boundary": "blind defect pass/fail detection only; no repair effectiveness or rollback score is inferred; Task B and Task C remain separate",
    }
    for method in sorted({row["method_code"] for row in consensus_rows}):
        subset = [row for row in consensus_rows if row["method_code"] == method]
        metrics = task_a_metrics(subset)
        summary["methods"][method] = {
            "method_name": method_names.get(method, method),
            "metrics": metrics,
            "cluster_bootstrap_95ci": judgment_cluster_bootstrap(subset, samples=10000, seed=20260825),
            "fault_strata": per_fault_summary(subset),
        }
    for seed in sorted({int(row["order_seed"]) for row in seed_rows}):
        summary["seed_level_methods"][str(seed)] = {}
        for method in sorted({row["method_code"] for row in seed_rows}):
            subset = [row for row in seed_rows if int(row["order_seed"]) == seed and row["method_code"] == method]
            summary["seed_level_methods"][str(seed)][method] = task_a_metrics(subset)

    paired: dict[str, Any] = {}
    methods = sorted(summary["methods"])
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            paired[f"{left}_vs_{right}"] = paired_common_coverage_test(
                [row for row in consensus_rows if row["method_code"] == left],
                [row for row in consensus_rows if row["method_code"] == right],
                samples=100000,
                seed=20260825,
            )
    summary["paired_common_coverage_tests"] = paired
    write_json(results_dir / "task_a_method_summary.json", summary)

    csv_path = results_dir / "task_a_method_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["method_code", "method_name", "n", "faults", "clean_controls", "tp", "tn", "fp", "fn", "manual_review", "precision", "recall", "f1", "false_acceptance_rate", "false_rejection_rate", "coverage", "manual_review_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            entry = summary["methods"][method]
            metrics = entry["metrics"]
            confusion = metrics["decisive_confusion"]
            writer.writerow({
                "method_code": method,
                "method_name": entry["method_name"],
                "n": metrics["n"],
                "faults": metrics["faults"],
                "clean_controls": metrics["clean_controls"],
                "tp": confusion["tp"], "tn": confusion["tn"], "fp": confusion["fp"], "fn": confusion["fn"],
                "manual_review": metrics["manual_review"]["total"],
                "precision": metrics["precision"],
                "recall": metrics["recall_full_fault_denominator"],
                "f1": metrics["f1_full_fault_denominator"],
                "false_acceptance_rate": metrics["false_acceptance_rate"],
                "false_rejection_rate": metrics["false_rejection_rate"],
                "coverage": metrics["coverage"],
                "manual_review_rate": metrics["manual_review_rate"],
            })

    lines = [
        "# Sanitized Blind RQ4 Task-A Results",
        "",
        "Primary estimates use a conservative consensus over three fixed request-order repeats. A row is decisive only when all three valid statuses agree; every other outcome is routed to manual review.",
        "",
        "| Method | n | TP/TN/FP/FN | Review | Precision | Recall | F1 | FAR | FRR | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        entry, m = summary["methods"][method], summary["methods"][method]["metrics"]
        c = m["decisive_confusion"]
        pct = lambda value: "N/A" if value is None else f"{100 * value:.1f}%"
        lines.append(f"| {method} ({entry['method_name']}) | {m['n']} | {c['tp']}/{c['tn']}/{c['fp']}/{c['fn']} | {m['manual_review']['total']} | {pct(m['precision'])} | {pct(m['recall_full_fault_denominator'])} | {pct(m['f1_full_fault_denominator'])} | {pct(m['false_acceptance_rate'])} | {pct(m['false_rejection_rate'])} | {pct(m['coverage'])} |")
    lines.extend(["", "Claim boundary: these results evaluate sanitized blind Task-A defect detection only. They do not establish repair success or rollback safety; Task B and Task C require separate evidence.", ""])
    (results_dir / "task_a_paper_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(canonical_json({"status": "SUMMARIZED", "consensus_rows": len(consensus_rows), "summary": str(results_dir / "task_a_method_summary.json")}))


def audit(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    execution_root = args.execution_root.resolve()
    config = read_json(execution_root / "execution_freeze.json")
    gate = verify_first_gate(root)
    index = request_index(root)
    by_id = {str(row["request_id"]): row for row in index}
    all_calls: list[dict[str, Any]] = []
    unique_request_audits: dict[str, dict[str, Any]] = {}
    hash_failures: list[dict[str, Any]] = []
    leakage_hits: list[dict[str, Any]] = []
    for seed in config["execution_policy"]["request_order_seeds"]:
        call_path = execution_root / "calls" / f"calls_{seed}.jsonl"
        calls = read_jsonl(call_path)
        all_calls.extend(calls)
        for call in calls:
            request_id = str(call["request_id"])
            source = by_id[request_id]
            template_path = root / "task_a" / "model_visible" / str(source["request_path"])
            source_hash = sha256_file(template_path)
            payload = materialize_execution_payload(
                read_json(template_path), root / "task_a" / "model_visible", config
            )
            rendered_hash = sha256_bytes(canonical_json(payload).encode("utf-8"))
            if source_hash != source["request_sha256"] or call.get("source_template_sha256") != source_hash or call.get("rendered_request_sha256") != rendered_hash:
                hash_failures.append({"order_seed": seed, "request_id": request_id, "approved_template": source["request_sha256"], "actual_template": source_hash, "recorded_template": call.get("source_template_sha256"), "reconstructed_rendered": rendered_hash, "recorded_rendered": call.get("rendered_request_sha256")})
            if request_id not in unique_request_audits:
                hits = recursive_forbidden_hits(payload, origin=f"executed-request:{request_id}")
                leakage_hits.extend(hits)
                unique_request_audits[request_id] = {"rendered_request_sha256": rendered_hash, "hits": len(hits)}

    expected_calls = 4032
    if len(all_calls) != expected_calls:
        raise RuntimeError(f"expected {expected_calls} call records, found {len(all_calls)}")
    duplicates = len(all_calls) - len({(row["order_seed"], row["request_id"]) for row in all_calls})
    results_dir = execution_root / "results"
    summary_path = results_dir / "task_a_method_summary.json"
    summary = read_json(summary_path)
    denominator_failures = []
    for method, entry in summary["methods"].items():
        m = entry["metrics"]
        if (m["n"], m["faults"], m["clean_controls"]) != (168, 144, 24):
            denominator_failures.append({"method": method, "n": m["n"], "faults": m["faults"], "clean": m["clean_controls"]})
    report = {
        "version": "1.0",
        "audited_at_kst": now_kst(),
        "status": "PASS" if not (hash_failures or leakage_hits or duplicates or denominator_failures) else "FAIL",
        "first_gate": gate,
        "executed_requests": {"call_records": len(all_calls), "unique_templates": len(unique_request_audits), "repeats": 3, "hash_failures": len(hash_failures), "duplicate_seed_request_rows": duplicates},
        "recursive_leakage_audit": {"unique_exact_rendered_requests_audited": len(unique_request_audits), "forbidden_or_revealing_hits": len(leakage_hits), "hits": leakage_hits},
        "denominators": {"expected_per_method": {"n": 168, "faults": 144, "clean_controls": 24}, "failures": denominator_failures},
        "request_hash_failures": hash_failures,
        "failure_rows_preserved": sum(row.get("status") not in {"PASS", "FAIL", "MANUAL_REVIEW"} for row in all_calls),
        "retries_preserved": sum(max(0, len(row.get("attempts") or []) - 1) for row in all_calls),
        "information_boundary": "Task A only; no Task-B or Task-C inputs or scores",
    }
    write_json(execution_root / "audit" / "executed_leakage_audit.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"executed-request audit failed: {report}")

    output_hashes = {}
    for path in sorted(results_dir.glob("*")):
        if path.is_file():
            output_hashes[str(path.relative_to(execution_root)).replace("\\", "/")] = sha256_file(path)
    code_paths = [
        REPO_ROOT / "jurisdrive" / "rq4_sanitized.py",
        REPO_ROOT / "jurisdrive" / "rq4_task_a.py",
        Path(__file__).resolve(),
        REPO_ROOT / "tests" / "test_rq4_sanitized_execution.py",
    ]
    completion = {
        "version": "1.0",
        "experiment_id": "rq4_sanitized_blind_v1_task_a",
        "status": "TASK_A_COMPLETE_PENDING_INDEPENDENT_AUDIT",
        "completed_at_kst": now_kst(),
        "input_root": str(root),
        "execution_root": str(execution_root),
        "configuration_sha256": config["configuration_sha256"],
        "inputs": {"first_gate_manifest_sha256": gate["manifest_sha256"], "request_index_sha256": config["approved_request_index_sha256"], "orders": config["order_file_sha256"]},
        "code_hashes": {str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path) for path in code_paths},
        "output_hashes": output_hashes,
        "audit_sha256": sha256_file(execution_root / "audit" / "executed_leakage_audit.json"),
        "model_calls": len(all_calls),
        "api_attempts": sum(len(row.get("attempts") or []) for row in all_calls),
        "retries": report["retries_preserved"],
        "invalid_parse_timeout_or_failure_rows": report["failure_rows_preserved"],
        "denominators": summary["denominators"],
        "claim_boundaries": [
            "Task-A estimates concern blind defect pass/fail detection under the frozen sanitized information budgets.",
            "Manual reviews and failed/invalid responses remain in full denominators and are never silently dropped.",
            "No Task-B repair-effectiveness or Task-C rollback-safety conclusion is supported by this experiment.",
            "Paired tests are restricted to common decisive coverage and use judgment-cluster-aware inference.",
        ],
        "handoff": "An independent audit of leakage, denominators, hashes, and statistics is required before publication use.",
    }
    write_json(execution_root / "completion_manifest.json", completion)
    print(canonical_json({"status": report["status"], "completion_manifest": str(execution_root / "completion_manifest.json")}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    common.add_argument("--execution-root", type=Path, required=True)
    freeze_parser = sub.add_parser("freeze", parents=[common])
    freeze_parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    freeze_parser.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B-FP8")
    freeze_parser.add_argument("--model-revision", default="9d1823d2dee688a6b25e77009dc727688c44936e")
    freeze_parser.add_argument("--served-model", default="qwen35-vlm")
    freeze_parser.add_argument("--container", default="qwen35-vlm-server")
    freeze_parser.add_argument("--image-digest", default="vllm/vllm-openai:v0.23.0@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f")
    freeze_parser.add_argument("--model-seed", type=int, default=20260825)
    freeze_parser.add_argument("--order-seeds", type=int, nargs=3, default=[2026082501, 2026082502, 2026082503])
    freeze_parser.add_argument("--concurrency", type=int, default=8)
    freeze_parser.add_argument("--timeout", type=float, default=300.0)
    freeze_parser.add_argument("--retries", type=int, default=2)
    freeze_parser.add_argument("--predecessor-failure-manifest", type=Path)
    freeze_parser.add_argument(
        "--server-compose",
        type=Path,
        default=REPO_ROOT.parents[1] / "wsl-vllm-qwen-vlm" / "docker-compose.yml",
    )
    execute_parser = sub.add_parser("execute", parents=[common])
    execute_parser.add_argument("--order-seed", type=int, required=True)
    sub.add_parser("summarize", parents=[common])
    sub.add_parser("audit", parents=[common])
    return result


def main() -> None:
    args = parser().parse_args()
    {"freeze": freeze, "execute": execute, "summarize": summarize, "audit": audit}[args.command](args)


if __name__ == "__main__":
    main()
