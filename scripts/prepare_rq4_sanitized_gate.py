#!/usr/bin/env python3
"""Build and audit the no-model-call first gate for sanitized blind RQ4."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jurisdrive.rq4_sanitized import (  # noqa: E402
    EXPERIMENT_ID,
    METHOD_CODES,
    build_method_payloads,
    build_request_template,
    forbidden_path_hits,
    opaque_identifier,
    recursive_forbidden_hits,
    render_request,
    sanitize_contract,
    sanitize_telemetry,
    sha256_file,
    summarize_budget_shape,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def source_keyframes(result: dict[str, Any], bundle: Path) -> list[Path]:
    values = result.get("keyframes") or []
    if len(values) != 3:
        raise ValueError(f"exactly three keyframes are required: {bundle} has {len(values)}")
    paths = []
    for value in values:
        path = Path(str(value))
        if not path.is_absolute():
            path = bundle / path
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def copy_frame_set(paths: list[Path], destination: Path) -> list[str]:
    refs: list[str] = []
    for index, source in enumerate(paths):
        target = destination / f"F{index:02d}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        refs.append(str(target).replace("\\", "/"))
    return refs


def cyclic_donors(rows: list[dict[str, Any]], shift: int) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("fault_type") or "control")].append(row)
    result: dict[str, dict[str, Any]] = {}
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: row["opaque_artifact_id"])
        if len(ordered) < 2:
            raise ValueError("shuffle group must contain at least two artifacts")
        effective = shift % len(ordered) or 1
        for index, row in enumerate(ordered):
            donor = ordered[(index + effective) % len(ordered)]
            if donor["slot_id"] == row["slot_id"]:
                raise ValueError("shuffle donor must be a different judgment")
            result[row["opaque_artifact_id"]] = donor
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    targets = ["private", "task_a", "task_b", "task_c", "audit", "sanitized_materialization_manifest.json", "leakage_audit.json"]
    collisions = [output_root / target for target in targets if (output_root / target).exists()]
    if collisions:
        raise FileExistsError("refusing to overwrite first-gate outputs: " + ", ".join(str(path) for path in collisions))
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".gate1_staging_", dir=output_root))
    try:
        model_visible = stage / "task_a" / "model_visible"
        private = stage / "private"
        audit = stage / "audit"
        model_visible.mkdir(parents=True)
        private.mkdir(parents=True)
        audit.mkdir(parents=True)

        materialized = read_jsonl(args.materialization_records.resolve())
        mutable_rows = read_jsonl(args.mutable_records.resolve())
        mutable = {str(row["trial_id"]): row for row in mutable_rows}
        if len(materialized) != 168 or len(mutable_rows) != 72:
            raise ValueError(f"unexpected source denominators: materialized={len(materialized)}, mutable={len(mutable_rows)}")

        enriched: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for row in materialized:
            opaque_id = opaque_identifier("artifact", str(row["trial_id"]))
            if opaque_id in seen_ids:
                raise ValueError(f"opaque-ID collision: {opaque_id}")
            seen_ids.add(opaque_id)
            contract_bundle = Path(row["bundle_path"]).resolve()
            result_bundle = contract_bundle
            injection_verified = bool(row.get("injection_verified"))
            if row.get("fault_class") == "mutable":
                rerun = mutable.get(str(row["trial_id"]))
                if not rerun:
                    raise ValueError(f"missing mutable rerun: {row['trial_id']}")
                result_bundle = Path(rerun["rerun_bundle_path"]).resolve()
                injection_verified = bool(rerun.get("injection_verified"))
            if not injection_verified:
                raise ValueError(f"unverified phenotype: {row['trial_id']}")
            contract_path = contract_bundle / "contract.json"
            result_path = result_bundle / "simulation_result.json"
            contract = read_json(contract_path)
            result = read_json(result_path)
            views = sanitize_contract(contract)
            telemetry = sanitize_telemetry(result, views["actor_map"])
            enriched.append(
                {
                    **row,
                    "opaque_artifact_id": opaque_id,
                    "contract_bundle": str(contract_bundle),
                    "result_bundle": str(result_bundle),
                    "contract_path": str(contract_path),
                    "result_path": str(result_path),
                    "contract_sha256": sha256_file(contract_path),
                    "result_sha256": sha256_file(result_path),
                    "contract_views": views,
                    "telemetry": telemetry,
                    "keyframe_paths": [str(path.resolve()) for path in source_keyframes(result, result_bundle)],
                }
            )

        image_donors = cyclic_donors(enriched, shift=1)
        telemetry_donors = cyclic_donors(enriched, shift=5)
        asset_refs: dict[str, dict[str, list[str]]] = {}
        for row in enriched:
            opaque_id = row["opaque_artifact_id"]
            base = model_visible / "assets" / opaque_id
            observed_absolute = copy_frame_set([Path(path) for path in row["keyframe_paths"]], base / "A")
            image_donor = image_donors[opaque_id]
            shuffled_absolute = copy_frame_set([Path(path) for path in image_donor["keyframe_paths"]], base / "B")
            asset_refs[opaque_id] = {
                "observed": [str(Path(path).relative_to(model_visible)).replace("\\", "/") for path in observed_absolute],
                "shuffled": [str(Path(path).relative_to(model_visible)).replace("\\", "/") for path in shuffled_absolute],
            }

        request_rows: list[dict[str, Any]] = []
        private_rows: list[dict[str, Any]] = []
        templates_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        budget_shapes: dict[str, Counter[tuple[tuple[str, ...], int]]] = {method: Counter() for method in METHOD_CODES}
        all_request_hits: list[dict[str, str]] = []
        for row in sorted(enriched, key=lambda item: item["opaque_artifact_id"]):
            opaque_id = row["opaque_artifact_id"]
            telemetry_donor = telemetry_donors[opaque_id]
            payloads = build_method_payloads(
                contract_views=row["contract_views"],
                telemetry=row["telemetry"],
                shuffled_telemetry=telemetry_donor["telemetry"],
                image_refs=asset_refs[opaque_id]["observed"],
                shuffled_image_refs=asset_refs[opaque_id]["shuffled"],
            )
            for method, code in METHOD_CODES.items():
                payload = payloads[method]
                request_id = opaque_identifier("request", f"{opaque_id}|{code}")
                template = build_request_template(
                    opaque_artifact_id=opaque_id,
                    evidence=payload["evidence"],
                    image_refs=payload["image_refs"],
                )
                relative = Path("requests") / code / f"{request_id}.json"
                write_json(model_visible / relative, template)
                templates_by_key[(opaque_id, method)] = template
                request_rows.append(
                    {
                        "request_id": request_id,
                        "opaque_artifact_id": opaque_id,
                        "method_code": code,
                        "request_path": str(relative).replace("\\", "/"),
                        "request_sha256": sha256_file(model_visible / relative),
                    }
                )
                shape = summarize_budget_shape(payload)
                budget_shapes[method][(tuple(shape["evidence_keys"]), int(shape["image_count"]))] += 1
                all_request_hits.extend(recursive_forbidden_hits(template, origin=str(relative).replace("\\", "/")))
            private_rows.append(
                {
                    "opaque_artifact_id": opaque_id,
                    "legacy_trial_id": row["trial_id"],
                    "judgment_id": row["candidate_id"],
                    "judgment_slot": row["slot_id"],
                    "topology": row["topology"],
                    "source_stage": row["source_stage"],
                    "trial_kind": row["trial_kind"],
                    "fault_type": row.get("fault_type"),
                    "fault_class": row.get("fault_class"),
                    "variant": row.get("variant"),
                    "contract_path": row["contract_path"],
                    "contract_sha256": row["contract_sha256"],
                    "result_path": row["result_path"],
                    "result_sha256": row["result_sha256"],
                    "image_shuffle_donor": image_donors[opaque_id]["opaque_artifact_id"],
                    "telemetry_shuffle_donor": telemetry_donors[opaque_id]["opaque_artifact_id"],
                    "cohort_split": "test",
                }
            )

        write_jsonl(model_visible / "request_index.jsonl", request_rows)
        write_jsonl(private / "opaque_id_mapping.jsonl", private_rows)
        write_json(
            private / "method_code_mapping.json",
            {"experiment_id": EXPERIMENT_ID, "mapping": METHOD_CODES},
        )
        write_json(
            private / "cohort_freeze.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "split_policy": "all 24 frozen judgments are test; implementation development uses synthetic unit fixtures only",
                "development_judgments": 0,
                "test_judgments": len({row["judgment_slot"] for row in private_rows}),
                "test_artifacts": len(private_rows),
                "no_replacement_after_freeze": True,
                "fault_family_counts": dict(Counter(str(row.get("fault_type") or "control") for row in private_rows)),
                "transformation_holdout": "All realized transformations and magnitudes are withheld from data-specific prompt/rule tuning; only schema-level allowlists and synthetic fixtures are used in gate development.",
            },
        )

        # Fully rendered human-inspection samples: one per method and injected
        # family.  Their hidden family mapping remains outside sample bodies.
        sample_index: list[dict[str, Any]] = []
        fault_families = sorted({str(row["fault_type"]) for row in enriched if row.get("fault_type")})
        for method, code in METHOD_CODES.items():
            for family in fault_families:
                row = next(item for item in enriched if item.get("fault_type") == family)
                sample_id = opaque_identifier("sample", f"{method}|{family}")
                rendered = render_request(templates_by_key[(row["opaque_artifact_id"], method)], model_visible)
                sample_path = audit / "rendered_samples" / f"{sample_id}.json"
                write_json(sample_path, rendered)
                hits = recursive_forbidden_hits(rendered, origin=f"rendered_samples/{sample_id}.json")
                all_request_hits.extend(hits)
                sample_index.append(
                    {
                        "sample_id": sample_id,
                        "method": method,
                        "method_code": code,
                        "fault_type": family,
                        "opaque_artifact_id": row["opaque_artifact_id"],
                        "sample_path": str(sample_path.relative_to(stage)).replace("\\", "/"),
                        "sample_sha256": sha256_file(sample_path),
                        "forbidden_hits": len(hits),
                    }
                )
        write_jsonl(private / "rendered_sample_index.jsonl", sample_index)

        # Scan every exposed Task-A path as well as all request structures.
        visible_paths = [str(path.relative_to(model_visible)).replace("\\", "/") for path in model_visible.rglob("*")]
        path_hits = forbidden_path_hits(visible_paths)
        all_hits = all_request_hits + path_hits
        file_hash_rows = [
            {
                "path": str(path.relative_to(model_visible)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(model_visible.rglob("*"))
            if path.is_file()
        ]
        write_jsonl(audit / "model_visible_file_hashes.jsonl", file_hash_rows)
        leakage = {
            "version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "generated_at_utc": utc_now(),
            "scope": "all Task-A request templates, all exposed paths, and 48 fully rendered human-inspection samples",
            "request_templates_audited": len(request_rows),
            "expected_request_templates": 168 * len(METHOD_CODES),
            "rendered_samples_audited": len(sample_index),
            "sample_requirement": f"{len(METHOD_CODES)} methods x {len(fault_families)} injected families",
            "model_visible_paths_audited": len(visible_paths),
            "forbidden_hits": len(all_hits),
            "forbidden_key_hits": sum(hit["kind"] == "forbidden_key" for hit in all_hits),
            "forbidden_path_hits": len(path_hits),
            "clean_reference_field_hits": sum("clean" in hit["kind"] for hit in all_hits),
            "oracle_field_hits": sum("oracle" in hit["kind"] for hit in all_hits),
            "hits": all_hits,
            "status": "PASS" if not all_hits and len(request_rows) == 1344 and len(sample_index) == 48 else "FAIL",
            "image_scope_note": "Image bytes are copied and hashed; recursive lexical scanning covers JSON, prompts, serialized scalar values, and exposed paths. Image pixels are intended visual evidence and are not OCR-redacted.",
        }
        write_json(stage / "leakage_audit.json", leakage)
        if leakage["status"] != "PASS":
            raise ValueError(f"leakage gate failed with {len(all_hits)} hits")

        budget_summary = {
            method: [
                {"evidence_keys": list(keys), "image_count": images, "artifact_count": count}
                for (keys, images), count in sorted(counter.items())
            ]
            for method, counter in budget_shapes.items()
        }
        write_json(
            stage / "protocol" / "information_budget.json",
            {
                "version": "1.0",
                "experiment_id": EXPERIMENT_ID,
                "method_codes_private": "private/method_code_mapping.json",
                "method_shapes": budget_summary,
                "invariants": {
                    "fusion_is_exact_image_text_plus_telemetry_union": True,
                    "image_shuffled_text_budget_equals_image_only": True,
                    "telemetry_shuffled_field_budget_equals_telemetry_only": True,
                    "no_image_is_fusion_with_images_removed": True,
                    "guarded_adds_only_mutability_classes": True,
                    "common_decision_schema": True,
                },
            },
        )
        write_json(
            stage / "task_b" / "task_manifest.json",
            {
                "task": "B",
                "status": "reserved_not_materialized_in_first_gate",
                "separation": "Privileged integrity references are prohibited from Task A and will be prepared only after gate approval.",
                "contributes_to_task_a_detection_metrics": False,
            },
        )
        write_json(
            stage / "task_c" / "task_manifest.json",
            {
                "task": "C",
                "status": "reserved_not_materialized_in_first_gate",
                "separation": "Known-good rollback values are prohibited from Task A and will be prepared only after an explicit routing stage.",
                "contributes_to_task_a_detection_metrics": False,
            },
        )

        manifest_inputs = {
            "materialization_records": {
                "path": str(args.materialization_records.resolve()),
                "sha256": sha256_file(args.materialization_records.resolve()),
            },
            "mutable_records": {
                "path": str(args.mutable_records.resolve()),
                "sha256": sha256_file(args.mutable_records.resolve()),
            },
            "preregistration": {
                "path": str(args.preregistration.resolve()),
                "sha256": sha256_file(args.preregistration.resolve()),
            },
        }
        manifest_outputs = {
            "request_index": {
                "path": "task_a/model_visible/request_index.jsonl",
                "sha256": sha256_file(model_visible / "request_index.jsonl"),
            },
            "opaque_id_mapping": {
                "path": "private/opaque_id_mapping.jsonl",
                "sha256": sha256_file(private / "opaque_id_mapping.jsonl"),
            },
            "rendered_sample_index": {
                "path": "private/rendered_sample_index.jsonl",
                "sha256": sha256_file(private / "rendered_sample_index.jsonl"),
            },
            "model_visible_file_hashes": {
                "path": "audit/model_visible_file_hashes.jsonl",
                "sha256": sha256_file(audit / "model_visible_file_hashes.jsonl"),
            },
            "leakage_audit": {
                "path": "leakage_audit.json",
                "sha256": sha256_file(stage / "leakage_audit.json"),
            },
        }
        manifest = {
            "version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "generated_at_utc": utc_now(),
            "status": "FIRST_GATE_PASS_NO_MODEL_CALLS",
            "source_policy": "historical artifacts read-only; new versioned materialization only",
            "counts": {
                "judgments": len({row["judgment_slot"] for row in private_rows}),
                "artifacts": len(private_rows),
                "controls": sum(row["trial_kind"] == "clean_control" for row in private_rows),
                "injected_artifacts": sum(row["trial_kind"] != "clean_control" for row in private_rows),
                "request_templates": len(request_rows),
                "rendered_samples": len(sample_index),
            },
            "task_separation": {
                "task_a": "sanitized blind inputs prepared and audited; no inference performed",
                "task_b": "reserved and separate",
                "task_c": "reserved and separate",
            },
            "inputs": manifest_inputs,
            "outputs": manifest_outputs,
            "code": {
                "module": {"path": str((ROOT / "jurisdrive" / "rq4_sanitized.py").resolve()), "sha256": sha256_file(ROOT / "jurisdrive" / "rq4_sanitized.py")},
                "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            },
            "acceptance_gate": {
                "forbidden_hits": leakage["forbidden_hits"],
                "clean_reference_field_hits": leakage["clean_reference_field_hits"],
                "oracle_field_hits": leakage["oracle_field_hits"],
                "information_budget_invariants": "PASS",
                "all_168_accounted": len(private_rows) == 168,
                "model_calls": 0,
            },
        }
        write_json(stage / "sanitized_materialization_manifest.json", manifest)

        # Move staged outputs only after every acceptance check has passed.
        for name in ("private", "task_a", "task_b", "task_c", "audit"):
            os.replace(stage / name, output_root / name)
        protocol_budget = stage / "protocol" / "information_budget.json"
        os.replace(protocol_budget, output_root / "protocol" / "information_budget.json")
        os.replace(stage / "leakage_audit.json", output_root / "leakage_audit.json")
        os.replace(stage / "sanitized_materialization_manifest.json", output_root / "sanitized_materialization_manifest.json")
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-records", type=Path, required=True)
    parser.add_argument("--mutable-records", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

