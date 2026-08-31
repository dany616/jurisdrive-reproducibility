"""Sanitized request construction for the preregistered blind RQ4 study.

This module deliberately separates model-visible Task-A data from the private
label/oracle mapping used for later scoring.  It performs no model calls.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "rq4_sanitized_blind_v1"

METHOD_CODES: dict[str, str] = {
    "text_only_contract": "M00",
    "image_only_vlm": "M01",
    "telemetry_only": "M02",
    "telemetry_plus_image": "M03",
    "image_shuffled": "M04",
    "telemetry_shuffled": "M05",
    "no_image_fusion_prompt": "M06",
    "guarded_blind_loop": "M07",
}

# Exact historical markers and semantic equivalents that must never appear in
# Task-A values.  Generic words are additionally prohibited in exposed paths
# and keys below, while base64 image data is checked against this exact list.
FORBIDDEN_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("map_lane_mismatch_marker", re.compile(r"map[\s_-]*lane[\s_-]*mismatch[\s_-]*fault", re.I)),
    ("collision_omission_marker", re.compile(r"collision[\s_-]*omission[\s_-]*fault", re.I)),
    ("pose_perturbation_marker", re.compile(r"pose[\s_-]*perturbation[\s_-]*fault", re.I)),
    ("controlled_actor_target_marker", re.compile(r"controlled\s+actor[\s_-]*target\s+fault", re.I)),
    ("controlled_event_order_marker", re.compile(r"controlled\s+event[\s_-]*order\s+fault", re.I)),
    ("fault_type_field_text", re.compile(r"\bfault[\s_-]*type\b", re.I)),
    ("trial_kind_field_text", re.compile(r"\btrial[\s_-]*kind\b", re.I)),
    ("expected_disposition_field_text", re.compile(r"\bexpected[\s_-]*disposition\b", re.I)),
    ("oracle_value_field_text", re.compile(r"\boracle[\s_-]*value\b", re.I)),
    ("clean_reference_text", re.compile(r"\bclean[\s_-]*(?:contract|bundle|reference|counterpart|keyframe|result|hash)", re.I)),
    ("injection_reason_text", re.compile(r"\binjection[\s_-]*reason\b", re.I)),
    ("deterministic_failure_reason_text", re.compile(r"\bdeterministic[\s_-]*failure[\s_-]*reason\b", re.I)),
)

FORBIDDEN_KEY_NORMALIZED: frozenset[str] = frozenset(
    {
        "faulttype",
        "faultclass",
        "trialkind",
        "expecteddisposition",
        "oraclevalue",
        "oraclecontract",
        "oracleresult",
        "cleancontractsha256",
        "cleanresultsha256",
        "cleanbundlepath",
        "cleancounterpart",
        "cleanreference",
        "faultmanifest",
        "injectionreason",
        "injectedvalue",
        "groundtruth",
        "isfault",
        "expectedlabel",
        "deterministicfailurereason",
        "failurereason",
    }
)

FORBIDDEN_PATH_PATTERN = re.compile(
    r"(?:^|[/_.-])(?:fault|oracle|clean(?:bundle|contract|reference|counterpart)?|"
    r"trial[\s_-]*kind|expected[\s_-]*disposition|ground[\s_-]*truth|injection)(?:$|[/_.-])",
    re.I,
)

LANE_REFERENCE_ALLOWLIST = {
    "lane_relative",
    "same_lane",
    "adjacent_lane",
    "left_lane",
    "right_lane",
    "centerline",
    "intersection_approach",
}

SYSTEM_PROMPT = (
    "You are a conservative road-scenario assurance evaluator. Assess only the "
    "evidence supplied in this request against its declared scenario expectations. "
    "Do not assume a hidden reference. Return PASS only when the supplied evidence "
    "supports the declared expectations, FAIL when it contradicts them, and "
    "MANUAL_REVIEW when the evidence is insufficient or ambiguous."
)

USER_INSTRUCTION = (
    "Assess this artifact. Report a status, non-privileged issue codes, and a short "
    "rationale grounded only in the supplied content."
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "issue_codes", "rationale"],
    "properties": {
        "status": {"type": "string", "enum": ["PASS", "FAIL", "MANUAL_REVIEW"]},
        "issue_codes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "ACTOR_RELATION",
                    "TRAJECTORY_GEOMETRY",
                    "EVENT_SEQUENCE",
                    "COLLISION_EVIDENCE",
                    "MODALITY_ALIGNMENT",
                    "INSUFFICIENT_EVIDENCE",
                ],
            },
            "uniqueItems": True,
        },
        "rationale": {"type": "string", "maxLength": 600},
    },
}


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opaque_identifier(namespace: str, source_key: str, *, length: int = 20) -> str:
    digest = hashlib.sha256(f"{EXPERIMENT_ID}|{namespace}|{source_key}".encode("utf-8")).hexdigest()
    prefix = {"artifact": "A", "request": "R", "sample": "S"}.get(namespace, "X")
    return prefix + digest[:length].upper()


def _safe_scalar(value: Any, *, field: str) -> Any:
    if not isinstance(value, str):
        return value
    if any(pattern.search(value) for _, pattern in FORBIDDEN_VALUE_PATTERNS):
        return "unrecognized_reference"
    if field == "lane_position" and value not in LANE_REFERENCE_ALLOWLIST:
        return "unrecognized_reference"
    return value


def _field_value(field: Any, *, field_name: str) -> Any:
    if isinstance(field, Mapping) and "value" in field:
        return _safe_scalar(field.get("value"), field=field_name)
    return _safe_scalar(field, field=field_name)


def _mutability(field: Any) -> str:
    provenance = str(field.get("provenance") or "unknown") if isinstance(field, Mapping) else "unknown"
    return "immutable" if provenance == "observed" else "mutable" if provenance in {"inferred", "defaulted"} else "review"


def sanitize_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Create allowlisted contract views and a neutral actor-ID mapping."""

    actors = list(contract.get("actors") or [])
    actor_map = {str(row.get("id")): f"actor_{index + 1:02d}" for index, row in enumerate(actors)}
    sanitized_actors: list[dict[str, Any]] = []
    visual_actors: list[dict[str, Any]] = []
    mutability: list[dict[str, str]] = []
    for row in actors:
        old_id = str(row.get("id"))
        actor_id = actor_map[old_id]
        item = {
            "actor_id": actor_id,
            "role": _safe_scalar(row.get("role"), field="role"),
            "vehicle_type": _field_value(row.get("vehicle_type"), field_name="vehicle_type"),
            "blueprint": _field_value(row.get("blueprint"), field_name="blueprint"),
            "lane_position": _field_value(row.get("lane_position"), field_name="lane_position"),
            "initial_speed_mps": _field_value(row.get("initial_speed_mps"), field_name="initial_speed_mps"),
        }
        sanitized_actors.append(item)
        visual_actors.append(
            {
                "actor_id": actor_id,
                "role": item["role"],
                "vehicle_type": item["vehicle_type"],
                "lane_position": item["lane_position"],
            }
        )
        for field_name in ("vehicle_type", "blueprint", "lane_position", "initial_speed_mps"):
            mutability.append(
                {
                    "field": f"actors.{actor_id}.{field_name}",
                    "class": _mutability(row.get(field_name)),
                }
            )

    map_binding = contract.get("map_binding") or {}
    topology = _field_value(contract.get("topology"), field_name="topology")
    map_view = {
        "archetype": _field_value(map_binding.get("archetype"), field_name="map_archetype"),
        "runtime_map": _field_value(map_binding.get("carla_map"), field_name="runtime_map"),
    }
    mutability.extend(
        [
            {"field": "map.archetype", "class": _mutability(map_binding.get("archetype"))},
            {"field": "map.runtime_map", "class": _mutability(map_binding.get("carla_map"))},
            {"field": "topology", "class": _mutability(contract.get("topology"))},
        ]
    )

    maneuvers = []
    for old_id, field in sorted((contract.get("maneuver_by_actor") or {}).items()):
        if str(old_id) in actor_map:
            actor_id = actor_map[str(old_id)]
            maneuvers.append({"actor_id": actor_id, "maneuver": _field_value(field, field_name="maneuver")})
            mutability.append({"field": f"maneuvers.{actor_id}", "class": _mutability(field)})

    events: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(contract.get("event_sequence") or [], key=lambda item: int(item.get("order", 0)))):
        actor = actor_map.get(str(row.get("actor_id"))) if row.get("actor_id") is not None else None
        target = actor_map.get(str(row.get("target_id"))) if row.get("target_id") is not None else None
        events.append(
            {
                "event_id": f"event_{index + 1:02d}",
                "order": int(row.get("order", index + 1)),
                "kind": _safe_scalar(row.get("kind"), field="event_kind"),
                "actor_id": actor,
                "target_id": target,
            }
        )
        mutability.append({"field": f"events.event_{index + 1:02d}", "class": _mutability(row.get("description"))})

    collisions = []
    for row in contract.get("collision_constraints") or []:
        collisions.append(
            {
                "actor_id": actor_map.get(str(row.get("actor_id"))),
                "target_id": actor_map.get(str(row.get("target_id"))),
                "required": bool(row.get("required")),
            }
        )
    if collisions:
        mutability.append(
            {
                "field": "collision_constraints",
                "class": "immutable" if all(str(row.get("provenance")) == "observed" for row in contract.get("collision_constraints") or []) else "review",
            }
        )

    contract_view = {
        "actors": sanitized_actors,
        "map": map_view,
        "topology": topology,
        "maneuvers": maneuvers,
        "events": events,
        "collision_constraints": collisions,
        "sensors": {
            "collision": bool((contract.get("sensors") or {}).get("collision")),
            "rgb": bool((contract.get("sensors") or {}).get("rgb")),
            "telemetry_hz": (contract.get("sensors") or {}).get("telemetry_hz"),
        },
        "fixed_delta_seconds": contract.get("fixed_delta_seconds"),
        "duration_seconds": contract.get("duration_seconds"),
    }
    visual_expectations = {
        "actors": visual_actors,
        "topology": topology,
        "maneuvers": maneuvers,
        "collision_constraints": collisions,
    }
    expected_constraints = {
        "topology": topology,
        "event_sequence": events,
        "collision_constraints": collisions,
    }
    return {
        "actor_map": actor_map,
        "contract": contract_view,
        "visual_expectations": visual_expectations,
        "expected_constraints": expected_constraints,
        "mutability_classes": sorted(mutability, key=lambda row: row["field"]),
    }


def _finite(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _round(value: Any, digits: int = 5) -> float | None:
    converted = _finite(value)
    return round(converted, digits) if converted is not None else None


def sanitize_telemetry(result: Mapping[str, Any], actor_map: Mapping[str, str]) -> dict[str, Any]:
    """Summarize numeric/event telemetry without verdicts or failure strings."""

    by_actor: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in result.get("actor_states") or []:
        neutral = actor_map.get(str(row.get("actor_id")))
        if neutral:
            by_actor[neutral].append(row)
    trajectories: list[dict[str, Any]] = []
    for actor_id in sorted(by_actor):
        rows = sorted(by_actor[actor_id], key=lambda row: (float(row.get("timestamp_seconds") or 0), int(row.get("frame") or 0)))
        speeds = [value for value in (_finite(row.get("speed_mps")) for row in rows) if value is not None]
        first, last = rows[0], rows[-1]
        trajectories.append(
            {
                "actor_id": actor_id,
                "samples": len(rows),
                "first_frame": int(first.get("frame") or 0),
                "last_frame": int(last.get("frame") or 0),
                "start_time_seconds": _round(first.get("timestamp_seconds")),
                "end_time_seconds": _round(last.get("timestamp_seconds")),
                "speed_mps": {
                    "first": _round(speeds[0]) if speeds else None,
                    "last": _round(speeds[-1]) if speeds else None,
                    "minimum": _round(min(speeds)) if speeds else None,
                    "maximum": _round(max(speeds)) if speeds else None,
                    "mean": _round(sum(speeds) / len(speeds)) if speeds else None,
                },
                "start_location": {axis: _round((first.get("location") or {}).get(axis)) for axis in ("x", "y", "z")},
                "end_location": {axis: _round((last.get("location") or {}).get(axis)) for axis in ("x", "y", "z")},
                "start_yaw_degrees": _round((first.get("rotation") or {}).get("yaw")),
                "end_yaw_degrees": _round((last.get("rotation") or {}).get("yaw")),
            }
        )

    collisions = []
    for row in result.get("collisions") or []:
        actor = actor_map.get(str(row.get("actor_id")))
        other = actor_map.get(str(row.get("other_actor_id")))
        if actor and other:
            impulse = row.get("impulse") or {}
            collisions.append(
                {
                    "frame": int(row.get("frame") or 0),
                    "actor_id": actor,
                    "other_actor_id": other,
                    "impulse_xyz": {axis: _round(impulse.get(axis)) for axis in ("x", "y", "z")},
                }
            )

    lane_observations: list[dict[str, Any]] = []
    event_observation: dict[str, Any] = {"first_state_frame": None, "first_collision_frame": None}
    for constraint in result.get("constraint_results") or []:
        name = str(constraint.get("name") or "")
        observed = constraint.get("observed")
        if name == "lane_topology_valid" and isinstance(observed, Mapping):
            for old_id, row in sorted((observed.get("lane_checks") or {}).items()):
                neutral = actor_map.get(str(old_id))
                if neutral and isinstance(row, Mapping):
                    lane_observations.append(
                        {
                            "actor_id": neutral,
                            "road_id": row.get("road_id"),
                            "lane_id": row.get("lane_id"),
                            "projected_distance_m": _round(row.get("projected_distance_m")),
                        }
                    )
        elif name == "event_order_valid" and isinstance(observed, Mapping):
            event_observation = {
                "first_state_frame": observed.get("first_state_frame"),
                "first_collision_frame": observed.get("first_collision_frame"),
            }

    return {
        "executed": bool(result.get("executed")),
        "trajectory_summaries": trajectories,
        "collision_events": collisions,
        "minimum_ttc_seconds": _round(result.get("minimum_ttc_seconds")),
        "lane_observations": lane_observations,
        "event_timing": event_observation,
    }


def build_method_payloads(
    *,
    contract_views: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    shuffled_telemetry: Mapping[str, Any],
    image_refs: Sequence[str],
    shuffled_image_refs: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Build the eight preregistered Task-A method budgets and assert unions."""

    visual = {"visual_expectations": contract_views["visual_expectations"]}
    telemetry_evidence = {
        "expected_constraints": contract_views["expected_constraints"],
        "telemetry_summary": telemetry,
    }
    fusion = {**visual, **telemetry_evidence}
    payloads = {
        "text_only_contract": {"evidence": {"contract": contract_views["contract"]}, "image_refs": []},
        "image_only_vlm": {"evidence": visual, "image_refs": list(image_refs)},
        "telemetry_only": {"evidence": telemetry_evidence, "image_refs": []},
        "telemetry_plus_image": {"evidence": fusion, "image_refs": list(image_refs)},
        "image_shuffled": {"evidence": visual, "image_refs": list(shuffled_image_refs)},
        "telemetry_shuffled": {
            "evidence": {
                "expected_constraints": contract_views["expected_constraints"],
                "telemetry_summary": shuffled_telemetry,
            },
            "image_refs": [],
        },
        "no_image_fusion_prompt": {"evidence": fusion, "image_refs": []},
        "guarded_blind_loop": {
            "evidence": {**fusion, "mutability_classes": contract_views["mutability_classes"]},
            "image_refs": list(image_refs),
        },
    }
    validate_method_payloads(payloads)
    return payloads


def validate_method_payloads(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    if set(payloads) != set(METHOD_CODES):
        raise ValueError(f"method set mismatch: {sorted(payloads)}")
    image = payloads["image_only_vlm"]
    telemetry = payloads["telemetry_only"]
    fusion = payloads["telemetry_plus_image"]
    expected_union = {**dict(image["evidence"]), **dict(telemetry["evidence"])}
    if fusion["evidence"] != expected_union:
        raise ValueError("fusion evidence is not the exact image/text plus telemetry union")
    if fusion["image_refs"] != image["image_refs"]:
        raise ValueError("fusion and image-only image budgets differ")
    if payloads["image_shuffled"]["evidence"] != image["evidence"]:
        raise ValueError("image-shuffled textual budget differs from image-only")
    if set(payloads["telemetry_shuffled"]["evidence"]) != set(telemetry["evidence"]):
        raise ValueError("telemetry-shuffled field budget differs from telemetry-only")
    no_image = payloads["no_image_fusion_prompt"]
    if no_image["evidence"] != fusion["evidence"] or no_image["image_refs"]:
        raise ValueError("no-image control is not fusion evidence with images removed")
    guarded = payloads["guarded_blind_loop"]
    if {key: value for key, value in guarded["evidence"].items() if key != "mutability_classes"} != fusion["evidence"]:
        raise ValueError("guarded budget adds information beyond mutability classes")
    if set(guarded["evidence"]) - set(fusion["evidence"]) != {"mutability_classes"}:
        raise ValueError("guarded budget must add exactly mutability_classes")
    if guarded["image_refs"] != fusion["image_refs"]:
        raise ValueError("guarded and fusion image budgets differ")


def build_request_template(
    *, opaque_artifact_id: str, evidence: Mapping[str, Any], image_refs: Sequence[str]
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": USER_INSTRUCTION
            + "\nArtifact ID: "
            + opaque_artifact_id
            + "\nEvidence JSON:\n"
            + canonical_json(evidence),
        }
    ]
    for ref in image_refs:
        content.append({"type": "image_url", "image_url": {"url": f"asset-ref://{ref}"}})
    return {
        "model": "MODEL_ID_PENDING_FREEZE",
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "blind_assurance_verdict",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
    }


def render_request(template: Mapping[str, Any], model_visible_root: Path) -> dict[str, Any]:
    """Replace opaque asset references with data URIs without changing evidence."""

    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, str) and value.startswith("asset-ref://"):
            relative = value[len("asset-ref://") :]
            path = (model_visible_root / relative).resolve()
            root = model_visible_root.resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"asset escapes model-visible root: {relative}") from exc
            if not path.is_file():
                raise FileNotFoundError(path)
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        return value

    return convert(template)


def recursive_forbidden_hits(value: Any, *, origin: str, path: str = "$") -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child = f"{path}.{key_text}"
            if _normalized_key(key_text) in FORBIDDEN_KEY_NORMALIZED:
                hits.append({"origin": origin, "path": child, "kind": "forbidden_key", "match": key_text})
            for name, pattern in FORBIDDEN_VALUE_PATTERNS:
                if pattern.search(key_text):
                    hits.append({"origin": origin, "path": child, "kind": name, "match": key_text})
            hits.extend(recursive_forbidden_hits(item, origin=origin, path=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            hits.extend(recursive_forbidden_hits(item, origin=origin, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        for name, pattern in FORBIDDEN_VALUE_PATTERNS:
            match = pattern.search(value)
            if match:
                hits.append({"origin": origin, "path": path, "kind": name, "match": match.group(0)})
    return hits


def forbidden_path_hits(relative_paths: Iterable[str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for value in relative_paths:
        match = FORBIDDEN_PATH_PATTERN.search(value.replace("\\", "/"))
        if match:
            hits.append({"origin": "model_visible_path", "path": value, "kind": "revealing_path", "match": match.group(0)})
    return hits


def summarize_budget_shape(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evidence_keys": sorted(payload["evidence"]),
        "image_count": len(payload["image_refs"]),
        "serialized_evidence_bytes": len(canonical_json(payload["evidence"]).encode("utf-8")),
    }

