from __future__ import annotations

import copy
import csv
import json
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field, model_validator

from .io import read_json, sha256_file, write_json, write_jsonl
from .models import (
    ConstraintResult,
    ExecutionStatus,
    ProvenanceValue,
    ScenarioContractV1,
    SimulationResultV1,
    StrictModel,
)


Topology = Literal[
    "rear_end",
    "intersection_crossing_turning",
    "lane_change_side_swipe",
    "head_on_centerline_intrusion",
]
SelectionRoute = Literal["rule", "qwen"]

TOPOLOGIES: tuple[str, ...] = (
    "rear_end",
    "intersection_crossing_turning",
    "lane_change_side_swipe",
    "head_on_centerline_intrusion",
)
SELECTION_ROUTES: tuple[str, ...] = ("rule", "qwen")
ASSURANCE_METHODS: tuple[str, ...] = (
    "deterministic_telemetry_only",
    "image_only_vlm",
    "telemetry_plus_vlm_no_repair",
    "guarded_bounded_repair",
    "unconstrained_self_refinement",
)
MAX_REPAIR_ITERATIONS = 3

# Three repairable and three evidence-conflicting faults give the preregistered
# 72/72 split over 24 cases. Collision omission is induced through a mutable
# causal speed/pose field; the observed collision requirement itself is never
# edited.
FAULT_DEFINITIONS: dict[str, dict[str, str]] = {
    "actor_target_swap": {
        "fault_class": "immutable",
        "expected_disposition": "manual_review",
        "injection_layer": "telemetry",
    },
    "required_collision_omission": {
        "fault_class": "mutable",
        "expected_disposition": "repair",
        "injection_layer": "contract_then_rerun",
    },
    "event_order_violation": {
        "fault_class": "immutable",
        "expected_disposition": "manual_review",
        "injection_layer": "telemetry",
    },
    "speed_pose_perturbation": {
        "fault_class": "mutable",
        "expected_disposition": "repair",
        "injection_layer": "contract_then_rerun",
    },
    "map_lane_mismatch": {
        "fault_class": "mutable",
        "expected_disposition": "repair",
        "injection_layer": "contract_then_rerun",
    },
    "mismatched_keyframes": {
        "fault_class": "immutable",
        "expected_disposition": "manual_review",
        "injection_layer": "keyframe_binding",
    },
}


class PreregisteredCase(StrictModel):
    slot_id: str
    candidate_id: int | None = None
    scenario_id: str | None = None
    topology: Topology
    source_stage: SelectionRoute
    topology_confirmed: bool = False
    human_topology_confirmed: bool = False
    confirmation_basis: str | None = None
    contract_path: str | None = None
    clean_bundle_path: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_identifiers(self) -> "PreregisteredCase":
        if self.candidate_id is not None:
            expected = f"jurisdrive_{self.candidate_id}"
            if self.scenario_id is None:
                raise ValueError("scenario_id is required when candidate_id is set")
            if self.scenario_id != expected:
                raise ValueError(
                    f"scenario_id must be {expected!r} for candidate_id {self.candidate_id}"
                )
        elif self.scenario_id is not None:
            raise ValueError("candidate_id is required when scenario_id is set")
        return self


class ExperimentPreregistration(StrictModel):
    version: Literal["1.0"] = "1.0"
    experiment_id: str
    selection_frozen: bool = False
    frozen_at_utc: str | None = None
    base_seed: int = Field(default=20260823, ge=0)
    max_repair_iterations: Literal[3] = 3
    cases: list[PreregisteredCase]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_design(self) -> "ExperimentPreregistration":
        if len(self.cases) != 24:
            raise ValueError("the fidelity cohort must contain exactly 24 slots")
        slot_ids = [case.slot_id for case in self.cases]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("slot_id values must be unique")

        topology_counts = Counter(case.topology for case in self.cases)
        if topology_counts != Counter({topology: 6 for topology in TOPOLOGIES}):
            raise ValueError("each of the four topologies must contain exactly six cases")
        route_counts = Counter(case.source_stage for case in self.cases)
        if route_counts != Counter({route: 12 for route in SELECTION_ROUTES}):
            raise ValueError("the cohort must contain 12 rule and 12 Qwen cases")
        for topology in TOPOLOGIES:
            stratum = Counter(
                case.source_stage for case in self.cases if case.topology == topology
            )
            if stratum != Counter({route: 3 for route in SELECTION_ROUTES}):
                raise ValueError(
                    f"topology {topology!r} must contain three rule and three Qwen cases"
                )

        selected = [case for case in self.cases if case.candidate_id is not None]
        candidate_ids = [case.candidate_id for case in selected]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_id values must be unique")
        if self.selection_frozen:
            if not self.frozen_at_utc:
                raise ValueError("frozen_at_utc is required for a frozen selection")
            incomplete = [
                case.slot_id
                for case in self.cases
                if case.candidate_id is None
                or not (case.topology_confirmed or case.human_topology_confirmed)
                or not case.contract_path
            ]
            if incomplete:
                raise ValueError(
                    "frozen selection has incomplete/unconfirmed slots: "
                    + ", ".join(incomplete)
                )
        return self


def load_preregistration(path: Path) -> ExperimentPreregistration:
    return ExperimentPreregistration.model_validate(read_json(path))


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _case_blockers(
    case: PreregisteredCase,
    base_dir: Path,
    *,
    require_clean_bundle: bool = False,
) -> list[str]:
    blockers: list[str] = []
    if case.candidate_id is None or case.scenario_id is None:
        blockers.append("candidate_not_selected")
    if not (case.topology_confirmed or case.human_topology_confirmed):
        blockers.append("topology_not_confirmed")
    contract = _resolve_path(case.contract_path, base_dir)
    if contract is None or not contract.is_file():
        blockers.append("contract_missing")
    if require_clean_bundle:
        bundle = _resolve_path(case.clean_bundle_path, base_dir)
        if bundle is None or not bundle.is_dir():
            blockers.append("clean_bundle_missing")
    return blockers


def build_fidelity_schedule(
    preregistration: ExperimentPreregistration,
    *,
    config_dir: Path = Path("."),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_index = 0
    for case_index, case in enumerate(preregistration.cases):
        blockers = _case_blockers(case, config_dir)
        contract = _resolve_path(case.contract_path, config_dir)
        contract_sha256 = sha256_file(contract) if contract and contract.is_file() else None
        for seed_index in range(2):
            seed = preregistration.base_seed + case_index * 10 + seed_index
            for repeat_index in range(1, 3):
                run_index += 1
                rows.append(
                    {
                        "experiment_id": preregistration.experiment_id,
                        "denominator": "unique24_fidelity_96_runs",
                        "run_index": run_index,
                        "run_id": f"fid_{case.slot_id}_s{seed_index + 1}_r{repeat_index}",
                        "slot_id": case.slot_id,
                        "candidate_id": case.candidate_id,
                        "scenario_id": case.scenario_id,
                        "topology": case.topology,
                        "source_stage": case.source_stage,
                        "seed_index": seed_index + 1,
                        "seed": seed,
                        "repeat_index": repeat_index,
                        "contract_path": case.contract_path,
                        "contract_sha256": contract_sha256,
                        "ready_for_execution": not blockers,
                        "blockers": blockers,
                        "execution_status": "not_executed",
                        "contract_compile_pass": None,
                        "carla_launch_complete": None,
                        "run_complete": None,
                        "actor_target_correct": None,
                        "lane_topology_valid": None,
                        "event_order_valid": None,
                        "minimum_ttc_seconds": None,
                        "impact_relative_speed_mps": None,
                        "hard_constraint_pass": None,
                        "collision_signature": None,
                        "telemetry_sha256": None,
                        "map_asset_fallback": None,
                        "failure_reason": None,
                    }
                )
    if len(rows) != 96:
        raise AssertionError("fidelity schedule must contain 96 runs")
    return rows


def build_fault_plan(
    preregistration: ExperimentPreregistration,
    *,
    config_dir: Path = Path("."),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(preregistration.cases):
        blockers = _case_blockers(case, config_dir, require_clean_bundle=True)
        donor_case = preregistration.cases[(case_index + 1) % len(preregistration.cases)]
        donor_bundle_path = donor_case.clean_bundle_path
        common = {
            "experiment_id": preregistration.experiment_id,
            "denominator": "n6_clean24_fault144",
            "slot_id": case.slot_id,
            "candidate_id": case.candidate_id,
            "scenario_id": case.scenario_id,
            "topology": case.topology,
            "source_stage": case.source_stage,
            "clean_bundle_path": case.clean_bundle_path,
            "donor_bundle_path": None,
            "max_repair_iterations": preregistration.max_repair_iterations,
        }
        rows.append(
            {
                **common,
                "trial_id": f"ctrl_{case.slot_id}",
                "trial_kind": "clean_control",
                "fault_type": None,
                "fault_class": None,
                "expected_disposition": "pass",
                "injection_layer": None,
                "variant": None,
                "ready_for_materialization": not blockers,
                "blockers": blockers,
                "execution_status": "not_executed",
                "injection_verified": None,
            }
        )
        for fault_type, definition in FAULT_DEFINITIONS.items():
            variant = None
            if fault_type == "speed_pose_perturbation":
                variant = "speed" if case_index % 2 == 0 else "pose"
            fault_blockers = list(blockers)
            donor_path = donor_bundle_path if fault_type == "mismatched_keyframes" else None
            if not fault_blockers:
                contract_path = _resolve_path(case.contract_path, config_dir)
                bundle_path = _resolve_path(case.clean_bundle_path, config_dir)
                donor_bundle = _resolve_path(donor_path, config_dir)
                assert contract_path is not None and bundle_path is not None
                fault_blockers.extend(
                    fault_preflight_blockers(
                        contract_path,
                        bundle_path,
                        fault_type,
                        variant=variant,
                        donor_bundle=donor_bundle,
                    )
                )
            rows.append(
                {
                    **common,
                    "trial_id": f"fault_{case.slot_id}_{fault_type}",
                    "trial_kind": "fault",
                    "fault_type": fault_type,
                    "fault_class": definition["fault_class"],
                    "expected_disposition": definition["expected_disposition"],
                    "injection_layer": definition["injection_layer"],
                    "variant": variant,
                    "donor_bundle_path": donor_path,
                    "ready_for_materialization": not fault_blockers,
                    "blockers": fault_blockers,
                    "execution_status": "not_executed",
                    "injection_verified": None,
                }
            )
    faults = [row for row in rows if row["trial_kind"] == "fault"]
    classes = Counter(row["fault_class"] for row in faults)
    if len(rows) != 168 or len(faults) != 144:
        raise AssertionError("fault plan must contain 24 controls and 144 faults")
    if classes != Counter({"mutable": 72, "immutable": 72}):
        raise AssertionError("fault plan must contain 72 mutable and 72 immutable faults")
    return rows


def build_assurance_evaluation_schedule(
    fault_plan: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in fault_plan:
        for method in ASSURANCE_METHODS:
            rows.append(
                {
                    **trial,
                    "evaluation_id": f"{trial['trial_id']}__{method}",
                    "method": method,
                    "execution_status": "not_executed",
                    "detected": None,
                    "repair_triggered": None,
                    "repair_iterations": None,
                    "post_repair_passed": None,
                    "post_repair_regression": None,
                    "immutable_edit_attempted": None,
                    "immutable_edit_rejected": None,
                    "manual_review": None,
                }
            )
    return rows


def fault_preflight_blockers(
    contract_path: Path,
    clean_bundle: Path,
    fault_type: str,
    *,
    variant: str | None = None,
    donor_bundle: Path | None = None,
) -> list[str]:
    """Return materialization blockers without changing any artifact."""
    if fault_type not in FAULT_DEFINITIONS:
        return ["unknown_fault_type"]
    try:
        contract = ScenarioContractV1.model_validate(read_json(contract_path))
    except Exception:
        return ["contract_invalid"]
    data = contract.model_dump(mode="json")
    blockers: list[str] = []
    if fault_type == "required_collision_omission":
        if not (
            _mutable_actor_path(data, "lane_position", role="ego")
            or _mutable_actor_path(data, "lane_position")
            or _mutable_actor_path(data, "initial_speed_mps", role="ego")
            or _mutable_actor_path(data, "initial_speed_mps")
        ):
            blockers.append("mutable_collision_causal_field_unavailable")
    elif fault_type == "speed_pose_perturbation":
        selected_variant = variant or "speed"
        field = "initial_speed_mps" if selected_variant == "speed" else "lane_position"
        if not (
            _mutable_actor_path(data, field, role="ego")
            or _mutable_actor_path(data, field)
        ):
            blockers.append(f"mutable_{selected_variant}_unavailable")
    elif fault_type == "map_lane_mismatch":
        if not _field_is_mutable(data, "map_binding.carla_map") and not _mutable_actor_path(
            data, "lane_position"
        ):
            blockers.append("mutable_map_or_lane_unavailable")
    else:
        try:
            result = _load_executed_result(clean_bundle)
        except Exception:
            blockers.append("executed_clean_result_missing_or_invalid")
            result = None
        if fault_type == "actor_target_swap" and not contract.collision_constraints:
            blockers.append("collision_constraint_missing")
        if fault_type == "event_order_violation" and len(contract.event_sequence) < 2:
            has_runtime_order = any(
                item.name == "event_order_valid" for item in (result.constraint_results if result else [])
            )
            if not has_runtime_order:
                blockers.append("grounded_or_runtime_event_pair_missing")
        if fault_type == "mismatched_keyframes":
            if donor_bundle is None:
                blockers.append("donor_bundle_unassigned")
            else:
                try:
                    donor = _load_executed_result(donor_bundle)
                    if donor.scenario_id == contract.scenario_id:
                        blockers.append("donor_scenario_not_distinct")
                    if not donor.keyframes:
                        blockers.append("donor_keyframes_missing")
                except Exception:
                    blockers.append("donor_result_missing_or_invalid")
            if result is not None and not result.keyframes:
                blockers.append("source_keyframes_missing")
        elif result is not None and not result.keyframes:
            blockers.append("source_keyframes_missing")
        if result is not None and result.keyframes:
            try:
                _keyframe_source_paths(result.model_dump(mode="json"), clean_bundle)
            except FileNotFoundError:
                blockers.append("source_keyframe_file_missing")
    return list(dict.fromkeys(blockers))


def write_experiment_plan(
    config_path: Path,
    output_dir: Path,
    *,
    allow_pending: bool = False,
) -> dict[str, Any]:
    preregistration = load_preregistration(config_path)
    if not preregistration.selection_frozen and not allow_pending:
        raise ValueError(
            "selection is not frozen; pass allow_pending only to create a non-executable draft"
        )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {output_dir}")
    fidelity = build_fidelity_schedule(
        preregistration, config_dir=config_path.resolve().parent
    )
    faults = build_fault_plan(preregistration, config_dir=config_path.resolve().parent)
    evaluations = build_assurance_evaluation_schedule(faults)
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "fidelity_schedule.jsonl", fidelity)
    write_jsonl(output_dir / "fault_plan.jsonl", faults)
    write_jsonl(output_dir / "assurance_evaluation_schedule.jsonl", evaluations)
    manifest = {
        "version": "1.0",
        "experiment_id": preregistration.experiment_id,
        "selection_frozen": preregistration.selection_frozen,
        "execution_authorized": preregistration.selection_frozen
        and all(row["ready_for_execution"] for row in fidelity),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "denominators": {
            "unique_scenarios": 24,
            "fidelity_runs": 96,
            "clean_controls": 24,
            "fault_trials": 144,
            "mutable_faults": 72,
            "immutable_faults": 72,
            "assurance_methods": len(ASSURANCE_METHODS),
            "assurance_method_evaluations": 168 * len(ASSURANCE_METHODS),
            "max_repair_iterations": preregistration.max_repair_iterations,
        },
        "not_claimed": [
            "No row in this plan is evidence of CARLA execution.",
            "Pending or blocked rows must not enter metric denominators.",
            "The 200-run repeated runtime benchmark is a separate experiment.",
        ],
        "blocked_fidelity_rows": sum(not row["ready_for_execution"] for row in fidelity),
        "blocked_fault_rows": sum(
            not row["ready_for_materialization"] for row in faults
        ),
        "blocked_assurance_evaluation_rows": sum(
            not row["ready_for_materialization"] for row in evaluations
        ),
    }
    write_json(output_dir / "experiment_manifest.json", manifest)
    return manifest


def _field_is_mutable(contract_data: dict[str, Any], path: str) -> bool:
    current: Any = contract_data
    for part in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return False
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    return isinstance(current, dict) and current.get("provenance") in {
        ProvenanceValue.INFERRED.value,
        ProvenanceValue.DEFAULTED.value,
    }


def _get_path(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _set_field_value(data: dict[str, Any], path: str, value: Any) -> Any:
    if not _field_is_mutable(data, path):
        raise ValueError(f"fault target is not inferred/defaulted: {path}")
    target = _get_path(data, path)
    original = copy.deepcopy(target["value"])
    target["value"] = value
    return original


def _mutable_actor_path(
    contract_data: dict[str, Any], field: str, *, role: str | None = None
) -> str | None:
    for index, actor in enumerate(contract_data.get("actors", [])):
        if role and actor.get("role") != role:
            continue
        path = f"actors.{index}.{field}"
        if _field_is_mutable(contract_data, path):
            return path
    return None


def _mutable_collision_actor_path(contract_data: dict[str, Any], field: str) -> str | None:
    constraints = contract_data.get("collision_constraints") or []
    if not constraints:
        return None
    pair_ids = (constraints[0].get("actor_id"), constraints[0].get("target_id"))
    for pair_id in pair_ids:
        for index, actor in enumerate(contract_data.get("actors", [])):
            if actor.get("id") != pair_id:
                continue
            path = f"actors.{index}.{field}"
            if _field_is_mutable(contract_data, path):
                return path
    return None


def _replace_constraint(
    result_data: dict[str, Any],
    *,
    name: str,
    expected: Any,
    observed: Any,
    reason: str,
) -> None:
    rows = [row for row in result_data.get("constraint_results", []) if row["name"] != name]
    rows.append(
        ConstraintResult(
            name=name,
            passed=False,
            expected=expected,
            observed=observed,
            reason=reason,
        ).model_dump(mode="json")
    )
    result_data["constraint_results"] = rows
    result_data["status"] = ExecutionStatus.FAILED.value


def _load_executed_result(bundle: Path) -> SimulationResultV1:
    path = bundle / "simulation_result.json"
    if not path.is_file():
        raise FileNotFoundError(f"executed simulation_result.json is required: {path}")
    result = SimulationResultV1.model_validate(read_json(path))
    if not result.executed:
        raise ValueError("fault injection requires an executed clean result")
    return result


def _copy_keyframes(
    result_data: dict[str, Any],
    source_bundle: Path,
    output_dir: Path,
    *,
    prefix: str,
) -> list[str]:
    copied: list[str] = []
    for index, source in enumerate(_keyframe_source_paths(result_data, source_bundle)):
        destination = output_dir / "keyframes" / f"{prefix}_{index:02d}{source.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(output_dir)).replace("\\", "/"))
    return copied


def _keyframe_source_paths(
    result_data: dict[str, Any], source_bundle: Path
) -> list[Path]:
    sources: list[Path] = []
    for value in result_data.get("keyframes") or []:
        source = Path(str(value))
        if not source.is_absolute():
            source = source_bundle / source
        if not source.is_file():
            raise FileNotFoundError(f"keyframe is unavailable: {source}")
        sources.append(source)
    return sources


def _write_checksums(output_dir: Path) -> None:
    paths = sorted(path for path in output_dir.rglob("*") if path.is_file())
    lines = [
        f"{sha256_file(path)}  {str(path.relative_to(output_dir)).replace(chr(92), '/')}"
        for path in paths
        if path.name != "checksums.sha256"
    ]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_fault_bundle(
    source_bundle: Path,
    output_dir: Path,
    fault_type: str,
    *,
    donor_bundle: Path | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    if fault_type not in FAULT_DEFINITIONS:
        raise ValueError(f"unknown fault type: {fault_type}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fault bundle: {output_dir}")
    contract_path = source_bundle / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"contract.json is required: {contract_path}")
    contract = ScenarioContractV1.model_validate(read_json(contract_path))
    contract_data = contract.model_dump(mode="json")
    definition = FAULT_DEFINITIONS[fault_type]
    mutation: dict[str, Any] = {}
    active_result: dict[str, Any] | None = None
    oracle_result: dict[str, Any] | None = None
    requires_carla_rerun = definition["injection_layer"] == "contract_then_rerun"

    if fault_type == "required_collision_omission":
        path = _mutable_collision_actor_path(contract_data, "lane_position")
        path = path or _mutable_actor_path(contract_data, "lane_position", role="ego")
        path = path or _mutable_actor_path(contract_data, "lane_position")
        path = path or _mutable_actor_path(contract_data, "initial_speed_mps", role="ego")
        path = path or _mutable_actor_path(contract_data, "initial_speed_mps")
        if not path:
            raise ValueError("no mutable causal field is available to induce collision omission")
        original = _get_path(contract_data, path)["value"]
        injected = "collision_omission_fault" if path.endswith("lane_position") else 0.0
        _set_field_value(contract_data, path, injected)
        mutation = {
            "path": path,
            "oracle_value": original,
            "injected_value": injected,
            "verification_gate": "rerun must confirm that the required collision is absent",
        }
    elif fault_type == "speed_pose_perturbation":
        variant = variant or "speed"
        if variant == "speed":
            path = _mutable_collision_actor_path(contract_data, "initial_speed_mps")
            path = path or _mutable_actor_path(contract_data, "initial_speed_mps", role="ego")
            path = path or _mutable_actor_path(contract_data, "initial_speed_mps")
            if not path:
                raise ValueError("no mutable speed field is available")
            original = _get_path(contract_data, path)["value"]
            if not isinstance(original, (int, float)):
                raise ValueError(f"speed fault requires a numeric value at {path}")
            injected = float(original) + max(5.0, abs(float(original)) * 0.5)
        elif variant == "pose":
            path = _mutable_collision_actor_path(contract_data, "lane_position")
            path = path or _mutable_actor_path(contract_data, "lane_position", role="ego")
            path = path or _mutable_actor_path(contract_data, "lane_position")
            if not path:
                raise ValueError("no mutable lane-position field is available")
            original = _get_path(contract_data, path)["value"]
            injected = "pose_perturbation_fault"
        else:
            raise ValueError("speed_pose_perturbation variant must be speed or pose")
        _set_field_value(contract_data, path, injected)
        mutation = {
            "path": path,
            "oracle_value": original,
            "injected_value": injected,
            "variant": variant,
            "verification_gate": "CARLA rerun and contract-to-telemetry check required",
        }
    elif fault_type == "map_lane_mismatch":
        path = _mutable_collision_actor_path(contract_data, "lane_position") or ""
        if not path and _field_is_mutable(contract_data, "map_binding.carla_map"):
            path = "map_binding.carla_map"
        if not path:
            raise ValueError("no mutable map or lane field is available")
        original = _get_path(contract_data, path)["value"]
        if path == "map_binding.carla_map":
            injected = {"Town01": "Town05", "Town05": "Town04"}.get(
                str(original), "Town01"
            )
        else:
            injected = "map_lane_mismatch_fault"
        _set_field_value(contract_data, path, injected)
        mutation = {
            "path": path,
            "oracle_value": original,
            "injected_value": injected,
            "verification_gate": "selected map/lane asset and topology must be validated before run",
        }
    else:
        result = _load_executed_result(source_bundle)
        oracle_result = result.model_dump(mode="json")
        active_result = copy.deepcopy(oracle_result)
        constraints = contract_data.get("collision_constraints", [])
        if fault_type == "actor_target_swap":
            if not constraints:
                raise ValueError("actor-target swap requires a collision constraint")
            expected = {
                "actor_id": constraints[0]["actor_id"],
                "target_id": constraints[0]["target_id"],
            }
            observed = {
                "actor_id": expected["target_id"],
                "target_id": expected["actor_id"],
            }
            for collision in active_result.get("collisions") or []:
                if {
                    collision.get("actor_id"),
                    collision.get("other_actor_id"),
                } == {expected["actor_id"], expected["target_id"]}:
                    collision["actor_id"], collision["other_actor_id"] = (
                        collision["other_actor_id"],
                        collision["actor_id"],
                    )
            _replace_constraint(
                active_result,
                name="collision_target",
                expected=expected,
                observed=observed,
                reason="controlled actor-target swap fault",
            )
            mutation = {
                "path": "collision_constraints.0",
                "oracle_value": expected,
                "injected_value": observed,
            }
        elif fault_type == "event_order_violation":
            events = sorted(contract_data.get("event_sequence", []), key=lambda row: row["order"])
            expected = (
                [row["id"] for row in events]
                if len(events) >= 2
                else ["runtime_initial_state", "runtime_required_collision"]
            )
            observed = list(reversed(expected))
            _replace_constraint(
                active_result,
                name="event_order",
                expected=expected,
                observed=observed,
                reason="controlled event-order reversal fault",
            )
            mutation = {
                "path": "event_sequence",
                "oracle_value": expected,
                "injected_value": observed,
                "basis": "grounded_contract_events" if len(events) >= 2 else "runtime_event_order_constraint",
            }
        elif fault_type == "mismatched_keyframes":
            if donor_bundle is None:
                raise ValueError("mismatched keyframes require a donor bundle")
            donor = _load_executed_result(donor_bundle)
            if donor.scenario_id == result.scenario_id:
                raise ValueError("keyframe donor must be a different scenario")
            mutation = {
                "path": "simulation_result.keyframes",
                "oracle_value": list(active_result.get("keyframes") or []),
                "donor_scenario_id": donor.scenario_id,
            }
        else:
            raise AssertionError("unreachable")

    # Validate mutations before making the output directory. Immutable-layer
    # result faults are also schema validated; no synthetic dry-run metrics pass.
    faulty_contract = ScenarioContractV1.model_validate(contract_data)
    if active_result is not None and fault_type != "mismatched_keyframes":
        active_result = SimulationResultV1.model_validate(active_result).model_dump(mode="json")
        if not active_result.get("keyframes"):
            raise ValueError("immutable fault materialization requires clean keyframes")
        _keyframe_source_paths(active_result, source_bundle)
    if fault_type == "mismatched_keyframes":
        assert donor_bundle is not None
        donor_result = _load_executed_result(donor_bundle).model_dump(mode="json")
        if not donor_result.get("keyframes"):
            raise ValueError("mismatched keyframes require donor keyframes")
        _keyframe_source_paths(donor_result, donor_bundle)

    output_dir.mkdir(parents=True)
    write_json(output_dir / "oracle_contract.json", contract)
    write_json(output_dir / "contract.json", faulty_contract)
    if oracle_result is not None:
        write_json(output_dir / "oracle_simulation_result.json", oracle_result)
    if active_result is not None:
        if fault_type == "mismatched_keyframes":
            assert donor_bundle is not None
            donor = donor_result
            active_result["keyframes"] = _copy_keyframes(
                donor, donor_bundle, output_dir, prefix="donor"
            )
            mutation["injected_value"] = list(active_result["keyframes"])
        else:
            active_result["keyframes"] = _copy_keyframes(
                active_result, source_bundle, output_dir, prefix="source"
            )
        active_result = SimulationResultV1.model_validate(active_result).model_dump(
            mode="json"
        )
        write_json(output_dir / "simulation_result.json", active_result)
    evidence_graph = source_bundle / "evidence_graph.json"
    if evidence_graph.is_file():
        shutil.copy2(evidence_graph, output_dir / "evidence_graph.json")
    manifest = {
        "version": "1.0",
        "scenario_id": contract.scenario_id,
        "fault_type": fault_type,
        **definition,
        "variant": variant,
        "source_bundle": str(source_bundle.resolve()),
        "source_contract_sha256": sha256_file(contract_path),
        "requires_carla_rerun": requires_carla_rerun,
        "max_repair_iterations": MAX_REPAIR_ITERATIONS,
        "injection_verified": False if requires_carla_rerun else True,
        "mutation": mutation,
        "status": "awaiting_carla_rerun" if requires_carla_rerun else "materialized",
        "not_claimed": "Materialization alone is not detection or repair success.",
    }
    write_json(output_dir / "fault_manifest.json", manifest)
    _write_checksums(output_dir)
    return manifest


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, Any]:
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_acceptance_rate": _rate(fn, tp + fn),
        "false_rejection_rate": _rate(fp, tn + fp),
    }


def summarize_assurance_records(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    all_rows = list(rows)
    eligible = [
        row
        for row in all_rows
        if row.get("execution_status") == "completed"
        and row.get("injection_verified", True)
    ]
    tp = tn = fp = fn = 0
    for row in eligible:
        actual_fault = row.get("trial_kind") == "fault"
        detected = bool(row.get("detected"))
        if actual_fault and detected:
            tp += 1
        elif actual_fault:
            fn += 1
        elif detected:
            fp += 1
        else:
            tn += 1
    mutable = [row for row in eligible if row.get("fault_class") == "mutable"]
    immutable = [row for row in eligible if row.get("fault_class") == "immutable"]
    triggered = [row for row in mutable if row.get("repair_triggered")]
    iterations = [
        int(row["repair_iterations"])
        for row in eligible
        if row.get("repair_iterations") is not None
    ]
    by_fault: dict[str, dict[str, Any]] = {}
    for fault_type in FAULT_DEFINITIONS:
        stratum = [row for row in eligible if row.get("fault_type") == fault_type]
        by_fault[fault_type] = {
            "n": len(stratum),
            "detected": sum(bool(row.get("detected")) for row in stratum),
            "detection_rate": _rate(
                sum(bool(row.get("detected")) for row in stratum), len(stratum)
            ),
            "manual_review": sum(bool(row.get("manual_review")) for row in stratum),
        }
    by_method: dict[str, dict[str, Any]] = {}
    for method in sorted({str(row.get("method") or "unspecified") for row in eligible}):
        stratum = [
            row for row in eligible if str(row.get("method") or "unspecified") == method
        ]
        method_tp = sum(
            row.get("trial_kind") == "fault" and bool(row.get("detected"))
            for row in stratum
        )
        method_fn = sum(
            row.get("trial_kind") == "fault" and not bool(row.get("detected"))
            for row in stratum
        )
        method_fp = sum(
            row.get("trial_kind") != "fault" and bool(row.get("detected"))
            for row in stratum
        )
        method_tn = sum(
            row.get("trial_kind") != "fault" and not bool(row.get("detected"))
            for row in stratum
        )
        by_method[method] = {
            "n": len(stratum),
            **_binary_metrics(method_tp, method_tn, method_fp, method_fn),
            "manual_review_rate": _rate(
                sum(bool(row.get("manual_review")) for row in stratum), len(stratum)
            ),
        }
    return {
        "version": "1.0",
        "denominator": "completed_and_injection_verified_trials",
        "planned_rows": len(all_rows),
        "eligible_rows": len(eligible),
        "excluded_rows": len(all_rows) - len(eligible),
        **_binary_metrics(tp, tn, fp, fn),
        "repair": {
            "mutable_faults": len(mutable),
            "triggered": len(triggered),
            "trigger_rate": _rate(len(triggered), len(mutable)),
            "post_repair_passed": sum(
                bool(row.get("post_repair_passed")) for row in triggered
            ),
            "post_repair_pass_rate": _rate(
                sum(bool(row.get("post_repair_passed")) for row in triggered),
                len(triggered),
            ),
            "mean_iterations": statistics.fmean(iterations) if iterations else None,
            "post_repair_regressions": sum(
                bool(row.get("post_repair_regression")) for row in eligible
            ),
        },
        "guard": {
            "immutable_faults": len(immutable),
            "immutable_edit_attempts": sum(
                bool(row.get("immutable_edit_attempted")) for row in immutable
            ),
            "immutable_edits_rejected": sum(
                bool(row.get("immutable_edit_rejected")) for row in immutable
            ),
            "manual_review": sum(bool(row.get("manual_review")) for row in eligible),
            "manual_review_rate": _rate(
                sum(bool(row.get("manual_review")) for row in eligible), len(eligible)
            ),
        },
        "by_fault_type": by_fault,
        "by_method": by_method,
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    rows = [float(value) for value in values if value is not None]
    return statistics.fmean(rows) if rows else None


def summarize_fidelity_records(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    all_rows = list(rows)
    completed = [row for row in all_rows if row.get("execution_status") == "completed"]
    binary_fields = (
        "contract_compile_pass",
        "carla_launch_complete",
        "run_complete",
        "actor_target_correct",
        "lane_topology_valid",
        "event_order_valid",
        "hard_constraint_pass",
    )
    rates = {
        field: {
            "passed": sum(row.get(field) is True for row in completed),
            "assessed": sum(row.get(field) is not None for row in completed),
            "rate": _rate(
                sum(row.get(field) is True for row in completed),
                sum(row.get(field) is not None for row in completed),
            ),
        }
        for field in binary_fields
    }
    pairs: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        if row.get("slot_id") is not None and row.get("seed") is not None:
            pairs[(str(row["slot_id"]), int(row["seed"]))].append(row)
    complete_pairs = [pair for pair in pairs.values() if len(pair) == 2]
    metric_fields = (
        "actor_target_correct",
        "lane_topology_valid",
        "event_order_valid",
        "collision_signature",
    )
    metric_exact = sum(
        all(pair[0].get(field) == pair[1].get(field) for field in metric_fields)
        for pair in complete_pairs
    )
    telemetry_exact = sum(
        bool(pair[0].get("telemetry_sha256"))
        and pair[0].get("telemetry_sha256") == pair[1].get("telemetry_sha256")
        for pair in complete_pairs
    )
    failures = Counter(
        str(row.get("failure_reason"))
        for row in completed
        if row.get("failure_reason")
    )

    def strata(field: str) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for value in sorted({str(row.get(field) or "unspecified") for row in completed}):
            stratum = [
                row for row in completed if str(row.get(field) or "unspecified") == value
            ]
            full_pass = sum(
                all(row.get(metric) is True for metric in binary_fields) for row in stratum
            )
            output[value] = {
                "n": len(stratum),
                "unique_scenarios": len(
                    {row.get("scenario_id") for row in stratum if row.get("scenario_id")}
                ),
                "all_assessed_checks_passed": full_pass,
                "all_assessed_checks_pass_rate": _rate(full_pass, len(stratum)),
                "hard_constraint_pass_rate": _rate(
                    sum(row.get("hard_constraint_pass") is True for row in stratum),
                    sum(row.get("hard_constraint_pass") is not None for row in stratum),
                ),
            }
        return output

    return {
        "version": "1.0",
        "denominator": "completed_unique24_fidelity_runs",
        "planned_runs": len(all_rows),
        "completed_runs": len(completed),
        "unique_scenarios_completed": len(
            {row.get("scenario_id") for row in completed if row.get("scenario_id")}
        ),
        "rates": rates,
        "minimum_ttc_seconds_mean": _mean(
            row.get("minimum_ttc_seconds") for row in completed
        ),
        "impact_relative_speed_mps_mean": _mean(
            row.get("impact_relative_speed_mps") for row in completed
        ),
        "replay": {
            "complete_same_seed_pairs": len(complete_pairs),
            "exact_core_metric_pairs": metric_exact,
            "exact_core_metric_rate": _rate(metric_exact, len(complete_pairs)),
            "exact_telemetry_hash_pairs": telemetry_exact,
            "exact_telemetry_hash_rate": _rate(telemetry_exact, len(complete_pairs)),
        },
        "map_asset_fallback_runs": sum(
            bool(row.get("map_asset_fallback")) for row in completed
        ),
        "by_topology": strata("topology"),
        "by_source_stage": strata("source_stage"),
        "failure_taxonomy": dict(sorted(failures.items())),
    }


def write_summary_tables(output_dir: Path, summary: dict[str, Any], name: str) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite summary directory: {output_dir}")
    output_dir.mkdir(parents=True)
    write_json(output_dir / f"{name}_summary.json", summary)
    rows: list[dict[str, Any]] = []

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(f"{prefix}.{key}" if prefix else str(key), item)
        elif not isinstance(value, list):
            rows.append({"measure": prefix, "value": value})

    flatten("", summary)
    with (output_dir / f"{name}_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("measure", "value"))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows
