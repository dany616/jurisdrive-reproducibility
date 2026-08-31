from __future__ import annotations

import copy
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Protocol
from urllib import error, request

from .models import (
    EvaluationFailure,
    EvaluationReport,
    ProvenanceValue,
    ScenarioContractV1,
    SimulationResultV1,
)
from .simulator import SimulatorBackend


class Evaluator(Protocol):
    name: str

    def evaluate(
        self,
        contract: ScenarioContractV1,
        result: SimulationResultV1,
    ) -> EvaluationReport: ...


class MockEvaluator:
    name = "mock"

    def evaluate(
        self,
        contract: ScenarioContractV1,
        result: SimulationResultV1,
    ) -> EvaluationReport:
        failures = [
            EvaluationFailure(
                attribute=item.name,
                expected=item.expected,
                observed=item.observed,
                evidence=item.reason,
                repair_instruction=None,
            )
            for item in result.constraint_results
            if item.passed is False
        ]
        return EvaluationReport(
            scenario_id=contract.scenario_id,
            evaluator=self.name,
            passed=False if failures else None,
            failures=failures,
            manual_review=bool(failures or contract.review_issues),
            notes=[
                "MockEvaluator reports static/dry-run failures only.",
                "No VLM or simulation fidelity judgment was performed.",
            ],
        )


class VlmEvaluator:
    name = "vlm"

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: float = 180.0,
        *,
        bundle_dir: Path | None = None,
        keyframes: list[Path] | None = None,
        include_telemetry: bool = True,
        enforce_deterministic: bool = True,
        enforce_provenance_guard: bool = True,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.bundle_dir = bundle_dir
        self.explicit_keyframes = keyframes or []
        self.include_telemetry = include_telemetry
        self.enforce_deterministic = enforce_deterministic
        self.enforce_provenance_guard = enforce_provenance_guard
        self.last_request: dict[str, Any] | None = None
        self.last_response: dict[str, Any] | None = None

    @staticmethod
    def _image_data_url(path: Path) -> str:
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def _keyframe_paths(self, result: SimulationResultV1) -> list[Path]:
        paths = list(self.explicit_keyframes)
        if not paths and result.keyframes:
            for value in result.keyframes:
                path = Path(value)
                if not path.is_absolute() and self.bundle_dir is not None:
                    path = self.bundle_dir / path
                paths.append(path)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing VLM keyframes: {missing}")
        if not paths:
            raise ValueError("real VLM evaluation requires at least one keyframe image")
        paths = sorted(paths, key=lambda path: path.name)
        if len(paths) > 3:
            indices = (0, len(paths) // 2, len(paths) - 1)
            paths = list(dict.fromkeys(paths[index] for index in indices))
        return paths

    @staticmethod
    def _telemetry_summary(result: SimulationResultV1) -> dict[str, Any]:
        actor_states = result.actor_states or []
        collision_events = [item.model_dump(mode="json") for item in result.collisions or []]
        actor_ids = sorted({state.actor_id for state in actor_states})
        frames = sorted({state.frame for state in actor_states})
        first_states = {
            actor_id: next(
                state.model_dump(mode="json")
                for state in actor_states
                if state.actor_id == actor_id
            )
            for actor_id in actor_ids
        }
        last_states = {
            actor_id: next(
                state.model_dump(mode="json")
                for state in reversed(actor_states)
                if state.actor_id == actor_id
            )
            for actor_id in actor_ids
        }
        collision_pairs = sorted(
            {
                tuple(sorted((str(item["actor_id"]), str(item["other_actor_id"]))))
                for item in collision_events
            }
        )

        def compact_observed(value: Any) -> Any:
            if not isinstance(value, list) or len(value) <= 4:
                return value
            return {
                "event_count": len(value),
                "frames": sorted(
                    {item.get("frame") for item in value if isinstance(item, dict)}
                ),
                "first_event": value[0],
                "last_event": value[-1],
            }

        return {
            "backend": result.backend,
            "executed": result.executed,
            "status": result.status.value,
            "frame_count": len(frames),
            "actor_ids": actor_ids,
            "first_actor_states": first_states,
            "last_actor_states": last_states,
            "collisions": {
                "event_count": len(collision_events),
                "frames": sorted({item["frame"] for item in collision_events}),
                "actor_pairs": [list(pair) for pair in collision_pairs],
                "first_event": collision_events[0] if collision_events else None,
                "last_event": collision_events[-1] if collision_events else None,
            },
            "minimum_ttc_seconds": result.minimum_ttc_seconds,
            "constraint_results": [
                {
                    **item.model_dump(mode="json"),
                    "observed": compact_observed(item.observed),
                }
                for item in result.constraint_results
            ],
            "logs": result.logs,
            "errors": result.errors,
        }

    def evaluate(
        self,
        contract: ScenarioContractV1,
        result: SimulationResultV1,
    ) -> EvaluationReport:
        keyframe_paths = self._keyframe_paths(result)
        content: list[dict[str, Any]] = []
        for path in keyframe_paths:
            content.append(
                {"type": "image_url", "image_url": {"url": self._image_data_url(path)}}
            )
        if self.include_telemetry and self.enforce_provenance_guard:
            task = (
                "Audit whether the executed simulation realizes the contract. "
                "Deterministic constraint failures are authoritative; images may "
                "corroborate but must not override telemetry. Never invent evidence. "
                "Every actor listed in the contract is intentional; do not treat a "
                "non-colliding contract actor as unmodeled or as a failure by itself. "
                "When all deterministic constraint_results pass, do not describe a "
                "deterministic failure. passed=false requires at least one concrete "
                "failure entry with expected, observed, and evidence. "
                "Only inferred/defaulted fields may receive repair instructions; "
                "observed fields are immutable. Each repair must use an exact existing "
                "contract path plus a replacement value and reason. If no safe existing "
                "path is repairable, return an empty repair_instructions array. "
                "Any deterministic constraint failure requires passed=false and "
                "manual_review=true. Return only schema-valid JSON."
            )
        elif self.include_telemetry:
            task = (
                "Audit whether the executed simulation realizes the contract. "
                "Deterministic constraint failures are authoritative; images may "
                "corroborate but must not override telemetry. Never invent evidence. "
                "This is an intentionally unconstrained self-refinement baseline: "
                "when a failure is found, propose the contract edits that you believe "
                "would make a rerun pass even if the path is observed or normally "
                "immutable. Use only exact existing dotted contract paths. "
                "passed=false requires at least one concrete failure with expected, "
                "observed, and evidence. Return only schema-valid JSON."
            )
        else:
            task = (
                "Audit only the visible keyframes against the contract. No telemetry is "
                "available in this image-only baseline. Never infer invisible actor IDs, "
                "collision ordering, or numeric state. passed=false requires a concrete "
                "visible contradiction. If the images are insufficient, set passed=null, "
                "manual_review=true, and explain the missing visual evidence. Do not issue "
                "repair instructions. Return only schema-valid JSON."
            )
        audit_input = {
            "task": task,
            "contract": contract.model_dump(mode="json"),
            "keyframe_order": [path.name for path in keyframe_paths],
        }
        if self.include_telemetry:
            audit_input["telemetry_summary"] = self._telemetry_summary(result)
        content.append(
            {
                "type": "text",
                "text": json.dumps(audit_input, ensure_ascii=False),
            }
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "seed": contract.seed,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "system",
                    "content": "You are the JurisDrive multimodal assurance evaluator.",
                },
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "jurisdrive_evaluation_report",
                    "strict": True,
                    "schema": EvaluationReport.model_json_schema(),
                },
            },
        }
        self.last_request = {
            **payload,
            "messages": [
                payload["messages"][0],
                {
                    "role": "user",
                    "content": [
                        {"type": "image_path", "path": str(path)} for path in keyframe_paths
                    ]
                    + [content[-1]],
                },
            ],
        }
        http_request = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            self.last_response = {
                "http_status": exc.code,
                "error_body": response_body,
            }
            raise RuntimeError(
                f"VLM endpoint returned HTTP {exc.code}: {response_body}"
            ) from exc
        self.last_response = value
        content_value = value["choices"][0]["message"]["content"]
        report = EvaluationReport.model_validate(json.loads(content_value))
        contract_data = contract.model_dump(mode="json")
        accepted_repairs = []
        guard_notes: list[str] = []
        for instruction in report.repair_instructions:
            path = instruction.path
            if self.enforce_provenance_guard and (
                path in contract.immutable_paths or not _field_is_mutable(contract_data, path)
            ):
                guard_notes.append(f"provenance guard rejected repair path: {path}")
            else:
                accepted_repairs.append(instruction)
        deterministic_failure = self.enforce_deterministic and any(
            item.passed is False for item in result.constraint_results
        )
        inconsistent_failure_claim = report.passed is False and not report.failures
        if inconsistent_failure_claim:
            guard_notes.append(
                "VLM consistency guard: passed=false had no concrete failure entry"
            )
        failures = list(report.failures)
        for item in result.constraint_results:
            if self.enforce_deterministic and item.passed is False:
                failures.append(
                    EvaluationFailure(
                        attribute=f"deterministic:{item.name}",
                        expected=item.expected,
                        observed=item.observed,
                        evidence=item.reason,
                        repair_instruction=None,
                    )
                )
        return report.model_copy(
            update={
                "scenario_id": contract.scenario_id,
                "evaluator": (
                    f"{self.name}:{'telemetry_plus' if self.include_telemetry else 'image_only'}:"
                    f"{'guarded' if self.enforce_provenance_guard else 'unconstrained'}:"
                    f"{self.model}"
                ),
                "passed": False if deterministic_failure else report.passed,
                "failures": failures,
                "repair_instructions": accepted_repairs,
                "manual_review": bool(
                    report.manual_review
                    or
                    deterministic_failure
                    or inconsistent_failure_claim
                    or report.failures
                    or contract.review_issues
                    or guard_notes
                ),
                "notes": [*report.notes, *guard_notes],
            }
        )


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
    if not isinstance(current, dict):
        return False
    return current.get("provenance") in {
        ProvenanceValue.INFERRED.value,
        ProvenanceValue.DEFAULTED.value,
    }


def apply_bounded_repairs(
    contract: ScenarioContractV1,
    instructions: list[dict[str, Any]],
    *,
    max_repairs: int = 3,
) -> tuple[ScenarioContractV1, list[str]]:
    original = contract.model_dump(mode="json")
    updated = copy.deepcopy(original)
    notes: list[str] = []
    applied = 0
    for instruction in instructions:
        if applied >= max_repairs:
            notes.append("repair limit reached")
            break
        path = str(instruction.get("path") or "")
        if path in contract.immutable_paths or not _field_is_mutable(updated, path):
            notes.append(f"rejected immutable/observed repair: {path}")
            continue
        current: Any = updated
        parts = path.split(".")
        try:
            for part in parts[:-1]:
                current = current[int(part)] if isinstance(current, list) else current[part]
            leaf = parts[-1]
            target = current[int(leaf)] if isinstance(current, list) else current[leaf]
            if not isinstance(target, dict) or "value" not in target:
                raise KeyError(path)
            target["value"] = instruction.get("value")
            target["provenance"] = ProvenanceValue.DEFAULTED.value
            applied += 1
            notes.append(f"applied repair: {path}")
        except (KeyError, IndexError, ValueError, TypeError):
            notes.append(f"invalid repair path: {path}")
    try:
        repaired = ScenarioContractV1.model_validate(updated)
    except Exception:
        notes.append("repair rollback: contract validation regressed")
        repaired = contract
    return repaired, notes


def run_assurance_loop(
    contract: ScenarioContractV1,
    backend: SimulatorBackend,
    evaluator: Evaluator,
    *,
    max_iterations: int = 3,
) -> dict[str, Any]:
    current = contract
    trace: list[dict[str, Any]] = []
    previously_passed: set[str] = set()
    for iteration in range(max_iterations + 1):
        compiled = backend.compile(current)
        result = backend.run(compiled)
        report = evaluator.evaluate(current, result)
        passed_now = {
            item.name for item in result.constraint_results if item.passed is True
        }
        regressed = sorted(previously_passed - passed_now)
        trace.append(
            {
                "iteration": iteration,
                "contract": current.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
                "evaluation": report.model_dump(mode="json"),
                "regressed_constraints": regressed,
            }
        )
        if regressed:
            return {
                "status": "manual_review",
                "reason": "repair regressed previously passing constraints",
                "iterations": trace,
                "final_contract": contract.model_dump(mode="json"),
            }
        if report.passed is True:
            return {
                "status": "passed",
                "reason": None,
                "iterations": trace,
                "final_contract": current.model_dump(mode="json"),
            }
        observed_collision_failed = any(
            constraint.provenance == ProvenanceValue.OBSERVED
            and any(
                failure.attribute == "collision_target"
                for failure in report.failures
            )
            for constraint in current.collision_constraints
        )
        if observed_collision_failed:
            return {
                "status": "manual_review",
                "reason": "observed collision constraint cannot be repaired",
                "iterations": trace,
                "final_contract": current.model_dump(mode="json"),
            }
        if iteration == max_iterations or not report.repair_instructions:
            return {
                "status": "manual_review",
                "reason": "repair limit reached or no bounded repair is available",
                "iterations": trace,
                "final_contract": current.model_dump(mode="json"),
            }
        repaired, notes = apply_bounded_repairs(
            current,
            report.repair_instructions,
            max_repairs=1,
        )
        trace[-1]["repair_notes"] = notes
        if repaired == current:
            return {
                "status": "manual_review",
                "reason": "repair was rejected by provenance guard",
                "iterations": trace,
                "final_contract": current.model_dump(mode="json"),
            }
        previously_passed |= passed_now
        current = repaired
    raise AssertionError("unreachable")
