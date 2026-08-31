from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from jurisdrive.contract import bind_topology_profile, compile_contract
from jurisdrive.evidence import build_evidence_graph
from jurisdrive.experiments import (
    ExperimentPreregistration,
    build_assurance_evaluation_schedule,
    build_fault_plan,
    build_fidelity_schedule,
    materialize_fault_bundle,
    summarize_assurance_records,
    summarize_fidelity_records,
)
from jurisdrive.io import write_json


def record(candidate_id: int = 1) -> dict:
    return {
        "input_file": f"zeroshot_test_{candidate_id}.json",
        "source_text": "피고차량의 앞부분으로 원고차량의 뒷부분을 충격하였다.",
        "parsed": {"vehicle_type": "승용차", "accident_trajectory": "충돌"},
        "_manifest": {
            "candidate_id": candidate_id,
            "readiness_tier": "A_minimum_grounded",
        },
    }


def pending_config() -> dict:
    cases = []
    topologies = (
        "rear_end",
        "intersection_crossing_turning",
        "lane_change_side_swipe",
        "head_on_centerline_intrusion",
    )
    for topology in topologies:
        for route in ("rule", "qwen"):
            for index in range(3):
                cases.append(
                    {
                        "slot_id": f"{topology}_{route}_{index}",
                        "topology": topology,
                        "source_stage": route,
                    }
                )
    return {
        "experiment_id": "test",
        "selection_frozen": False,
        "base_seed": 100,
        "cases": cases,
    }


class ExperimentPlanTests(unittest.TestCase):
    def test_topology_binding_preserves_evidence_and_defaults_physics(self) -> None:
        source = record()
        graph = build_evidence_graph(source)
        contract = compile_contract(
            graph,
            source_text=source["source_text"],
            readiness_tier="A_minimum_grounded",
        )
        evidence_ids = list(contract.collision_constraints[0].evidence_ids)
        bound = bind_topology_profile(
            contract, "rear_end", evidence_ids=evidence_ids
        )
        self.assertEqual(bound.topology.value, "rear_end")
        self.assertEqual(bound.topology.provenance.value, "inferred")
        self.assertEqual(bound.topology.evidence_ids, evidence_ids)
        ego = next(actor for actor in bound.actors if actor.role == "ego")
        target = next(actor for actor in bound.actors if actor.role == "target")
        self.assertEqual(ego.initial_speed_mps.provenance.value, "defaulted")
        self.assertGreater(ego.initial_speed_mps.value, target.initial_speed_mps.value)

    def test_balanced_schedule_and_fault_denominators(self) -> None:
        config = ExperimentPreregistration.model_validate(pending_config())
        schedule = build_fidelity_schedule(config)
        faults = build_fault_plan(config)
        evaluations = build_assurance_evaluation_schedule(faults)
        self.assertEqual(len(schedule), 96)
        self.assertEqual(len({row["seed"] for row in schedule}), 48)
        self.assertEqual(len(faults), 168)
        self.assertEqual(sum(row["trial_kind"] == "clean_control" for row in faults), 24)
        self.assertEqual(sum(row["fault_class"] == "mutable" for row in faults), 72)
        self.assertEqual(sum(row["fault_class"] == "immutable" for row in faults), 72)
        self.assertEqual(len(evaluations), 840)
        self.assertEqual(len({row["method"] for row in evaluations}), 5)
        self.assertTrue(all(not row["ready_for_execution"] for row in schedule))

    def test_unbalanced_preregistration_is_rejected(self) -> None:
        value = pending_config()
        value["cases"][0]["source_stage"] = "qwen"
        with self.assertRaises(ValidationError):
            ExperimentPreregistration.model_validate(value)

    def test_frozen_selection_does_not_require_future_clean_bundle(self) -> None:
        value = pending_config()
        value["selection_frozen"] = True
        value["frozen_at_utc"] = "2026-08-23T00:00:00Z"
        for candidate_id, case in enumerate(value["cases"], start=1):
            case.update(
                {
                    "candidate_id": candidate_id,
                    "scenario_id": f"jurisdrive_{candidate_id}",
                    "human_topology_confirmed": True,
                    "contract_path": f"contracts/jurisdrive_{candidate_id}.json",
                    "clean_bundle_path": None,
                }
            )
        config = ExperimentPreregistration.model_validate(value)
        self.assertTrue(config.selection_frozen)


class FaultMaterializationTests(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        source = record()
        graph = build_evidence_graph(source)
        contract = compile_contract(
            graph,
            source_text=source["source_text"],
            readiness_tier="A_minimum_grounded",
        )
        bundle = root / "clean"
        bundle.mkdir()
        write_json(bundle / "contract.json", contract)
        write_json(bundle / "evidence_graph.json", graph)
        return bundle

    def test_mutable_fault_preserves_oracle_and_requires_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._bundle(root)
            output = root / "fault"
            manifest = materialize_fault_bundle(
                bundle, output, "speed_pose_perturbation", variant="speed"
            )
            oracle = (output / "oracle_contract.json").read_bytes()
            faulty = (output / "contract.json").read_bytes()
            self.assertNotEqual(oracle, faulty)
            self.assertTrue(manifest["requires_carla_rerun"])
            self.assertFalse(manifest["injection_verified"])
            self.assertEqual(manifest["fault_class"], "mutable")
            self.assertTrue((output / "checksums.sha256").is_file())

    def test_event_order_fault_uses_runtime_order_for_single_event_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._bundle(root)
            (bundle / "frame.png").write_bytes(b"test-frame")
            write_json(
                bundle / "simulation_result.json",
                {
                    "scenario_id": "jurisdrive_1",
                    "backend": "carla",
                    "executed": True,
                    "status": "passed",
                    "actor_states": [],
                    "collisions": [],
                    "minimum_ttc_seconds": None,
                    "constraint_results": [
                        {
                            "name": "event_order_valid",
                            "passed": True,
                            "expected": [
                                "runtime_initial_state",
                                "runtime_required_collision",
                            ],
                            "observed": [
                                "runtime_initial_state",
                                "runtime_required_collision",
                            ],
                        }
                    ],
                    "keyframes": ["frame.png"],
                },
            )
            manifest = materialize_fault_bundle(
                bundle, root / "fault", "event_order_violation"
            )
            self.assertEqual(
                manifest["mutation"]["basis"],
                "runtime_event_order_constraint",
            )
            faulty_result = json.loads(
                (root / "fault" / "simulation_result.json").read_text(encoding="utf-8")
            )
            event_order = next(
                row
                for row in faulty_result["constraint_results"]
                if row["name"] == "event_order"
            )
            self.assertFalse(event_order["passed"])

    def test_immutable_fault_changes_only_copied_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._bundle(root)
            contract = json.loads(
                (bundle / "contract.json").read_text(encoding="utf-8")
            )
            expected = contract["collision_constraints"][0]
            (bundle / "frame.png").write_bytes(b"test-frame")
            write_json(
                bundle / "simulation_result.json",
                {
                    "scenario_id": "jurisdrive_1",
                    "backend": "carla",
                    "executed": True,
                    "status": "passed",
                    "actor_states": [],
                    "collisions": [
                        {
                            "frame": 10,
                            "actor_id": expected["actor_id"],
                            "other_actor_id": expected["target_id"],
                            "impulse": {"x": 1.0, "y": 0.0, "z": 0.0},
                        }
                    ],
                    "minimum_ttc_seconds": 0.2,
                    "constraint_results": [
                        {
                            "name": "collision_target",
                            "passed": True,
                            "expected": expected,
                            "observed": expected,
                        }
                    ],
                    "keyframes": ["frame.png"],
                },
            )
            original_contract = (bundle / "contract.json").read_bytes()
            original_result = (bundle / "simulation_result.json").read_bytes()
            manifest = materialize_fault_bundle(
                bundle, root / "fault", "actor_target_swap"
            )
            self.assertEqual((bundle / "contract.json").read_bytes(), original_contract)
            self.assertEqual((bundle / "simulation_result.json").read_bytes(), original_result)
            self.assertEqual(manifest["fault_class"], "immutable")
            self.assertFalse(manifest["requires_carla_rerun"])
            faulty_result = json.loads(
                (root / "fault" / "simulation_result.json").read_text(encoding="utf-8")
            )
            collision_check = next(
                row
                for row in faulty_result["constraint_results"]
                if row["name"] == "collision_target"
            )
            self.assertFalse(collision_check["passed"])


class ExperimentSummaryTests(unittest.TestCase):
    def test_assurance_summary_excludes_unverified_injections(self) -> None:
        rows = [
            {"trial_kind": "clean_control", "execution_status": "completed", "detected": False},
            {
                "trial_kind": "fault",
                "fault_type": "actor_target_swap",
                "fault_class": "immutable",
                "execution_status": "completed",
                "injection_verified": True,
                "detected": True,
                "manual_review": True,
                "immutable_edit_attempted": True,
                "immutable_edit_rejected": True,
            },
            {
                "trial_kind": "fault",
                "fault_type": "required_collision_omission",
                "fault_class": "mutable",
                "execution_status": "completed",
                "injection_verified": False,
                "detected": True,
            },
        ]
        summary = summarize_assurance_records(rows)
        self.assertEqual(summary["eligible_rows"], 2)
        self.assertEqual(summary["confusion"], {"tp": 1, "tn": 1, "fp": 0, "fn": 0})
        self.assertEqual(summary["guard"]["immutable_edits_rejected"], 1)

    def test_fidelity_replay_summary_uses_same_seed_pairs(self) -> None:
        base = {
            "slot_id": "rear_end_r01",
            "scenario_id": "jurisdrive_1",
            "seed": 42,
            "execution_status": "completed",
            "contract_compile_pass": True,
            "carla_launch_complete": True,
            "run_complete": True,
            "actor_target_correct": True,
            "lane_topology_valid": True,
            "event_order_valid": True,
            "hard_constraint_pass": True,
            "collision_signature": "vehicle_1>vehicle_2@10",
            "telemetry_sha256": "abc",
        }
        summary = summarize_fidelity_records([base, copy.deepcopy(base)])
        self.assertEqual(summary["replay"]["complete_same_seed_pairs"], 1)
        self.assertEqual(summary["replay"]["exact_core_metric_rate"], 1.0)
        self.assertEqual(summary["replay"]["exact_telemetry_hash_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
