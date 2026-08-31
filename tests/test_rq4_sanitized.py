from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jurisdrive.rq4_sanitized import (
    METHOD_CODES,
    build_method_payloads,
    build_request_template,
    forbidden_path_hits,
    opaque_identifier,
    recursive_forbidden_hits,
    render_request,
    sanitize_contract,
    sanitize_telemetry,
)


class Rq4SanitizedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = {
            "actors": [
                {
                    "id": "legacy_vehicle_1",
                    "role": "ego",
                    "vehicle_type": {"value": "sedan", "provenance": "observed"},
                    "blueprint": {"value": "vehicle.generic", "provenance": "defaulted"},
                    "lane_position": {"value": "map_lane_mismatch_fault", "provenance": "defaulted"},
                    "initial_speed_mps": {"value": 15.0, "provenance": "defaulted"},
                },
                {
                    "id": "legacy_vehicle_2",
                    "role": "target",
                    "vehicle_type": {"value": "truck", "provenance": "observed"},
                    "blueprint": {"value": "vehicle.generic", "provenance": "defaulted"},
                    "lane_position": {"value": "lane_relative", "provenance": "defaulted"},
                    "initial_speed_mps": {"value": 0.0, "provenance": "defaulted"},
                },
            ],
            "map_binding": {
                "archetype": {"value": "straight_road", "provenance": "inferred"},
                "carla_map": {"value": "runtime_map", "provenance": "defaulted"},
            },
            "topology": {"value": "rear_end", "provenance": "inferred"},
            "maneuver_by_actor": {
                "legacy_vehicle_1": {"value": "closure", "provenance": "defaulted"},
                "legacy_vehicle_2": {"value": "stopped", "provenance": "defaulted"},
            },
            "event_sequence": [
                {"id": "legacy_event", "kind": "collision", "actor_id": "legacy_vehicle_1", "target_id": "legacy_vehicle_2", "order": 1, "description": {"value": "impact", "provenance": "observed"}}
            ],
            "collision_constraints": [
                {"actor_id": "legacy_vehicle_1", "target_id": "legacy_vehicle_2", "required": True, "provenance": "observed"}
            ],
            "sensors": {"collision": True, "rgb": True, "telemetry_hz": 20},
            "fixed_delta_seconds": 0.05,
            "duration_seconds": 20.0,
        }
        self.result = {
            "executed": True,
            "actor_states": [
                {"frame": 1, "actor_id": "legacy_vehicle_1", "timestamp_seconds": 0.05, "location": {"x": 0, "y": 0, "z": 0}, "rotation": {"yaw": 0}, "speed_mps": 15},
                {"frame": 2, "actor_id": "legacy_vehicle_1", "timestamp_seconds": 0.10, "location": {"x": 1, "y": 0, "z": 0}, "rotation": {"yaw": 0}, "speed_mps": 14},
                {"frame": 1, "actor_id": "legacy_vehicle_2", "timestamp_seconds": 0.05, "location": {"x": 2, "y": 0, "z": 0}, "rotation": {"yaw": 0}, "speed_mps": 0},
            ],
            "collisions": [{"frame": 2, "actor_id": "legacy_vehicle_1", "other_actor_id": "legacy_vehicle_2", "impulse": {"x": 1, "y": 2, "z": 3}}],
            "minimum_ttc_seconds": 0.2,
            "constraint_results": [
                {"name": "lane_topology_valid", "passed": False, "observed": {"lane_checks": {"legacy_vehicle_1": {"road_id": 1, "lane_id": 2, "projected_distance_m": 3.5, "within_driving_lane_tolerance": False}}}, "reason": "map_lane_mismatch_fault"},
                {"name": "event_order_valid", "passed": True, "observed": {"first_state_frame": 1, "first_collision_frame": 2}, "reason": None},
            ],
            "status": "failed",
            "errors": ["controlled actor-target fault"],
        }

    def test_contract_and_telemetry_are_allowlisted(self) -> None:
        views = sanitize_contract(self.contract)
        telemetry = sanitize_telemetry(self.result, views["actor_map"])
        combined = {"views": views, "telemetry": telemetry}
        self.assertEqual(recursive_forbidden_hits(combined, origin="test"), [])
        self.assertEqual(views["contract"]["actors"][0]["lane_position"], "unrecognized_reference")
        self.assertNotIn("status", telemetry)
        self.assertNotIn("constraint_results", telemetry)
        self.assertEqual(set(views["actor_map"].values()), {"actor_01", "actor_02"})

    def test_budget_union_and_guard_delta(self) -> None:
        views = sanitize_contract(self.contract)
        telemetry = sanitize_telemetry(self.result, views["actor_map"])
        payloads = build_method_payloads(
            contract_views=views,
            telemetry=telemetry,
            shuffled_telemetry={**telemetry, "minimum_ttc_seconds": 9.9},
            image_refs=["assets/A123/A/F00.png", "assets/A123/A/F01.png", "assets/A123/A/F02.png"],
            shuffled_image_refs=["assets/A123/B/F00.png", "assets/A123/B/F01.png", "assets/A123/B/F02.png"],
        )
        self.assertEqual(set(payloads), set(METHOD_CODES))
        fusion = payloads["telemetry_plus_image"]["evidence"]
        image = payloads["image_only_vlm"]["evidence"]
        telemetry_only = payloads["telemetry_only"]["evidence"]
        self.assertEqual(fusion, {**image, **telemetry_only})
        guarded = payloads["guarded_blind_loop"]["evidence"]
        self.assertEqual(set(guarded) - set(fusion), {"mutability_classes"})

    def test_recursive_audit_catches_fields_values_and_paths(self) -> None:
        value = {"nested": [{"oracle_value": 3}, "controlled event-order fault"]}
        hits = recursive_forbidden_hits(value, origin="test")
        self.assertGreaterEqual(len(hits), 2)
        self.assertTrue(forbidden_path_hits(["requests/fault_actor_target/item.json"]))
        self.assertFalse(forbidden_path_hits(["requests/M00/R0123.json"]))

    def test_opaque_ids_and_rendered_request(self) -> None:
        opaque = opaque_identifier("artifact", "fault_rear_end_r01_actor_target_swap")
        self.assertNotIn("rear", opaque.lower())
        self.assertNotIn("fault", opaque.lower())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            asset = root / "assets" / opaque / "A" / "F00.png"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
            template = build_request_template(
                opaque_artifact_id=opaque,
                evidence={"visual_expectations": {"actors": []}},
                image_refs=[str(asset.relative_to(root)).replace("\\", "/")],
            )
            self.assertEqual(recursive_forbidden_hits(template, origin="template"), [])
            rendered = render_request(template, root)
            body = json.dumps(rendered)
            self.assertIn("data:image/png;base64,", body)
            self.assertNotIn("asset-ref://", body)


if __name__ == "__main__":
    unittest.main()
