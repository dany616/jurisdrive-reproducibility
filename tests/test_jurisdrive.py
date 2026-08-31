from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from jurisdrive.assurance import MockEvaluator, VlmEvaluator, apply_bounded_repairs
from jurisdrive.contract import compile_contract
from jurisdrive.evidence import (
    OpenAICompatibleResolver,
    build_evidence_graph,
    validate_evidence_spans,
)
from jurisdrive.gold import (
    binary_metrics,
    blank_human_annotation,
    blinded_annotation_task,
    cohens_kappa,
    weighted_binary_metrics,
)
from jurisdrive.io import candidate_source_path, load_candidate
from jurisdrive.models import (
    ContractStatus,
    EvidenceGraphV1,
    ExecutionStatus,
    SimulationResultV1,
)
from jurisdrive.pipeline import build_graph_batch
from jurisdrive.simulator import DryRunBackend, render_scenic, write_bundle


def record(candidate_id: int, text: str, tier: str = "A_minimum_grounded") -> dict:
    return {
        "input_file": f"zeroshot_test_{candidate_id}.json",
        "source_text": text,
        "parsed": {"vehicle_type": "승용차", "accident_trajectory": "충돌"},
        "_manifest": {"candidate_id": candidate_id, "readiness_tier": tier},
    }


class EvidenceGraphTests(unittest.TestCase):
    def test_remote_resolver_requires_exact_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "exact model ID"):
                build_graph_batch(
                    [],
                    Path(temporary),
                    resolver_endpoint="http://127.0.0.1:8000/v1",
                )

    def test_manifest_source_resolves_portably_from_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            full_run = Path(temporary)
            source = full_run / "ambiguous_done" / "car_to_car" / "case.json"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps({"source_text": "test"}),
                encoding="utf-8",
            )
            row = {
                "candidate_id": 1,
                "source_stage": "llm",
                "result_file": "case.json",
                "source_path": "/unavailable/historical/case.json",
            }
            self.assertEqual(
                candidate_source_path(row, full_run_dir=full_run),
                source,
            )
            self.assertEqual(
                load_candidate(row, full_run_dir=full_run)["source_text"],
                "test",
            )

    def test_resolver_disables_thinking_for_bounded_json(self) -> None:
        source_text = "도로에 주차 중이던 아반떼 승용차와 싼타페 승용차를 순차로 들이받았다."
        graph = build_evidence_graph(record(87, source_text))
        captured: dict = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "agent_id": "vehicle_a",
                                            "target_id": "vehicle_b",
                                            "quote": "exact quote",
                                            "confidence": 0.5,
                                        }
                                    )
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(http_request, timeout):
            captured.update(json.loads(http_request.data.decode("utf-8")))
            captured["url"] = http_request.full_url
            self.assertEqual(timeout, 120.0)
            return FakeResponse()

        resolver = OpenAICompatibleResolver("http://127.0.0.1:8000", "test-model")
        with patch("jurisdrive.evidence.request.urlopen", side_effect=fake_urlopen):
            resolver.resolve(graph, source_text)
        self.assertEqual(
            captured["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertEqual(captured["url"], "http://127.0.0.1:8000/v1/chat/completions")

        resolver_v1 = OpenAICompatibleResolver("http://127.0.0.1:8000/v1", "test-model")
        with patch("jurisdrive.evidence.request.urlopen", side_effect=fake_urlopen):
            resolver_v1.resolve(graph, source_text)
        self.assertEqual(captured["url"], "http://127.0.0.1:8000/v1/chat/completions")

    def test_korean_collision_roles_and_exact_spans(self) -> None:
        text = (
            "원고차량이 직진하던 중 피고차량이 차선을 변경하여 "
            "피고차량의 오른쪽 측면으로 원고차량 왼쪽 측면을 접촉하였다."
        )
        graph = build_evidence_graph(record(34, text))
        vehicles = {node.id: node.label for node in graph.nodes if node.type == "vehicle"}
        collision = next(edge for edge in graph.edges if edge.relation == "collides_with")
        self.assertIn("피고차량", vehicles[collision.source])
        self.assertIn("원고차량", vehicles[collision.target])
        self.assertEqual(validate_evidence_spans(graph, text), [])
        self.assertFalse(graph.critical_unresolved)

    def test_missing_agent_is_not_guessed(self) -> None:
        text = "도로에 주차 중이던 아반떼 승용차와 싼타페 승용차를 순차로 들이받았다."
        graph = build_evidence_graph(record(87, text))
        self.assertIn("collision_agent", graph.critical_unresolved)
        self.assertFalse(any(edge.relation == "collides_with" for edge in graph.edges))

    def test_extra_schema_field_is_rejected(self) -> None:
        graph = build_evidence_graph(
            record(1, "피고차량의 앞부분으로 원고차량의 뒷부분을 충격하였다.")
        )
        payload = graph.model_dump(mode="json")
        payload["unexpected"] = True
        with self.assertRaises(ValidationError):
            EvidenceGraphV1.model_validate(payload)


class ContractAndDryRunTests(unittest.TestCase):
    def _contract(self, tier: str = "A_minimum_grounded"):
        text = "피고차량의 앞부분으로 원고차량의 뒷부분을 충격하였다."
        graph = build_evidence_graph(record(1, text, tier))
        return graph, compile_contract(graph, source_text=text, readiness_tier=tier)

    def test_tier_c_never_compiles(self) -> None:
        _, contract = self._contract("C_reextract_or_review")
        self.assertEqual(contract.status, ContractStatus.NEEDS_REVIEW)
        compiled = DryRunBackend().compile(contract)
        self.assertFalse(compiled["compile_valid"])

    def test_string_enum_is_python38_compatible(self) -> None:
        self.assertEqual(str(ContractStatus.READY), "ready")

    def test_dry_run_has_no_simulation_metrics(self) -> None:
        _, contract = self._contract()
        result = DryRunBackend().run(DryRunBackend().compile(contract))
        self.assertFalse(result.executed)
        self.assertEqual(result.status, ExecutionStatus.NOT_EXECUTED)
        self.assertIsNone(result.actor_states)
        self.assertIsNone(result.collisions)
        self.assertIsNone(result.minimum_ttc_seconds)
        self.assertIsNone(result.keyframes)

    def test_scenic_uses_vendored_carla_map_parameter(self) -> None:
        _, contract = self._contract()
        source = render_scenic(contract)
        self.assertIn(
            "/opt/CARLA_0.9.13/CarlaUE4/Content/Carla/Maps/OpenDrive/", source
        )
        self.assertIn(
            f'param carla_map = "{contract.map_binding.carla_map.value}"', source
        )
        self.assertNotIn(" at 0@0", source)
        self.assertNotIn("= new Car", source)
        self.assertIn("ego = Car", source)
        self.assertIn("position sampled on the bound road network", source)

    def test_not_executed_model_rejects_metrics(self) -> None:
        with self.assertRaises(ValidationError):
            SimulationResultV1(
                scenario_id="x",
                backend="dry-run",
                executed=False,
                status="not_executed",
                actor_states=[],
                collisions=None,
                minimum_ttc_seconds=None,
                constraint_results=[],
                keyframes=None,
            )

    def test_bundle_checksums(self) -> None:
        graph, contract = self._contract()
        backend = DryRunBackend()
        compiled = backend.compile(contract)
        result = backend.run(compiled)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = write_bundle(Path(temporary), graph, contract, compiled, result)
            for line in (bundle / "checksums.sha256").read_text().splitlines():
                expected, name = line.split("  ", 1)
                actual = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
                self.assertEqual(expected, actual)

    def test_observed_repair_is_rejected(self) -> None:
        _, contract = self._contract()
        repaired, notes = apply_bounded_repairs(
            contract,
            [{"path": "collision_constraints.0", "value": "changed"}],
        )
        self.assertEqual(repaired, contract)
        self.assertTrue(any("rejected" in note for note in notes))

    def test_mock_evaluator_does_not_claim_pass(self) -> None:
        _, contract = self._contract("C_reextract_or_review")
        backend = DryRunBackend()
        report = MockEvaluator().evaluate(contract, backend.run(backend.compile(contract)))
        self.assertFalse(report.passed)
        self.assertTrue(report.manual_review)
        self.assertTrue(any("No VLM" in note for note in report.notes))

    def test_vlm_evaluator_requires_real_keyframe(self) -> None:
        _, contract = self._contract()
        backend = DryRunBackend()
        result = backend.run(backend.compile(contract))
        evaluator = VlmEvaluator("http://127.0.0.1:1", "test-model")
        with self.assertRaises(ValueError):
            evaluator.evaluate(contract, result)

    def test_vlm_evaluator_caps_keyframes_at_before_during_after(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative_paths = []
            for frame in (10, 20, 30, 40, 50, 60, 70):
                path = root / f"frame_{frame:08d}.png"
                path.write_bytes(b"png")
                relative_paths.append(path.name)
            evaluator = VlmEvaluator(
                "http://127.0.0.1:1", "test-model", bundle_dir=root
            )
            selected = evaluator._keyframe_paths(
                SimpleNamespace(keyframes=list(reversed(relative_paths)))
            )
            self.assertEqual(
                [path.name for path in selected],
                [
                    "frame_00000010.png",
                    "frame_00000040.png",
                    "frame_00000070.png",
                ],
            )

    def test_vlm_telemetry_compacts_repeated_collisions(self) -> None:
        collisions = [
            {
                "frame": frame,
                "actor_id": "vehicle_1",
                "other_actor_id": "vehicle_2",
                "impulse": {"x": 1.0, "y": 0.0, "z": 0.0},
            }
            for frame in range(10, 16)
        ]
        result = SimulationResultV1(
            scenario_id="x",
            backend="carla",
            executed=True,
            status="passed",
            actor_states=[],
            collisions=collisions,
            minimum_ttc_seconds=0.5,
            constraint_results=[
                {
                    "name": "collision_target",
                    "passed": True,
                    "expected": {
                        "actor_id": "vehicle_1",
                        "target_id": "vehicle_2",
                    },
                    "observed": collisions,
                }
            ],
            keyframes=[],
        )
        summary = VlmEvaluator._telemetry_summary(result)
        self.assertEqual(summary["collisions"]["event_count"], 6)
        self.assertEqual(
            summary["constraint_results"][0]["observed"]["event_count"], 6
        )


class GoldMetricTests(unittest.TestCase):
    def test_human_annotation_inputs_exclude_prediction_metadata(self) -> None:
        source = {
            "candidate_id": 7,
            "source_file_sha256": "a" * 64,
            "source_text": "사람이 직접 검수할 원문",
            "stratum": "qwen_car",
            "predicted_label": "car_to_car",
            "rule": {"label": "ambiguous"},
            "qwen": {"label": "car_to_car"},
        }
        blinded = blinded_annotation_task(source)
        template = blank_human_annotation(source)
        self.assertEqual(
            set(blinded),
            {"candidate_id", "source_file_sha256", "source_text"},
        )
        for forbidden in ("stratum", "predicted_label", "rule", "qwen"):
            self.assertNotIn(forbidden, blinded)
            self.assertNotIn(forbidden, template)

    def test_kappa_and_binary_metrics(self) -> None:
        self.assertAlmostEqual(
            cohens_kappa(
                ["car_to_car", "car_to_car", "not_car_to_car", "not_car_to_car"],
                ["car_to_car", "not_car_to_car", "not_car_to_car", "not_car_to_car"],
            ),
            0.5,
        )
        metrics = binary_metrics(
            ["car_to_car", "car_to_car", "not_car_to_car", "not_car_to_car"],
            ["car_to_car", "not_car_to_car", "car_to_car", "not_car_to_car"],
        )
        self.assertEqual(metrics["confusion"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertEqual(metrics["f1"], 0.5)
        weighted = weighted_binary_metrics(
            ["car_to_car", "not_car_to_car", "not_car_to_car"],
            ["car_to_car", "car_to_car", "not_car_to_car"],
            [1.0, 1.0, 8.0],
        )
        self.assertAlmostEqual(weighted["precision"], 0.5)
        self.assertAlmostEqual(weighted["recall"], 1.0)
        self.assertAlmostEqual(weighted["mcc"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
