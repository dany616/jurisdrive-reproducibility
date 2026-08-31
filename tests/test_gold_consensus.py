from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jurisdrive.gold_consensus import (
    apply_additional_review,
    evaluate_graph_semantics,
    evaluate_selective_protocol,
    freeze_dual_human_consensus,
    mcnemar_exact,
    normalize_label,
    read_jsonl_index,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class GoldConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tasks = self.root / "tasks.jsonl"
        self.annotator_a = self.root / "annotator_one.jsonl"
        self.annotator_b = self.root / "annotator_two.jsonl"
        tasks = [
            {
                "candidate_id": candidate_id,
                "stratum": "s1" if candidate_id <= 3 else "s2",
                "source_file_sha256": f"source-{candidate_id}",
                "source_text": f"source text {candidate_id}",
            }
            for candidate_id in range(1, 6)
        ]
        labels_a = ["car_to_car", "not_car_to_car", "uncertain", "car_to_car", "car_to_car"]
        labels_b = ["car_to_car", "not_car_to_car", "uncertain", "uncertain", "not_car_to_car"]
        rows_a = []
        rows_b = []
        for task, left, right in zip(tasks, labels_a, labels_b):
            common = {
                "candidate_id": task["candidate_id"],
                "stratum": task["stratum"],
                "source_file_sha256": task["source_file_sha256"],
                "vehicle_count": 2 if left == "car_to_car" else 1,
                "collision_agent": "vehicle-a" if left == "car_to_car" else None,
                "collision_target": "vehicle-b" if left == "car_to_car" else None,
                "legal_status": "accepted_fact",
                "evidence_quotes": [f"source text {task['candidate_id']}"],
            }
            rows_a.append({**common, "label": left, "annotator_id": "annotator_one"})
            right_common = dict(common)
            if right != left:
                right_common.update(
                    {
                        "vehicle_count": 2 if right == "car_to_car" else 1,
                        "collision_agent": "vehicle-a" if right == "car_to_car" else None,
                        "collision_target": "vehicle-b" if right == "car_to_car" else None,
                    }
                )
            rows_b.append({**right_common, "label": right, "annotator_id": "annotator_two"})
        write_jsonl(self.tasks, tasks)
        write_jsonl(self.annotator_a, rows_a)
        write_jsonl(self.annotator_b, rows_b)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _freeze(self, name: str = "freeze") -> tuple[Path, dict]:
        output = self.root / name
        manifest = freeze_dual_human_consensus(
            tasks_path=self.tasks,
            annotator_a_path=self.annotator_a,
            annotator_b_path=self.annotator_b,
            output_dir=output,
            expected_total=5,
            expected_consensus=2,
            expected_review=3,
            semantic_review_sample_size=2,
            semantic_sample_seed=9,
        )
        return output, manifest

    def test_label_normalization_supports_paper_and_legacy_enums(self) -> None:
        self.assertEqual(normalize_label("ACCEPT"), "ACCEPT")
        self.assertEqual(normalize_label("car_to_car"), "ACCEPT")
        self.assertEqual(normalize_label("not_car_to_car"), "REJECT")
        self.assertEqual(normalize_label("uncertain"), "UNRESOLVED")
        self.assertEqual(normalize_label("abstain"), "UNRESOLVED")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            normalize_label("maybe")

    def test_freeze_keeps_uncertain_and_disagreement_out_of_binary_gold(self) -> None:
        output, manifest = self._freeze()
        consensus = read_jsonl_index(output / "consensus_gold_743.jsonl")
        review = read_jsonl_index(output / "additional_review_queue_157.jsonl")
        full = read_jsonl_index(output / "full_selective_reference_900.jsonl")
        self.assertEqual({row["gold_label"] for row in consensus.values()}, {"ACCEPT", "REJECT"})
        self.assertEqual(set(consensus), {1, 2})
        self.assertEqual(set(review), {3, 4, 5})
        self.assertTrue(all(row["gold_label"] == "UNRESOLVED" for row in review.values()))
        self.assertEqual(review[3]["review_reason"], "common_uncertain")
        self.assertEqual(review[4]["review_reason"], "one_annotator_uncertain")
        self.assertEqual(review[5]["review_reason"], "binary_label_disagreement")
        self.assertEqual(len(full), 5)
        self.assertEqual(manifest["counts"]["consensus"], 2)
        self.assertEqual(manifest["counts"]["additional_review"], 3)
        self.assertEqual(manifest["semantic_sampling"]["all_consensus_accept"], 1)
        self.assertEqual(manifest["counts"]["semantic_review_tasks"], 3)

    def test_freeze_digest_is_reproducible_and_inputs_are_not_modified(self) -> None:
        before = (self.annotator_a.read_bytes(), self.annotator_b.read_bytes())
        _, first = self._freeze("freeze-one")
        _, second = self._freeze("freeze-two")
        self.assertEqual(first["freeze_digest"], second["freeze_digest"])
        self.assertEqual(before, (self.annotator_a.read_bytes(), self.annotator_b.read_bytes()))

    def test_custom_protocol_version_and_statement_are_bound(self) -> None:
        statement = self.root / "protocol.md"
        statement.write_text("author-approved protocol statement\n", encoding="utf-8")
        output = self.root / "freeze-v2"
        manifest = freeze_dual_human_consensus(
            tasks_path=self.tasks,
            annotator_a_path=self.annotator_a,
            annotator_b_path=self.annotator_b,
            output_dir=output,
            expected_total=5,
            expected_consensus=2,
            expected_review=3,
            semantic_review_sample_size=2,
            semantic_sample_seed=9,
            protocol_version="test-protocol-v2",
            protocol_statement={"path": str(statement), "status": "author-approved"},
        )
        self.assertEqual(manifest["protocol_version"], "test-protocol-v2")
        self.assertEqual(manifest["protocol_statement"]["status"], "author-approved")

        hybrid = self.root / "hybrid-v2.jsonl"
        write_jsonl(
            hybrid,
            [
                {"candidate_id": 1, "prediction": "ACCEPT"},
                {"candidate_id": 2, "prediction": "REJECT"},
                {"candidate_id": 3, "prediction": "UNRESOLVED"},
                {"candidate_id": 4, "prediction": "UNRESOLVED"},
                {"candidate_id": 5, "prediction": "UNRESOLVED"},
            ],
        )
        payload = evaluate_selective_protocol(
            consensus_path=output / "consensus_gold_743.jsonl",
            full_reference_path=output / "full_selective_reference_900.jsonl",
            prediction_paths={"hybrid": hybrid},
            bootstrap_samples=10,
            bootstrap_seed=1,
            forced_reject_from=None,
            protocol_version="test-protocol-v2",
        )
        self.assertEqual(payload["protocol_version"], "test-protocol-v2")

    def test_freeze_refuses_count_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 743"):
            freeze_dual_human_consensus(
                tasks_path=self.tasks,
                annotator_a_path=self.annotator_a,
                annotator_b_path=self.annotator_b,
                output_dir=self.root / "bad-freeze",
                expected_total=5,
                expected_consensus=743,
                expected_review=None,
                semantic_review_sample_size=0,
            )

    def test_additional_review_can_remain_unresolved(self) -> None:
        output, _ = self._freeze()
        adjudication = self.root / "adjudication.jsonl"
        write_jsonl(
            adjudication,
            [
                {"candidate_id": 3, "label": "UNRESOLVED", "notes": "insufficient evidence"},
                {"candidate_id": 4, "label": "ACCEPT"},
                {"candidate_id": 5, "label": "REJECT"},
            ],
        )
        final_path = self.root / "final" / "full_adjudicated_gold.jsonl"
        manifest = apply_additional_review(
            full_reference_path=output / "full_selective_reference_900.jsonl",
            adjudication_path=adjudication,
            output_path=final_path,
            manifest_path=self.root / "final" / "manifest.json",
            expected_review=3,
        )
        final = read_jsonl_index(final_path)
        self.assertEqual(final[3]["gold_label"], "UNRESOLVED")
        self.assertEqual(final[3]["reference_status"], "adjudicated_unresolved")
        self.assertEqual(final[4]["gold_label"], "ACCEPT")
        self.assertEqual(final[5]["gold_label"], "REJECT")
        self.assertEqual(manifest["counts"], {"total": 5, "ACCEPT": 2, "REJECT": 2, "UNRESOLVED": 1})

    def test_selective_metrics_bootstrap_and_forced_binary_baseline(self) -> None:
        output, _ = self._freeze()
        hybrid = self.root / "hybrid.jsonl"
        all_reject = self.root / "all_reject.jsonl"
        write_jsonl(
            hybrid,
            [
                {"candidate_id": 1, "prediction": "car_to_car"},
                {"candidate_id": 2, "prediction": "not_car_to_car"},
                {"candidate_id": 3, "prediction": "abstain"},
                {"candidate_id": 4, "prediction": "UNRESOLVED"},
                {"candidate_id": 5, "prediction": "uncertain"},
            ],
        )
        write_jsonl(
            all_reject,
            [
                {"candidate_id": candidate_id, "prediction": "REJECT"}
                for candidate_id in range(1, 6)
            ],
        )
        payload = evaluate_selective_protocol(
            consensus_path=output / "consensus_gold_743.jsonl",
            full_reference_path=output / "full_selective_reference_900.jsonl",
            prediction_paths={"hybrid": hybrid, "all_reject": all_reject},
            bootstrap_samples=200,
            bootstrap_seed=17,
            forced_reject_from="hybrid",
        )
        hybrid_result = payload["methods"]["hybrid"]
        self.assertEqual(
            hybrid_result["consensus_evaluation"]["binary_metrics_on_covered"]["f1"],
            1.0,
        )
        self.assertEqual(hybrid_result["full_set_selective_evaluation"]["coverage"], 0.4)
        self.assertEqual(
            hybrid_result["full_set_selective_evaluation"]["unresolved_detection"]["recall"],
            1.0,
        )
        self.assertEqual(
            hybrid_result["full_set_selective_evaluation"]["binary_reference_evaluation"]
            ["binary_metrics_on_covered"]["confusion"],
            {"tp": 1, "tn": 1, "fp": 0, "fn": 0},
        )
        forced = payload["methods"]["hybrid_forced_reject"]
        self.assertEqual(forced["full_set_selective_evaluation"]["coverage"], 1.0)
        self.assertEqual(
            forced["full_set_selective_evaluation"]["unresolved_detection"]["recall"],
            0.0,
        )
        self.assertEqual(
            hybrid_result["consensus_evaluation"]["bootstrap_95_ci"]["samples"], 200
        )

    def test_mcnemar_is_restricted_to_common_coverage(self) -> None:
        truth = {1: "ACCEPT", 2: "REJECT", 3: "ACCEPT"}
        left = {1: "ACCEPT", 2: "UNRESOLVED", 3: "REJECT"}
        right = {1: "REJECT", 2: "REJECT", 3: "ACCEPT"}
        result = mcnemar_exact(truth, left, right)
        self.assertEqual(result["common_coverage_n"], 2)
        self.assertEqual(result["a_correct_b_wrong"], 1)
        self.assertEqual(result["a_wrong_b_correct"], 1)
        self.assertEqual(result["two_sided_exact_p"], 1.0)

    def test_graph_semantics_does_not_treat_exact_offsets_as_semantic_accuracy(self) -> None:
        references = self.root / "semantic.jsonl"
        predictions = self.root / "graphs.jsonl"
        write_jsonl(
            references,
            [
                {
                    "candidate_id": 1,
                    "semantic_review_status": "human_semantic_review_complete",
                    "semantic_reference": {
                        "vehicle_entities": ["vehicle-a", "vehicle-b"],
                        "collision_agent": "vehicle-a",
                        "collision_target": "vehicle-b",
                        "legal_status": "accepted_fact",
                        "evidence_span_sufficient": True,
                    },
                }
            ],
        )
        write_jsonl(
            predictions,
            [
                {
                    "candidate_id": 1,
                    "vehicle_entities": ["vehicle-a", "vehicle-b"],
                    "collision_agent": "vehicle-a",
                    "collision_target": "vehicle-b",
                    "legal_status": "accepted_fact",
                    "evidence_span_sufficient": True,
                    "unsupported_relation_count": 0,
                    "relation_count": 2,
                    "resolver_status": "resolved",
                    "exact_offset_valid": True,
                }
            ],
        )
        result = evaluate_graph_semantics(references, predictions)
        self.assertEqual(result["vehicle_entity"]["f1"], 1.0)
        self.assertEqual(result["evidence_span_semantic_sufficiency_accuracy"], 1.0)
        self.assertIn("not counted", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
