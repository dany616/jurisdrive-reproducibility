from __future__ import annotations

import unittest

from jurisdrive.rq4_task_a import (
    conservative_repeat_consensus,
    paired_common_coverage_test,
    task_a_metrics,
    validate_verdict,
)
from scripts.run_rq4_sanitized_task_a import vllm_compatible_response_schema


class SanitizedTaskAExecutionTests(unittest.TestCase):
    def test_vllm_schema_compatibility_keeps_validator_constraint(self) -> None:
        schema = vllm_compatible_response_schema()
        self.assertNotIn("uniqueItems", schema["properties"]["issue_codes"])
        invalid, error = validate_verdict(
            {
                "status": "FAIL",
                "issue_codes": ["EVENT_SEQUENCE", "EVENT_SEQUENCE"],
                "rationale": "Duplicate.",
            }
        )
        self.assertIsNone(invalid)
        self.assertEqual(error, "duplicate issue_codes")

    def test_verdict_schema_accepts_only_frozen_values(self) -> None:
        valid, error = validate_verdict(
            {"status": "PASS", "issue_codes": [], "rationale": "Supported."}
        )
        self.assertIsNone(error)
        self.assertEqual(valid["status"], "PASS")
        invalid, error = validate_verdict(
            {"status": "REPAIR", "issue_codes": [], "rationale": "No."}
        )
        self.assertIsNone(invalid)
        self.assertIn("invalid status", error)

    def test_conservative_repeat_consensus_abstains_on_any_disagreement(self) -> None:
        self.assertEqual(
            conservative_repeat_consensus(["FAIL", "FAIL", "FAIL"]), "FAIL"
        )
        self.assertEqual(
            conservative_repeat_consensus(["FAIL", "PASS", "FAIL"]),
            "MANUAL_REVIEW",
        )
        self.assertEqual(
            conservative_repeat_consensus(["PASS", None, "PASS"]),
            "MANUAL_REVIEW",
        )

    def test_metrics_keep_reviews_in_full_denominators(self) -> None:
        rows = [
            {"is_fault": True, "status": "FAIL"},
            {"is_fault": True, "status": "PASS"},
            {"is_fault": True, "status": "MANUAL_REVIEW"},
            {"is_fault": False, "status": "PASS"},
            {"is_fault": False, "status": "FAIL"},
            {"is_fault": False, "status": "MANUAL_REVIEW"},
        ]
        metrics = task_a_metrics(rows)
        self.assertEqual(metrics["n"], 6)
        self.assertEqual(metrics["decisive_confusion"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertAlmostEqual(metrics["recall_full_fault_denominator"], 1 / 3)
        self.assertAlmostEqual(metrics["specificity_full_clean_denominator"], 1 / 3)
        self.assertAlmostEqual(metrics["coverage"], 4 / 6)
        self.assertAlmostEqual(metrics["manual_review_rate"], 2 / 6)

    def test_paired_test_uses_only_common_decisive_rows(self) -> None:
        left = []
        right = []
        for index in range(24):
            artifact = f"A{index:02d}"
            common = {
                "opaque_artifact_id": artifact,
                "judgment_slot": f"J{index:02d}",
                "is_fault": bool(index % 2),
            }
            left.append({**common, "status": "FAIL" if index % 2 else "PASS"})
            right.append(
                {
                    **common,
                    "status": "MANUAL_REVIEW" if index == 0 else "FAIL" if index % 2 else "PASS",
                }
            )
        result = paired_common_coverage_test(left, right, samples=100, seed=7)
        self.assertEqual(result["common_decisive_artifacts"], 23)
        self.assertEqual(result["common_judgment_clusters"], 23)


if __name__ == "__main__":
    unittest.main()
