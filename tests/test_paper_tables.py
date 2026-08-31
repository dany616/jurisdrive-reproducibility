from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_ieee_iotj_tables import DEFAULT_GOLD, build_tables


class PaperTableTests(unittest.TestCase):
    def test_tables_preserve_measured_and_pending_denominators(self) -> None:
        self.assertTrue(DEFAULT_GOLD.is_file())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = build_tables(DEFAULT_GOLD, output)
            self.assertEqual(manifest["denominator_guard"]["binary_consensus"], 743)
            self.assertEqual(manifest["denominator_guard"]["fidelity_runs"], 96)
            self.assertEqual(manifest["denominator_guard"]["assurance_artifacts"], 168)

            with (output / "table_ii_selective_gold_metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                metrics = {row["Method"]: row for row in csv.DictReader(handle)}
            self.assertEqual(metrics["Selective Hybrid"]["Consensus n/covered"], "743/659")
            self.assertEqual(metrics["Qwen Only"]["TP/TN/FP/FN"], "277/415/8/4")

            text = (output / "table_iii_grounding_fidelity_assurance.csv").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("PENDING", text)
            self.assertIn("96", text)
            self.assertIn("168", text)
            self.assertIn("72 mutable + 72 immutable", text)


if __name__ == "__main__":
    unittest.main()
