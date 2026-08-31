from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "analysis" / "audit_current_data.py"
SPEC = importlib.util.spec_from_file_location("audit_current_data", MODULE_PATH)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


class AuditCurrentDataTest(unittest.TestCase):
    def test_readiness_tier_a(self) -> None:
        parsed = {
            "vehicle_type": "승용차 2대",
            "accident_trajectory": "후행 차량이 선행 차량을 추돌",
            "road_type": "교차로",
        }
        self.assertEqual(audit_module.readiness_tier(parsed), "A_minimum_grounded")

    def test_readiness_tier_b(self) -> None:
        parsed = {
            "vehicle_type": "승용차 2대",
            "accident_trajectory": "후행 차량이 선행 차량을 추돌",
        }
        self.assertEqual(audit_module.readiness_tier(parsed), "B_defaults_needed")

    def test_readiness_tier_c(self) -> None:
        parsed = {"accident_trajectory": "충돌"}
        self.assertEqual(audit_module.readiness_tier(parsed), "C_reextract_or_review")

    def test_nonempty(self) -> None:
        self.assertFalse(audit_module.nonempty(None))
        self.assertFalse(audit_module.nonempty(""))
        self.assertTrue(audit_module.nonempty("교차로"))


if __name__ == "__main__":
    unittest.main()

