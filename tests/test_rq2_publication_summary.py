from __future__ import annotations

import unittest

from scripts.summarize_rq2_publication import (
    _collision_evidence_quotes,
    _quote_alignment,
)


class Rq2PublicationSummaryTests(unittest.TestCase):
    def test_quote_alignment_requires_exact_containment(self) -> None:
        self.assertTrue(_quote_alignment(["vehicle A struck vehicle B"], ["A struck vehicle B"]))
        self.assertTrue(_quote_alignment(["struck vehicle B"], ["vehicle A struck vehicle B"]))
        self.assertFalse(_quote_alignment(["vehicle A changed lanes"], ["vehicle A struck vehicle B"]))
        self.assertFalse(_quote_alignment([], ["vehicle A struck vehicle B"]))

    def test_collision_quotes_only_use_supported_collision_edges(self) -> None:
        graph = {
            "evidence": [
                {"id": "collision", "quote": "vehicle A struck vehicle B"},
                {"id": "maneuver", "quote": "vehicle A changed lanes"},
            ],
            "edges": [
                {
                    "relation": "collides_with",
                    "supported": True,
                    "evidence_ids": ["collision"],
                },
                {
                    "relation": "collides_with",
                    "supported": False,
                    "evidence_ids": ["maneuver"],
                },
                {
                    "relation": "precedes",
                    "supported": True,
                    "evidence_ids": ["maneuver"],
                },
            ],
        }
        self.assertEqual(
            _collision_evidence_quotes(graph), ["vehicle A struck vehicle B"]
        )


if __name__ == "__main__":
    unittest.main()
