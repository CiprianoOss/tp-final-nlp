import unittest

from fireball_narrator.evaluation.metrics import (
    information_retention_accuracy,
    semantic_preservation_score,
    thematic_adherence_rate,
    token_compression_ratio,
)


class MetricTests(unittest.TestCase):
    def test_rates(self):
        self.assertEqual(thematic_adherence_rate([True, False, True]), 2 / 3)
        self.assertEqual(information_retention_accuracy([True, True]), 1.0)
        self.assertEqual(token_compression_ratio(100, 25), 0.75)

    def test_fact_preservation(self):
        score = semantic_preservation_score(
            ["Potion is red", "Door is locked"],
            ["door is locked"],
        )
        self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()

