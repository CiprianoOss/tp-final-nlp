import unittest

from fireball_narrator.evaluation.evaluate_retention import (
    answer_is_correct,
    fact_is_preserved,
)


class RetentionEvaluationTests(unittest.TestCase):
    def test_closed_answer_matching_uses_word_boundaries(self):
        self.assertTrue(answer_is_correct("The liquid was red.", ["red"]))
        self.assertFalse(answer_is_correct("The liquid was redder.", ["red"]))

    def test_fact_keywords_must_all_survive(self):
        fact = {"keywords": ["silver bucket", "red liquid"]}
        self.assertTrue(
            fact_is_preserved(fact, "Turn 5: silver bucket held red liquid")
        )
        self.assertFalse(fact_is_preserved(fact, "Turn 5: silver bucket was empty"))


if __name__ == "__main__":
    unittest.main()

