import unittest

from fireball_narrator.decoding import build_bad_words_ids


class FakeTokenizer:
    def encode(self, value, add_special_tokens=False):
        self.assert_no_special_tokens = not add_special_tokens
        return [ord(character) for character in value]


class DecodingTests(unittest.TestCase):
    def test_builds_distinct_variants(self):
        sequences = build_bad_words_ids(FakeTokenizer(), ["jetpack"])
        self.assertIn([ord(character) for character in "jetpack"], sequences)
        self.assertIn([ord(character) for character in " jetpack"], sequences)
        self.assertEqual(len(sequences), len({tuple(item) for item in sequences}))


if __name__ == "__main__":
    unittest.main()

