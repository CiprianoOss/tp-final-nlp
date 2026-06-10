import unittest

from fireball_narrator.compression.caveman import compress_text, token_compression_ratio


class CompressionTests(unittest.TestCase):
    def test_preserves_negation_numbers_and_content(self):
        compressed = compress_text(
            "The red potion was not under the table after turn 5."
        )
        self.assertIn("red", compressed)
        self.assertIn("not", compressed)
        self.assertIn("under", compressed)
        self.assertIn("5", compressed)

    def test_ratio(self):
        ratio = token_compression_ratio("one two three four", "one three")
        self.assertEqual(ratio, 0.5)


if __name__ == "__main__":
    unittest.main()

