import unittest

from fireball_narrator.data.formatting import record_to_chat


class FormattingTests(unittest.TestCase):
    def test_state_to_narration_chat(self):
        record = {
            "speaker_id": "player-1",
            "utterance_history": ["Player: I strike the skeleton."],
            "combat_state_after": [
                {
                    "name": "Arin",
                    "race": "Human",
                    "class": "Fighter 3",
                    "hp": "<8/20 HP; Injured>",
                    "effects": "",
                }
            ],
            "caster_after": {
                "name": "Arin",
                "race": "Human",
                "class": "Fighter 3",
                "hp": "<8/20 HP; Injured>",
                "attacks": "Longsword",
            },
            "targets_after": [],
            "commands_norm": ["!attack longsword"],
            "automation_results": ["Arin hits Skeleton for 7 damage."],
            "after_utterances": ["Arin's blade cracks through the old bones."],
        }

        example = record_to_chat(record)

        self.assertIsNotNone(example)
        self.assertEqual(example["messages"][-1]["role"], "assistant")
        self.assertIn("Resolved action", example["messages"][1]["content"])
        self.assertEqual(example["metadata"]["task"], "state_to_narration")

    def test_missing_target_is_skipped(self):
        self.assertIsNone(
            record_to_chat({"after_utterances": [], "automation_results": ["result"]})
        )


if __name__ == "__main__":
    unittest.main()

