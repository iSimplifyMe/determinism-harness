"""Tests for the logprobs exposure probe's classification helper."""
import unittest

from harness.probe_logprobs import find_logprob_keys


class TestFindLogprobKeys(unittest.TestCase):
    def test_plain_payload_has_none(self):
        payload = {"model": "m", "message": {"role": "a", "content": "x"}}
        self.assertEqual(find_logprob_keys(payload), [])

    def test_nested_logprob_fields_located(self):
        payload = {
            "message": {"content": "x"},
            "logprobs": [{"token": "x", "logprob": -0.1,
                          "top_logprobs": [{"token": "y", "logprob": -2.0}]}],
        }
        keys = find_logprob_keys(payload)
        self.assertIn("logprobs", keys)
        self.assertTrue(any("top_logprobs" in k for k in keys))

    def test_case_insensitive_match(self):
        self.assertEqual(
            find_logprob_keys({"choices": [{"LogProbs": {}}]}),
            ["choices[0].LogProbs"],
        )


if __name__ == "__main__":
    unittest.main()
