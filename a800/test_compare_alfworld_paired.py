"""Paired summaries must compare the same cases and starting observations."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    from compare_alfworld_paired import compare_paired_results
except ImportError:
    compare_paired_results = None


class PairedSummaryTest(unittest.TestCase):
    def test_counts_paired_success_and_feedback_by_family(self):
        self.assertIsNotNone(compare_paired_results, "paired summary is missing")
        cases = [{"source_id": f"game-{i}", "task_family": family, "prior_case": i == 0}
                 for i, family in enumerate(("a", "a", "b"))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, guide = root / "base", root / "guide"
            base.mkdir()
            guide.mkdir()
            for index, (old, new) in enumerate(((False, True), (True, False), (False, False))):
                for folder, won in ((base, old), (guide, new)):
                    result = {"source_id": f"game-{index}", "initial_observation_sha256": f"hash-{index}",
                              "success": won, "trace": [{"feedback": "Nothing happens."}]}
                    (folder / f"episode-{index:03d}.json").write_text(json.dumps(result))
            result = compare_paired_results({"cases": cases}, base, guide)
            self.assertEqual((result["total"], result["base_success"], result["guide_success"]), (3, 1, 1))
            self.assertEqual((result["improved"], result["worsened"], result["both_failed"]), (1, 1, 1))
            self.assertEqual(result["families"]["a"]["total"], 2)
            self.assertIn("subsets", result)
            self.assertEqual(result["subsets"]["previous"]["improved"], 1)
            self.assertEqual(result["subsets"]["new"]["total"], 2)
            self.assertEqual(result["subsets"]["new"]["worsened"], 1)
            self.assertEqual((result["base_no_effect"], result["guide_no_effect"]), (3, 3))
            bad = json.loads((guide / "episode-000.json").read_text())
            bad["initial_observation_sha256"] = "different"
            (guide / "episode-000.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                compare_paired_results({"cases": cases}, base, guide)


if __name__ == "__main__":
    unittest.main()
