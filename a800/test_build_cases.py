"""The fixed smoke cases can be recovered from a regenerated validation set."""
import unittest

try:
    from build_cases import extract_cases
except ImportError:
    extract_cases = None


class BuildCasesTest(unittest.TestCase):
    def test_extracts_fixed_cases_in_difficulty_order(self):
        self.assertIsNotNone(extract_cases, "extract_cases has not been implemented")
        rows = [
            {"id": "bw-v1-val-0370", "plan_len": 8, "init": [["C"]], "goal": [["C", "table"]]},
            {"id": "bw-v1-val-0326", "plan_len": 4, "init": [["A"]], "goal": [["A", "table"]]},
            {"id": "bw-v1-val-0166", "plan_len": 6, "init": [["B"]], "goal": [["B", "table"]]},
        ]
        cases = extract_cases(rows)
        self.assertEqual([case["source_id"] for case in cases],
                         ["bw-v1-val-0326", "bw-v1-val-0166", "bw-v1-val-0370"])
        self.assertEqual([case["oracle_steps"] for case in cases], [4, 6, 8])
        self.assertEqual(cases[0]["init"], [["A"]])
        with self.assertRaises(ValueError):
            extract_cases(rows[:-1])
        with self.assertRaises(ValueError):
            extract_cases(rows + [rows[0]])


if __name__ == "__main__":
    unittest.main()
