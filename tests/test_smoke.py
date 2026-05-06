import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import constraint_filter as cf
import score_submissions as scorer


class SmokeTest(unittest.TestCase):
    def test_eval_data_loads(self):
        challenges = json.loads((ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_challenges.json").read_text())
        solutions = json.loads((ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_solutions.json").read_text())
        self.assertEqual(len(challenges), 120)
        self.assertEqual(len(solutions), 120)
        self.assertEqual(sum(len(v) for v in solutions.values()), 172)

    def test_constraint_filter_ranks_valid_copy(self):
        examples = [
            {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]},
            {"input": [[2, 0], [0, 2]], "output": [[2, 0], [0, 2]]},
        ]
        test_input = [[3, 0], [0, 3]]
        good = [[3, 0], [0, 3]]
        bad = [[3, 3], [3, 3]]
        ranked = cf.rank_attempts([bad, good], test_input, examples)
        self.assertEqual(ranked[0][0], good)
        self.assertEqual(ranked[0][1], 0)

    def test_sample_submission_scores_without_crash(self):
        sample = json.loads((ROOT / "data" / "arc-prize-2026-arc-agi-2" / "sample_submission.json").read_text())
        solutions = json.loads((ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_solutions.json").read_text())
        results = scorer.score_submission(sample, solutions)
        self.assertEqual(set(results), set(sample))


if __name__ == "__main__":
    unittest.main(verbosity=2)
