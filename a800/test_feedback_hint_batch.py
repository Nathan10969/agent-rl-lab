"""CPU checks for the bounded four-case GPU job setup."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_continue_batch import fake_cluster


SCRIPT = Path(__file__).with_name("feedback_hint.sbatch")


class FeedbackHintBatchTest(unittest.TestCase):
    def test_job_carries_baseline_environment_and_control_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            work = fake_cluster(directory)
            python = Path(directory) / ".alfworld-venv" / "bin" / "python"
            python.write_text('#!/bin/sh\nprintf "sampler=%s\\ncpath=%s\\nargs=%s\\n" '
                              '"$VLLM_USE_FLASHINFER_SAMPLER" "$CPATH" "$*"\n')
            control = Path(directory) / "old-control"
            result = subprocess.run(["bash", str(SCRIPT), directory, str(work), str(control)],
                                    capture_output=True, text=True,
                                    env=dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="1", CPATH="/existing"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("sampler=0\n", result.stdout)
            self.assertIn(f"cpath={directory}/python-headers/include/python3.10:"
                          f"{directory}/python-headers/include:/existing\n", result.stdout)
            self.assertIn("feedback_hint_eval.py", result.stdout)
            self.assertIn(f"--control-dir {control}", result.stdout)

    def test_missing_headers_prevent_model_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            work = fake_cluster(directory, headers=False)
            result = subprocess.run(["bash", str(SCRIPT), directory, str(work), str(Path(directory) / "old")],
                                    capture_output=True, text=True, env=os.environ.copy())
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Python headers", result.stderr)
            self.assertNotIn("sampler=", result.stdout)


if __name__ == "__main__":
    unittest.main()
