"""Check that the continuation job carries its required runtime environment."""
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("continue_alfworld.sbatch")


def fake_cluster(root, headers=True):
    root = Path(root)
    (root / "weight.bin").write_bytes(b"known weight")
    digest = hashlib.sha256(b"known weight").hexdigest()
    (root / "official-shards.sha256").write_text(f"{digest}  weight.bin\n")
    if headers:
        include = root / "python-headers" / "include"
        (include / "python3.10").mkdir(parents=True)
        (include / "python3.10" / "Python.h").write_text("header")
        (include / "x86_64-linux-gnu" / "python3.10").mkdir(parents=True)
        (include / "x86_64-linux-gnu" / "python3.10" / "pyconfig.h").write_text("config")
    python = root / ".alfworld-venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "sampler=%s\\ncpath=%s\\n" "$VLLM_USE_FLASHINFER_SAMPLER" "$CPATH"\n')
    python.chmod(0o755)
    return root / "work"


class ContinueBatchTest(unittest.TestCase):
    def test_job_restores_the_original_sampler_and_python_header_path(self):
        with tempfile.TemporaryDirectory() as directory:
            work = fake_cluster(directory)
            environment = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="1", CPATH="/existing/include")
            result = subprocess.run(["bash", str(SCRIPT), directory, str(work)],
                                    capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("sampler=0\n", result.stdout)
            self.assertIn(f"cpath={directory}/python-headers/include/python3.10:"
                          f"{directory}/python-headers/include:/existing/include\n", result.stdout)

    def test_job_stops_early_when_python_headers_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            work = fake_cluster(directory, headers=False)
            result = subprocess.run(["bash", str(SCRIPT), directory, str(work)],
                                    capture_output=True, text=True, env=os.environ.copy())
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Python headers", result.stderr)
            self.assertNotIn("sampler=", result.stdout)


if __name__ == "__main__":
    unittest.main()
