"""Checks for the bounded local inference-server lifecycle (no GPU needed)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from pilot_server import free_local_port, ready_server
except ImportError:
    free_local_port = ready_server = None
try:
    from pilot_server import build_eval_commands
except ImportError:
    build_eval_commands = None
try:
    from pilot_server import verify_server_reference
except ImportError:
    verify_server_reference = None


class PilotServerTest(unittest.TestCase):
    def test_server_reference_allows_only_a_different_ephemeral_port(self):
        self.assertIsNotNone(verify_server_reference, "server configuration checking is missing")
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "alfworld"
            reference.mkdir()
            prior = ["vllm", "serve", "model", "--port", "123", "--max-model-len", "4096"]
            (reference.parent / "config.json").write_text(json.dumps({"server_command": prior}))
            same = ["vllm", "serve", "model", "--port", "456", "--max-model-len", "4096"]
            verify_server_reference(reference, same)
            with self.assertRaises(ValueError):
                verify_server_reference(reference, same[:-1] + ["2048"])
            with self.assertRaises(ValueError):
                verify_server_reference(reference, ["vllm", "serve", "other", "--port", "456",
                                                    "--max-model-len", "4096"])

    def test_alfworld_only_forwards_guide_and_reference_without_changing_common_args(self):
        self.assertIsNotNone(build_eval_commands, "evaluation command construction is missing")
        common = ["python", "evaluate.py", "--model", "model", "--max-steps", "20"]
        commands = build_eval_commands(common, Path("cases.json"), Path("data"), Path("run"),
                                       only_alfworld=True, action_guide=True,
                                       reference_dir=Path("old/alfworld"))
        self.assertEqual(list(commands), ["alfworld"])
        command = commands["alfworld"]
        self.assertEqual(command[:len(common)], common)
        self.assertIn("--alfworld-action-guide", command)
        self.assertEqual(command[command.index("--reference-alfworld-dir") + 1], "old/alfworld")
        default = build_eval_commands(common, Path("cases.json"), Path("data"), Path("run"))
        self.assertEqual(list(default), ["blocksworld", "alfworld"])
        self.assertNotIn("--alfworld-action-guide", default["alfworld"])
        self.assertNotIn("--reference-alfworld-dir", default["alfworld"])

    def test_server_is_ready_and_stopped_even_when_the_caller_fails(self):
        self.assertIsNotNone(ready_server, "ready_server has not been implemented")
        port = free_local_port()
        code = (
            "from http.server import HTTPServer, BaseHTTPRequestHandler\n"
            "import sys\n"
            "class H(BaseHTTPRequestHandler):\n"
            " def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b'{}')\n"
            "HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "server.log").open("w") as log:
                with self.assertRaisesRegex(RuntimeError, "caller failed"):
                    with ready_server([sys.executable, "-u", "-c", code, str(port)],
                                      f"http://127.0.0.1:{port}/v1/models", log, 10) as process:
                        self.assertIsNone(process.poll())
                        raise RuntimeError("caller failed")
                self.assertIsNotNone(process.poll())

    def test_server_exit_is_reported_instead_of_waiting_for_timeout(self):
        self.assertIsNotNone(ready_server, "ready_server has not been implemented")
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "server.log").open("w") as log:
                with self.assertRaisesRegex(RuntimeError, "exited"):
                    with ready_server([sys.executable, "-c", "import sys; sys.exit(7)"],
                                      f"http://127.0.0.1:{free_local_port()}/v1/models", log, 5):
                        pass


if __name__ == "__main__":
    unittest.main()
