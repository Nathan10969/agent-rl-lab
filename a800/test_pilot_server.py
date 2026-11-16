"""Checks for the bounded local inference-server lifecycle (no GPU needed)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from pilot_server import free_local_port, ready_server
except ImportError:
    free_local_port = ready_server = None


class PilotServerTest(unittest.TestCase):
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
