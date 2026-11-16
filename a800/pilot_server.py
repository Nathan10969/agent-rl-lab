"""Start one bounded local model server and run both small baseline sets."""
import argparse
import json
import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from baseline_eval import save_json_new
from gpu_monitor import GPUMonitor, identify_allocated_gpu, summarize_memory


def free_local_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def ready_server(command, ready_url, log, timeout):
    """Run a server in its own process group and always shut that group down."""
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"model server exited with code {process.returncode}")
            try:
                with urlopen(ready_url, timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError, OSError):
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError("model server did not become ready within the time limit")
        yield process
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    port = free_local_port()
    url = f"http://127.0.0.1:{port}"
    model_name = "qwen27b-fp8"
    server_command = [str(args.vllm), "serve", str(args.model_path), "--host", "127.0.0.1",
                      "--port", str(port), "--served-model-name", model_name,
                      "--max-model-len", "4096", "--max-num-seqs", "1",
                      "--gpu-memory-utilization", "0.8", "--enforce-eager", "--language-model-only"]
    eval_script = Path(__file__).resolve().parent / "evaluate.py"
    common = [str(args.eval_python), str(eval_script), "--model", model_name, "--url", url,
              "--max-steps", str(args.max_steps)]
    commands = {
        "blocksworld": common + ["--task", "blocksworld", "--cases", str(args.cases),
                                 "--output-dir", str(args.output_dir / "blocksworld")],
        "alfworld": common + ["--task", "alfworld", "--data-dir", str(args.alfworld_data),
                              "--output-dir", str(args.output_dir / "alfworld")],
    }
    save_json_new(args.output_dir / "config.json", {
        "server_command": server_command, "evaluation_commands": commands,
        "max_model_len": 4096, "max_steps": args.max_steps,
        "temperature": 0, "thinking": False, "text_only": True,
    })

    started = time.monotonic()
    monitor = None
    result = {"status": "error", "tasks": {}}
    try:
        uuid = identify_allocated_gpu()
        if not uuid:
            raise RuntimeError("could not identify the allocated GPU")
        monitor = GPUMonitor(uuid)
        monitor.start()
        result["gpu_uuid"] = uuid
        with (args.output_dir / "server.log").open("x", encoding="utf-8") as server_log:
            with ready_server(server_command, url + "/v1/models", server_log, timeout=600):
                print("Model server ready; evaluating fixed development cases.", flush=True)
                for task, command in commands.items():
                    with (args.output_dir / f"{task}.log").open("x", encoding="utf-8") as log:
                        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                   timeout=1200, check=False)
                    task_result = {"exit_code": completed.returncode}
                    summary_path = args.output_dir / task / "summary.json"
                    if summary_path.exists():
                        task_result["summary"] = json.loads(summary_path.read_text())
                    result["tasks"][task] = task_result
                    print(f"{task}: exit={completed.returncode}", flush=True)
        result["status"] = "ok" if all(task["exit_code"] == 0 for task in result["tasks"].values()) else "error"
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    finally:
        if monitor:
            monitor.stop()
            result["gpu_memory"] = summarize_memory(monitor.samples)
            result["gpu_monitor_errors"] = monitor.errors
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save_json_new(args.output_dir / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
