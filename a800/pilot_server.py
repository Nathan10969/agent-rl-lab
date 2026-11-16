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


def build_eval_commands(common, cases, data_dir, output_dir, only_alfworld=False,
                        action_guide=False, reference_dir=None, paired_manifest=None):
    if paired_manifest is not None:
        if not only_alfworld or action_guide:
            raise ValueError("paired evaluation requires ALFWorld-only and controls its own guide")
        base = common + ["--task", "alfworld", "--data-dir", str(data_dir),
                         "--output-dir", str(output_dir / "alfworld-base"),
                         "--alfworld-manifest", str(paired_manifest)]
        guided = common + ["--task", "alfworld", "--data-dir", str(data_dir),
                           "--output-dir", str(output_dir / "alfworld-guide"),
                           "--alfworld-manifest", str(paired_manifest), "--alfworld-action-guide",
                           "--reference-alfworld-dir", str(output_dir / "alfworld-base")]
        return {"alfworld-base": base, "alfworld-guide": guided}
    commands = {
        "blocksworld": common + ["--task", "blocksworld", "--cases", str(cases),
                                 "--output-dir", str(output_dir / "blocksworld")],
        "alfworld": common + ["--task", "alfworld", "--data-dir", str(data_dir),
                              "--output-dir", str(output_dir / "alfworld")],
    }
    if action_guide:
        commands["alfworld"].append("--alfworld-action-guide")
    if reference_dir is not None:
        commands["alfworld"] += ["--reference-alfworld-dir", str(reference_dir)]
    if only_alfworld:
        del commands["blocksworld"]
    return commands


def verify_server_reference(reference_dir, server_command):
    """Allow the local ephemeral port to change, but not the server settings."""
    previous = json.loads((Path(reference_dir).parent / "config.json").read_text())["server_command"]

    def without_port(command):
        if command.count("--port") != 1:
            raise ValueError("server command must contain exactly one port")
        index = command.index("--port")
        if index + 1 >= len(command):
            raise ValueError("server port is missing")
        return command[:index] + command[index + 2:]

    if without_port(previous) != without_port(server_command):
        raise ValueError("baseline server settings or model path do not match")


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
    parser.add_argument("--only-alfworld", action="store_true")
    parser.add_argument("--alfworld-action-guide", action="store_true")
    parser.add_argument("--reference-alfworld-dir", type=Path)
    parser.add_argument("--paired-alfworld-manifest", type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.paired_alfworld_manifest and (not args.only_alfworld or args.alfworld_action_guide):
        parser.error("paired evaluation requires --only-alfworld and controls its own guide")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    port = free_local_port()
    url = f"http://127.0.0.1:{port}"
    model_name = "qwen27b-fp8"
    server_command = [str(args.vllm), "serve", str(args.model_path), "--host", "127.0.0.1",
                      "--port", str(port), "--served-model-name", model_name,
                      "--max-model-len", "4096", "--max-num-seqs", "1",
                      "--gpu-memory-utilization", "0.8", "--enforce-eager", "--language-model-only"]
    if args.reference_alfworld_dir:
        verify_server_reference(args.reference_alfworld_dir, server_command)
    eval_script = Path(__file__).resolve().parent / "evaluate.py"
    common = [str(args.eval_python), str(eval_script), "--model", model_name, "--url", url,
              "--max-steps", str(args.max_steps)]
    commands = build_eval_commands(common, args.cases, args.alfworld_data, args.output_dir,
                                   only_alfworld=args.only_alfworld,
                                   action_guide=args.alfworld_action_guide,
                                   reference_dir=args.reference_alfworld_dir,
                                   paired_manifest=args.paired_alfworld_manifest)
    save_json_new(args.output_dir / "config.json", {
        "server_command": server_command, "evaluation_commands": commands,
        "max_model_len": 4096, "max_steps": args.max_steps,
        "temperature": 0, "thinking": False, "text_only": True,
        "reference_server_matched": args.reference_alfworld_dir is not None,
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
                                                   timeout=2400 if args.paired_alfworld_manifest else 1200,
                                                   check=False)
                    task_result = {"exit_code": completed.returncode}
                    summary_path = args.output_dir / task / "summary.json"
                    if summary_path.exists():
                        task_result["summary"] = json.loads(summary_path.read_text())
                    result["tasks"][task] = task_result
                    print(f"{task}: exit={completed.returncode}", flush=True)
                    if args.paired_alfworld_manifest and task == "alfworld-base" and completed.returncode:
                        raise RuntimeError("baseline failed; guided arm was not started")
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
