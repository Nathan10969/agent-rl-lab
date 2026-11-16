"""CPU-only checks for the real Blocksworld baseline loop."""
import sys
import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from baseline_eval import run_blocksworld
except ImportError:
    run_blocksworld = None
try:
    from baseline_eval import request_action
except ImportError:
    request_action = None
try:
    from baseline_eval import run_alfworld
except ImportError:
    run_alfworld = None
try:
    from baseline_eval import ALFWORLD_ACTION_GUIDE
except ImportError:
    ALFWORLD_ACTION_GUIDE = None
try:
    from baseline_eval import select_block_cases
except ImportError:
    select_block_cases = None
try:
    from baseline_eval import save_json_new
except ImportError:
    save_json_new = None
try:
    from baseline_eval import summarize
except ImportError:
    summarize = None
try:
    from gpu_monitor import find_gpu_uuid, parse_gpu_memory
except ImportError:
    find_gpu_uuid = parse_gpu_memory = None
try:
    from gpu_monitor import summarize_memory
except ImportError:
    summarize_memory = None
try:
    from alfworld_probe import build_alfworld_config
except ImportError:
    build_alfworld_config = None
try:
    from alfworld_probe import select_alfworld_games
except ImportError:
    select_alfworld_games = None
try:
    from alfworld_probe import verify_alfworld_reference
except ImportError:
    verify_alfworld_reference = None


class BaselineEvalTest(unittest.TestCase):
    def test_alfworld_cli_counts_won_and_records_task_name(self):
        script = Path(__file__).resolve().parent / "evaluate.py"
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                response = json.dumps({"choices": [{"message": {"content": "look"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                module = root / "fake" / "alfworld" / "agents" / "environment" / "alfred_tw_env.py"
                module.parent.mkdir(parents=True)
                # Make the fake a regular package, so an installed alfworld cannot
                # shadow this test double when the test runs on the cluster.
                for package in (module.parent, module.parent.parent, module.parent.parent.parent):
                    (package / "__init__.py").touch()
                module.write_text(
                    'from pathlib import Path\n'
                    'class Env:\n'
                    ' def reset(self): return ["task"], {}\n'
                    ' def step(self, actions): return ["won"], [1], [True], {"won": [True]}\n'
                    ' def close(self): pass\n'
                    'class AlfredTWEnv:\n'
                    ' def __init__(self, config, train_eval):\n'
                    '  base = Path(config["dataset"]["eval_id_data_path"])\n'
                    '  self.game_files = [str(base / f"task{i}-object" / "trial" / "game.tw-pddl") for i in range(3)]\n'
                    ' def init_env(self, batch_size): return Env()\n'
                )
                output = root / "run"
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(root / "fake")
                command = [sys.executable, str(script), "--task", "alfworld", "--model", "fake",
                           "--url", f"http://127.0.0.1:{server.server_port}", "--data-dir", str(root / "data"),
                           "--output-dir", str(output), "--max-steps", "3"]
                run = subprocess.run(command, capture_output=True, text=True, timeout=15, env=environment)
                self.assertEqual(run.returncode, 0, run.stderr)
                summary = json.loads((output / "summary.json").read_text())
                self.assertIn("task", summary)
                self.assertEqual(summary["task"], "alfworld")
                self.assertEqual((summary["success"], summary["total"]), (3, 3))
                guided_output = root / "guided"
                guided_command = command.copy()
                guided_command[guided_command.index("--output-dir") + 1] = str(guided_output)
                guided_command += ["--alfworld-action-guide", "--reference-alfworld-dir", str(output)]
                guided = subprocess.run(guided_command, capture_output=True, text=True, timeout=15, env=environment)
                self.assertEqual(guided.returncode, 0, guided.stderr)
                guided_config = json.loads((guided_output / "config.json").read_text())
                self.assertEqual(guided_config["prompt_variant"], "static-action-guide")
                self.assertTrue(json.loads((guided_output / "reference_check.json").read_text())["matched"])
                self.assertEqual(len(received), 6)
                self.assertNotIn(ALFWORLD_ACTION_GUIDE, received[0]["messages"][0]["content"])
                self.assertIn(ALFWORLD_ACTION_GUIDE, received[3]["messages"][0]["content"])
                self.assertEqual(received[0]["temperature"], received[3]["temperature"])

                mismatch_command = guided_command.copy()
                mismatch_command[mismatch_command.index("--output-dir") + 1] = str(root / "mismatch")
                mismatch_command[mismatch_command.index("--max-steps") + 1] = "4"
                mismatch = subprocess.run(mismatch_command, capture_output=True, text=True, timeout=15,
                                          env=environment)
                self.assertNotEqual(mismatch.returncode, 0)
                self.assertEqual(len(received), 6, "a mismatched comparison must stop before model requests")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_alfworld_configuration_uses_only_fixed_validation_split(self):
        self.assertIsNotNone(build_alfworld_config, "build_alfworld_config has not been implemented")
        config = build_alfworld_config(Path("/tmp/alfworld-example"), 20)
        self.assertEqual(config["dataset"]["eval_id_data_path"],
                         "/tmp/alfworld-example/json_2.1.1/valid_seen")
        self.assertEqual(config["dagger"]["training"]["max_nb_steps_per_episode"], 20)
        self.assertFalse(config["env"]["domain_randomization"])

    def test_alfworld_cases_cover_distinct_task_families(self):
        self.assertIsNotNone(select_alfworld_games, "select_alfworld_games has not been implemented")
        root = Path("/example/valid_seen")
        games = [root / "look_at_obj-clock" / "trial2" / "game.tw-pddl",
                 root / "look_at_obj-bowl" / "trial1" / "game.tw-pddl",
                 root / "pick_and_place-mug" / "trial1" / "game.tw-pddl",
                 root / "pick_clean-bowl" / "trial1" / "game.tw-pddl"]
        selected = select_alfworld_games(games, 3)
        self.assertEqual([p.parent.parent.name.split("-", 1)[0] for p in selected],
                         ["look_at_obj", "pick_and_place", "pick_clean"])
        with self.assertRaises(ValueError):
            select_alfworld_games(games[:2], 3)

    def test_alfworld_reference_rejects_different_cases_or_generation_settings(self):
        self.assertIsNotNone(verify_alfworld_reference, "reference verification has not been implemented")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            reference = root / "baseline"
            reference.mkdir()
            config = {"task": "alfworld", "model": "model", "max_steps": 20,
                      "temperature": 0, "max_new_tokens": 48, "thinking": False}
            (reference / "config.json").write_text(json.dumps(config))
            games = [data / "a" / "game.tw-pddl", data / "b" / "game.tw-pddl"]
            for index, game in enumerate(games):
                (reference / f"episode-{index:03d}.json").write_text(
                    json.dumps({"source_id": str(game.relative_to(data))}))
            self.assertEqual(verify_alfworld_reference(reference, data, games, "model", 20),
                             ["a/game.tw-pddl", "b/game.tw-pddl"])
            with self.assertRaises(ValueError):
                verify_alfworld_reference(reference, data, games[::-1], "model", 20)
            with self.assertRaises(ValueError):
                verify_alfworld_reference(reference, data, games, "model", 21)
            config["temperature"] = 0.8
            (reference / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                verify_alfworld_reference(reference, data, games, "model", 20)

    def test_peak_memory_uses_observed_samples_only(self):
        self.assertIsNotNone(summarize_memory, "summarize_memory has not been implemented")
        self.assertIsNone(summarize_memory([]))
        self.assertEqual(summarize_memory([(1000, 81920), (5600, 81920), (5000, 81920)]),
                         {"initial_used_mib": 1000, "sampled_peak_used_mib": 5600,
                          "total_mib": 81920, "samples": 3})

    def test_gpu_memory_is_matched_to_the_actual_process_uuid(self):
        self.assertIsNotNone(find_gpu_uuid, "GPU telemetry helpers have not been implemented")
        rows = "12, GPU-aaaa\n38, GPU-bbbb\n"
        self.assertEqual(find_gpu_uuid(rows, 38), "GPU-bbbb")
        self.assertIsNone(find_gpu_uuid(rows, 99))
        self.assertEqual(parse_gpu_memory("1234, 81920\n"), (1234, 81920))

    def test_blockworld_cli_runs_real_episodes_and_refuses_overwrite(self):
        script = Path(__file__).resolve().parent / "evaluate.py"
        self.assertTrue(script.is_file(), "evaluate.py has not been implemented")

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                action = "pickup(B)" if len(body["messages"]) == 1 else "stack(B, A)"
                response = json.dumps({"choices": [{"message": {"content": action}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cases = [{"source_id": str(n), "oracle_steps": n, "init": [["A"], ["B"]],
                          "goal": [["B", "A"]]} for n in (4, 6, 8)]
                case_file = root / "cases.json"
                case_file.write_text(json.dumps({"cases": cases}))
                output = root / "run"
                command = [sys.executable, str(script), "--task", "blocksworld", "--model", "fake",
                           "--url", f"http://127.0.0.1:{server.server_port}", "--cases", str(case_file),
                           "--output-dir", str(output), "--max-steps", "3"]
                first = subprocess.run(command, capture_output=True, text=True, timeout=15)
                self.assertEqual(first.returncode, 0, first.stderr)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual((summary["success"], summary["total"]), (3, 3))
                second = subprocess.run(command, capture_output=True, text=True, timeout=15)
                self.assertNotEqual(second.returncode, 0)
                self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_summary_does_not_count_unfinished_as_success(self):
        self.assertIsNotNone(summarize, "summarize has not been implemented")
        episodes = [{"success": True, "trace": [{"format_valid": True, "legal": True}]},
                    {"success": False, "trace": [{"format_valid": False, "legal": False}]}]
        summary = summarize(episodes)
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["actions"], 2)
        self.assertEqual(summary["format_valid_actions"], 1)
        self.assertEqual(summary["legal_actions"], 1)

    def test_saved_results_never_overwrite_previous_files(self):
        self.assertIsNotNone(save_json_new, "save_json_new has not been implemented")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_json_new(path, {"success": True})
            with self.assertRaises(FileExistsError):
                save_json_new(path, {"success": False})
            self.assertTrue(json.loads(path.read_text())["success"])

    def test_fixed_development_cases_cover_three_lengths(self):
        self.assertIsNotNone(select_block_cases, "select_block_cases has not been implemented")
        cases = [{"source_id": "a", "oracle_steps": 4}, {"source_id": "b", "oracle_steps": 4},
                 {"source_id": "c", "oracle_steps": 6}, {"source_id": "d", "oracle_steps": 8}]
        self.assertEqual([case["source_id"] for case in select_block_cases(cases)], ["a", "c", "d"])
        with self.assertRaises(ValueError):
            select_block_cases(cases[:3])

    def test_two_legal_actions_finish_the_real_environment(self):
        self.assertIsNotNone(run_blocksworld, "run_blocksworld has not been implemented")
        case = {"source_id": "two-block", "init": [["A"], ["B"]], "goal": [["B", "A"]]}
        actions = iter(["pickup(B)", "stack(B, A)"])
        result = run_blocksworld(case, lambda messages: next(actions), max_steps=3)
        self.assertTrue(result["success"])
        self.assertEqual([step["legal"] for step in result["trace"]], [True, True])
        self.assertEqual(result["termination_reason"], "goal_reached")

    def test_illegal_action_feedback_is_visible_to_the_next_decision(self):
        case = {"source_id": "recover", "init": [["A", "B"], ["C"]], "goal": [["B", "C"]]}
        actions = iter(["pickup(B)", "unstack(B, A)", "stack(B, C)"])
        seen = []

        def policy(messages):
            seen.append(messages)
            return next(actions)

        result = run_blocksworld(case, policy, max_steps=3)
        self.assertTrue(result["success"])
        self.assertFalse(result["trace"][0]["legal"])
        self.assertIn("失败", seen[1][-1]["content"])

    def test_step_limit_is_not_mistaken_for_success(self):
        case = {"source_id": "limit", "init": [["A"], ["B"]], "goal": [["B", "A"]]}
        result = run_blocksworld(case, lambda messages: "stack(B, A)", max_steps=2)
        self.assertFalse(result["success"])
        self.assertEqual(result["termination_reason"], "step_limit")

    def test_format_errors_are_distinct_from_illegal_valid_syntax(self):
        case = {"source_id": "format", "init": [["A"], ["B"]], "goal": [["B", "A"]]}
        actions = iter(["fly(B)", "stack(B, A)"])
        result = run_blocksworld(case, lambda messages: next(actions), max_steps=2)
        self.assertIn("format_valid", result["trace"][0])
        self.assertFalse(result["trace"][0]["format_valid"])
        self.assertTrue(result["trace"][1]["format_valid"])
        self.assertFalse(result["trace"][1]["legal"])

    def test_http_action_uses_non_thinking_mode_and_preserves_request(self):
        self.assertIsNotNone(request_action, "request_action has not been implemented")
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"choices": [{"message": {"content": " pickup(B) "}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            action = request_action(f"http://127.0.0.1:{server.server_port}", "model", [{"role": "user", "content": "go"}])
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertEqual(action, "pickup(B)")
        self.assertEqual(received[0]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(received[0]["messages"][0]["content"], "go")
        self.assertEqual(received[0]["temperature"], 0)

    def test_alfworld_done_without_won_is_not_success(self):
        self.assertIsNotNone(run_alfworld, "run_alfworld has not been implemented")

        class Environment:
            def reset(self):
                return ["Task: put the apple away."], {}

            def step(self, actions):
                return ["Time limit reached."], [0], [True], {"won": [False]}

        result = run_alfworld(Environment(), lambda messages: "look", max_steps=4)
        self.assertFalse(result["success"])
        self.assertEqual(result["termination_reason"], "environment_done")

    def test_alfworld_guide_changes_only_the_initial_prompt(self):
        self.assertIsNotNone(ALFWORLD_ACTION_GUIDE, "static ALFWorld guide has not been implemented")

        class Environment:
            def __init__(self):
                self.steps = 0

            def reset(self):
                return ["Room observation and task."], {}

            def step(self, actions):
                self.steps += 1
                return [f"Feedback {self.steps}"], [0], [self.steps == 2], {"won": [False]}

        def record_run(guide):
            calls = []

            def policy(messages):
                calls.append(messages)
                return "look"

            result = run_alfworld(Environment(), policy, max_steps=20, action_guide=guide)
            return calls, result

        plain, plain_result = record_run("")
        guided, guided_result = record_run(ALFWORLD_ACTION_GUIDE)
        old_prefix = ("You are interacting with a text household environment. "
                      "Output exactly one environment command per turn, with no explanation; "
                      "wait for feedback before the next command.\n\n")
        self.assertEqual(plain[0][0]["content"], old_prefix + "Room observation and task.")
        self.assertEqual(guided[0][0]["content"],
                         old_prefix + ALFWORLD_ACTION_GUIDE + "\n\n" + "Room observation and task.")
        self.assertEqual(plain[1][1:], guided[1][1:])
        self.assertEqual(plain_result, guided_result)
        self.assertIn("move <object> to <receptacle>", ALFWORLD_ACTION_GUIDE)
        self.assertNotIn("alarmclock 2", ALFWORLD_ACTION_GUIDE)

    def test_alfworld_feedback_is_visible_before_next_action(self):
        self.assertIsNotNone(run_alfworld, "run_alfworld has not been implemented")

        class Environment:
            def __init__(self):
                self.steps = 0

            def reset(self):
                return ["Task: put the apple away."], {}

            def step(self, actions):
                self.steps += 1
                return (["You open the fridge." if self.steps == 1 else "You won!"],
                        [int(self.steps == 2)], [self.steps == 2], {"won": [self.steps == 2]})

        seen = []
        actions = iter(["open fridge 1", "put apple 1 in fridge 1"])

        def policy(messages):
            seen.append(messages)
            return next(actions)

        result = run_alfworld(Environment(), policy, max_steps=4)
        self.assertTrue(result["success"])
        self.assertIn("open the fridge", seen[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
