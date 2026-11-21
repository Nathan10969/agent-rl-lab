"""CPU-only checks for replaying and continuing a saved ALFWorld episode."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from baseline_eval import ALFWORLD_ACTION_GUIDE, run_alfworld

try:
    from continue_alfworld import continue_episode, replay_prefix
except ImportError:
    continue_episode = replay_prefix = None
try:
    from continue_alfworld import load_reference, summarize_continuations
except ImportError:
    load_reference = summarize_continuations = None
try:
    from continue_alfworld import run_experiment
except ImportError:
    run_experiment = None


def source_episode():
    return {
        "success": False,
        "initial_observation_sha256": hashlib.sha256(b"initial task").hexdigest(),
        "trace": [
            {"action": f"action {step}", "feedback": f"feedback {step}",
             "score": 0.0, "won": False, "done": step == 20}
            for step in range(1, 21)
        ],
    }


class Environment:
    def __init__(self, *, initial="initial task", changed_step=None, changed_score=None,
                 early_done=None, win_step=None):
        self.initial = initial
        self.changed_step = changed_step
        self.changed_score = changed_score
        self.early_done = early_done
        self.win_step = win_step
        self.actions = []

    def reset(self):
        return [self.initial], {}

    def step(self, actions):
        self.actions.extend(actions)
        step = len(self.actions)
        feedback = "changed" if step == self.changed_step else f"feedback {step}"
        score = 1.0 if step == self.changed_score else 0.0
        won = step == self.win_step
        done = step == self.early_done or won or step == 50
        return [feedback], [score], [done], {"won": [won]}

    def close(self):
        pass


class ContinueAlfworldTest(unittest.TestCase):
    def test_model_smoke_request_precedes_continuation_and_is_recorded(self):
        self.assertIsNotNone(run_experiment, "run_experiment is not implemented")
        calls = []

        @contextmanager
        def policy_context():
            def policy(history):
                calls.append(len(history))
                return "action 1" if len(history) == 1 else "finish"
            yield policy

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run_experiment([(0, source_episode())], lambda index: Environment(win_step=21),
                           policy_context, output, old_success=1, total=2)
            smoke_path = output / "smoke.json"
            self.assertTrue(smoke_path.is_file(), "model smoke result was not recorded")
            smoke = json.loads(smoke_path.read_text())
            self.assertEqual(calls, [1, 41])
            self.assertEqual(smoke["case_index"], 0)
            self.assertEqual(smoke["original_first_action"], "action 1")
            self.assertEqual(smoke["smoke_action"], "action 1")
            self.assertTrue(smoke["matches_original"])

    def test_failed_model_smoke_stops_before_any_continuation(self):
        self.assertIsNotNone(run_experiment, "run_experiment is not implemented")

        @contextmanager
        def policy_context():
            def fail(history):
                raise RuntimeError("model smoke failed")
            yield fail

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaisesRegex(RuntimeError, "model smoke failed"):
                run_experiment([(0, source_episode())], lambda index: Environment(),
                               policy_context, output, old_success=1, total=2)
            self.assertTrue((output / "failure.json").is_file())
            self.assertEqual(list(output.glob("episode-*.json")), [])

    def test_preflight_cli_checks_a_realistic_manifest_without_starting_inference(self):
        script = Path(__file__).resolve().parent / "continue_alfworld.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake" / "alfworld" / "agents" / "environment"
            fake.mkdir(parents=True)
            for package in (fake, fake.parent, fake.parent.parent):
                (package / "__init__.py").touch()
            (fake / "alfred_tw_env.py").write_text(
                'from pathlib import Path\n'
                'class Env:\n'
                ' def __init__(self): self.n=0\n'
                ' def reset(self): return ["initial task"], {}\n'
                ' def step(self, actions):\n'
                '  self.n+=1\n'
                '  return [f"feedback {self.n}"], [0], [False], {"won": [False]}\n'
                ' def close(self): pass\n'
                'class AlfredTWEnv:\n'
                ' def __init__(self, config, train_eval):\n'
                '  root=Path(config["dataset"]["eval_id_data_path"])\n'
                '  self.game_files=[str(p) for p in root.glob("*/trial/game.tw-pddl")]\n'
                ' def init_env(self, batch_size): return Env()\n', encoding="utf-8")
            data, reference = root / "data", root / "reference"
            reference.mkdir()
            cases = []
            for i in range(30):
                game = data / "json_2.1.1" / "valid_seen" / f"game-{i}" / "trial" / "game.tw-pddl"
                game.parent.mkdir(parents=True)
                game.write_text(f"game-{i}", encoding="utf-8")
                case_id = str(game.relative_to(data))
                cases.append({"source_id": case_id, "task_family": "family",
                              "game_sha256": hashlib.sha256(game.read_bytes()).hexdigest()})
                episode = source_episode()
                episode["source_id"] = case_id
                if i >= 18:
                    episode["success"] = True
                    episode["trace"][-1]["won"] = True
                (reference / f"episode-{i:03d}.json").write_text(json.dumps(episode), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            config = {"task": "alfworld", "model": "qwen27b-fp8", "max_steps": 20,
                      "temperature": 0, "max_new_tokens": 48, "thinking": False,
                      "prompt_variant": "static-action-guide", "per_case_seed": 0,
                      "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      "action_guide_sha256": hashlib.sha256(ALFWORLD_ACTION_GUIDE.encode()).hexdigest()}
            (reference / "config.json").write_text(json.dumps(config), encoding="utf-8")
            output = root / "preflight"
            command = [sys.executable, str(script), "--manifest", str(manifest), "--reference-dir", str(reference),
                       "--data-dir", str(data), "--output-dir", str(output), "--preflight-only", "--case-index", "0"]
            environment = dict(os.environ, PYTHONPATH=str(root / "fake"))
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((output / "preflight.json").is_file(), "preflight CLI did not write its result")
            self.assertEqual(json.loads((output / "preflight.json").read_text())["case_indices"], [0])
            self.assertFalse((output / "server.log").exists())

    def test_all_prefixes_are_verified_before_starting_model(self):
        self.assertIsNotNone(run_experiment, "run_experiment is not implemented")
        failures = [(0, source_episode()), (2, source_episode())]
        created = []
        events = []

        def create_env(index):
            events.append(f"environment {index}")
            env = Environment(win_step=21)
            created.append(env)
            return env

        @contextmanager
        def policy_context():
            events.append("model started")
            yield lambda history: "finish"

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_experiment(failures, create_env, policy_context, output, old_success=1, total=3)
            self.assertEqual(events[:3], ["environment 0", "environment 2", "model started"])
            self.assertEqual(len(created), 4)
            self.assertEqual(summary["new_successes"], 2)
            self.assertEqual(summary["overall_success"], 3)
            self.assertEqual(json.loads((output / "preflight.json").read_text())["verified"], 2)
            self.assertTrue((output / "episode-000.json").is_file())
            self.assertTrue((output / "episode-002.json").is_file())
            with self.assertRaises(FileExistsError):
                run_experiment(failures, create_env, policy_context, output, old_success=1, total=3)

    def test_bad_prefix_stops_before_any_model_requests(self):
        self.assertIsNotNone(run_experiment, "run_experiment is not implemented")
        failures = [(0, source_episode()), (2, source_episode())]
        starts = []

        @contextmanager
        def policy_context():
            starts.append(True)
            yield lambda history: "finish"

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaisesRegex(ValueError, "step 3"):
                run_experiment(failures, lambda index: Environment(changed_step=3 if index == 2 else None),
                               policy_context, output, old_success=1, total=3)
            self.assertEqual(starts, [])
            self.assertTrue((output / "failure.json").is_file())
            self.assertEqual(list(output.glob("episode-*.json")), [])

    def test_reference_requires_same_guide_settings_manifest_and_cases(self):
        self.assertIsNotNone(load_reference, "load_reference is not implemented")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "old"
            reference.mkdir()
            manifest = {"cases": [{"source_id": f"game-{i}", "task_family": "family"}
                                  for i in range(3)]}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            config = {"task": "alfworld", "model": "qwen27b-fp8", "max_steps": 20,
                      "temperature": 0, "max_new_tokens": 48, "thinking": False,
                      "prompt_variant": "static-action-guide", "per_case_seed": 0,
                      "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      "action_guide_sha256": hashlib.sha256(ALFWORLD_ACTION_GUIDE.encode()).hexdigest()}
            (reference / "config.json").write_text(json.dumps(config), encoding="utf-8")
            for i in range(3):
                episode = source_episode()
                episode["source_id"] = f"game-{i}"
                episode["success"] = i == 1
                if episode["success"]:
                    episode["trace"][-1]["won"] = True
                (reference / f"episode-{i:03d}.json").write_text(json.dumps(episode), encoding="utf-8")
            loaded, failures = load_reference(manifest_path, reference, expected_total=3, expected_failed=2)
            self.assertEqual(loaded, manifest)
            self.assertEqual([index for index, _ in failures], [0, 2])
            config["temperature"] = 0.7
            (reference / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "settings"):
                load_reference(manifest_path, reference, expected_total=3, expected_failed=2)
            config["temperature"] = 0
            config["action_guide_sha256"] = "changed"
            (reference / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "settings"):
                load_reference(manifest_path, reference, expected_total=3, expected_failed=2)

    def test_summary_never_counts_missing_or_errored_results_as_failures(self):
        self.assertIsNotNone(summarize_continuations, "summarize_continuations is not implemented")
        complete = [{"status": "complete", "success": True},
                    {"status": "complete", "success": False}]
        summary = summarize_continuations(complete, old_success=1, total=3)
        self.assertEqual(summary["new_successes"], 1)
        self.assertEqual(summary["remaining_failures"], 1)
        self.assertEqual(summary["overall_success"], 2)
        incomplete = summarize_continuations([complete[0], {"status": "incomplete", "success": None}],
                                             old_success=1, total=3)
        self.assertEqual(incomplete["new_successes"], 1)
        self.assertIsNone(incomplete["overall_success"])
        self.assertIsNone(incomplete["remaining_failures"])
        missing = summarize_continuations([complete[0]], old_success=1, total=3)
        self.assertEqual(missing["missing"], 1)
        self.assertIsNone(missing["overall_success"])

    def test_replay_rebuilds_the_exact_original_messages(self):
        self.assertIsNotNone(replay_prefix, "replay_prefix is not implemented")
        original_calls = []
        source = source_episode()

        def old_policy(messages):
            original_calls.append(messages)
            return f"action {len(original_calls)}"

        original = run_alfworld(Environment(early_done=20), old_policy, max_steps=20,
                                action_guide=ALFWORLD_ACTION_GUIDE)
        self.assertEqual(original["trace"], source["trace"])
        env = Environment()
        messages = replay_prefix(env, source)
        self.assertEqual(env.actions, [f"action {i}" for i in range(1, 21)])
        self.assertEqual(messages[:-2], original_calls[-1])
        self.assertEqual(messages[-2:], [{"role": "assistant", "content": "action 20"},
                                        {"role": "user", "content": "feedback 20"}])

    def test_replay_rejects_changed_initial_observation(self):
        self.assertIsNotNone(replay_prefix, "replay_prefix is not implemented")
        env = Environment(initial="different task")
        with self.assertRaisesRegex(ValueError, "initial observation"):
            replay_prefix(env, source_episode())
        self.assertEqual(env.actions, [])

    def test_replay_rejects_changed_feedback_or_score(self):
        self.assertIsNotNone(replay_prefix, "replay_prefix is not implemented")
        for env in (Environment(changed_step=3), Environment(changed_score=3)):
            with self.subTest(env=env):
                with self.assertRaisesRegex(ValueError, "step 3"):
                    replay_prefix(env, source_episode())
                self.assertEqual(len(env.actions), 3)

    def test_replay_rejects_won_or_ended_before_continuation(self):
        self.assertIsNotNone(replay_prefix, "replay_prefix is not implemented")
        for env in (Environment(early_done=20), Environment(win_step=10)):
            with self.subTest(env=env):
                with self.assertRaises(ValueError):
                    replay_prefix(env, source_episode())

    def test_replay_rejects_wrong_source_episode(self):
        self.assertIsNotNone(replay_prefix, "replay_prefix is not implemented")
        for source in (dict(source_episode(), success=True),
                       dict(source_episode(), trace=source_episode()["trace"][:19])):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    replay_prefix(Environment(), source)

    def test_continuation_starts_at_21_and_stops_on_success(self):
        self.assertIsNotNone(continue_episode, "continue_episode is not implemented")
        source = source_episode()
        env = Environment(win_step=22)
        messages = replay_prefix(env, source)
        seen = []

        def policy(history):
            seen.append(history)
            return "new action"

        result = continue_episode(env, messages, source, policy, max_steps=50)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["success"])
        self.assertEqual(result["success_step"], 22)
        self.assertEqual(len(result["trace"]), 22)
        self.assertEqual(len(seen[0]), 41)
        self.assertEqual(seen[0][-1]["content"], "feedback 20")
        self.assertEqual(seen[1][-1]["content"], "feedback 21")

    def test_continuation_step_limit_is_not_success(self):
        self.assertIsNotNone(continue_episode, "continue_episode is not implemented")
        source = source_episode()
        env = Environment()
        messages = replay_prefix(env, source)
        result = continue_episode(env, messages, source, lambda history: "look", max_steps=50)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["success"])
        self.assertIsNone(result["success_step"])
        self.assertEqual(len(result["trace"]), 50)

    def test_model_error_is_incomplete_not_a_failed_task(self):
        self.assertIsNotNone(continue_episode, "continue_episode is not implemented")
        source = source_episode()
        env = Environment()
        messages = replay_prefix(env, source)

        def broken_policy(history):
            raise RuntimeError("server unavailable")

        result = continue_episode(env, messages, source, broken_policy, max_steps=50)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["success"])
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(len(result["trace"]), 20)


if __name__ == "__main__":
    unittest.main()
