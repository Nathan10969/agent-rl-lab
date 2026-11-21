"""CPU tests for the four-case no-effect reminder pilot."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

try:
    from feedback_hint_eval import REMINDER, add_reminders
except ImportError:
    REMINDER = add_reminders = None
try:
    from feedback_hint_eval import select_paired_cases
except ImportError:
    select_paired_cases = None
try:
    from feedback_hint_eval import validate_control_setup
except ImportError:
    validate_control_setup = None
try:
    from feedback_hint_eval import run_hint_pilot
except ImportError:
    run_hint_pilot = None

from test_continue_alfworld import Environment, source_episode


def history_after_22_steps():
    messages = [{"role": "user", "content": "original task and instructions"}]
    for step in range(1, 23):
        feedback = "Nothing happens." if step in (2, 20, 21) else f"observed {step}"
        messages.extend([{"role": "assistant", "content": f"action {step}"},
                         {"role": "user", "content": feedback}])
    return messages


class FeedbackHintTest(unittest.TestCase):
    def test_preflight_cli_replays_one_fixed_case_without_starting_a_model(self):
        script = Path(__file__).with_name("feedback_hint_eval.py")
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
                '  obs="Nothing happens." if self.n == 20 else f"feedback {self.n}"\n'
                '  return [obs], [0], [False], {"won": [False]}\n'
                ' def close(self): pass\n'
                'class AlfredTWEnv:\n'
                ' def __init__(self, config, train_eval):\n'
                '  root=Path(config["dataset"]["eval_id_data_path"])\n'
                '  self.game_files=[str(p) for p in root.glob("*/trial/game.tw-pddl")]\n'
                ' def init_env(self, batch_size): return Env()\n', encoding="utf-8")
            data, reference, control = root / "data", root / "baseline" / "guide", root / "control"
            reference.mkdir(parents=True)
            control.mkdir()
            cases = []
            failure_indices = set(range(16)) | {24, 29}
            for i in range(30):
                game = data / "json_2.1.1" / "valid_seen" / f"game-{i}" / "trial" / "game.tw-pddl"
                game.parent.mkdir(parents=True)
                game.write_text(f"game-{i}", encoding="utf-8")
                case_id = str(game.relative_to(data))
                cases.append({"source_id": case_id, "task_family": "family",
                              "game_sha256": hashlib.sha256(game.read_bytes()).hexdigest()})
                episode = source_episode()
                episode["source_id"] = case_id
                episode["trace"][-1]["feedback"] = "Nothing happens."
                if i not in failure_indices:
                    episode["success"] = True
                    episode["trace"][-1]["won"] = True
                (reference / f"episode-{i:03d}.json").write_text(json.dumps(episode), encoding="utf-8")
                if i in (0, 9, 24, 29):
                    saved = {"case_index": i, "source_id": case_id, "status": "complete", "success": False,
                             "prefix_verified": True,
                             "trace": copy.deepcopy(episode["trace"]) + [{"action": "old"}] * 30}
                    (control / f"episode-{i:03d}.json").write_text(json.dumps(saved), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            from baseline_eval import ALFWORLD_ACTION_GUIDE
            old_config = {"task": "alfworld", "model": "qwen27b-fp8", "max_steps": 20,
                          "temperature": 0, "max_new_tokens": 48, "thinking": False,
                          "prompt_variant": "static-action-guide", "per_case_seed": 0,
                          "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                          "action_guide_sha256": hashlib.sha256(ALFWORLD_ACTION_GUIDE.encode()).hexdigest()}
            (reference / "config.json").write_text(json.dumps(old_config), encoding="utf-8")
            command = ["vllm", "serve", "model", "--port", "100", "--max-model-len", "4096"]
            (reference.parent / "config.json").write_text(json.dumps({"server_command": command}), encoding="utf-8")
            prior = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                     "reference_config_sha256": hashlib.sha256((reference / "config.json").read_bytes()).hexdigest(),
                     "max_steps": 50, "original_max_steps": 20, "temperature": 0, "max_new_tokens": 48,
                     "thinking": False, "max_model_len": 4096, "per_case_seed": 0,
                     "flashinfer_sampler": "0", "server_command": command}
            (control / "config.json").write_text(json.dumps(prior), encoding="utf-8")
            (control / "preflight.json").write_text(json.dumps({"verified": 18}), encoding="utf-8")
            (control / "smoke.json").write_text(json.dumps({"matches_original": True}), encoding="utf-8")
            output = root / "preflight"
            args = [sys.executable, str(script), "--manifest", str(manifest), "--reference-dir", str(reference),
                    "--control-dir", str(control), "--data-dir", str(data), "--output-dir", str(output),
                    "--preflight-only", "--case-index", "9"]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30,
                                    env=dict(os.environ, PYTHONPATH=str(root / "fake")))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((output / "preflight.json").is_file(), "preflight did not write a result")
            self.assertEqual(json.loads((output / "preflight.json").read_text())["case_indices"], [9])
            self.assertFalse((output / "server.log").exists())

    def test_pilot_uses_raw_replay_and_only_model_facing_reminders(self):
        self.assertIsNotNone(run_hint_pilot, "run_hint_pilot is not implemented")
        source = source_episode()
        source["source_id"] = "case-zero"
        source["trace"][19]["feedback"] = "Nothing happens."
        control = {"trace": copy.deepcopy(source["trace"]) + [{"action": "old next"}] * 30,
                   "success": False}
        calls, events = [], []

        class HintEnv(Environment):
            def step(self, actions):
                obs, scores, dones, info = super().step(actions)
                if len(self.actions) in (20, 21):
                    obs[0] = "Nothing happens."
                return obs, scores, dones, info

        def create_env(index):
            events.append("environment")
            return HintEnv(win_step=22)

        @contextmanager
        def policy_context():
            events.append("model")
            def policy(messages):
                calls.append(copy.deepcopy(messages))
                return "action 1" if len(messages) == 1 else "new action"
            yield policy

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            summary = run_hint_pilot([(0, source, control)], create_env, policy_context, output)
            self.assertEqual(events[:2], ["environment", "model"])
            self.assertEqual(summary["control_successes"], 0)
            self.assertEqual(summary["reminder_successes"], 1)
            self.assertEqual(summary["incomplete"], 0)
            result = json.loads((output / "episode-000.json").read_text(encoding="utf-8"))
            self.assertEqual(result["reminder_steps"], [20, 21])
            self.assertEqual(result["success_step"], 22)
            self.assertEqual(result["trace"][19]["feedback"], "Nothing happens.")
            self.assertEqual(result["trace"][20]["feedback"], "Nothing happens.")
            self.assertNotIn(REMINDER, result["trace"][20]["feedback"])
            self.assertIn(REMINDER, calls[1][40]["content"])
            self.assertIn(REMINDER, calls[2][42]["content"])
            self.assertEqual(source["trace"][19]["feedback"], "Nothing happens.")
            with self.assertRaises(FileExistsError):
                run_hint_pilot([(0, source, control)], create_env, policy_context, output)

    def test_bad_replay_blocks_the_model_and_does_not_count_as_failure(self):
        self.assertIsNotNone(run_hint_pilot, "run_hint_pilot is not implemented")
        source = source_episode()
        control = {"trace": copy.deepcopy(source["trace"]) + [{"action": "old next"}] * 30,
                   "success": False}
        starts = []

        @contextmanager
        def policy_context():
            starts.append(True)
            yield lambda messages: "look"

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with self.assertRaisesRegex(ValueError, "step 3"):
                run_hint_pilot([(0, source, control)], lambda index: Environment(changed_step=3),
                               policy_context, output)
            self.assertEqual(starts, [])
            self.assertTrue((output / "failure.json").is_file())
            self.assertEqual(list(output.glob("episode-*.json")), [])

    def test_model_error_is_incomplete_and_not_a_failed_trial(self):
        self.assertIsNotNone(run_hint_pilot, "run_hint_pilot is not implemented")
        source = source_episode()
        control = {"trace": copy.deepcopy(source["trace"]) + [{"action": "old next"}] * 30,
                   "success": False}

        @contextmanager
        def policy_context():
            def policy(messages):
                if len(messages) == 1:
                    return "action 1"
                raise RuntimeError("model service error")
            yield policy

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            summary = run_hint_pilot([(0, source, control)], lambda index: Environment(),
                                     policy_context, output)
            self.assertEqual(summary["incomplete"], 1)
            self.assertIsNone(summary["reminder_successes"])
            result = json.loads((output / "episode-000.json").read_text())
            self.assertEqual(result["status"], "incomplete")
            self.assertIsNone(result["success"])

    def test_control_setup_must_match_baseline_model_and_generation(self):
        self.assertIsNotNone(validate_control_setup, "validate_control_setup is not implemented")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "baseline" / "guide"
            reference.mkdir(parents=True)
            (reference / "config.json").write_text('{"task":"alfworld"}', encoding="utf-8")
            command = ["vllm", "serve", "model", "--port", "100", "--max-model-len", "4096"]
            (reference.parent / "config.json").write_text(json.dumps({"server_command": command}), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text('{"cases":[]}', encoding="utf-8")
            control = root / "control"
            control.mkdir()
            old_command = command.copy()
            old_command[old_command.index("--port") + 1] = "200"
            config = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      "reference_config_sha256": hashlib.sha256((reference / "config.json").read_bytes()).hexdigest(),
                      "max_steps": 50, "original_max_steps": 20, "temperature": 0, "max_new_tokens": 48,
                      "thinking": False, "max_model_len": 4096, "per_case_seed": 0,
                      "flashinfer_sampler": "0", "server_command": old_command}
            (control / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (control / "preflight.json").write_text(json.dumps({"verified": 18}), encoding="utf-8")
            (control / "smoke.json").write_text(json.dumps({"matches_original": True}), encoding="utf-8")
            validate_control_setup(manifest, reference, control)
            config["temperature"] = 0.8
            (control / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "control"):
                validate_control_setup(manifest, reference, control)

    def test_control_must_be_the_same_failed_fifty_step_episode(self):
        self.assertIsNotNone(select_paired_cases, "select_paired_cases is not implemented")
        source = source_episode()
        source["source_id"] = "case-zero"
        control = {"case_index": 0, "source_id": "case-zero", "status": "complete",
                   "success": False, "prefix_verified": True,
                   "trace": copy.deepcopy(source["trace"]) + [{"action": "look"}] * 30}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode-000.json"
            path.write_text(json.dumps(control), encoding="utf-8")
            selected = select_paired_cases([(0, source)], directory, case_indices=(0,))
            self.assertEqual([(i, old["source_id"], result["success"]) for i, old, result in selected],
                             [(0, "case-zero", False)])
            control["trace"][7]["action"] = "changed"
            path.write_text(json.dumps(control), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "control"):
                select_paired_cases([(0, source)], directory, case_indices=(0,))

    def test_only_no_effect_feedback_from_step_twenty_is_augmented(self):
        self.assertIsNotNone(add_reminders, "add_reminders is not implemented")
        history = history_after_22_steps()
        original = copy.deepcopy(history)
        augmented, steps = add_reminders(history)
        self.assertEqual(steps, [20, 21])
        self.assertEqual(history, original, "model-facing hints must not mutate saved raw feedback")
        self.assertEqual(augmented[4], original[4], "the old step 2 feedback must remain unchanged")
        self.assertEqual(augmented[40]["content"], "Nothing happens.\n" + REMINDER)
        self.assertEqual(augmented[42]["content"], "Nothing happens.\n" + REMINDER)
        self.assertEqual(augmented[44], original[44], "normal feedback must not get a reminder")
        self.assertEqual(augmented[0], original[0], "task instructions must remain unchanged")
        self.assertEqual(augmented[39], original[39], "assistant action must remain unchanged")

    def test_rejects_bad_history_instead_of_modifying_the_wrong_role(self):
        self.assertIsNotNone(add_reminders, "add_reminders is not implemented")
        history = history_after_22_steps()
        history[40]["role"] = "assistant"
        with self.assertRaisesRegex(ValueError, "role"):
            add_reminders(history)
        with self.assertRaises(ValueError):
            add_reminders(history[:20])


if __name__ == "__main__":
    unittest.main()
