"""CPU tests for the per-action no-progress count reminder."""
import copy
import inspect
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from feedback_hint_eval import main as generic_main, run_hint_pilot
from test_continue_alfworld import Environment, source_episode

try:
    from counted_hint_eval import add_counted_reminders
except ImportError:
    add_counted_reminders = None


def sample_history():
    messages = [{"role": "user", "content": "original task"}]
    for step in range(1, 24):
        action = "go to drawer 4" if step in (2, 4, 7, 19, 20, 22) else ("look" if step in (21, 23) else f"action {step}")
        feedback = "Nothing happens." if step in (2, 7, 19, 20, 21, 22, 23) else f"feedback {step}"
        messages.extend([{"role": "assistant", "content": action}, {"role": "user", "content": feedback}])
    return messages


class CountedHintTest(unittest.TestCase):
    def test_counts_exact_action_failures_across_whole_visible_history(self):
        self.assertIsNotNone(add_counted_reminders, "add_counted_reminders is not implemented")
        history = sample_history()
        before = copy.deepcopy(history)
        augmented, steps = add_counted_reminders(history)
        self.assertEqual(history, before, "real feedback and saved history must not be mutated")
        self.assertEqual(steps, [20, 22, 23])
        self.assertEqual(augmented[4], before[4], "step 2 must never be augmented")
        self.assertIn('"go to drawer 4"', augmented[40]["content"])
        self.assertIn("4 times", augmented[40]["content"])
        self.assertEqual(augmented[42], before[42], "first failed look must not trigger a repeat reminder")
        self.assertIn("5 times", augmented[44]["content"])
        self.assertIn('"look"', augmented[46]["content"])
        self.assertIn("2 times", augmented[46]["content"])

    def test_bad_role_or_short_history_is_rejected(self):
        self.assertIsNotNone(add_counted_reminders, "add_counted_reminders is not implemented")
        history = sample_history()
        history[40]["role"] = "assistant"
        with self.assertRaisesRegex(ValueError, "role"):
            add_counted_reminders(history)
        with self.assertRaises(ValueError):
            add_counted_reminders(history[:20])

    def test_pilot_accepts_a_transform_without_changing_default_pilot(self):
        self.assertIn("reminder_transform", inspect.signature(run_hint_pilot).parameters,
                      "pilot does not yet accept a reminder transform")
        source = source_episode()
        control = {"trace": copy.deepcopy(source["trace"]) + [{"action": "old"}] * 30, "success": False}
        calls = []

        def transform(messages):
            result = copy.deepcopy(messages)
            result[40]["content"] += "\ncustom reminder"
            return result, [20]

        @contextmanager
        def policy_context():
            def policy(messages):
                calls.append(copy.deepcopy(messages))
                return "action 1" if len(messages) == 1 else "new action"
            yield policy

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            summary = run_hint_pilot([(0, source, control)], lambda index: Environment(win_step=21),
                                     policy_context, output, reminder_transform=transform)
            self.assertEqual(summary["reminder_successes"], 1)
            self.assertIn("custom reminder", calls[1][40]["content"])
            result = json.loads((output / "episode-000.json").read_text())
            self.assertEqual(result["reminder_steps"], [20])
            self.assertEqual(result["trace"][19]["feedback"], "feedback 20")

    def test_generic_entry_point_can_receive_counted_reminder_configuration(self):
        signature = inspect.signature(generic_main).parameters
        for name in ("reminder_transform", "reminder_description", "reminder_mode"):
            self.assertIn(name, signature, f"generic entry point lacks {name}")


if __name__ == "__main__":
    unittest.main()
