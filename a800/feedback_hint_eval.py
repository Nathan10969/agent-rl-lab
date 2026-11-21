"""Compare an explicit no-progress reminder with saved ALFWorld continuations."""
import argparse
import copy
import hashlib
import inspect
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

from alfworld_manifest import match_manifest_games
from alfworld_probe import build_alfworld_config
from baseline_eval import request_action, save_json_new
from continue_alfworld import continue_episode, load_reference, replay_prefix
from pilot_server import free_local_port, ready_server, verify_server_reference


REMINDER = "Reminder: the environment reported that the previous action had no apparent effect."
CASE_INDICES = (0, 9, 24, 29)


def validate_control_setup(manifest_path, reference_dir, control_dir):
    """Require the original continuation's model and sampling setup to match."""
    manifest_path, reference_dir, control_dir = map(Path, (manifest_path, reference_dir, control_dir))
    config = json.loads((control_dir / "config.json").read_text(encoding="utf-8"))
    expected = {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "reference_config_sha256": hashlib.sha256((reference_dir / "config.json").read_bytes()).hexdigest(),
                "max_steps": 50, "original_max_steps": 20, "temperature": 0, "max_new_tokens": 48,
                "thinking": False, "max_model_len": 4096, "per_case_seed": 0, "flashinfer_sampler": "0"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("control generation settings do not match")
    verify_server_reference(reference_dir, config["server_command"])
    preflight = json.loads((control_dir / "preflight.json").read_text(encoding="utf-8"))
    smoke = json.loads((control_dir / "smoke.json").read_text(encoding="utf-8"))
    if preflight.get("verified") != 18 or smoke.get("matches_original") is not True:
        raise ValueError("control prefix checks or model smoke did not pass")
    return config


def add_reminders(messages):
    """Copy a dialogue and augment only exact no-effect feedback from step 20."""
    if len(messages) < 41 or len(messages) % 2 != 1:
        raise ValueError("expected the initial prompt and at least 20 complete action/feedback pairs")
    if any(message.get("role") != ("assistant" if index % 2 else "user")
           for index, message in enumerate(messages)):
        raise ValueError("unexpected conversation role")
    result = copy.deepcopy(messages)
    steps = []
    for index in range(40, len(result), 2):
        content = result[index]["content"]
        if content.strip() == "Nothing happens.":
            result[index]["content"] = content + "\n" + REMINDER
            steps.append(index // 2)
    return result, steps


def select_paired_cases(failures, control_dir, case_indices=CASE_INDICES):
    """Select fixed failed cases only when their saved 50-step controls match."""
    failures = dict(failures)
    if not case_indices or len(set(case_indices)) != len(case_indices):
        raise ValueError("selected case indices must be non-empty and unique")
    selected = []
    for index in case_indices:
        if index not in failures:
            raise ValueError(f"case {index} is not an original failure")
        source = failures[index]
        control = json.loads((Path(control_dir) / f"episode-{index:03d}.json").read_text(encoding="utf-8"))
        trace = control["trace"]
        if (control.get("case_index") != index or control.get("source_id") != source.get("source_id")
                or control.get("status") != "complete" or control.get("success") is not False
                or control.get("prefix_verified") is not True or len(trace) != 50
                or trace[:20] != source["trace"]):
            raise ValueError(f"control for case {index} does not match the saved failed trajectory")
        selected.append((index, source, control))
    return selected


def run_hint_pilot(selected, create_env, policy_context, output_dir, metadata=None,
                   reminder_transform=add_reminders):
    """Replay every case before generating; save raw feedback and paired results."""
    selected = list(selected)
    if not selected or len({index for index, _, _ in selected}) != len(selected):
        raise ValueError("selected cases must be non-empty and unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        if metadata is not None:
            save_json_new(output_dir / "config.json", metadata)
        smoke_messages = None
        for index, source, _ in selected:
            env = create_env(index)
            try:
                messages = replay_prefix(env, source)
            finally:
                env.close()
            if smoke_messages is None:
                smoke_messages = messages[:1]
            print(f"verified prefix: case {index}", flush=True)
        save_json_new(output_dir / "preflight.json", {"verified": len(selected),
                                                       "case_indices": [i for i, _, _ in selected]})
        with policy_context() as policy:
            smoke_action = policy(copy.deepcopy(smoke_messages))
            if not isinstance(smoke_action, str) or not smoke_action.strip():
                raise RuntimeError("model smoke returned an empty action")
            original_action = selected[0][1]["trace"][0]["action"]
            save_json_new(output_dir / "smoke.json", {"case_index": selected[0][0],
                          "original_first_action": original_action, "smoke_action": smoke_action,
                          "matches_original": smoke_action == original_action})
            print("model generation smoke passed", flush=True)
            for index, source, control in selected:
                env = create_env(index)
                reminder_steps = set()
                try:
                    messages = replay_prefix(env, source)

                    def hinted_policy(history):
                        augmented, steps = reminder_transform(history)
                        reminder_steps.update(steps)
                        return policy(augmented)

                    result = continue_episode(env, messages, source, hinted_policy, max_steps=50)
                finally:
                    env.close()
                first_action = result["trace"][20]["action"] if len(result["trace"]) > 20 else None
                control_action = control["trace"][20]["action"]
                result.update({"case_index": index, "source_id": source.get("source_id"),
                               "prefix_verified": True, "reminder_steps": sorted(reminder_steps),
                               "control_success": control["success"], "control_first_action": control_action,
                               "first_action_changed": (first_action != control_action
                                                        if first_action is not None else None)})
                save_json_new(output_dir / f"episode-{index:03d}.json", result)
                results.append(result)
                print(f"case {index}: status={result['status']} success={result['success']}", flush=True)
        incomplete = sum(r["status"] != "complete" for r in results)
        summary = {"selected_cases": len(selected), "recorded": len(results), "incomplete": incomplete,
                   "control_successes": sum(control["success"] for _, _, control in selected),
                   "reminder_successes": (sum(r["success"] for r in results) if incomplete == 0 else None),
                   "first_actions_changed": sum(r["first_action_changed"] is True for r in results)}
        save_json_new(output_dir / "summary.json", summary)
        return summary
    except Exception as error:
        save_json_new(output_dir / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                                                     "completed_continuations": len(results)})
        raise


def main(reminder_transform=add_reminders, reminder_description=REMINDER,
         reminder_mode="generic"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    if args.case_index is not None and not args.preflight_only:
        parser.error("--case-index may only be used with --preflight-only")

    manifest, failures = load_reference(args.manifest, args.reference_dir)
    validate_control_setup(args.manifest, args.reference_dir, args.control_dir)
    selected = select_paired_cases(failures, args.control_dir)
    if any(source["trace"][-1]["feedback"].strip() != "Nothing happens."
           for _, source, _ in selected):
        raise ValueError("selected cases must all trigger the reminder at the 20-step boundary")
    if args.case_index is not None:
        selected = [case for case in selected if case[0] == args.case_index]
        if len(selected) != 1:
            parser.error("--case-index must identify a selected case")
    os.environ["ALFWORLD_DATA"] = str(args.data_dir.resolve())
    random.seed(0)
    import numpy as np
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    np.random.seed(0)
    world = AlfredTWEnv(build_alfworld_config(args.data_dir, 50), train_eval="eval_in_distribution")
    games = match_manifest_games(manifest, args.data_dir, world.game_files)

    def create_env(index):
        random.seed(0)
        np.random.seed(0)
        world.game_files = [games[index]]
        return world.init_env(batch_size=1)

    if args.preflight_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        try:
            for index, source, _ in selected:
                env = create_env(index)
                try:
                    messages = replay_prefix(env, source)
                    _, steps = reminder_transform(messages)
                    if steps != [20]:
                        raise ValueError(f"unexpected reminder boundary in case {index}")
                finally:
                    env.close()
                print(f"verified prefix and reminder boundary: case {index}", flush=True)
            save_json_new(args.output_dir / "preflight.json", {"verified": len(selected),
                          "case_indices": [i for i, _, _ in selected]})
        except Exception as error:
            save_json_new(args.output_dir / "failure.json", {"error_type": type(error).__name__,
                                                               "error": str(error)})
            raise
        return

    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise ValueError("the original baseline disabled the FlashInfer sampler")
    original = json.loads((args.reference_dir.parent / "config.json").read_text(encoding="utf-8"))
    command = list(original["server_command"])
    if command.count("--port") != 1:
        raise ValueError("original server command has no unique port")
    port = free_local_port()
    command[command.index("--port") + 1] = str(port)
    verify_server_reference(args.reference_dir, command)
    url = f"http://127.0.0.1:{port}"

    @contextmanager
    def policy_context():
        with (args.output_dir / "server.log").open("x", encoding="utf-8") as server_log:
            with ready_server(command, url + "/v1/models", server_log, timeout=600):
                print("model server ready", flush=True)
                yield lambda messages: request_action(url, "qwen27b-fp8", messages)

    transform_file = Path(inspect.getfile(reminder_transform))
    metadata = {"case_indices": list(CASE_INDICES), "reminder": reminder_description,
                "reminder_mode": reminder_mode,
                "transform_sha256": hashlib.sha256(transform_file.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "control_config_sha256": hashlib.sha256((args.control_dir / "config.json").read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "server_command": command, "max_steps": 50, "original_max_steps": 20,
                "temperature": 0, "max_new_tokens": 48, "thinking": False, "max_model_len": 4096,
                "per_case_seed": 0, "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
                "cpath": os.environ.get("CPATH")}
    summary = run_hint_pilot(selected, create_env, policy_context, args.output_dir, metadata=metadata,
                             reminder_transform=reminder_transform)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["reminder_successes"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
