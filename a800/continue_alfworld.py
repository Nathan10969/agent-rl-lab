"""Replay a saved 20-step ALFWorld episode and continue its exact dialogue."""
import argparse
import copy
import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

from alfworld_manifest import match_manifest_games
from alfworld_probe import build_alfworld_config
from baseline_eval import ALFWORLD_ACTION_GUIDE, request_action, save_json_new
from pilot_server import free_local_port, ready_server, verify_server_reference


def load_reference(manifest_path, reference_dir, expected_total=30, expected_failed=18):
    """Reject a changed prompt, sampling setup, case list, or incomplete baseline."""
    manifest_path, reference_dir = Path(manifest_path), Path(reference_dir)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    cases = manifest["cases"]
    config = json.loads((reference_dir / "config.json").read_text(encoding="utf-8"))
    expected = {"task": "alfworld", "model": "qwen27b-fp8", "max_steps": 20,
                "temperature": 0, "max_new_tokens": 48, "thinking": False,
                "prompt_variant": "static-action-guide", "per_case_seed": 0,
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "action_guide_sha256": hashlib.sha256(ALFWORLD_ACTION_GUIDE.encode()).hexdigest()}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("original generation settings or guide do not match")
    if len(cases) != expected_total or len({case["source_id"] for case in cases}) != expected_total:
        raise ValueError("manifest cases are missing or duplicated")
    if len(list(reference_dir.glob("episode-*.json"))) != expected_total:
        raise ValueError("original episode files are incomplete")
    failures = []
    for index, case in enumerate(cases):
        source = json.loads((reference_dir / f"episode-{index:03d}.json").read_text(encoding="utf-8"))
        trace = source["trace"]
        if (source["source_id"] != case["source_id"] or source.get("success") not in (True, False)
                or not trace or bool(trace[-1]["won"]) != source["success"]):
            raise ValueError(f"original episode {index} does not match the manifest or outcome")
        if source["success"] is False:
            if len(trace) != 20:
                raise ValueError(f"failed episode {index} does not contain 20 steps")
            failures.append((index, source))
    if len(failures) != expected_failed:
        raise ValueError("unexpected number of failed original episodes")
    return manifest, failures


def summarize_continuations(results, old_success=12, total=30):
    """Only report full 50-step totals when every continuation is complete."""
    expected = total - old_success
    if expected < 0 or len(results) > expected:
        raise ValueError("invalid number of continuation results")
    for result in results:
        if not ((result["status"] == "complete" and type(result["success"]) is bool)
                or (result["status"] == "incomplete" and result["success"] is None)):
            raise ValueError("continuation status and success are inconsistent")
    new_successes = sum(result["status"] == "complete" and result["success"] for result in results)
    incomplete = sum(result["status"] == "incomplete" for result in results)
    missing = expected - len(results)
    complete = not incomplete and not missing
    return {"total": total, "original_successes": old_success, "original_failures": expected,
            "continuations_recorded": len(results), "incomplete": incomplete, "missing": missing,
            "new_successes": new_successes,
            "remaining_failures": expected - new_successes if complete else None,
            "overall_success": old_success + new_successes if complete else None}


def replay_prefix(env, source):
    """Return the original conversation only if all 20 saved steps match."""
    trace = source["trace"]
    if source.get("success") is not False or len(trace) != 20:
        raise ValueError("source must be an unsuccessful 20-step episode")
    if any(step["won"] or (index < 19 and step["done"])
           for index, step in enumerate(trace)):
        raise ValueError("source ended before its 20-step boundary")

    observations, _ = env.reset()
    initial = observations[0]
    digest = hashlib.sha256(initial.encode("utf-8")).hexdigest()
    if digest != source["initial_observation_sha256"]:
        raise ValueError("initial observation does not match")
    instruction = ("You are interacting with a text household environment. "
                   "Output exactly one environment command per turn, with no explanation; "
                   "wait for feedback before the next command.\n\n")
    messages = [{"role": "user", "content": instruction + ALFWORLD_ACTION_GUIDE + "\n\n" + initial}]
    for number, saved in enumerate(trace, 1):
        observations, scores, dones, info = env.step([saved["action"]])
        feedback = observations[0]
        if (feedback != saved["feedback"] or float(scores[0]) != saved["score"]
                or bool(info["won"][0]) != saved["won"]):
            raise ValueError(f"feedback, score or won differs at step {number}")
        if bool(dones[0]):
            raise ValueError(f"extended environment ended at step {number}")
        messages.extend([{"role": "assistant", "content": saved["action"]},
                         {"role": "user", "content": feedback}])
    return messages


def continue_episode(env, messages, source, policy, max_steps=50):
    """Continue a verified prefix; an execution error is never a task failure."""
    trace = copy.deepcopy(source["trace"])
    messages = copy.deepcopy(messages)
    if len(trace) != 20 or max_steps <= len(trace):
        raise ValueError("continuation needs a 20-step prefix and a larger limit")
    for number in range(len(trace) + 1, max_steps + 1):
        try:
            action = policy(copy.deepcopy(messages))
            observations, scores, dones, info = env.step([action])
            feedback, won, done = observations[0], bool(info["won"][0]), bool(dones[0])
        except Exception as error:
            return {"status": "incomplete", "success": None, "success_step": None,
                    "error_type": type(error).__name__, "error": str(error), "trace": trace}
        trace.append({"action": action, "feedback": feedback, "score": float(scores[0]),
                      "won": won, "done": done})
        messages.extend([{"role": "assistant", "content": action},
                         {"role": "user", "content": feedback}])
        if won or done:
            return {"status": "complete", "success": won, "success_step": number if won else None,
                    "termination_reason": "goal_reached" if won else "environment_done", "trace": trace}
    return {"status": "complete", "success": False, "success_step": None,
            "termination_reason": "step_limit", "trace": trace}


def run_experiment(failures, create_env, policy_context, output_dir, old_success=12, total=30,
                   metadata=None):
    """Verify every prefix before starting inference; keep every result separately."""
    failures = list(failures)
    if len(failures) != total - old_success or len({index for index, _ in failures}) != len(failures):
        raise ValueError("continuation cases are missing or duplicated")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    smoke_messages = None
    try:
        if metadata is not None:
            save_json_new(output_dir / "config.json", metadata)
        for count, (index, source) in enumerate(failures, 1):
            env = create_env(index)
            try:
                messages = replay_prefix(env, source)
            finally:
                env.close()
            if smoke_messages is None:
                smoke_messages = messages[:1]
            print(f"prefix verified {count}/{len(failures)}: case {index}", flush=True)
        save_json_new(output_dir / "preflight.json", {"verified": len(failures),
                                                       "case_indices": [index for index, _ in failures]})
        with policy_context() as policy:
            smoke_action = policy(copy.deepcopy(smoke_messages))
            if not isinstance(smoke_action, str) or not smoke_action.strip():
                raise RuntimeError("model smoke returned an empty action")
            original_action = failures[0][1]["trace"][0]["action"]
            save_json_new(output_dir / "smoke.json", {"case_index": failures[0][0],
                          "original_first_action": original_action, "smoke_action": smoke_action,
                          "matches_original": smoke_action == original_action})
            print("model generation smoke passed", flush=True)
            for count, (index, source) in enumerate(failures, 1):
                env = create_env(index)
                try:
                    messages = replay_prefix(env, source)
                    result = continue_episode(env, messages, source, policy, max_steps=50)
                finally:
                    env.close()
                result.update({"case_index": index, "source_id": source.get("source_id"),
                               "prefix_verified": True, "original_step_limit": 20,
                               "new_step_limit": 50})
                save_json_new(output_dir / f"episode-{index:03d}.json", result)
                results.append(result)
                print(f"continued {count}/{len(failures)}: case {index}, "
                      f"status={result['status']}, success={result['success']}", flush=True)
        summary = summarize_continuations(results, old_success=old_success, total=total)
        save_json_new(output_dir / "summary.json", summary)
        return summary
    except Exception as error:
        save_json_new(output_dir / "failure.json", {"error_type": type(error).__name__,
                                                     "error": str(error),
                                                     "completed_continuations": len(results)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    if args.case_index is not None and not args.preflight_only:
        parser.error("--case-index may only be used with --preflight-only")

    manifest, failures = load_reference(args.manifest, args.reference_dir)
    if args.case_index is not None:
        failures = [(index, source) for index, source in failures if index == args.case_index]
        if len(failures) != 1:
            parser.error("--case-index must identify a failed original episode")
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
            for count, (index, source) in enumerate(failures, 1):
                env = create_env(index)
                try:
                    replay_prefix(env, source)
                finally:
                    env.close()
                print(f"prefix verified {count}/{len(failures)}: case {index}", flush=True)
            save_json_new(args.output_dir / "preflight.json", {"verified": len(failures),
                          "case_indices": [index for index, _ in failures]})
        except Exception as error:
            save_json_new(args.output_dir / "failure.json", {"error_type": type(error).__name__,
                                                               "error": str(error)})
            raise
        return

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
                print("model server ready; continuing verified prefixes", flush=True)
                yield lambda messages: request_action(url, "qwen27b-fp8", messages)

    metadata = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "reference_config_sha256": hashlib.sha256((args.reference_dir / "config.json").read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "server_command": command, "max_steps": 50, "original_max_steps": 20,
                "temperature": 0, "max_new_tokens": 48, "thinking": False,
                "max_model_len": 4096, "per_case_seed": 0,
                "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
                "cpath": os.environ.get("CPATH")}
    summary = run_experiment(failures, create_env, policy_context, args.output_dir, metadata=metadata)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["overall_success"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
