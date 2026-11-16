"""Minimal, inference-only closed-loop baseline for the archived tasks."""
import copy
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "blocksworld"))
from blocksworld import parse_plan
from recovery_samples import build_recovery_sample
from step_env import BlocksworldEnv, format_feedback


def save_json_new(path, value):
    """Write one result once, never replacing an earlier experiment."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def summarize(episodes):
    """Keep per-task success separate from per-action syntax and legality."""
    actions = [turn for episode in episodes for turn in episode["trace"]]
    summary = {"total": len(episodes), "success": sum(bool(e["success"]) for e in episodes),
               "actions": len(actions)}
    if actions and all("format_valid" in turn and "legal" in turn for turn in actions):
        summary["format_valid_actions"] = sum(turn["format_valid"] for turn in actions)
        summary["legal_actions"] = sum(turn["legal"] for turn in actions)
    return summary


def select_block_cases(cases):
    """Choose the first fixed development case at each agreed difficulty."""
    selected = []
    for length in (4, 6, 8):
        case = next((item for item in cases if item["oracle_steps"] == length), None)
        if case is None:
            raise ValueError(f"no development case with an oracle length of {length}")
        selected.append(case)
    return selected


def request_action(base_url, model, messages, timeout=120):
    """Ask the local inference server for one deterministic text action."""
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 48,
               "stop": ["\n"], "chat_template_kwargs": {"enable_thinking": False}}
    request = Request(base_url.rstrip("/") + "/v1/chat/completions",
                      data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    text = result["choices"][0]["message"]["content"]
    if not isinstance(text, str):
        raise ValueError("the inference server did not return a text action")
    return text.strip()


def run_blocksworld(case, policy, max_steps):
    """Run one actual environment; policy sees past actions and feedback."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    messages = [build_recovery_sample(case["init"], case["goal"])["messages"][0]]
    env = BlocksworldEnv(case["init"], case["goal"])
    trace = []
    for _ in range(max_steps):
        action = policy(copy.deepcopy(messages))
        parsed, _ = parse_plan(action)
        result = env.step_text(action)
        feedback = format_feedback(result)
        trace.append({"action": action, "format_valid": parsed is not None and len(parsed) == 1,
                      "legal": result["legal"], "done": result["done"],
                      "feedback": feedback})
        messages.extend([{"role": "assistant", "content": action},
                         {"role": "user", "content": feedback}])
        if result["done"]:
            break
    success = trace[-1]["done"]
    return {"source_id": case["source_id"], "success": success,
            "termination_reason": "goal_reached" if success else "step_limit", "trace": trace}


def run_alfworld(env, policy, max_steps):
    """Run one ALFWorld game; distinguish a won game from a time limit."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    observations, _ = env.reset()
    instruction = ("You are interacting with a text household environment. "
                   "Output exactly one environment command per turn, with no explanation; "
                   "wait for feedback before the next command.\n\n")
    messages = [{"role": "user", "content": instruction + observations[0]}]
    trace = []
    won = done = False
    for _ in range(max_steps):
        action = policy(copy.deepcopy(messages))
        observations, scores, dones, info = env.step([action])
        won, done = bool(info["won"][0]), bool(dones[0])
        feedback = observations[0]
        trace.append({"action": action, "feedback": feedback, "score": float(scores[0]),
                      "won": won, "done": done})
        messages.extend([{"role": "assistant", "content": action},
                         {"role": "user", "content": feedback}])
        if won or done:
            break
    reason = "goal_reached" if won else "environment_done" if done else "step_limit"
    return {"success": won, "termination_reason": reason, "trace": trace}
