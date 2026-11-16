"""逐步生成纠错示范：错误动作必须可重放，但不能作为训练答案。"""
import pytest

import recovery_samples
from blocksworld import action_to_text, solve_bfs
from step_env import BlocksworldEnv, format_feedback


def test_one_error_and_bfs_recovery_match_environment():
    stacks, goal = [["A"], ["B"]], [("A", "B")]
    sample = recovery_samples.build_recovery_sample(stacks, goal)
    messages = sample["messages"]
    env = BlocksworldEnv(stacks, goal)

    assert [m["content"] for m in messages if m["role"] == "assistant"] == [
        "pickup(A)", "pickup(B)", "stack(A, B)"
    ]
    assert [m["train"] for m in messages if m["role"] == "assistant"] == [True, False, True]
    first = env.step_text(messages[1]["content"])
    assert first["legal"] and messages[2]["content"] == format_feedback(first)
    failed = env.step_text(messages[3]["content"])
    assert not failed["legal"] and failed["state"] == first["state"]
    assert messages[4]["content"] == format_feedback(failed)
    plan = solve_bfs(failed["state"], goal)
    assert [action_to_text(a) for a in plan] == ["stack(A, B)"]
    final = env.step_text(messages[5]["content"])
    assert final["legal"] and final["done"]
    assert messages[6]["content"] == format_feedback(final)


def test_recovery_works_when_first_action_is_unstack():
    sample = recovery_samples.build_recovery_sample([["A", "B"], ["C"]], [("A", "C")])
    messages = sample["messages"]
    assert messages[1]["content"] == "unstack(B, A)"
    assert messages[3]["train"] is False
    env = BlocksworldEnv([["A", "B"], ["C"]], [("A", "C")])
    for i in range(1, len(messages), 2):
        result = env.step_text(messages[i]["content"])
        assert result["legal"] is messages[i]["train"]
        assert messages[i + 1]["content"] == format_feedback(result)
    assert result["done"]


def test_already_solved_problem_is_rejected():
    with pytest.raises(ValueError, match="已经完成"):
        recovery_samples.build_recovery_sample([["A", "B"]], [("B", "A")])
