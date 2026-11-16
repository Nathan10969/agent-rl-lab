"""多轮环境的第一步：输入动作，检查反馈与状态。"""
import pytest

import step_env
from step_env import BlocksworldEnv


def make_env():
    return BlocksworldEnv([["A"], ["B"]], [("A", "B")])


def test_pickup_returns_new_state_without_completing_goal():
    env = make_env()
    result = env.step(("pickup", "A"))
    assert result == {"legal": True, "reason": "", "state": {"on": {"B": "table"}, "holding": "A"}, "done": False}


def test_illegal_action_keeps_state_and_explains_why():
    env = make_env()
    first = env.step(("pickup", "A"))
    result = env.step(("pickup", "B"))
    assert result["legal"] is False
    assert "手里已经拿着 A" in result["reason"]
    assert result["state"] == first["state"]
    assert result["done"] is False


def test_stack_reaches_goal_and_ends_episode():
    env = make_env()
    env.step(("pickup", "A"))
    result = env.step(("stack", "A", "B"))
    assert result["legal"] is True and result["done"] is True
    assert result["state"] == {"on": {"A": "B", "B": "table"}, "holding": None}
    with pytest.raises(RuntimeError, match="结束"):
        env.step(("pickup", "B"))


def test_returned_state_cannot_mutate_environment():
    env = make_env()
    returned = env.step(("pickup", "A"))
    returned["state"]["on"]["B"] = "A"
    actual = env.step(("stack", "A", "B"))
    assert actual["legal"] is True and actual["done"] is True


def test_malformed_action_is_rejected_without_state_change():
    env = make_env()
    with pytest.raises(ValueError, match="动作"):
        env.step(())
    assert env.step(("pickup", "A"))["legal"] is True


def test_text_action_is_parsed_and_executed():
    env = make_env()
    result = env.step_text("pickup(A)")
    assert result["legal"] is True
    assert result["state"]["holding"] == "A"


@pytest.mark.parametrize("text", ["pickup()", "", "pickup(A)\nstack(A, B)"])
def test_bad_or_multiple_text_actions_are_rejected_without_state_change(text):
    env = make_env()
    result = env.step_text(text)
    assert result["legal"] is False
    assert result["reason"]
    assert result["state"] == {"on": {"A": "table", "B": "table"}, "holding": None}
    assert env.step_text("pickup(A)")["legal"] is True


def test_text_action_obeys_terminal_state():
    env = make_env()
    env.step_text("pickup(A)")
    env.step_text("stack(A, B)")
    with pytest.raises(RuntimeError, match="结束"):
        env.step_text("pickup()")


def test_format_feedback_describes_a_held_block():
    result = make_env().step_text("pickup(A)")
    assert step_env.format_feedback(result) == (
        "动作执行成功。\n当前状态：\nB 在桌上\n手里拿着 A\n目标尚未完成。"
    )


def test_format_feedback_includes_failure_reason_and_unchanged_state():
    env = make_env()
    env.step_text("pickup(A)")
    result = env.step_text("pickup(B)")
    text = step_env.format_feedback(result)
    assert text.startswith("动作执行失败：")
    assert result["reason"] in text
    assert "B 在桌上\n手里拿着 A\n目标尚未完成。" in text


def test_format_feedback_describes_completed_goal_in_sorted_order():
    env = make_env()
    env.step_text("pickup(A)")
    result = env.step_text("stack(A, B)")
    assert step_env.format_feedback(result) == (
        "动作执行成功。\n当前状态：\nA 在 B 上面\nB 在桌上\n手是空的\n目标已完成。"
    )
