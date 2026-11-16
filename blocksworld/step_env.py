"""积木多轮交互的最小环境：每次执行一个结构化动作。"""
from blocksworld import apply_action, check_action, goal_reached, make_state, parse_plan


def format_feedback(result):
    """把一次动作的结果翻译成可读文字，不修改环境或结果。"""
    if result["legal"]:
        lines = ["动作执行成功。"]
    else:
        lines = [f"动作执行失败：{result['reason']}"]

    lines.append("当前状态：")
    state = result["state"]
    for block, location in sorted(state["on"].items()):
        position = "在桌上" if location == "table" else f"在 {location} 上面"
        lines.append(f"{block} {position}")
    holding = state["holding"]
    lines.append(f"手里拿着 {holding}" if holding is not None else "手是空的")
    lines.append("目标已完成。" if result["done"] else "目标尚未完成。")
    return "\n".join(lines)


class BlocksworldEnv:
    def __init__(self, stacks, goal):
        self._state = make_state(stacks)
        self._goal = tuple(tuple(item) for item in goal)
        self._done = goal_reached(self._state, self._goal)

    def step(self, action):
        """输入例如 ("pickup", "A")；返回是否合法、原因、新状态和目标是否完成。"""
        if self._done:
            raise RuntimeError("本局已经结束，不能继续执行动作")
        if not isinstance(action, tuple) or not action or not all(isinstance(x, str) for x in action):
            raise ValueError("动作必须是非空的字符串元组，例如 ('pickup', 'A')")

        legal, reason = check_action(self._state, action)
        if legal:
            self._state = apply_action(self._state, action)
        self._done = goal_reached(self._state, self._goal)
        state_snapshot = {"on": dict(self._state["on"]), "holding": self._state["holding"]}
        return {"legal": legal, "reason": reason, "state": state_snapshot, "done": self._done}

    def step_text(self, text):
        """接收一条动作文字；格式错误或多个动作时返回反馈，不改变状态。"""
        if self._done:
            raise RuntimeError("本局已经结束，不能继续执行动作")
        if not isinstance(text, str):
            raise ValueError("动作文字必须是字符串")
        actions, error = parse_plan(text)
        if actions is None or len(actions) != 1:
            reason = error if actions is None else "每轮只能输入一个动作"
            state_snapshot = {"on": dict(self._state["on"]), "holding": self._state["holding"]}
            return {"legal": False, "reason": reason, "state": state_snapshot, "done": self._done}
        return self.step(actions[0])
