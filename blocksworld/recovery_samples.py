"""教学用纠错示范：先故意犯一次错误，再从当前状态用 BFS 找解。"""
from blocksworld import action_to_text, make_state, solve_bfs
from problems import RULES, describe_stacks
from step_env import BlocksworldEnv, format_feedback


def build_recovery_sample(stacks, goal):
    """从手空的 2–5 块积木题构造一条示范；只制造一种非法 pickup。"""
    blocks = [block for stack in stacks for block in stack]
    if (not 2 <= len(blocks) <= 5 or len(set(blocks)) != len(blocks)
            or any(not isinstance(b, str) or len(b) != 1 or b not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in blocks)
            or any(not stack for stack in stacks)):
        raise ValueError("需要 2–5 块不重复的大写字母积木，且每摞不能为空")
    if not goal or any(len(fact) != 2 or fact[0] not in blocks
                       or fact[1] not in blocks + ["table"] for fact in goal):
        raise ValueError("目标必须是现有积木的位置关系")

    initial_state = make_state(stacks)
    initial_plan = solve_bfs(initial_state, goal)
    if initial_plan == []:
        raise ValueError("目标已经完成，不能制造中途纠错")
    if initial_plan is None:
        raise ValueError("目标无解")

    goal_lines = [f"- {x} 在桌上" if y == "table" else f"- {x} 在 {y} 上面" for x, y in goal]
    prompt = "\n".join([RULES, "", "初始状态：", *describe_stacks(stacks), "- 手是空的",
                         "", "目标：", *goal_lines, "", "每轮只输出一个动作，等待环境反馈。"])
    messages = [{"role": "user", "content": prompt}]
    env = BlocksworldEnv(stacks, goal)

    def record(action, train):
        text = action_to_text(action)
        result = env.step_text(text)
        if result["legal"] != train:
            raise RuntimeError("环境结果与动作标注不一致")
        messages.append({"role": "assistant", "content": text, "train": train})
        messages.append({"role": "user", "content": format_feedback(result)})
        return result

    first = record(initial_plan[0], True)
    held = first["state"]["holding"]
    other = next(block for block in sorted(blocks) if block != held)
    failed = record(("pickup", other), False)
    if failed["state"] != first["state"]:
        raise RuntimeError("非法动作不应改变环境状态")

    recovery_plan = solve_bfs(failed["state"], goal)
    if recovery_plan is None:
        raise RuntimeError("错误后的状态无解")
    for action in recovery_plan:
        final = record(action, True)
    if not recovery_plan or not final["done"]:
        raise RuntimeError("纠正计划没有完成目标")
    return {"init": stacks, "goal": goal, "messages": messages}
