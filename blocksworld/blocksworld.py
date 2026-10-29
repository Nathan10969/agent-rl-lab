"""
Blocksworld 的规则、模拟器和求解器。第 1 级（生成 SFT 数据、做评测）和第 2 级（GRPO 奖励）共用这一份代码。

状态：state = {"on": {"A": "table", "B": "A"}, "holding": None}
  "on" 记录每块积木下面是什么：另一块积木，或者桌子 "table"。拿在手里的积木不在 "on" 里。
  "holding" 是手里的积木，手空时为 None。
动作：("pickup", "A")、("putdown", "A")、("unstack", "B", "A")、("stack", "A", "B")
  写成文字是 pickup(A)、putdown(A)、unstack(B, A)、stack(A, B)。
目标：[("A", "B")] 表示“A 在 B 上面”。和 PlanBench 一样只写“谁在谁上面”，没提到的积木放哪都行。

自检：python test_blocksworld.py
"""
import re
from collections import deque

TABLE = "table"
ARITY = {"pickup": 1, "putdown": 1, "unstack": 2, "stack": 2}  # 每个动作要几个参数


# ---------- 1. 状态 ----------

def make_state(stacks):
    """从“一摞一摞”的写法建状态，手是空的。
    每一摞从下往上写，例如 [["A", "B"], ["C"]] 表示 B 在 A 上面，C 单独在桌上。"""
    on = {}
    for stack in stacks:
        below = TABLE
        for block in stack:
            on[block] = below
            below = block
    return {"on": on, "holding": None}


def stacks_of(state):
    """make_state 的反过程：返回每一摞（从下往上），按最底下那块的名字排序。手里的积木不算。"""
    above = {below: block for block, below in state["on"].items() if below != TABLE}
    stacks = []
    for bottom in sorted(b for b, below in state["on"].items() if below == TABLE):
        stack = [bottom]
        while stack[-1] in above:
            stack.append(above[stack[-1]])
        stacks.append(stack)
    return stacks


def blocks_of(state):
    """所有积木的名字，包括手里那块。"""
    names = set(state["on"])
    if state["holding"] is not None:
        names.add(state["holding"])
    return sorted(names)


def is_clear(state, x):
    """x 上面没有积木，并且 x 不在手里。"""
    return state["holding"] != x and x not in state["on"].values()


# ---------- 2. 动作：合法条件和执行 ----------

def action_to_text(action):
    return f"{action[0]}({', '.join(action[1:])})"


def check_action(state, action):
    """返回 (是否合法, 不合法的原因)。原因写清楚，以后可以统计模型最常犯哪种错。"""
    name, args = action[0], action[1:]
    text = action_to_text(action)
    if name not in ARITY:
        return False, f"{text}：没有这个动作"
    if len(args) != ARITY[name]:
        return False, f"{text}：参数个数不对"
    for b in args:
        if b not in blocks_of(state):
            return False, f"{text}：没有积木 {b}"
    hand = state["holding"]
    on = state["on"]

    if name == "pickup":
        x = args[0]
        if hand is not None:
            return False, f"{text}：手里已经拿着 {hand}"
        if on[x] != TABLE:
            return False, f"{text}：{x} 在 {on[x]} 上面，不在桌上，应该用 unstack"
        if not is_clear(state, x):
            return False, f"{text}：{x} 上面还有积木"
    elif name == "unstack":
        x, y = args
        if hand is not None:
            return False, f"{text}：手里已经拿着 {hand}"
        if on[x] != y:
            return False, f"{text}：{x} 不在 {y} 上面"
        if not is_clear(state, x):
            return False, f"{text}：{x} 上面还有积木"
    elif name == "putdown":
        x = args[0]
        if hand != x:
            return False, f"{text}：手里没有拿着 {x}"
    elif name == "stack":
        x, y = args
        if hand != x:
            return False, f"{text}：手里没有拿着 {x}"
        if x == y:
            return False, f"{text}：不能把积木放在自己上面"
        if not is_clear(state, y):
            return False, f"{text}：{y} 上面还有积木"
    return True, ""


def apply_action(state, action):
    """执行一个已经确认合法的动作，返回新状态（不修改原来的 state）。"""
    name, args = action[0], action[1:]
    on = dict(state["on"])
    holding = state["holding"]
    if name in ("pickup", "unstack"):
        holding = args[0]
        del on[args[0]]
    elif name == "putdown":
        on[args[0]] = TABLE
        holding = None
    elif name == "stack":
        on[args[0]] = args[1]
        holding = None
    return {"on": on, "holding": holding}


def goal_reached(state, goal):
    return all(state["on"].get(x) == y for x, y in goal)


# ---------- 3. 模拟器：逐步执行一份计划 ----------

def simulate(init, plan, goal):
    """按顺序执行 plan（动作列表），返回一个字典：
        success   是否每一步都合法，并且最后达成目标
        steps_ok  成功执行了几步
        error     失败原因；成功时为空字符串
        final     执行到最后（或出错前）的状态
    遇到第一个非法动作就停下，不再往后执行。"""
    state = init
    for i, action in enumerate(plan):
        ok, reason = check_action(state, action)
        if not ok:
            return {"success": False, "steps_ok": i, "error": f"第 {i + 1} 步不合法：{reason}", "final": state}
        state = apply_action(state, action)
    if not goal_reached(state, goal):
        return {"success": False, "steps_ok": len(plan), "error": "每一步都合法，但没有达成目标", "final": state}
    return {"success": True, "steps_ok": len(plan), "error": "", "final": state}


# ---------- 4. 把模型输出的文字解析成动作 ----------

ACTION_RE = re.compile(r"^(pickup|putdown|unstack|stack)\(\s*([A-Z])\s*(?:,\s*([A-Z])\s*)?\)$")


def parse_plan(text):
    """每行一个动作，空行跳过。返回 (动作列表, 错误)；有一行看不懂就返回 (None, 错误)。
    格式要求是严格的：第 2 级给 GRPO 打分时，格式不对也算失败。"""
    plan = []
    for i, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        m = ACTION_RE.match(line)
        if m is None:
            return None, f"第 {i + 1} 行看不懂：{line!r}"
        name, x, y = m.groups()
        args = (x,) if y is None else (x, y)
        if len(args) != ARITY[name]:
            return None, f"第 {i + 1} 行参数个数不对：{line!r}"
        plan.append((name, *args))
    if not plan:
        return None, "没有任何动作"
    return plan, ""


# ---------- 5. 求解器：广度优先搜索，找最短计划 ----------

def possible_actions(state):
    """当前状态下所有合法的动作。直接按规则列出来，不逐个去试 check_action，搜索会快很多。
    test_blocksworld.py 会核对这里列出的动作和 check_action 的判断完全一致。"""
    hand = state["holding"]
    clear = [b for b in blocks_of(state) if is_clear(state, b)]
    if hand is None:
        return [("pickup", x) if state["on"][x] == TABLE else ("unstack", x, state["on"][x]) for x in clear]
    return [("putdown", hand)] + [("stack", hand, y) for y in clear]


def state_key(state):
    """把状态变成能放进 set 的样子，用来记录哪些状态已经搜过。"""
    return tuple(sorted(state["on"].items())), state["holding"]


def solve_bfs(init, goal):
    """返回一份最短计划（动作列表）；无解时返回 None。
    广度优先：先把 1 步能到的状态都试完，再试 2 步的……所以第一次碰到目标时，步数一定最少。"""
    if goal_reached(init, goal):
        return []
    queue = deque([(init, [])])
    seen = {state_key(init)}
    while queue:
        state, plan = queue.popleft()
        for action in possible_actions(state):
            nxt = apply_action(state, action)
            key = state_key(nxt)
            if key in seen:
                continue
            if goal_reached(nxt, goal):
                return plan + [action]
            seen.add(key)
            queue.append((nxt, plan + [action]))
    return None
