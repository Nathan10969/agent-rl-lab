"""
用手写的例子检查规则、模拟器、求解器和去重，全部通过后才去生成数据。
求解器和模拟器都可能有 bug，所以这里的正确答案是人工写好的，不靠它们互相验证。

运行：python test_blocksworld.py
"""
import random

from blocksworld import (TABLE, apply_action, check_action, make_state, parse_plan, possible_actions,
                         simulate, solve_bfs, stacks_of, state_key)
from problems import canonical, fixes_every_block, random_state

# 和 Codex 例子里一样的局面：B 压在 A 上，C 单独在桌上；目标是 A 在 B 上面
INIT = make_state([["A", "B"], ["C"]])
GOAL = [("A", "B")]


def run(plan_text, init=INIT, goal=GOAL):
    plan, err = parse_plan(plan_text)
    assert plan is not None, err
    return simulate(init, plan, goal)


def test_success():
    """标准解法：先把 B 拿开，再把 A 放到 B 上。"""
    r = run("unstack(B, A)\nputdown(B)\npickup(A)\nstack(A, B)")
    assert r["success"], r["error"]


def test_other_correct_plan():
    """同一道题不止一种正确计划：多绕两步也算成功，评测时不能要求和参考答案一字不差。"""
    r = run("unstack(B, A)\nputdown(B)\npickup(C)\nputdown(C)\npickup(A)\nstack(A, B)")
    assert r["success"], r["error"]


def test_illegal_actions():
    """每种非法动作都要被拦下，并且给出对应的原因。"""
    cases = [
        ("pickup(A)", "上面还有积木"),  # A 被 B 压着
        ("pickup(B)", "应该用 unstack"),  # B 在 A 上面，不在桌上
        ("stack(A, B)", "手里没有拿着 A"),  # 手是空的
        ("putdown(B)", "手里没有拿着 B"),
        ("unstack(C, A)", "C 不在 A 上面"),
        ("unstack(B, A)\npickup(C)", "手里已经拿着 B"),
        ("unstack(B, A)\nstack(B, B)", "不能把积木放在自己上面"),
        ("pickup(D)", "没有积木 D"),
    ]
    for text, expected in cases:
        r = run(text)
        assert not r["success"] and expected in r["error"], (text, r["error"])


def test_goal_not_reached():
    """每一步都合法，但停得太早。"""
    r = run("unstack(B, A)\nputdown(B)")
    assert not r["success"] and "没有达成目标" in r["error"]


def test_parse():
    """解析是严格的：大小写、参数个数、写法不对都算看不懂。"""
    assert parse_plan("stack(A,B)")[0] == [("stack", "A", "B")]  # 逗号后面没空格也接受
    for bad in ["pick up A", "pickup(A, B)", "stack(A)", "PICKUP(A)", "pickup(a)", ""]:
        assert parse_plan(bad)[0] is None, bad


def test_sussman_anomaly():
    """经典的 Sussman anomaly（1975）：先完成任何一个子目标，都会挡住另一个。最短计划是 6 步。"""
    init = make_state([["A", "C"], ["B"]])
    goal = [("A", "B"), ("B", "C")]
    plan = solve_bfs(init, goal)
    assert len(plan) == 6, plan
    assert simulate(init, plan, goal)["success"]


def test_solver_actions_are_legal():
    """求解器为了快，自己列出合法动作；这里在 300 个随机状态上核对它和 check_action 完全一致。"""
    rng = random.Random(0)
    for _ in range(300):
        state = random_state(rng.randint(2, 5), rng)
        for _ in range(rng.randint(0, 3)):  # 随机走几步，让手里有时拿着积木
            state = apply_action(state, rng.choice(possible_actions(state)))
        blocks = sorted(state["on"]) + ([state["holding"]] if state["holding"] else [])
        candidates = [(a, x) for a in ("pickup", "putdown") for x in blocks]
        candidates += [(a, x, y) for a in ("unstack", "stack") for x in blocks for y in blocks]
        legal = {c for c in candidates if check_action(state, c)[0]}
        listed = set(possible_actions(state))
        assert listed == legal, (stacks_of(state), state["holding"], listed ^ legal)


def test_canonical():
    """只是把积木换了名字的两道题，结构签名相同；结构真的不同，签名就不同。"""
    a = canonical(make_state([["A", "B"], ["C"]]), [("A", "B")])
    b = canonical(make_state([["C", "A"], ["B"]]), [("C", "A")])  # 改名：A→C，B→A，C→B
    c = canonical(make_state([["A", "B", "C"]]), [("A", "B")])
    assert a == b and a != c



def test_complete_goal_checks_table():
    """完整目标连“在桌上”也要满足：A 在 B 上了，但 C 被放到 A 上面，部分目标算成功，完整目标不算。"""
    plan = "unstack(B, A)\nputdown(B)\npickup(A)\nstack(A, B)\npickup(C)\nstack(C, A)"
    assert run(plan, goal=[("A", "B")])["success"]
    r = run(plan, goal=[("A", "B"), ("B", TABLE), ("C", TABLE)])
    assert not r["success"] and "没有达成目标" in r["error"]


def all_arrangements(n):
    """从“全部在桌上”出发，把能到达的状态都走一遍，收集手空时的状态，就是 n 块积木的全部摆法。"""
    start = make_state([[b] for b in "ABCDEFG"[:n]])
    seen, todo, found = {state_key(start)}, [start], []
    while todo:
        state = todo.pop()
        if state["holding"] is None:
            found.append(state)
        for action in possible_actions(state):
            nxt = apply_action(state, action)
            if state_key(nxt) not in seen:
                seen.add(state_key(nxt))
                todo.append(nxt)
    return found


def test_fixes_every_block():
    """只写“谁在谁上面”时，恰好 n−1 条关系才会定死整个摆法。3、4 块积木的全部摆法穷举核对。"""
    assert fixes_every_block([("A", "B"), ("B", "C")], 3)  # Codex 的例子：C 只能在桌上
    assert not fixes_every_block([("A", "B")], 3)  # C 可以在桌上，也可以在 A 上面
    for n, expected in ((3, 13), (4, 73)):
        arrangements = all_arrangements(n)
        assert len(arrangements) == expected  # n 块积木摆法总数的已知值
        on_sets = [{(x, y) for x, y in s["on"].items() if y != TABLE} for s in arrangements]
        for goal in on_sets:
            if goal:
                matches = sum(goal <= other for other in on_sets)  # 有几个摆法满足这个目标
                assert (matches == 1) == fixes_every_block(sorted(goal), n), goal


def test_canonical_complete_vs_partial():
    """“A 在 B 上”和“A 在 B 上、B 和 C 在桌上”是不同的要求，签名不同；
    3 块积木时“A 在 B 上、B 在 C 上”已经隐含“C 在桌上”，两种写法是同一道题，签名必须相同。"""
    init = make_state([["A"], ["B"], ["C"]])
    assert canonical(init, [("A", "B")]) != canonical(init, [("A", "B"), ("B", TABLE), ("C", TABLE)])
    assert canonical(init, [("A", "B"), ("B", "C")]) == canonical(init, [("A", "B"), ("B", "C"), ("C", TABLE)])


if __name__ == "__main__":
    tests = [f for name, f in globals().items() if name.startswith("test_")]
    for t in tests:
        t()
        print("✓", t.__name__, "—", t.__doc__.strip().splitlines()[0])
    print(f"全部 {len(tests)} 项通过")
