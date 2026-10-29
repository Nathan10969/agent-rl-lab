"""
用手写的例子检查规则、模拟器、求解器和去重，全部通过后才去生成数据。
求解器和模拟器都可能有 bug，所以这里的正确答案是人工写好的，不靠它们互相验证。

运行：python test_blocksworld.py
"""
import itertools
import random

from blocksworld import (TABLE, all_arrangements, apply_action, check_action, goal_reached, make_state,
                         parse_plan, possible_actions, simulate, solve_bfs, stacks_of)
from make_dataset import rename_problem
from planbench import parse_pddl, to_problem
from problems import canonical, enumerate_problems, fixes_every_block, random_state

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


def test_enumerate_problems_3_blocks():
    """3 块积木：写全目标 26 道，部分目标 10 道。两个数都能手算：
    写全：13 种初始 × 13 种目标，去掉相同的 13 种，剩 156 种；改名有 6 种方式，每道题正好被数了 6 次，156 ÷ 6 = 26。
    部分：只能写 1 条“X 在 Y 上面”，13 种初始 × 6 种 (X, Y) = 78，去掉一开始就满足的 18 种，剩 60；60 ÷ 6 = 10。"""
    problems = enumerate_problems(3)
    assert sum(p["goal_type"] == "complete" for p in problems) == 26
    assert sum(p["goal_type"] == "partial" for p in problems) == 10


def test_enumerate_problems_matches_brute_force():
    """换一种笨办法再列一遍：初始摆法用全部摆法；部分目标从所有“谁在谁上面”的组合里挑，至少一种摆法能做到就算。
    两种办法列出的题必须完全一样，才说明“初始摆法只取每种形状一个代表”没有漏题，也没有重复。"""
    for n in (3, 4):
        arrangements = all_arrangements(n)
        on_sets = [{(x, y) for x, y in s["on"].items() if y != TABLE} for s in arrangements]
        blocks = "ABCDEFG"[:n]
        pairs = [(x, y) for x in blocks for y in blocks if x != y]
        partial_goals = [list(sub) for k in range(1, n - 1) for sub in itertools.combinations(pairs, k)
                         if any(set(sub) <= s for s in on_sets)]
        complete_goals = [sorted(s["on"].items()) for s in arrangements]
        expected = {canonical(init, goal) for init in arrangements for goal in complete_goals + partial_goals
                    if not goal_reached(init, goal)}
        got = [p["canonical"] for p in enumerate_problems(n)]
        assert len(got) == len(set(got))
        assert set(got) == expected


def test_rename_keeps_structure():
    """随机改名后，初始状态和目标一起变，结构签名不变，最短步数也不变。"""
    rng = random.Random(0)
    for p in enumerate_problems(4)[::20]:
        init, goal = rename_problem(p["init"], p["goal"], rng)
        assert canonical(init, goal) == p["canonical"]
        assert len(solve_bfs(init, goal)) == len(solve_bfs(p["init"], p["goal"]))


PLANBENCH_INSTANCE_1 = """(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b c)
(ontable c)
(ontable d)
(clear a)
(clear b)
(clear d)
)
(:goal
(and
(on c b))
)
)"""


def test_planbench_parse():
    """PlanBench generated_basic/instance-1.pddl：b 在 c 上，其余在桌上；目标 c 在 b 上。
    a、b、c、d 按字母顺序改成 A、B、C、D。目标里出现 on / ontable 以外的谓词要报错，不能悄悄丢掉。"""
    init, goal = to_problem(*parse_pddl(PLANBENCH_INSTANCE_1))
    assert stacks_of(init) == [["A"], ["C", "B"], ["D"]] and goal == [("C", "B")]
    try:
        to_problem(*parse_pddl(PLANBENCH_INSTANCE_1.replace("(on c b))", "(on c b) (clear a))")))
    except AssertionError:
        pass
    else:
        raise AssertionError("目标里的 clear 应该报错")


if __name__ == "__main__":
    tests = [f for name, f in globals().items() if name.startswith("test_")]
    for t in tests:
        t()
        print("✓", t.__name__, "—", t.__doc__.strip().splitlines()[0])
    print(f"全部 {len(tests)} 项通过")
