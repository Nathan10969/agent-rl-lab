"""
随机出题、去重，以及把一道题写成一条 SFT 样本。

一条样本的字段：
  id          样本编号
  n_blocks    积木数
  goal_type   "complete"：写出每一块积木的位置，包括“X 在桌上”；"partial"：只写几条“谁在谁上面”，没提到的积木放哪都行
  init        初始状态，一摞一摞从下往上写
  goal        目标，[["A", "B"]] 表示 A 在 B 上面，[["B", "table"]] 表示 B 在桌上
  plan_len    求解器找到的最短计划有几步，用来按难度统计
  canonical   结构签名：两道题只是把积木换了名字，这一项就相同，用来去重
  messages    user / assistant 两条消息，SFT 直接用这一项
"""
import itertools

from blocksworld import TABLE, action_to_text, goal_reached, make_state, stacks_of

NAMES = "ABCDEFG"

RULES = """你在控制一只机械臂摆积木。规则：
- 机械臂一次只能拿一块积木，并且只能拿上面没有其他积木的积木。
- pickup(X)：手空时，从桌上拿起 X。
- unstack(X, Y)：手空时，把 X 从 Y 上面拿起来。
- putdown(X)：把手里的 X 放到桌上。
- stack(X, Y)：把手里的 X 放到 Y 上面，Y 上面必须没有积木。"""


# ---------- 1. 随机出题 ----------

def random_state(n, rng):
    """随机摆 n 块积木，手是空的。逐块决定：另起一摞，或者叠到已有某一摞的最上面。
    这样抽到的摆法不是均匀分布，但足够多样；需要时再换成均匀采样。"""
    blocks = list(NAMES[:n])
    rng.shuffle(blocks)
    stacks = []
    for b in blocks:
        if not stacks or rng.random() < 0.4:
            stacks.append([b])
        else:
            rng.choice(stacks).append(b)
    return make_state(stacks)


def random_problem(n, goal_type, rng):
    """随机出一道 n 块积木的题，返回 (初始状态, 目标)。目标取自另一个随机摆法，所以一定能做到。
      complete：写出目标摆法里每一块积木的位置，包括“X 在桌上”
      partial ：只写其中几条“谁在谁上面”，没提到的积木放哪都行
    一开始就已经满足目标的题不要。"""
    while True:
        init = random_state(n, rng)
        target = random_state(n, rng)
        if goal_type == "complete":
            goal = sorted(target["on"].items())
        else:
            facts = sorted((x, y) for x, y in target["on"].items() if y != TABLE)
            if not facts:
                continue  # 目标摆法里所有积木都在桌上，写不出“谁在谁上面”，重抽
            goal = sorted(rng.sample(facts, rng.randint(1, len(facts))))
            if fixes_every_block(goal, n):
                continue  # 这样写其实已经定死了每块积木的位置，和某道 complete 题是同一道题，重抽
        if not goal_reached(init, goal):
            return init, goal


def fixes_every_block(goal, n):
    """只写“谁在谁上面”的目标，什么时候其实已经定死了每一块积木的位置？
    n 块积木写了 n−1 条这样的关系，就只能叠成一摞，最底下那块只能在桌上。
    例：3 块积木，目标“A 在 B 上面、B 在 C 上面”，C 只能在桌上，和写全了的目标是同一个要求。
    test_blocksworld.py 用 3、4 块积木的全部摆法穷举核对了这条规律。"""
    return len(goal) == n - 1


def complete_form(goal, blocks):
    """如果只写“谁在谁上面”的目标其实已经定死了每块积木的位置，就补上省略的那句“X 在桌上”。
    这样同一个要求的两种写法会变成同一个样子，去重时才认得出来。"""
    goal = sorted(tuple(fact) for fact in goal)
    if any(y == TABLE for _, y in goal) or not fixes_every_block(goal, len(blocks)):
        return goal
    supported = {x for x, _ in goal}
    bottom = next(b for b in blocks if b not in supported)  # 唯一一块没写下面是什么的，就是最底下那块
    return sorted(goal + [(bottom, TABLE)])


# ---------- 2. 去重：只是改了积木名字的题算同一道 ----------

def canonical(init, goal):
    """把积木改名的所有方式都试一遍，取字典序最小的那种写法，作为这道题的结构签名。
    7 块积木最多 5040 种改名方式，直接暴力枚举就够了。
    比较之前先用 complete_form 补上省略的“在桌上”，所以同一个要求的两种写法会得到同一个签名。"""
    blocks = sorted(init["on"])
    goal = complete_form(goal, blocks)
    best = None
    for perm in itertools.permutations(blocks):
        rename = dict(zip(blocks, perm))
        rename[TABLE] = TABLE
        sig = (tuple(sorted((rename[x], rename[y]) for x, y in init["on"].items())),
               tuple(sorted((rename[x], rename[y]) for x, y in goal)))
        if best is None or sig < best:
            best = sig
    return best


def signature_text(sig):
    """把签名写成一行字，方便存进 JSON，例如 "A>B,B>table | A>B"。"""
    init_part, goal_part = sig
    return ",".join(f"{x}>{y}" for x, y in init_part) + " | " + ",".join(f"{x}>{y}" for x, y in goal_part)


# ---------- 3. 写成 SFT 样本 ----------

def describe_stacks(stacks):
    """把一摞一摞的摆法写成几行字：每摞先写最底下那块在桌上，再从下往上写谁在谁上面。"""
    lines = []
    for stack in stacks:
        lines.append(f"- {stack[0]} 在桌上")
        for below, above in zip(stack, stack[1:]):
            lines.append(f"- {above} 在 {below} 上面")
    return lines


def format_prompt(init, goal, goal_type):
    """user 消息：规则、初始状态、目标。两种目标的写法不同，末尾用一句话说明是哪一种。"""
    lines = [RULES, "", "初始状态："]
    lines += describe_stacks(stacks_of(init))
    lines.append("- 手是空的")
    lines += ["", "目标："]
    if goal_type == "complete":
        lines += describe_stacks(stacks_of({"on": dict(goal), "holding": None}))
        lines.append("（每一块积木的位置都已写出）")
    else:
        lines += [f"- {x} 在 {y} 上面" for x, y in goal]
        lines.append("（没有提到的积木放在哪里都可以）")
    lines += ["", "请给出动作序列，每行一个动作，不要输出其他内容。"]
    return "\n".join(lines)


def build_sample(sample_id, init, goal, goal_type, plan):
    return {
        "id": sample_id,
        "n_blocks": len(init["on"]),
        "goal_type": goal_type,
        "init": stacks_of(init),
        "goal": [list(fact) for fact in goal],
        "plan_len": len(plan),
        "canonical": signature_text(canonical(init, goal)),
        "messages": [
            {"role": "user", "content": format_prompt(init, goal, goal_type)},
            {"role": "assistant", "content": "\n".join(action_to_text(a) for a in plan)},
        ],
    }
