"""
生成 30 道调试题（3、4、5 块积木各 10 道，complete 和 partial 目标各一半），给人逐条检查。

每道题都走一遍：随机出题 → 求解器找最短计划 → 计划写成文字 → 再解析回来 → 模拟器重放验证。
任何一步不对就直接报错，不会悄悄存进文件。
之后生成正式的训练、验证、测试集时，会按 canonical 排除这些调试题。

调试题属于“小型人工检查样例”，按仓库约定放在 examples/，可以提交 Git：
  examples/debug_v1.jsonl  第一版（goal_type 还叫 full，完整目标没写“在桌上”），原样保留
  examples/debug_v2.jsonl  这一版
正式数据集以后写到 data/，它被 .gitignore 忽略，用同样的代码和 seed 可以重新生成。

运行：python make_debug_set.py   （几秒钟）
"""
import json
import random
from collections import Counter
from pathlib import Path

from blocksworld import parse_plan, simulate, solve_bfs
from problems import build_sample, canonical, random_problem

SEED = 0
PER_SIZE = 10  # 每种积木数出几道
OUT = Path(__file__).parent / "examples" / "debug_v2.jsonl"


def make_one(n, goal_type, rng, seen):
    """出一道和前面结构都不同的题，并验证它的参考计划。"""
    for _ in range(1000):
        init, goal = random_problem(n, goal_type, rng)
        key = canonical(init, goal)
        if key not in seen:
            break
    else:
        raise RuntimeError(f"{n} 块、{goal_type} 目标连抽 1000 次都是重复题：不同的题不够，如实报告，不凑数")
    seen.add(key)

    plan = solve_bfs(init, goal)
    assert plan is not None, "目标取自一个真实摆法，不可能无解"
    sample = build_sample("", init, goal, goal_type, plan)

    # 双重检查：从存下来的文字重新解析、重新执行，确认这条样本本身是对的
    parsed, err = parse_plan(sample["messages"][1]["content"])
    assert parsed is not None, err
    result = simulate(init, parsed, goal)
    assert result["success"], result["error"]
    return sample


def main():
    rng = random.Random(SEED)
    seen, samples = set(), []
    for n in (3, 4, 5):
        for i in range(PER_SIZE):
            sample = make_one(n, "complete" if i % 2 == 0 else "partial", rng, seen)
            sample["id"] = f"debug-v2-{len(samples):03d}"
            samples.append(sample)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    for s in samples[:2]:
        print(f"===== {s['id']}：{s['n_blocks']} 块，{s['goal_type']} 目标，最短 {s['plan_len']} 步 =====")
        print("[user]\n" + s["messages"][0]["content"])
        print("[assistant]\n" + s["messages"][1]["content"] + "\n")

    print("===== 统计 =====")
    for (n, g), c in sorted(Counter((s["n_blocks"], s["goal_type"]) for s in samples).items()):
        lens = sorted(s["plan_len"] for s in samples if s["n_blocks"] == n and s["goal_type"] == g)
        print(f"{n} 块 {g:8s}：{c} 道，最短计划步数 {lens}")
    print(f"共 {len(samples)} 道，结构全部不同，已写入 {OUT}")


if __name__ == "__main__":
    main()
