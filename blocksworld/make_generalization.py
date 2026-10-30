"""
生成 6–7 块积木的泛化测试题：只用于评测，不进 SFT、GRPO、自我训练的题池。
训练题只有 3–5 块，这里看模型能不能把学到的做法用到更多积木上。

6 块有 4,051 种摆法，7 块有 37,633 种，题太多，没法像 3–5 块那样全部列出来，所以随机抽，再按签名去重。
抽法尽量让每道题被抽到的机会和全部列举时一样，免得泛化成绩里混进“目标分布变了”的影响：
  complete：初始摆法、目标摆法各从全部摆法里均匀抽一个
  partial ：目标摆法均匀抽一个，要求它有 2 到 n−1 摞；写出它全部“谁在谁上面”，不写“在桌上”
            依据：k 条“谁在谁上面”组成的目标，正好对应一个有 n−k 摞的摆法，一一对应（test_blocksworld.py 穷举核对）
            1 摞时写满 n−1 条，定死整个摆法，和 complete 重复；n 摞时全在桌上，写不出“谁在谁上面”
  一开始就已经满足目标的题不要。
每格（积木数 × 目标类型）100 道，共 400 道。

用法：python make_generalization.py
"""
import collections
import json
import random
from pathlib import Path

from blocksworld import TABLE, all_arrangements, goal_reached, simulate, solve_bfs, stacks_of
from make_dataset import git_commit, sha256
from problems import build_sample, canonical, signature_text

VERSION = "v1"
SEED = 20261030
SIZES = (6, 7)
PER_CELL = 100
HERE = Path(__file__).parent
SPLIT_DIR = HERE / "splits" / VERSION
DATA_DIR = HERE.parent / "data" / "blocksworld" / VERSION


def sample_problem(arrangements, n, goal_type, rng):
    """按上面说的抽法抽一道题，返回 (init, goal)。"""
    while True:
        init, target = rng.choice(arrangements), rng.choice(arrangements)
        if goal_type == "complete":
            goal = sorted(target["on"].items())
        elif 2 <= len(stacks_of(target)) <= n - 1:
            goal = sorted((x, y) for x, y in target["on"].items() if y != TABLE)
        else:
            continue
        if not goal_reached(init, goal):
            return init, goal


def is_planbench_generalization_shape(init, goal, n):
    """PlanBench generated/ 里 6–7 块的题都是同一种结构：全部在桌上，目标叠成一摞。"""
    return all(y == TABLE for y in init["on"].values()) and sum(y != TABLE for _, y in goal) == n - 1


def main():
    rng = random.Random(SEED)
    problems, seen, duplicates = [], set(), 0
    for n in SIZES:
        arrangements = all_arrangements(n)
        for goal_type in ("complete", "partial"):
            kept = 0
            while kept < PER_CELL:
                init, goal = sample_problem(arrangements, n, goal_type, rng)
                sig = signature_text(canonical(init, goal))
                if sig in seen:
                    duplicates += 1
                    continue
                seen.add(sig)
                problems.append((init, goal, goal_type))
                kept += 1
    rng.shuffle(problems)  # 不打乱的话文件是按格排的，只取前 100 条就会全是 6 块写全目标

    samples = []
    for i, (init, goal, goal_type) in enumerate(problems):
        plan = solve_bfs(init, goal)
        assert plan and simulate(init, plan, goal)["success"]
        samples.append(build_sample(f"bw-{VERSION}-gen-{i:04d}", init, goal, goal_type, plan))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_file = DATA_DIR / "generalization.jsonl"
    data_file.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
    sig_file = SPLIT_DIR / "generalization.txt"
    sig_file.write_text("".join(sorted(s["canonical"] + "\n" for s in samples)))

    manifest = {
        "version": VERSION,
        "seed": SEED,
        "sizes": list(SIZES),
        "per_cell": PER_CELL,
        "code_commit": git_commit(),
        "count": len(samples),
        "by_group": dict(sorted(collections.Counter(f"n={s['n_blocks']} {s['goal_type']}" for s in samples).items())),
        "plan_len": dict(sorted(collections.Counter(s["plan_len"] for s in samples).items())),
        "duplicates_skipped": duplicates,
        "planbench_generalization_shape": sum(is_planbench_generalization_shape(i, g, len(i["on"]))
                                              for i, g, _ in problems),
        "data_file": str(data_file.relative_to(HERE.parent)),
        "data_sha256": sha256(data_file),
        "signatures_file": str(sig_file.relative_to(HERE)),
        "signatures_sha256": sha256(sig_file),
        "notes": ["只用于评测；以后自生成题目或自我训练筛题时，也要排除这些结构。",
                  "抽法让每道带名字的题机会相同；和 3–5 块按结构列举相比，少数有对称性的结构机会略小。",
                  "PlanBench 泛化集是另一份评测集，两份评测集有重合不算泄漏；这里只记录有几道是同一种结构。"],
    }
    (SPLIT_DIR / "generalization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "notes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
