"""
生成正式的 SFT 数据：3–5 块积木的全部题，去掉留出的结构，再分 train / val / test。

步骤：
  1. enumerate_problems 列出 3–5 块的全部题（4,070 道）
  2. 去掉 splits/planbench_reserved.txt 里的 263 个结构，以及 examples/ 里两版调试题的结构
     调试题的签名按它们实际的初始状态和目标重新计算，不用文件里存的旧字段（v1 有几条和现在的规则不一致）
  3. 按 (积木数, 目标类型) 分组，每组打乱后按 80 / 10 / 10 切成 train / val / test
  4. 每道题随机改名：初始状态和目标用同一套改名，再用 BFS 求最短计划，最后交给模拟器重放检查
  5. 写出 data/blocksworld/v1/*.jsonl（被 Git 忽略），以及 splits/v1/ 下的签名列表和 manifest.json（提交到 Git）

排除的是当前签名规则认得出的同构重复。相关但不相同的题（例如一道题的目标是另一道的一部分）仍可能分在不同集合；
PlanBench 是否出现在模型的预训练数据里，这里也管不了。

用法：python make_dataset.py
"""
import collections
import hashlib
import json
import random
import subprocess
from pathlib import Path

from blocksworld import TABLE, make_state, simulate, solve_bfs
from problems import build_sample, canonical, enumerate_problems, signature_text

VERSION = "v1"
SEED = 20261029
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
HERE = Path(__file__).parent
RESERVED = HERE / "splits" / "planbench_reserved.txt"
DEBUG_FILES = [HERE / "examples" / "debug_v1.jsonl", HERE / "examples" / "debug_v2.jsonl"]
SPLIT_DIR = HERE / "splits" / VERSION
DATA_DIR = HERE.parent / "data" / "blocksworld" / VERSION


def read_signatures(path):
    return {line for line in path.read_text().splitlines() if line and not line.startswith("#")}


def debug_signatures():
    """两版调试题的结构签名，按实际的初始状态和目标重新算。"""
    sigs = set()
    for path in DEBUG_FILES:
        for line in path.read_text().splitlines():
            s = json.loads(line)
            sigs.add(signature_text(canonical(make_state(s["init"]), [tuple(f) for f in s["goal"]])))
    return sigs


def split_problems(problems, rng):
    """按 (积木数, 目标类型) 分组，每组打乱后切成 80 / 10 / 10，最后每份再整体打乱。
    组很小时 val、test 可能只分到 0 或 1 道，照实记录在 manifest 里。"""
    groups = collections.defaultdict(list)
    for p in problems:
        groups[(len(p["init"]["on"]), p["goal_type"])].append(p)
    splits = {name: [] for name in RATIOS}
    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        n_val = round(len(group) * RATIOS["val"])
        n_test = round(len(group) * RATIOS["test"])
        splits["val"] += group[:n_val]
        splits["test"] += group[n_val:n_val + n_test]
        splits["train"] += group[n_val + n_test:]
    for name in RATIOS:
        rng.shuffle(splits[name])  # 不打乱的话文件是按组排的，只取前 100 条就会全是 3 块积木
    return splits


def rename_problem(init, goal, rng):
    """随机改名：初始状态和目标用同一套改名。改名不改变结构，所以签名不变。"""
    blocks = sorted(init["on"])
    rename = dict(zip(blocks, rng.sample(blocks, len(blocks))))
    rename[TABLE] = TABLE
    new_init = {"on": {rename[x]: rename[y] for x, y in init["on"].items()}, "holding": None}
    return new_init, sorted((rename[x], rename[y]) for x, y in goal)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit():
    """生成数据时代码的版本。有没提交的改动就标 dirty，这时 commit 号不能完全代表所用代码。"""
    run = lambda *args: subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD") + (" (dirty)" if run("status", "--porcelain", ".") else "")


def main():
    pool = [p for n in (3, 4, 5) for p in enumerate_problems(n)]
    pool_sigs = {signature_text(p["canonical"]) for p in pool}
    reserved, debug = read_signatures(RESERVED), debug_signatures()
    assert reserved <= pool_sigs and debug <= pool_sigs  # 不在题库里，说明规则或转换出了问题
    remaining = [p for p in pool if signature_text(p["canonical"]) not in reserved | debug]
    print(f"题库 {len(pool)}，留出 PlanBench {len(reserved)}、调试题 {len(debug)}（两者重叠 {len(reserved & debug)}），"
          f"剩 {len(remaining)}")

    splits = split_problems(remaining, random.Random(SEED))
    rename_rng = random.Random(SEED + 1)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": VERSION,
        "seed": SEED,
        "ratios": RATIOS,
        "code_commit": git_commit(),
        "pool": len(pool),
        "excluded": {"planbench": len(reserved), "debug": len(debug), "overlap": len(reserved & debug),
                     "planbench_file": str(RESERVED.relative_to(HERE)), "planbench_sha256": sha256(RESERVED)},
        "remaining": len(remaining),
        "splits": {},
        "notes": ["排除的是当前签名规则认得出的同构重复；相关但不相同的题仍可能分在不同集合。",
                  "严格留出 PlanBench 后，3 块积木的部分目标题在训练池里一道都没有，4 块的部分目标题也很少。",
                  "data/ 被 Git 忽略；数据可用同一份代码和 seed 重新生成，再用这里的 sha256 核对是否一致。"],
    }
    for name in RATIOS:  # 按固定顺序 train、val、test 改名，随机数才能复现
        samples = []
        for i, p in enumerate(splits[name]):
            init, goal = rename_problem(p["init"], p["goal"], rename_rng)
            plan = solve_bfs(init, goal)
            assert plan and simulate(init, plan, goal)["success"]
            assert len(plan) == len(solve_bfs(p["init"], p["goal"]))  # 改名不该改变最短步数
            sample = build_sample(f"bw-{VERSION}-{name}-{i:04d}", init, goal, p["goal_type"], plan)
            assert sample["canonical"] == signature_text(p["canonical"])
            samples.append(sample)
        data_file = DATA_DIR / f"{name}.jsonl"
        data_file.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
        sig_file = SPLIT_DIR / f"{name}.txt"
        sig_file.write_text("".join(sorted(s["canonical"] + "\n" for s in samples)))
        manifest["splits"][name] = {
            "count": len(samples),
            "by_group": dict(sorted(collections.Counter(f"n={s['n_blocks']} {s['goal_type']}" for s in samples).items())),
            "plan_len": dict(sorted(collections.Counter(s["plan_len"] for s in samples).items())),
            "data_file": str(data_file.relative_to(HERE.parent)),
            "data_sha256": sha256(data_file),
            "signatures_file": str(sig_file.relative_to(HERE)),
            "signatures_sha256": sha256(sig_file),
        }
    split_sigs = [read_signatures(SPLIT_DIR / f"{name}.txt") for name in RATIOS]
    assert sum(map(len, split_sigs)) == len(set.union(*split_sigs)) == len(remaining)  # 三份互不重叠，加起来正好是全部
    (SPLIT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
