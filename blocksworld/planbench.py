"""
读 PlanBench 的 Blocksworld 题（PDDL 文件），转换成我们的格式，写出要留出的结构签名。
这些结构不进 SFT、GRPO、自我训练的题池，PlanBench 才能当作外部考试。

PlanBench：https://github.com/karthikv792/LLMs-Planning（MIT License）。只读它 instances/ 下的数据，不运行它的代码。
规则已核对：它的四个动作和我们一致；task_1 的 500 份官方参考计划在我们的模拟器里全部成功，没有一份比 BFS 更长。

用法：python planbench.py <LLMs-Planning 仓库目录>
输出：splits/planbench_reserved.txt，每行一个结构签名
"""
import re
import subprocess
import sys
from pathlib import Path

from blocksworld import TABLE
from problems import canonical, signature_text

# 基础题（task_1 出题用 generated_basic，3 块版用 generated_basic_3），连同 prompt 里当示例的那一道都留出；
# generated 是泛化题，其中 3–5 块的也留出。mystery/ 下面是同一批题换了谓词名字，结构完全相同，不用另外读。
FOLDERS = ("generated_basic", "generated_basic_3", "generated")
OUT = Path(__file__).parent / "splits" / "planbench_reserved.txt"


def parse_pddl(text):
    """从一个 PDDL 题目文件里取出积木名、初始事实和目标事实。事实写成元组，例如 ("on", "b", "c")。"""
    text = text.lower()
    objects = re.search(r"\(:objects([^)]*)\)", text).group(1).split()
    init_part = text[text.index("(:init"):text.index("(:goal")]
    goal_part = text[text.index("(:goal"):]
    facts = lambda part: [tuple(m.split()) for m in re.findall(r"\(([a-z][a-z\- ]*)\)", part)]
    return objects, facts(init_part), [f for f in facts(goal_part) if f[0] != "and"]


def to_problem(objects, init_facts, goal_facts):
    """把积木名按字母顺序改成 A、B、C…，返回我们格式的 (init, goal)。
    初始状态必须手空、每块积木都写了位置；目标只能有 on / ontable。不符合就报错，免得悄悄转错。"""
    rename = {o: "ABCDEFGHIJKL"[i] for i, o in enumerate(sorted(objects))}
    names = {f[0] for f in init_facts}
    assert "handempty" in names and "holding" not in names, names
    on = {}
    for f in init_facts:
        if f[0] == "ontable":
            on[rename[f[1]]] = TABLE
        elif f[0] == "on":
            on[rename[f[1]]] = rename[f[2]]
    assert len(on) == len(objects), on
    goal = []
    for f in goal_facts:
        assert f[0] in ("on", "ontable"), f
        goal.append((rename[f[1]], TABLE) if f[0] == "ontable" else (rename[f[1]], rename[f[2]]))
    return {"on": on, "holding": None}, sorted(goal)


def main():
    root = Path(sys.argv[1])
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    sigs, lines = set(), []
    for folder in FOLDERS:
        files = sorted((root / "plan-bench" / "instances" / "blocksworld" / folder).glob("instance-*.pddl"))
        assert files, folder
        kept = set()
        for f in files:
            objects, init_facts, goal_facts = parse_pddl(f.read_text())
            if 3 <= len(objects) <= 5:  # 只有 3–5 块会和我们的题库重叠
                kept.add(signature_text(canonical(*to_problem(objects, init_facts, goal_facts))))
        lines.append(f"# {folder}: {len(files)} 个文件，3–5 块的去重后 {len(kept)} 个结构")
        sigs |= kept
    header = ["# PlanBench Blocksworld 的结构签名：不进 SFT、GRPO、自我训练的题池",
              f"# 来源 https://github.com/karthikv792/LLMs-Planning commit {commit}",
              *lines,
              f"# 合并去重后 {len(sigs)} 个结构（官方 600 道基础题单独去重是 262 个，另 1 个只在泛化题里）"]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(header + sorted(sigs)) + "\n")
    print("\n".join(header))
    print("写出", OUT)


if __name__ == "__main__":
    main()
