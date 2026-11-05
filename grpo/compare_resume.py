"""比较连续运行与续训的逐答案 token、adapter 和完整训练状态。"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from resume_state import inspect_checkpoint


def compare_values(left, right):
    """递归比较嵌套状态；返回是否逐位相同和浮点／整数张量最大绝对差。"""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, None
        equal = torch.equal(left, right)
        diff = 0.0 if not left.numel() else (left.to(torch.float64) - right.to(torch.float64)).abs().max().item()
        return equal, diff
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False, None
        parts = [compare_values(left[k], right[k]) for k in left]
    elif isinstance(left, (list, tuple)) and type(left) is type(right):
        if len(left) != len(right):
            return False, None
        parts = [compare_values(a, b) for a, b in zip(left, right)]
    else:
        return left == right, 0.0 if left == right else None
    if any(diff is None for _, diff in parts):
        return False, None
    return all(equal for equal, _ in parts), max((diff for _, diff in parts), default=0.0)


def read_samples(run, start, end):
    rows = [json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()]
    selected = [row for row in rows if start <= row["step"] <= end]
    keys = [(row["step"], row["id"], row["k"]) for row in selected]
    if not selected or len(set(keys)) != len(keys) or {r["step"] for r in selected} != set(range(start, end + 1)):
        raise ValueError("逐答案记录缺失步骤或存在重复")
    return selected


def read_checkpoint(run, step):
    return read_checkpoint_path(run / f"checkpoint-{step:04d}")


def read_checkpoint_path(path):
    manifest = json.loads((path / "resume_manifest.json").read_text())
    state = inspect_checkpoint(path, manifest["compatibility"])
    return load_file(str(path / "adapter_model.safetensors")), state, manifest["compatibility"]


def adapter_config(path):
    """PEFT 将 target_modules 保存为列表，其顺序不影响 LoRA 含义。"""
    config = json.loads((path / "adapter_config.json").read_text())
    for key in ("target_modules", "exclude_modules"):
        if isinstance(config.get(key), list):
            config[key] = sorted(config[key])
    return config


def compare_runs(continuous, resumed, start, end):
    continuous, resumed = Path(continuous), Path(resumed)
    if continuous.resolve() == resumed.resolve():
        raise ValueError("不能把同一次运行与自身比较")
    if start < 2 or end < start:
        raise ValueError("对照步骤范围无效")
    continuous_run = json.loads((continuous / "config.json").read_text())
    resumed_run = json.loads((resumed / "config.json").read_text())
    if continuous_run.get("resume_from"):
        raise ValueError("连续运行不应来自续训存档")
    if not resumed_run.get("resume_from"):
        raise ValueError("缺少续训来源；无法证明发生过恢复")
    source = Path(resumed_run["resume_from"])
    a, b = read_samples(continuous, start, end), read_samples(resumed, start, end)
    keys = lambda rows: [(x["step"], x["id"], x["k"]) for x in rows]  # noqa: E731
    ids_equal = keys(a) == keys(b)
    def field_equal(field):
        return ids_equal and all(x[field] == y[field] for x, y in zip(a, b))
    a_model, a_state, a_config = read_checkpoint(continuous, end)
    b_model, b_state, b_config = read_checkpoint(resumed, end)
    a_start_model, a_start_state, a_start_config = read_checkpoint(continuous, start - 1)
    b_start_model, b_start_state, b_start_config = read_checkpoint_path(source)
    if b_start_state["completed_step"] != start - 1:
        raise ValueError("续训来源与对照起始步骤不一致")
    start_adapter_equal, start_adapter_diff = compare_values(a_start_model, b_start_model)
    start_adapter_config_equal = adapter_config(continuous / f"checkpoint-{start - 1:04d}") == adapter_config(source)
    start_state_equal, start_state_diff = compare_values(a_start_state, b_start_state)
    start_config_equal = a_start_config == b_start_config
    for rows, config in ((a, a_config), (b, b_config)):
        group_size = config.get("group_size")
        prompts = config.get("prompts_per_step")
        if not isinstance(group_size, int) or not isinstance(prompts, int) or group_size < 1 or prompts < 1:
            raise ValueError("存档缺少有效的答案数量配置")
        for step in range(start, end + 1):
            group = [row for row in rows if row["step"] == step]
            ids = {row["id"] for row in group}
            if len(group) != group_size * prompts or len(ids) != prompts:
                raise ValueError(f"step {step} 的答案或题目数量不完整")
            for question_id in ids:
                if {row["k"] for row in group if row["id"] == question_id} != set(range(group_size)):
                    raise ValueError(f"step {step} 的答案序号或数量不完整")
    adapter_equal, adapter_diff = compare_values(a_model, b_model)
    optimizer_equal, optimizer_diff = compare_values(a_state["optimizer"], b_state["optimizer"])
    rng_equal, _ = compare_values(
        {k: a_state[k] for k in ("python_rng", "torch_cpu_rng", "torch_mps_rng")},
        {k: b_state[k] for k in ("python_rng", "torch_cpu_rng", "torch_mps_rng")},
    )
    progress_equal = all(a_state[k] == b_state[k] for k in ("completed_step", "order", "parameter_names", "device"))
    return {"continuous_run": str(continuous), "resumed_run": str(resumed), "source_checkpoint": str(source),
            "start_step": start, "end_step": end, "continuous_samples": len(a), "resumed_samples": len(b),
            "start_adapter_equal": start_adapter_equal, "start_adapter_max_abs_diff": start_adapter_diff,
            "start_adapter_config_equal": start_adapter_config_equal,
            "start_state_equal": start_state_equal, "start_state_max_abs_diff": start_state_diff,
            "start_compatibility_equal": start_config_equal,
            "question_ids_equal": ids_equal, "tokens_equal": field_equal("token_ids"),
            "rewards_equal": field_equal("reward"), "advantages_equal": field_equal("advantage"),
            "texts_equal": field_equal("text"), "compatibility_equal": a_config == b_config,
            "adapter_equal": adapter_equal, "adapter_max_abs_diff": adapter_diff,
            "adapter_config_equal": adapter_config(continuous / f"checkpoint-{end:04d}") == adapter_config(resumed / f"checkpoint-{end:04d}"),
            "optimizer_equal": optimizer_equal, "optimizer_max_abs_diff": optimizer_diff,
            "rng_equal": rng_equal, "progress_equal": progress_equal}


def main():
    parser = argparse.ArgumentParser(description="比较连续和续训结果")
    parser.add_argument("--continuous", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"输出已存在，拒绝覆盖：{out}")
    result = compare_runs(args.continuous, args.resumed, args.start, args.end)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
