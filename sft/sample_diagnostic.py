"""
进 RL 之前的采样诊断。GRPO 每道题采样一组答案，按组内谁比平均好来学；
一组全对或全错时没有差别，这道题就学不到东西。所以要先看：随机采样时，有多少题“有对有错”。

对全量 SFT 的每个 checkpoint，在 train 和 val 各取前 N 道题：
  贪心   每道题生成 1 次，设定和 train_full.py 的评测一样（每批 2 道），用来和全量训练的结果核对
  采样   每道题在温度 T 下采样 K 次，统计：
         采样成功率   N×K 次里成功的比例。RL 训练时模型就是这样“随机地答”
         pass@K      K 次里至少成功一次的题有几道：当前模型在这种采样设定下探索得到成功的范围。
                     只是诊断指标，不是 RL 的上限，RL 之后可能超过它
         全对 / 全错 / 有对有错   各几道题；只有“有对有错”的题在 GRPO 里有学习信号
         不同答案数   K 次里平均有几种不同的计划。太少说明模型几乎每次都说同一句话，RL 没东西可探索
         最短比例     成功的答案里，步数等于 BFS 最短步数的比例
         没写结束符   写满 max_new_tokens 还没停下的条数

只读：checkpoint 和数据都不改；结果写到新的 --out 目录，目录已存在就拒绝运行。
test、6–7 块泛化题、PlanBench 都不碰。

运行（要在 Claude Code 外面的终端里，沙箱里用不了 MPS）：
  .venv/bin/python -u sft/sample_diagnostic.py --run-dir outputs/sft/full-20261029-codex-01 \\
      --out outputs/sft/sample-diag-20261029-01
"""
import argparse
import json
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import MODEL_PATH, REPO, load_jsonl
from eval_plans import generate, score
from blocksworld import parse_plan  # eval_plans 已经把 blocksworld/ 加进了 import 路径

SEED = 0
MAX_NEW_TOKENS = 160  # 和评测一样
GREEDY_BATCH = 2  # 和 train_full.py 一样，贪心结果才能逐字核对
PROMPTS_PER_BATCH = 2  # 采样时每批 2 道题 × K 次


def parse_args():
    parser = argparse.ArgumentParser(description="RL 之前的采样诊断")
    parser.add_argument("--run-dir", required=True, help="全量 SFT 的输出目录，里面有 checkpoint-XXXX/")
    parser.add_argument("--out", required=True, help="本次诊断的输出目录，必须不存在")
    parser.add_argument("--steps", type=int, nargs="+", default=[126, 252, 378], help="要诊断的 checkpoint 步数")
    parser.add_argument("--n", type=int, default=100, help="train 和 val 各取前几道题")
    parser.add_argument("--k", type=int, default=8, help="每道题采样几次")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="mps", help="mps 或 cpu（cpu 只用来在沙箱里做很小的冒烟测试）")
    return parser.parse_args()


@torch.no_grad()
def sample(model, tokenizer, rows, k, temperature):
    """每道题采样 k 次。返回 (每道题 k 条文字的列表, 没写出结束符的条数)。
    模型自带的 generation_config 是 temperature 0.7、top_p 0.8、top_k 20、repetition_penalty 1.1，都会把分布变窄。
    RL 采样要的是模型本来的分布，所以 top_p=1.0、top_k=0（都表示不截断）、repetition_penalty=1.0，只用温度控制随机程度。"""
    model.eval()
    tokenizer.padding_side = "left"
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    groups, no_eos = [], 0
    for i in range(0, len(rows), PROMPTS_PER_BATCH):
        chunk = rows[i:i + PROMPTS_PER_BATCH]
        prompts = [tokenizer.apply_chat_template(r["messages"][:1], tokenize=False, add_generation_prompt=True)
                   for r in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, do_sample=True, num_return_sequences=k, temperature=temperature, top_p=1.0,
                             top_k=0, repetition_penalty=1.0, max_new_tokens=MAX_NEW_TOKENS,
                             pad_token_id=tokenizer.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        no_eos += sum(not (eos_ids & set(ids.tolist())) for ids in new)
        texts = tokenizer.batch_decode(new, skip_special_tokens=True)
        # 输出按题排好：第 1 道题的 k 条，接着第 2 道题的 k 条
        groups += [texts[j:j + k] for j in range(0, len(texts), k)]
        del enc, out, new
        if model.device.type == "mps":
            torch.mps.empty_cache()
    return groups, no_eos


def summarize(rows, greedy_texts, groups, no_eos):
    """给每道题判分，返回 (汇总, 每道题的明细)。"""
    details = []
    for row, g_text, texts in zip(rows, greedy_texts, groups):
        scores = [score(row, t) for t in texts]
        optimal = [len(parse_plan(t)[0]) == row["plan_len"] for t, s in zip(texts, scores) if s["success"]]
        details.append({"id": row["id"], "n_blocks": row["n_blocks"], "goal_type": row["goal_type"],
                        "plan_len": row["plan_len"], "greedy": {**score(row, g_text), "text": g_text},
                        "wins": sum(s["success"] for s in scores), "optimal_wins": sum(optimal),
                        "distinct": len({t.strip() for t in texts}),
                        "samples": [{**s, "text": t} for s, t in zip(scores, texts)]})
    k = len(groups[0])
    wins = [d["wins"] for d in details]
    summary = {
        "problems": len(rows), "k": k,
        "greedy_success": sum(d["greedy"]["success"] for d in details),
        "sample_format": sum(s["format"] for d in details for s in d["samples"]),
        "sample_legal": sum(s["legal"] for d in details for s in d["samples"]),
        "sample_success": sum(wins),
        "pass_at_k": sum(w > 0 for w in wins),
        "all_correct": sum(w == k for w in wins),
        "all_wrong": sum(w == 0 for w in wins),
        "mixed": sum(0 < w < k for w in wins),
        "mean_distinct": sum(d["distinct"] for d in details) / len(details),
        "optimal_wins": sum(d["optimal_wins"] for d in details),
        "no_eos": no_eos,
    }
    return summary, details


def report(step, split, s, same_as_full_run=None):
    total = s["problems"] * s["k"]

    def pct(x):
        return f"{x / total:.1%}"

    lines = [f"step {step} {split}（{s['problems']} 道，每道采样 {s['k']} 次）",
             f"  贪心成功 {s['greedy_success']}/{s['problems']}"
             + (f"；和全量训练时的评测逐字相同 {same_as_full_run}/{s['problems']}" if same_as_full_run is not None else ""),
             f"  采样：格式 {pct(s['sample_format'])}  合法 {pct(s['sample_legal'])}  成功 {pct(s['sample_success'])}"
             f"  pass@{s['k']} {s['pass_at_k']}/{s['problems']}",
             f"  全对 {s['all_correct']}  全错 {s['all_wrong']}  有对有错 {s['mixed']}",
             f"  平均每道 {s['mean_distinct']:.1f} 种不同答案；成功答案里最短的占 "
             f"{s['optimal_wins'] / max(s['sample_success'], 1):.0%}；没写结束符 {s['no_eos']}/{total}"]
    print("\n".join(lines), flush=True)


def main():
    args = parse_args()
    run_dir, out = (REPO / args.run_dir).resolve(), (REPO / args.out).resolve()
    if out.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{out}")
    checkpoints = {step: run_dir / f"checkpoint-{step:04d}" for step in args.steps}
    missing = [str(p) for p in checkpoints.values() if not (p / "adapter_model.safetensors").exists()]
    if missing:
        raise SystemExit(f"找不到 checkpoint：{missing}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS 不可用；为避免意外在 CPU 上跑很久，已停止。")

    # 全量训练最后选中的 checkpoint 在 val 上有逐题的贪心结果，拿来核对这里加载和生成的方式没变
    full_run = json.loads((run_dir / "results.json").read_text())
    full_run_texts = {d["id"]: d["text"] for d in json.loads((run_dir / "selected_eval.json").read_text())["details"]}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    splits = {"train": load_jsonl("train", args.n), "val": load_jsonl("val", args.n)}
    out.mkdir(parents=True)
    config = {"run_dir": str(run_dir), "checkpoints": {s: str(p) for s, p in checkpoints.items()},
              "model_path": MODEL_PATH, "n": args.n, "k": args.k, "seed": SEED, "device": args.device,
              "greedy": {"batch_size": GREEDY_BATCH, "max_new_tokens": MAX_NEW_TOKENS},
              "sampling": {"temperature": args.temperature, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                           "prompts_per_batch": PROMPTS_PER_BATCH, "max_new_tokens": MAX_NEW_TOKENS},
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                           "peft": peft.__version__}}
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    results = {}
    for step, path in checkpoints.items():
        t0 = time.time()
        base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, local_files_only=True)
        model = PeftModel.from_pretrained(base.to(args.device), path).to(args.device)
        details_all = {}
        for split, rows in splits.items():
            greedy_texts = generate(model, tokenizer, rows, batch_size=GREEDY_BATCH, max_new_tokens=MAX_NEW_TOKENS)
            torch.manual_seed(SEED)  # 每个 checkpoint 的每份题都从同一个随机种子开始采样
            groups, no_eos = sample(model, tokenizer, rows, args.k, args.temperature)
            summary, details = summarize(rows, greedy_texts, groups, no_eos)
            same = None
            if split == "val" and step == full_run["selected"]["step"]:
                same = sum(full_run_texts[r["id"]] == t for r, t in zip(rows, greedy_texts))
                summary["greedy_same_as_full_run"] = same
            results[f"step{step}_{split}"] = summary
            details_all[split] = details
            report(step, split, summary, same)
        (out / f"details-step{step:04d}.json").write_text(json.dumps(details_all, ensure_ascii=False, indent=1) + "\n")
        # 每做完一个 checkpoint 就写一次汇总，中途停下也留得下已完成的部分
        (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(f"step {step} 用时 {time.time() - t0:.0f} 秒\n", flush=True)
        del model, base
        if args.device == "mps":
            torch.mps.empty_cache()
    print("写出", out)


if __name__ == "__main__":
    main()
