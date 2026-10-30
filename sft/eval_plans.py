"""
让模型为一批题生成计划，再用模拟器判分。三个层次分开记：
  format   格式正确：每一行都是看得懂的动作
  legal    每步合法：格式正确，并且按顺序执行没有非法动作（但不一定达成目标）
  success  任务成功：每步合法，并且最后达成目标
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "blocksworld"))
from blocksworld import make_state, parse_plan, simulate  # noqa: E402


def score(sample, text):
    plan, _ = parse_plan(text)
    if plan is None:
        return {"format": False, "legal": False, "success": False}
    result = simulate(make_state(sample["init"]), plan, [tuple(f) for f in sample["goal"]])
    return {"format": True, "legal": result["steps_ok"] == len(plan), "success": result["success"]}


@torch.no_grad()
def generate(model, tokenizer, samples, batch_size=16, max_new_tokens=160):
    """贪心解码：每一步都取概率最大的 token，同样的输入得到同样的输出。
    模型自带的 generation_config 默认随机采样，还有 repetition_penalty=1.1（压低已出现过的 token）。
    动作序列里 "stack(" 这类片段本来就要反复出现，所以这里显式关掉采样，并把 repetition_penalty 设回 1.0。"""
    model.eval()
    tokenizer.padding_side = "left"  # 批量生成时在左边补齐，每道题的提示才都结束在同一列，接着往右生成
    texts = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        prompts = [tokenizer.apply_chat_template(s["messages"][:1], tokenize=False, add_generation_prompt=True)
                   for s in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id)
        texts += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    if model.device.type == "mps":
        torch.mps.empty_cache()  # 生成时每批长度都不一样，用完把 MPS 缓存还给系统（见 sft/mem_probe.py）
    return texts


def evaluate(model, tokenizer, samples):
    """返回 (汇总, 每道题生成的文字)。汇总里是三个层次各有几道题达到，以及总题数。"""
    texts = generate(model, tokenizer, samples)
    scores = [score(s, t) for s, t in zip(samples, texts)]
    summary = {k: sum(sc[k] for sc in scores) for k in ("format", "legal", "success")}
    summary["total"] = len(samples)
    return summary, texts
