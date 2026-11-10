"""积木任务的 DAPO 教学训练入口；不覆盖旧实验，也不自动运行长训练。"""
import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sft"))

from dapo_core import collect_groups, soft_length_penalty  # noqa: E402
from dapo_update import optimize_batch  # noqa: E402
from grpo_core import completion_mask, group_advantages, policy_logprobs, reward, token_logprobs  # noqa: E402


def save_adapter_step(model, run_dir, step):
    """每完成一批保存 adapter；先写临时目录，已有存档绝不主动覆盖。"""
    run_dir = Path(run_dir)
    destination = run_dir / f"checkpoint-{step:04d}"
    if destination.exists():
        raise FileExistsError(f"存档已存在：{destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step:04d}.partial-", dir=run_dir))
    model.save_pretrained(stage)
    if destination.exists():
        raise FileExistsError(f"存档已存在：{destination}")
    stage.rename(destination)
    return destination


def check_sampling_logp(sampled, recomputed, mask, tolerance=1e-2):
    """仅对生成的有效 token 比较采样时与重算的 log 概率。"""
    if sampled.shape != recomputed.shape or sampled.shape != mask.shape or not mask.bool().any():
        raise ValueError("采样概率检查形状不一致或没有有效 token")
    gap = (sampled - recomputed).abs()[mask.bool()].max().item()
    if not torch.isfinite(torch.tensor(gap)) or gap > tolerance:
        raise ValueError(f"采样和重算 log 概率不一致：最大差 {gap}")
    return gap


def prepare_group(row, prompt_ids, prompt_mask, comp_ids, texts, eos_ids, score_fn,
                  max_length, buffer):
    """分开记录原始成功、长度惩罚和截断 mask；优势用带惩罚的组内奖励。"""
    n = len(texts)
    if n < 2 or comp_ids.ndim != 2 or comp_ids.shape[0] != n or prompt_ids.shape[0] != n:
        raise ValueError("题组的提示、答案和文本数量不一致")
    if prompt_mask.shape != prompt_ids.shape or comp_ids.shape[1] != max_length:
        raise ValueError("提示 mask 或答案长度不一致")
    eos_tensor = torch.tensor(sorted(eos_ids), device=comp_ids.device)
    eos_hit = torch.isin(comp_ids, eos_tensor).any(dim=1)
    token_mask = completion_mask(comp_ids, eos_ids)
    lengths = token_mask.sum(dim=1).to(torch.int64).tolist()
    judged = [score_fn(row, text) for text in texts]
    success = [int(result) for result, _ in judged]
    if any(result not in (0, 1) for result, _ in judged):
        raise ValueError("模拟器必须返回 0/1 的任务成功奖励")
    penalties = [soft_length_penalty(int(length), max_length, buffer) for length in lengths]
    shaped = [float(result) + penalty for result, penalty in zip(success, penalties)]
    # 没有结束符表示触及生成上限：不对不完整答案的 token 反向传播。
    loss_mask = token_mask * eos_hit.unsqueeze(1).to(token_mask.dtype)
    return {"id": row["id"], "success": success, "details": [detail for _, detail in judged],
            "texts": texts, "lengths": lengths, "truncated": (~eos_hit).tolist(),
            "penalties": penalties, "shaped_rewards": shaped,
            "advantage": group_advantages(shaped).to(comp_ids.device),
            "prompt_ids": prompt_ids, "prompt_mask": prompt_mask, "comp_ids": comp_ids,
            "mask": loss_mask}


def parse_args():
    parser = argparse.ArgumentParser(description="积木 DAPO 教学实现，输出目录必须是新目录")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=1, help="采样并更新多少批，默认 1")
    parser.add_argument("--target-groups", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=20, help="每批最多尝试多少道题")
    parser.add_argument("--updates", type=int, default=2, help="同一批答案更新几次")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--length-buffer", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--check-sampling-logp", action="store_true", help="保存生成 logits 并检查采样概率与重算对齐")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{run_dir}")
    if (args.steps < 1 or args.target_groups < 1 or args.group_size < 2 or args.updates < 1
            or args.max_attempts < args.target_groups or args.lr <= 0
            or not 0 < args.length_buffer <= args.max_new_tokens):
        raise SystemExit("参数无效：检查步数、组数、采样预算、学习率与长度区间")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS 不可用，已停止")

    import peft
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from data import DATA_DIR, MODEL_PATH, load_jsonl
    from full_sft_utils import verify_file_hash
    from train_grpo import START, PROMPT_PAD, sha256, tokenizer_fingerprints, adapter_fingerprints

    if not (START / "adapter_model.safetensors").exists():
        raise SystemExit(f"找不到 SFT 起点 adapter：{START}")
    manifest = json.loads((REPO / "blocksworld/splits/v1/manifest.json").read_text())
    train_sha = verify_file_hash(DATA_DIR / "train.jsonl", manifest["splits"]["train"]["data_sha256"])
    train = load_jsonl("train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    longest = max(len(tokenizer.apply_chat_template(row["messages"][:1], add_generation_prompt=True)) for row in train)
    if longest > PROMPT_PAD:
        raise SystemExit(f"题目超过固定 prompt 长度 {PROMPT_PAD}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    order = list(range(len(train)))
    random.shuffle(order)
    rows = iter(train[i] for i in order)
    run_dir.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), REPO / "grpo/dapo_core.py", REPO / "grpo/dapo_update.py",
                  REPO / "grpo/grpo_core.py", REPO / "grpo/train_grpo.py", REPO / "sft/data.py",
                  REPO / "sft/eval_plans.py", REPO / "sft/full_sft_utils.py", REPO / "blocksworld/blocksworld.py"]
    config = {"settings": vars(args), "train_sha256": train_sha,
              "model_sha256": sha256(Path(MODEL_PATH) / "model.safetensors"),
              "start_adapter_files": adapter_fingerprints(START),
              "tokenizer_sha256": tokenizer_fingerprints(MODEL_PATH, tokenizer.chat_template),
              "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in code_paths},
              "versions": {"python": sys.version, "torch": torch.__version__, "peft": peft.__version__,
                           "transformers": transformers.__version__},
              "algorithm": {"eps_low": 0.2, "eps_high": 0.28, "temperature": 1.0, "kl_beta": 0,
                            "max_grad_norm": 1.0, "dropout": "off"},
              "design": "原始 0/1 成功率筛组；加长度惩罚算优势；无 EOS 的答案不参与 loss；token 平均；无 KL",
              "note": "教学实现；每批及最终保存 adapter，不支持精确续训；不是 DAPO 论文实验复现"}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, local_files_only=True).to(args.device)
    model = PeftModel.from_pretrained(model, START, is_trainable=True)
    model.eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(tokenizer.pad_token_id)

    def candidates():
        for row in rows:
            prompt = tokenizer.apply_chat_template(row["messages"][:1], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=PROMPT_PAD).to(args.device)
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=True, num_return_sequences=args.group_size,
                                     temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                     max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                                     return_dict_in_generate=True, output_logits=args.check_sampling_logp)
            new = out.sequences[:, PROMPT_PAD:]
            comp = F.pad(new, (0, args.max_new_tokens - new.shape[1]), value=tokenizer.pad_token_id)
            prompt_ids = inputs["input_ids"].repeat_interleave(args.group_size, 0)
            prompt_mask = inputs["attention_mask"].repeat_interleave(args.group_size, 0)
            gap = None
            if args.check_sampling_logp:
                with torch.no_grad():
                    sampled = token_logprobs(torch.stack(out.logits, dim=1), new)
                    recomputed = policy_logprobs(model, prompt_ids, prompt_mask, comp)[:, :new.shape[1]]
                    gap = check_sampling_logp(sampled, recomputed, completion_mask(comp, eos_ids)[:, :new.shape[1]])
            texts = tokenizer.batch_decode(comp, skip_special_tokens=True)
            group = prepare_group(row, prompt_ids, prompt_mask, comp,
                                  texts, eos_ids, reward, args.max_new_tokens, args.length_buffer)
            record = {k: group[k] for k in ("id", "success", "details", "texts", "lengths", "truncated",
                                             "penalties", "shaped_rewards")}
            record["token_ids"] = [seq[:length].tolist() for seq, length in zip(comp, group["lengths"])]
            record["advantages"] = group["advantage"].tolist()
            record["trained_token_counts"] = group["mask"].sum(dim=1).to(torch.int64).tolist()
            record["sampling_logp_max_gap"] = gap
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            yield group

    candidate_stream = candidates()
    for step in range(1, args.steps + 1):
        started = time.monotonic()
        groups, stats = collect_groups(candidate_stream, args.target_groups, args.max_attempts)
        sampled = time.monotonic()
        if not groups:
            print(f"第 {step} 批预算耗尽或题库用完，没有更新：{stats}", flush=True)
            (run_dir / "incomplete.json").write_text(json.dumps({"step": step, **stats}, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        if sum(int(g["mask"].sum().item()) for g in groups) == 0:
            raise SystemExit("选中的答案全部截断，没有有效 token；已停止")
        history = optimize_batch(model, optimizer, groups, policy_logprobs, updates=args.updates)
        save_adapter_step(model, run_dir, step)
        record = {"step": step, "sampling": stats, "selected_ids": [g["id"] for g in groups], "updates": history,
                  "seconds": {"sampling": round(sampled - started, 3), "updates": round(time.monotonic() - sampled, 3)}}
        with (run_dir / "steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"第 {step} 批：{stats}；更新统计：{history}", flush=True)
        if args.device == "mps":
            torch.mps.empty_cache()
    model.save_pretrained(run_dir / "adapter-final")
    print(f"完成；adapter 已写入 {run_dir / 'adapter-final'}（不含完整续训状态）", flush=True)


if __name__ == "__main__":
    main()
