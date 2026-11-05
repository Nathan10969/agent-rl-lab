"""
第一版 GRPO：从全量 SFT 的 step 378 出发，用模拟器的 0/1 成功奖励，接着训练同一份 LoRA。
目的是把“采样 → 打分 → 组内优势 → 反向传播 → 保存”串起来，每一步都看得见。
10–20 步的结果只用来检查流程，不能用来判断 GRPO 有没有效果。

设定（2026-11-01 和 Codex 对齐）：
  起点     outputs/sft/full-20261029-codex-01/checkpoint-0378 的 adapter
  奖励     动作全部合法且最后达成目标 = 1，否则 = 0；不加长度奖励（grpo_core.reward）
  采样     温度 1.0、top_p 1.0、top_k 0、repetition_penalty 1.0、最多 160 个新 token，和采样诊断相同
  每步     4 道 train 题，每道 G = 8 个答案，共 32 个；用这 32 个答案更新一次参数
  优势     组内 (奖励 − 平均) / (总体标准差 + 1e-4)；全对或全错的组优势全 0，不补采（补采是 DAPO 的做法，留到 2b）
  loss     ratio + clip(ε = 0.2)，答案内 token 平均、再对答案平均。每批只更新一次，所以 ratio 恒为 1，clip 不会触发
  KL       不加（β = 0），所以不需要另放一份参考模型
  dropout  关掉（model.eval()）：算 loss 用的概率分布要和采样时是同一个
  优化器   AdamW，lr 1e-5（常见量级，没有调过），weight decay 0，梯度范数裁剪到 1.0

只用 train 题。val、test、6–7 块泛化题、PlanBench 都不碰。
--smoke：2 步 × 每步 2 道题，另外检查 (a) 采样时和训练时的 log 概率对得上 (b) 保存再加载后输出一致。
原模型参数没变、可训练的只有 LoRA，这两项每次运行都检查。

运行（要在 Claude Code 外面的终端里，沙箱里用不了 MPS）：
  .venv/bin/python -u grpo/train_grpo.py --run-dir outputs/grpo/smoke-01 --smoke
  .venv/bin/python -u grpo/train_grpo.py --run-dir outputs/grpo/trial-01 --steps 20
"""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sft"))

START = REPO / "outputs" / "sft" / "full-20261029-codex-01" / "checkpoint-0378"
GROUP = 8  # G：每道题采样几个答案
PROMPTS_PER_STEP = 4
GEN_PROMPTS_PER_CALL = 2  # 一次 generate 放 2 道题 × 8 个答案，和采样诊断一样
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 160
PROMPT_PAD = 256  # train 题目 206–241 个 token，全部左边补齐到 256：形状固定，MPS 内存不乱涨
LR = 1e-5
CLIP_EPS = 0.2
MAX_GRAD_NORM = 1.0
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description="第一版 GRPO，拒绝覆盖已有运行目录")
    p.add_argument("--run-dir", required=True, help="必须不存在的本次输出目录")
    p.add_argument("--steps", type=int, default=20, help="更新几步（--smoke 时固定 2 步）")
    p.add_argument("--save-every", type=int, default=10, help="每几步存一次 adapter；最后一步总会存")
    p.add_argument("--resume-from", help="从完整训练存档继续；输出仍写入新的 --run-dir")
    p.add_argument("--stop-after", type=int, help="在指定 step 存档后停止；--steps 仍表示总目标步数")
    p.add_argument("--smoke", action="store_true", help="2 步 × 每步 2 道题，外加检查；不能当作训练结果")
    p.add_argument("--device", default="mps", help="mps；cpu 只允许和 --smoke 一起用，在沙箱里检查流程")
    return p.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_randomness(seed):
    """让独立进程从相同的 Python 和 PyTorch 随机状态开始。"""
    import torch
    random.seed(seed)
    torch.manual_seed(seed)


def tokenizer_fingerprints(model_dir, chat_template):
    """记录 tokenizer 本地资产及最终加载的模板；新增模板同样改变指纹。"""
    model_dir = Path(model_dir)
    extensions = {".json", ".txt", ".jinja", ".model", ".tiktoken", ".vocab", ".bpe"}
    files = sorted(p for p in model_dir.iterdir() if p.is_file() and p.suffix in extensions)
    templates = model_dir / "chat_templates"
    if templates.is_dir():
        files.extend(sorted(p for p in templates.rglob("*.jinja") if p.is_file()))
    result = {str(p.relative_to(model_dir)): sha256(p) for p in files}
    encoded = json.dumps(chat_template, ensure_ascii=False, sort_keys=True).encode("utf-8")
    result["loaded_chat_template"] = hashlib.sha256(encoded).hexdigest()
    return result


def adapter_fingerprints(path):
    """权重和配置都影响 LoRA 行为，必须一起核对。"""
    path = Path(path)
    return {name: sha256(path / name) for name in ("adapter_model.safetensors", "adapter_config.json")}


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{run_dir}")
    if args.resume_from and not (Path(args.resume_from) / "resume_manifest.json").is_file():
        raise SystemExit("缺少完整续训存档标记；旧 adapter 无法精确续训。")
    if args.device == "cpu" and not args.smoke:
        raise SystemExit("CPU 只用于 --smoke；正式试跑请用 mps。")
    steps = 2 if args.smoke else args.steps
    prompts_per_step = 2 if args.smoke else PROMPTS_PER_STEP
    if steps < 1 or args.save_every < 1:
        raise SystemExit("--steps 和 --save-every 必须大于 0。")
    if args.stop_after is not None and not 1 <= args.stop_after <= steps:
        raise SystemExit("--stop-after 必须在 1 到 --steps 之间。")
    stop_at = args.stop_after if args.stop_after is not None else steps

    # 检查完输出目录再导入模型相关的库，--help 和防覆盖检查不用等
    import peft
    import torch
    import torch.nn.functional as F
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from data import DATA_DIR, MODEL_PATH, load_jsonl
    from full_sft_utils import verify_file_hash
    from grpo_core import completion_mask, group_advantages, grpo_loss, policy_logprobs, reward, token_logprobs
    from resume_state import inspect_checkpoint, restore_training_state, save_checkpoint

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS 不可用；为避免意外在 CPU 上跑很久，已停止。")
    if not (START / "adapter_model.safetensors").exists():
        raise SystemExit(f"找不到起点 adapter：{START}")

    manifest = json.loads((REPO / "blocksworld" / "splits" / "v1" / "manifest.json").read_text())
    train_sha = verify_file_hash(DATA_DIR / "train.jsonl", manifest["splits"]["train"]["data_sha256"])
    train = load_jsonl("train")
    if len(train) != 3021:
        raise SystemExit("train 条数不是 3021，已停止。")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    longest = max(len(tokenizer.apply_chat_template(r["messages"][:1], add_generation_prompt=True)) for r in train)
    if longest > PROMPT_PAD:
        raise SystemExit(f"最长的题目 {longest} 个 token，超过 PROMPT_PAD={PROMPT_PAD}，已停止。")
    if steps * prompts_per_step > len(train):
        raise SystemExit("步数太多，train 题不够每步都用新题。")

    run_dir.mkdir(parents=True)
    log_path = run_dir / "run.log"

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def append_jsonl(name, obj):
        with (run_dir / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def free_cache():
        if device == "mps":
            torch.mps.empty_cache()

    def driver_gib():
        if device != "mps":
            return None
        torch.mps.synchronize()
        return round(torch.mps.driver_allocated_memory() / 1024 ** 3, 2)

    @torch.no_grad()
    def sample(rows, keep_logits=False):
        """每道题采 GROUP 个答案，按题排好：第 1 题的 8 个，接着第 2 题的 8 个……
        返回 prompt_ids / prompt_mask [N, 256]、comp_ids [N, 160]；keep_logits 时还有采样那一刻每个 token 的 log 概率。"""
        model.eval()
        tokenizer.padding_side = "left"
        parts = {"prompt_ids": [], "prompt_mask": [], "comp_ids": [], "sampled_logp": []}
        for i in range(0, len(rows), GEN_PROMPTS_PER_CALL):
            chunk = rows[i:i + GEN_PROMPTS_PER_CALL]
            prompts = [tokenizer.apply_chat_template(r["messages"][:1], tokenize=False, add_generation_prompt=True)
                       for r in chunk]
            enc = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=PROMPT_PAD).to(device)
            out = model.generate(**enc, do_sample=True, num_return_sequences=GROUP, temperature=TEMPERATURE,
                                 top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=MAX_NEW_TOKENS,
                                 pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True,
                                 output_logits=keep_logits)
            new = out.sequences[:, PROMPT_PAD:]  # 只要新写出来的部分；所有答案都停了就提前结束，可能不到 160
            room = MAX_NEW_TOKENS - new.shape[1]
            if keep_logits:  # output_logits 是没经过任何处理的原始输出；温度 1、不截断时采样就是按它的 softmax
                parts["sampled_logp"].append(F.pad(token_logprobs(torch.stack(out.logits, dim=1), new, TEMPERATURE),
                                                   (0, room)))
            parts["comp_ids"].append(F.pad(new, (0, room), value=tokenizer.pad_token_id))
            parts["prompt_ids"].append(enc["input_ids"].repeat_interleave(GROUP, dim=0))
            parts["prompt_mask"].append(enc["attention_mask"].repeat_interleave(GROUP, dim=0))
            del enc, out, new
            free_cache()
        return {k: torch.cat(v) for k, v in parts.items() if v}

    def update(batch, adv, mask, n_groups):
        """有学习信号的组逐组反向传播（一次放一组 8 个答案），梯度累加起来，再更新一次参数。
        返回 (loss, 裁剪前的梯度范数, 反向传播了几组)。"""
        model.eval()  # 关掉 dropout；eval 只改 dropout 的行为，梯度照样能算
        loss_sum, used = 0.0, 0
        for g in range(n_groups):
            sl = slice(g * GROUP, (g + 1) * GROUP)
            if not adv[sl].any():
                continue  # 全对或全错：优势全 0，乘出来的梯度就是 0，不用算
            logp = policy_logprobs(model, batch["prompt_ids"][sl], batch["prompt_mask"][sl], batch["comp_ids"][sl],
                                   TEMPERATURE)
            loss = grpo_loss(logp, logp.detach(), adv[sl], mask[sl], n_groups * GROUP, CLIP_EPS)
            loss.backward()
            loss_sum, used = loss_sum + loss.item(), used + 1
            del logp, loss
            free_cache()
        if not used:
            return 0.0, 0.0, 0  # 这一步没有任何一组有对有错，不更新
        grad_norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), MAX_GRAD_NORM).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss_sum, grad_norm, used

    @torch.no_grad()
    def all_logprobs(m, batch):
        m.eval()
        return torch.cat([policy_logprobs(m, batch["prompt_ids"][i:i + GROUP], batch["prompt_mask"][i:i + GROUP],
                                          batch["comp_ids"][i:i + GROUP], TEMPERATURE)
                          for i in range(0, len(batch["comp_ids"]), GROUP)])

    @torch.no_grad()
    def distance(ref):
        return math.sqrt(sum(((p - ref[n]) ** 2).sum().item() for n, p in trainable.items()))

    try:
        code = [Path(__file__), REPO / "grpo" / "grpo_core.py", REPO / "grpo" / "resume_state.py", REPO / "sft" / "eval_plans.py",
                REPO / "sft" / "data.py", REPO / "blocksworld" / "blocksworld.py"]
        git = lambda *a: subprocess.run(["git", *a], cwd=REPO, text=True, capture_output=True).stdout.strip()  # noqa: E731
        config = {"mode": "smoke" if args.smoke else "trial", "device": device, "dtype": "float32",
                  "model_path": MODEL_PATH, "model_sha256": sha256(Path(MODEL_PATH) / "model.safetensors"),
                  "start_adapter": str(START), "start_adapter_sha256": sha256(START / "adapter_model.safetensors"),
                  "train_sha256": train_sha, "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in code},
                  "git_head": git("rev-parse", "HEAD"), "git_worktree_dirty": bool(git("status", "--porcelain")),
                  "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                               "peft": peft.__version__},
                  "steps": steps, "prompts_per_step": prompts_per_step, "group_size": GROUP,
                  "reward": "1 if every action legal and goal reached else 0; no length term",
                  "advantage": "(r - group mean) / (population std + 1e-4)",
                  "loss": "clipped ratio, mean over tokens per answer then over answers; one update per batch",
                  "clip_eps": CLIP_EPS, "kl_beta": 0.0, "dropout_during_loss": False,
                  "sampling": {"temperature": TEMPERATURE, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                               "max_new_tokens": MAX_NEW_TOKENS, "prompt_pad": PROMPT_PAD,
                               "prompts_per_generate_call": GEN_PROMPTS_PER_CALL},
                  "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": 0.0, "max_grad_norm": MAX_GRAD_NORM},
                  "seed": SEED, "save_every": args.save_every}
        # 只保存影响训练轨迹的值；运行目录和人为停止点不属于模型配置。
        compatibility = {
            "model_sha256": config["model_sha256"], "start_adapter_sha256": config["start_adapter_sha256"],
            "start_adapter_files": adapter_fingerprints(START),
            "train_sha256": train_sha, "code_sha256": config["code_sha256"],
            "tokenizer_sha256": tokenizer_fingerprints(MODEL_PATH, tokenizer.chat_template),
            "versions": {**config["versions"], "python": list(sys.version_info[:3])},
            "device": device, "dtype": "float32", "steps": steps, "mode": config["mode"],
            "prompts_per_step": prompts_per_step, "group_size": GROUP,
            "sampling": config["sampling"], "optimizer": config["optimizer"],
            "advantage": config["advantage"], "loss": config["loss"],
            "clip_eps": CLIP_EPS, "kl_beta": 0.0, "seed": SEED, "save_every": args.save_every,
        }
        resume_state = inspect_checkpoint(args.resume_from, compatibility) if args.resume_from else None
        if resume_state:
            completed = resume_state["completed_step"]
            order_saved = resume_state["order"]
            if not isinstance(completed, int) or not 0 < completed < steps or completed >= stop_at:
                raise ValueError("续训 step 必须在保存的步骤之后且不超过总目标。")
            if len(order_saved) != len(train) or sorted(order_saved) != list(range(len(train))):
                raise ValueError("续训题目顺序无效。")
        config["resume_from"] = str(Path(args.resume_from).resolve()) if args.resume_from else None
        config["stop_after"] = stop_at
        (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        log(f"运行目录 {run_dir}；模式 {config['mode']}；{steps} 步 × 每步 {prompts_per_step} 道题 × {GROUP} 个答案")

        initialize_randomness(SEED)
        order = list(range(len(train)))
        random.Random(SEED).shuffle(order)
        base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, local_files_only=True).to(device)
        # is_trainable=True：不加的话 adapter 按“只推理”加载，LoRA 参数没有梯度，训练会静悄悄地什么都不改
        adapter_source = Path(args.resume_from) if args.resume_from else START
        model = PeftModel.from_pretrained(base, adapter_source, is_trainable=True).to(device)
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not trainable or not all("lora_" in n for n in trainable):
            raise SystemExit("可训练参数里有不是 LoRA 的，已停止。")
        n_lora = sum(p.numel() for p in trainable.values())
        n_all = sum(p.numel() for p in model.parameters())
        log(f"可训练：{len(trainable)} 个 LoRA 矩阵，{n_lora:,} 个数，占全部 {n_all:,} 的 {n_lora / n_all:.2%}；"
            f"driver {driver_gib()} GiB")
        frozen_probe = {n: p.detach().clone() for n, p in model.named_parameters()
                        if not p.requires_grad and n.endswith(("layers.0.self_attn.q_proj.base_layer.weight",
                                                               "layers.23.mlp.down_proj.base_layer.weight"))}
        start_params = {n: p.detach().clone() for n, p in trainable.items()}
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=LR, weight_decay=0.0)
        eos_ids = set(model.generation_config.eos_token_id)  # <|im_end|> 和 <|endoftext|>（补齐也用它）
        checks, history = {}, []

        first_step = 1
        if resume_state:
            order = resume_state["order"]
            first_step = resume_state["completed_step"] + 1
            log(f"从 {args.resume_from} 恢复，下一步 {first_step}；写入独立运行目录。")
            restore_training_state(resume_state, optimizer, list(trainable), device)

        for step in range(first_step, stop_at + 1):
            rows = [train[j] for j in order[(step - 1) * prompts_per_step: step * prompts_per_step]]
            prev = {n: p.detach().clone() for n, p in trainable.items()}

            # 1 采样
            t0 = time.time()
            batch = sample(rows, keep_logits=args.smoke and step == 1)
            mask = completion_mask(batch["comp_ids"], eos_ids)
            eos_hit = torch.isin(batch["comp_ids"], torch.tensor(sorted(eos_ids), device=device)).any(dim=1)
            texts = tokenizer.batch_decode(batch["comp_ids"], skip_special_tokens=True)

            # 2 奖励  3 优势
            t1 = time.time()
            judged = [reward(rows[i // GROUP], t) for i, t in enumerate(texts)]
            rewards = [r for r, _ in judged]
            adv = torch.cat([group_advantages(rewards[g * GROUP:(g + 1) * GROUP]) for g in range(len(rows))]).to(device)
            t2 = time.time()

            if args.smoke and step == 1:  # 检查 (a)：同一批答案，采样时的 log 概率 vs 训练时重算的
                recomputed = all_logprobs(model, batch)
                gap = (recomputed - batch["sampled_logp"]).abs() * mask
                checks["logprob_gap"] = {"max": gap.max().item(), "mean": (gap.sum() / mask.sum()).item(),
                                         "pass": gap.max().item() < 1e-2}
                log(f"[检查 a] 采样时 vs 重算的 token log 概率：最大差 {checks['logprob_gap']['max']:.2e}，"
                    f"平均差 {checks['logprob_gap']['mean']:.2e}（错开一格时会差好几个单位）")
                n_show = int(mask[0].sum().item()) + 2
                log("  第 1 个答案逐 token：位置  token id  算不算  采样时 log 概率  重算 log 概率  token 文字")
                for j in range(min(n_show, MAX_NEW_TOKENS)):
                    tid = batch["comp_ids"][0, j].item()
                    log(f"  {j:4d} {tid:9d}   {'算' if mask[0, j] else '·'}   {batch['sampled_logp'][0, j].item():12.4f}"
                        f"   {recomputed[0, j].item():12.4f}   {tokenizer.decode([tid])!r}")
                del recomputed, gap

            # 4 反向传播 + 更新
            t_update = time.time()
            loss, grad_norm, used = update(batch, adv, mask, len(rows))
            t3 = time.time()

            wins = [sum(rewards[g * GROUP:(g + 1) * GROUP]) for g in range(len(rows))]
            n_tokens = mask.sum(dim=1)
            record = {
                "step": step, "prompt_ids": [r["id"] for r in rows],
                "success": sum(rewards), "answers": len(rewards),
                "format": sum(s["format"] for _, s in judged), "legal": sum(s["legal"] for _, s in judged),
                "groups": {"mixed": sum(0 < w < GROUP for w in wins), "all_correct": sum(w == GROUP for w in wins),
                           "all_wrong": sum(w == 0 for w in wins)},
                "groups_backpropagated": used,
                "tokens_per_answer": {"mean": n_tokens.mean().item(), "max": int(n_tokens.max().item())},
                "no_eos": int((~eos_hit).sum().item()),
                "loss": loss, "grad_norm_before_clip": grad_norm,
                "lora_change_this_step": distance(prev), "lora_drift_from_start": distance(start_params),
                "lora_tensors_changed": sum(not torch.equal(p, prev[n]) for n, p in trainable.items()),
                "seconds": {"sample": round(t1 - t0, 1), "reward": round(t2 - t1, 1), "update": round(t3 - t_update, 1)},
                "driver_gib": driver_gib(),
            }
            history.append(record)
            append_jsonl("steps.jsonl", record)
            for i, (t, (r, s)) in enumerate(zip(texts, judged)):
                append_jsonl("samples.jsonl", {"step": step, "id": rows[i // GROUP]["id"], "k": i % GROUP,
                                               "reward": r, "advantage": adv[i].item(), **s,
                                               "n_tokens": int(n_tokens[i].item()), "eos_hit": bool(eos_hit[i]),
                                               "token_ids": batch["comp_ids"][i, :int(n_tokens[i].item())].tolist(),
                                               "text": t})
            g = record["groups"]
            log(f"step {step}/{steps}  成功 {record['success']:.0f}/{record['answers']}  "
                f"有对有错 {g['mixed']} 组 / 全对 {g['all_correct']} / 全错 {g['all_wrong']}  "
                f"loss {loss:+.2e}  梯度范数 {grad_norm:.3f}  LoRA 本步变化 {record['lora_change_this_step']:.2e}"
                f"（{record['lora_tensors_changed']}/{len(trainable)} 个矩阵变了）  "
                f"用时 采样 {record['seconds']['sample']}s 更新 {record['seconds']['update']}s  "
                f"driver {record['driver_gib']} GiB")
            for gi, row in enumerate(rows):
                sl = slice(gi * GROUP, (gi + 1) * GROUP)
                bits = "".join(str(int(r)) for r in rewards[sl])
                values = sorted({round(a, 3) for a in adv[sl].tolist()}, reverse=True)
                log(f"    {row['id']}  奖励 {bits}  优势 {' / '.join(f'{a:+.3f}' for a in values)}")

            if step % args.save_every == 0 or step == stop_at:
                checkpoint = run_dir / f"checkpoint-{step:04d}"
                save_checkpoint(checkpoint, model, optimizer, step, order, list(trainable), compatibility, device)
                log(f"step {step} 完整训练存档已保存")
            free_cache()

        # 原模型参数一个数都不能变
        checks["frozen_unchanged"] = all(torch.equal(p, frozen_probe[n]) for n, p in model.named_parameters()
                                         if n in frozen_probe) and len(frozen_probe) == 2
        log(f"[检查] 抽查的 2 个原模型矩阵没有变：{checks['frozen_unchanged']}")

        if args.smoke:  # 检查 (b)：把刚存的 adapter 装到一份新读入的原模型上，对最后一批答案算 log 概率
            first_group = {k: v[:GROUP] for k, v in batch.items() if k != "sampled_logp"}
            before = all_logprobs(model, first_group)
            fresh = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, local_files_only=True)
            reloaded = PeftModel.from_pretrained(fresh.to(device), run_dir / f"checkpoint-{stop_at:04d}").to(device)
            diff = ((all_logprobs(reloaded, first_group) - before).abs() * mask[:GROUP]).max().item()
            checks["reload_max_diff"] = diff
            log(f"[检查 b] 保存再加载后 log 概率最大差 {diff:.2e}")

        result = {"config": config, "checks": checks, "history": history,
                  "segment_first_step": first_step, "segment_last_step": stop_at,
                  "target_steps": steps, "stopped_early": stop_at < steps,
                  "total_success": sum(h["success"] for h in history),
                  "total_answers": sum(h["answers"] for h in history),
                  "mixed_groups": sum(h["groups"]["mixed"] for h in history),
                  "total_groups": len(history) * prompts_per_step}
        (run_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        log(f"本段完成：step {first_step}–{stop_at} 共采 {result['total_answers']} 个答案，成功 {result['total_success']:.0f}；"
            f"有对有错 {result['mixed_groups']}/{result['total_groups']} 组。写出 {run_dir}")
    except BaseException:
        log("运行未完成：\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
