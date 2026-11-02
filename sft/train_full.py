"""Blocksworld 全量 SFT：独立输出目录、分段存档、完整验证集对照。

只保存 adapter，不保存优化器和随机状态，因此 checkpoint 不能精确续训。
运行示例：.venv/bin/python -u sft/train_full.py --run-dir outputs/sft/full-run-001
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

from full_sft_utils import batch_count, checkpoint_steps, create_run_dir, generation_stop, pick_best_checkpoint, verify_file_hash


def parse_args():
    parser = argparse.ArgumentParser(description="Blocksworld 全量 SFT，拒绝覆盖已有运行目录")
    parser.add_argument("--run-dir", required=True, help="必须不存在的本次输出目录")
    parser.add_argument("--smoke", action="store_true", help="仅跑两步及两条验证，不能当作全量结果")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{run_dir}")

    # 在检查输出目录后才导入机器学习依赖，--help 和防覆盖检查不启动模型。
    import peft
    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from data import DATA_DIR, IGNORE, MODEL_PATH, REPO, collate, encode, load_jsonl
    from eval_plans import score
    from train_lora import LORA

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS 不可用；为避免意外启动漫长的 CPU 训练，已停止。")

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # 完全检查完成前不创建目录，也不加载模型。
    manifest = json.loads((REPO / "blocksworld" / "splits" / "v1" / "manifest.json").read_text())
    data_hashes = {name: verify_file_hash(DATA_DIR / f"{name}.jsonl", manifest["splits"][name]["data_sha256"])
                   for name in ("train", "val")}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    train_rows = load_jsonl("train", 16 if args.smoke else None)
    val_rows = load_jsonl("val", 2 if args.smoke else None)
    if (len(train_rows), len(val_rows)) != ((16, 2) if args.smoke else (3021, 378)):
        raise SystemExit("数据条数不符合预期，已停止。")
    train = [encode(row, tokenizer) for row in train_rows]
    val = [encode(row, tokenizer) for row in val_rows]
    max_length = max(len(ids) for ids, _ in train + val)
    if max_length > 320:
        raise SystemExit(f"样本长度 {max_length} 超过 320，未截断；已停止。")

    create_run_dir(run_dir)
    log_path = run_dir / "run.log"
    start_time = time.time()

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def write_json(name, obj):
        (run_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def free_cache():
        torch.mps.empty_cache()

    def driver_gib():
        torch.mps.synchronize()
        return round(torch.mps.driver_allocated_memory() / (1024 ** 3), 3)

    def val_loss(model):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), 8):
                ids, labels, mask = (t.to("mps") for t in collate(val[i:i + 8], tokenizer.pad_token_id, 320))
                n = (labels[:, 1:] != IGNORE).sum().item()
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                total += loss.item() * n
                count += n
                del ids, labels, mask, loss
                free_cache()
        if not count:
            raise ValueError("验证集没有参与 loss 的 token")
        return total / count

    def evaluate(model, rows, name):
        model.eval()
        tokenizer.padding_side = "left"
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        details = []
        for i in range(0, len(rows), 2):
            chunk = rows[i:i + 2]
            prompts = [tokenizer.apply_chat_template(row["messages"][:1], tokenize=False,
                                                     add_generation_prompt=True) for row in chunk]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("mps")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=160, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, repetition_penalty=1.0,
                                     pad_token_id=tokenizer.pad_token_id)
            generated = out[:, inputs["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, ids, text in zip(chunk, generated, texts):
                token_ids = ids.tolist()
                stopped = generation_stop(token_ids, eos_ids, 160)
                details.append({"id": row["id"], "text": text, **score(row, text),
                                **stopped})
            del inputs, out, generated
            free_cache()
            if (i + len(chunk)) % 40 == 0 or i + len(chunk) == len(rows):
                log(f"{name} 生成评测 {i + len(chunk)}/{len(rows)}")
        summary = {key: sum(row[key] for row in details) for key in
                   ("format", "legal", "success", "eos_hit", "reached_limit_without_eos")}
        summary["total"] = len(details)
        write_json(f"{name}_eval.json", {"summary": summary, "details": details})
        return summary

    try:
        seed, lr, interval = 0, 2e-4, (2 if args.smoke else 126)
        total_steps = batch_count(len(train), 8)
        points = checkpoint_steps(total_steps, interval)
        code_paths = [Path(__file__), REPO / "sft" / "full_sft_utils.py", REPO / "sft" / "data.py",
                      REPO / "sft" / "eval_plans.py", REPO / "sft" / "train_lora.py"]
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, text=True, capture_output=True)
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True)
        config = {"mode": "smoke" if args.smoke else "full", "model_path": MODEL_PATH,
                  "model_sha256": sha256(Path(MODEL_PATH) / "model.safetensors"),
                  "train_sha256": data_hashes["train"], "val_sha256": data_hashes["val"],
                  "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in code_paths},
                  "git_head": rev.stdout.strip(), "git_worktree_dirty": bool(status.stdout.strip()),
                  "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
                  "device": "mps", "dtype": "float32", "train_n": len(train), "val_n": len(val),
                  "batch_size": 8, "epochs": 1, "steps": total_steps, "checkpoint_steps": points,
                  "pad_to": 320, "max_observed_length": max_length, "lr": lr, "weight_decay": 0.0,
                  "seed": seed, "lora": LORA, "empty_cache_each_step": True,
                  "generation": {"batch_size": 2, "max_new_tokens": 160, "do_sample": False,
                                 "repetition_penalty": 1.0},
                  "mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
                  "mps_low_watermark_ratio": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
                  "checkpoint_resume": "adapter only; optimizer and RNG state not saved"}
        write_json("config.json", config)
        log(f"运行目录：{run_dir}；模式 {config['mode']}；训练 {len(train)} 条，共 {total_steps} 步；验证 {len(val)} 条")
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, local_files_only=True).to("mps")
        log(f"原模型加载完成；driver {driver_gib()} GiB")
        baseline_loss = val_loss(model)
        log(f"原模型 val loss {baseline_loss:.6f}；开始训练前完整生成评测")
        baseline = evaluate(model, val_rows, "baseline")
        log(f"原模型评测：{baseline}")
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
        order = list(range(len(train)))
        rng.shuffle(order)
        checkpoints, history = [], []
        train_start = time.time()
        for step, i in enumerate(range(0, len(order), 8), 1):
            model.train()
            batch = [train[j] for j in order[i:i + 8]]
            ids, labels, mask = (t.to("mps") for t in collate(batch, tokenizer.pad_token_id, 320))
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            record = {"step": step, "batch_size": len(batch), "train_loss": loss.item()}
            del ids, labels, mask, loss
            free_cache()
            record["driver_gib_after_step"] = driver_gib()
            history.append(record)
            log(f"step {step}/{total_steps} batch {record['batch_size']} loss {record['train_loss']:.4f} driver {record['driver_gib_after_step']:.2f} GiB")
            if step in points:
                checkpoint_dir = run_dir / f"checkpoint-{step:04d}"
                checkpoint_dir.mkdir(exist_ok=False)
                model.save_pretrained(checkpoint_dir)
                log(f"step {step} adapter 已保存；开始完整验证 loss")
                v = val_loss(model)
                if not math.isfinite(v):
                    raise ValueError(f"非有限验证 loss：{v}")
                checkpoints.append({"step": step, "val_loss": v, "adapter": str(checkpoint_dir)})
                write_json("progress.json", {"history": history, "checkpoints": checkpoints})
                log(f"step {step} val loss {v:.6f}")
        training_and_validation_seconds = round(time.time() - train_start, 1)
        best = pick_best_checkpoint(checkpoints)
        log(f"训练完成，选择 step {best['step']}（val loss {best['val_loss']:.6f}）；开始最终生成评测")
        # 释放训练模型后再加载所选 adapter，避免同时保留两份模型。
        del optimizer, model
        free_cache()
        base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, local_files_only=True).to("mps")
        model = PeftModel.from_pretrained(base, best["adapter"]).to("mps")
        final = evaluate(model, val_rows, "selected")
        result = {"config": config, "baseline_val_loss": baseline_loss, "baseline": baseline,
                  "checkpoints": checkpoints, "selected": best, "selected_eval": final,
                  "training_and_validation_seconds": training_and_validation_seconds,
                  "total_seconds": round(time.time() - start_time, 1)}
        write_json("results.json", result)
        log(f"完成；所选 checkpoint {best['step']}；验证评测 {final}")
    except BaseException:
        log("运行未完成：\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
