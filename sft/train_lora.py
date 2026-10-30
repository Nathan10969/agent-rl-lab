"""
Qwen2.5-0.5B-Instruct + LoRA 的小规模 SFT 试运行。目的是把流程跑通、每一步都看得见，不是追求成绩。

  1 数据     train 前 TRAIN_N 条用来训练；val 前 VAL_LOSS_N 条算 loss，前 GEN_N 条生成计划交给模拟器判分
             test、6–7 块泛化题、PlanBench 都不碰，也不拿来调参
  2 模型     只读加载本地的 0.5B 模型
  3 训练前   先测一次 val loss 和生成判分，作为“没微调”的对照
  4 LoRA     冻结原模型全部参数，只在注意力和 MLP 的每个线性层旁边加一对小矩阵来训练
  5 训练     AdamW，固定学习率，每 EVAL_EVERY 步算一次 val loss
  6 训练后   再测一次生成判分
  7 保存加载 只保存 LoRA 的小矩阵（adapter），再加载到一份新读入的原模型上，检查输出和保存前一致

运行：.venv/bin/python sft/train_lora.py
输出：outputs/sft/qwen0.5b-lora-trial/（被 Git 忽略）下的 adapter/ 和 results.json
"""
import hashlib
import json
import random
import time
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import IGNORE, MODEL_PATH, REPO, collate, encode, load_jsonl
from eval_plans import evaluate

# ---------- 试运行的设定 ----------
TRAIN_N = 512  # train 共 3,021 条，试运行只用前 512 条
BATCH_SIZE = 8  # 每一步用 8 条样本
EPOCHS = 1  # 512 ÷ 8 = 64 步
LR = 2e-4  # 只训练很小的 LoRA 矩阵，学习率可以比全量微调大；这是常见量级，不是调出来的
VAL_LOSS_N = 64
GEN_N = 32
EVAL_EVERY = 16
SEED = 0
# 每批都补到同样长度。train / val 最长 315 个 token。
# 不固定长度时，每批形状都不一样，MPS 用完的内存块对不上下一批，只能再要新的：
# 试运行涨到 77 GB 进了 swap；固定到 320 后不再涨，每步再清缓存只剩约 4 GB（sft/mem_probe.py 实测）
PAD_TO = 320
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
OUT = REPO / "outputs" / "sft" / "qwen0.5b-lora-trial"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def batches(encoded, pad_id, rng=None):
    """按 BATCH_SIZE 切成一批批；给了 rng 就先打乱顺序（训练用），不给就按原顺序（评测用）。"""
    order = list(range(len(encoded)))
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), BATCH_SIZE):
        yield collate([encoded[j] for j in order[i:i + BATCH_SIZE]], pad_id, PAD_TO)


def free_cache():
    """把 MPS 留着备用的内存还给系统。只是释放缓存，不影响计算结果。"""
    if DEVICE == "mps":
        torch.mps.empty_cache()


def driver_gb():
    """MPS 一共向系统要了多少内存（GB）；不是 MPS 时返回 None。"""
    return round(torch.mps.driver_allocated_memory() / 1024 ** 3, 2) if DEVICE == "mps" else None


def forward(model, batch):
    """前向计算。传入 labels 时，模型内部会把 labels 左移一位：用第 t 个位置的输出去预测第 t+1 个 token，
    再对所有 label 不是 -100 的位置求平均交叉熵，这就是 loss。"""
    input_ids, labels, attention_mask = (t.to(DEVICE) for t in batch)
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


@torch.no_grad()
def val_loss(model, encoded, pad_id):
    """val 上的平均 loss：把所有参与 loss 的 token 放在一起平均，而不是先每批平均、再把各批平均。"""
    model.eval()
    total, count = 0.0, 0
    for batch in batches(encoded, pad_id):
        n = (batch[1][:, 1:] != IGNORE).sum().item()  # 左移后参与 loss 的 token 数
        total += forward(model, batch).loss.item() * n
        count += n
    free_cache()
    return total / count


def load_base_model():
    # float32：在 Mac 的 MPS 上最稳；0.5B 参数 × 4 字节 ≈ 2 GB，内存足够
    return AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).to(DEVICE)


def main():
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------- 1 数据 ----------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    pad_id = tokenizer.pad_token_id
    train = [encode(s, tokenizer) for s in load_jsonl("train", TRAIN_N)]
    val_rows = load_jsonl("val", max(VAL_LOSS_N, GEN_N))
    val = [encode(s, tokenizer) for s in val_rows[:VAL_LOSS_N]]
    gen_rows = val_rows[:GEN_N]
    print(f"训练 {len(train)} 条，每步 {BATCH_SIZE} 条，共 {len(train) // BATCH_SIZE * EPOCHS} 步；"
          f"val 算 loss {len(val)} 条，生成判分 {len(gen_rows)} 条；设备 {DEVICE}")

    # ---------- 2 模型 ----------
    model = load_base_model()

    # ---------- 3 训练前的对照 ----------
    base_loss = val_loss(model, val, pad_id)
    base_gen, base_texts = evaluate(model, tokenizer, gen_rows)
    print(f"[没微调] val loss {base_loss:.4f}，生成判分 {base_gen}")

    # ---------- 4 LoRA ----------
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数 {trainable:,} / 全部 {total:,}（{trainable / total:.2%}）")
    # LoRA 的一对小矩阵里，B 一开始全是 0，所以刚加上时模型输出和原来一样，val loss 应该不变
    init_loss = val_loss(model, val, pad_id)
    print(f"[刚加上 LoRA] val loss {init_loss:.4f}（应该和没微调时一样）")

    # ---------- 5 训练 ----------
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    history = [{"step": 0, "val_loss": init_loss}]
    step = 0
    t_train = time.time()
    for _ in range(EPOCHS):
        for batch in batches(train, pad_id, rng):
            model.train()  # 训练模式：LoRA 的 dropout 生效
            loss = forward(model, batch).loss
            loss.backward()  # 算梯度：只有 LoRA 矩阵有梯度，原模型参数被冻结
            optimizer.step()  # 按梯度更新 LoRA 矩阵
            optimizer.zero_grad()  # 清掉这一步的梯度，免得加到下一步上
            step += 1
            record = {"step": step, "train_loss": loss.item()}
            del loss
            free_cache()
            if step % EVAL_EVERY == 0:
                record["val_loss"] = val_loss(model, val, pad_id)
            record["mps_driver_gb"] = driver_gb()
            history.append(record)
            print(f"step {step:3d}  train loss {record['train_loss']:.4f}"
                  + (f"  val loss {record['val_loss']:.4f}" if "val_loss" in record else "")
                  + (f"  MPS 内存 {record['mps_driver_gb']} GB" if record["mps_driver_gb"] is not None else ""))
    train_seconds = time.time() - t_train

    # ---------- 6 训练后 ----------
    sft_gen, sft_texts = evaluate(model, tokenizer, gen_rows)
    print(f"[SFT 后] 生成判分 {sft_gen}")

    # ---------- 7 保存和重新加载 ----------
    adapter_dir = OUT / "adapter"
    model.save_pretrained(adapter_dir)  # 只存 LoRA 矩阵和配置，不存原模型
    reloaded = PeftModel.from_pretrained(load_base_model(), adapter_dir).to(DEVICE)
    reload_gen, reload_texts = evaluate(reloaded, tokenizer, gen_rows)
    same_texts = sum(a == b for a, b in zip(sft_texts, reload_texts))
    with torch.no_grad():
        model.eval()
        reloaded.eval()
        batch = collate(val[:BATCH_SIZE], pad_id, PAD_TO)
        max_logit_diff = (forward(model, batch).logits - forward(reloaded, batch).logits).abs().max().item()
    print(f"[重新加载] 生成判分 {reload_gen}；{same_texts}/{len(gen_rows)} 条生成文字和保存前完全相同；"
          f"logits 最大差 {max_logit_diff:.2e}")

    # ---------- 记录 ----------
    weights = Path(MODEL_PATH) / "model.safetensors"
    results = {
        "model_path": MODEL_PATH,
        "model_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "device": DEVICE,
        "config": {"train_n": len(train), "batch_size": BATCH_SIZE, "epochs": EPOCHS, "steps": step, "lr": LR,
                   "val_loss_n": len(val), "gen_n": len(gen_rows), "seed": SEED, "lora": LORA,
                   "data": "data/blocksworld/v1（见 blocksworld/splits/v1/manifest.json）"},
        "trainable_params": trainable,
        "total_params": total,
        "base": {"val_loss": base_loss, "gen": base_gen},
        "lora_init_val_loss": init_loss,
        "history": history,
        "sft": {"val_loss": history[-1].get("val_loss"), "gen": sft_gen},
        "reload": {"gen": reload_gen, "same_texts": same_texts, "max_logit_diff": max_logit_diff},
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(time.time() - t_start, 1),
        "examples": [{"id": s["id"], "answer": s["messages"][1]["content"], "base": b, "sft": a}
                     for s, b, a in list(zip(gen_rows, base_texts, sft_texts))[:5]],
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("写出", OUT / "results.json")


if __name__ == "__main__":
    main()
