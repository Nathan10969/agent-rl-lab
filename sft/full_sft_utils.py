"""全量 SFT 中不依赖模型的安全小工具。"""
import hashlib
import math
from pathlib import Path


def batch_count(n, batch_size):
    if n < 0 or batch_size <= 0:
        raise ValueError("样本数不能为负，batch size 必须大于零")
    return (n + batch_size - 1) // batch_size


def create_run_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_file_hash(path, expected):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"文件指纹不匹配：{path}")
    return actual


def generation_stop(token_ids, eos_ids, max_new_tokens):
    stop = next((token for token in token_ids if token in eos_ids), None)
    return {"eos_hit": stop is not None, "stop_token_id": stop,
            "reached_limit_without_eos": stop is None and len(token_ids) >= max_new_tokens}


def checkpoint_steps(total_steps, interval):
    if total_steps <= 0 or interval <= 0:
        raise ValueError("总步数和保存间隔必须大于零")
    result = list(range(interval, total_steps + 1, interval))
    if not result or result[-1] != total_steps:
        result.append(total_steps)
    return result


def pick_best_checkpoint(records):
    if not records or any(not math.isfinite(row["val_loss"]) for row in records):
        raise ValueError("checkpoint 列表不能为空，验证 loss 必须是有限数")
    return min(records, key=lambda row: (row["val_loss"], row["step"]))
