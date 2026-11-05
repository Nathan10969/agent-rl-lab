"""GRPO 步骤边界的完整训练状态存档；只加载可信的本地存档。"""
import hashlib
import json
import random
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open


FILES = ("adapter_model.safetensors", "adapter_config.json", "training_state.pt")
MANIFEST = "resume_manifest.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter(path):
    """发布前确认 adapter 配置与 safetensors 结构可读取。"""
    try:
        config = json.loads((path / "adapter_config.json").read_text())
        if not isinstance(config, dict) or not config:
            raise ValueError("空配置")
        with safe_open(str(path / "adapter_model.safetensors"), framework="pt", device="cpu") as tensors:
            keys = list(tensors.keys())
            if not keys:
                raise ValueError("没有参数")
            for key in keys:
                tensors.get_slice(key).get_shape()
    except Exception as exc:
        raise ValueError("adapter 配置或权重无法读取") from exc


def read_training_state(path):
    """只以受限方式读取可信本地存档，并检查所需字段。"""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        required = {"completed_step", "order", "parameter_names", "device", "optimizer",
                    "python_rng", "torch_cpu_rng", "torch_mps_rng"}
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError("字段缺失")
        if not isinstance(state["completed_step"], int) or state["completed_step"] < 1:
            raise ValueError("step 无效")
        if not isinstance(state["order"], list) or not isinstance(state["parameter_names"], list):
            raise ValueError("题目或参数顺序无效")
        if not isinstance(state["optimizer"], dict) or not {"state", "param_groups"}.issubset(state["optimizer"]):
            raise ValueError("优化器状态无效")
        if state["device"] not in {"cpu", "mps"} or not isinstance(state["torch_cpu_rng"], torch.Tensor):
            raise ValueError("设备或 CPU 随机状态无效")
        if state["device"] == "mps" and not isinstance(state["torch_mps_rng"], torch.Tensor):
            raise ValueError("MPS 随机状态无效")
        return state
    except Exception as exc:
        raise ValueError("完整训练状态无法读取或字段无效") from exc


def save_checkpoint(path, model, optimizer, completed_step, order, parameter_names, compatibility, device):
    """先写临时目录并校验，再发布；调用方须确保已完成更新和梯度清理。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"存档已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{path.name}.partial-", dir=path.parent))
    model.save_pretrained(stage)
    validate_adapter(stage)
    if device == "mps":
        torch.mps.synchronize()
    state = {
        "completed_step": completed_step,
        "order": list(order),
        "parameter_names": list(parameter_names),
        "device": device,
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
    }
    torch.save(state, stage / "training_state.pt")
    read_training_state(stage / "training_state.pt")
    fingerprints = {name: file_sha256(stage / name) for name in FILES}
    manifest = {"format_version": 1, "complete": True,
                "compatibility": compatibility, "files": fingerprints}
    (stage / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if path.exists():
        raise FileExistsError(f"存档已存在，拒绝覆盖：{path}")
    stage.rename(path)


def inspect_checkpoint(path, compatibility):
    """先核对存档、指纹、配置，再读取状态；不会修改模型或优化器。"""
    path = Path(path)
    marker = path / MANIFEST
    if not marker.is_file():
        raise ValueError(f"缺少完整续训存档标记；旧 adapter 无法精确续训：{path}")
    try:
        manifest = json.loads(marker.read_text())
    except (ValueError, OSError) as exc:
        raise ValueError(f"完整存档标记无法读取：{path}") from exc
    if manifest.get("format_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("续训存档不完整或格式版本不支持")
    if manifest.get("compatibility") != compatibility:
        raise ValueError("续训配置或环境与存档不一致")
    hashes = manifest.get("files", {})
    if set(hashes) != set(FILES):
        raise ValueError("续训存档不完整：文件清单不符")
    for name in FILES:
        file = path / name
        if not file.is_file() or file_sha256(file) != hashes[name]:
            raise ValueError(f"续训存档缺少文件或指纹不符：{name}")
    validate_adapter(path)
    return read_training_state(path / "training_state.pt")


def restore_training_state(state, optimizer, parameter_names, device):
    """在模型和优化器重建完毕后调用，随机数状态最后恢复。"""
    if state.get("parameter_names") != list(parameter_names):
        raise ValueError("续训参数名称或顺序不一致")
    if state.get("device") != device:
        raise ValueError("续训设备与存档不一致")
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if device == "mps":
        torch.mps.set_rng_state(state["torch_mps_rng"])
