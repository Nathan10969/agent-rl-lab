"""Load one quantized model and generate once in a bounded GPU allocation."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from baseline_eval import save_json_new
from gpu_monitor import GPUMonitor, identify_allocated_gpu, summarize_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_data = (args.model_path / "config.json").read_bytes()
    config = json.loads(config_data)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        uuid = identify_allocated_gpu()
        identify_error = None if uuid else "GPU UUID could not be identified"
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        uuid, identify_error = None, str(error)
    monitor = GPUMonitor(uuid) if uuid else None
    if monitor:
        monitor.start()
    start = time.monotonic()
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams

        llm = LLM(model=str(args.model_path), max_model_len=2048, max_num_seqs=1,
                  gpu_memory_utilization=0.8, enforce_eager=True, trust_remote_code=False,
                  language_model_only=True)
        load_seconds = time.monotonic() - start
        tokenizer = llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "请只用一句话解释什么是强化学习。"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        generation_start = time.monotonic()
        output = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=32))[0].outputs[0]
        text = output.text.strip()
        if not text:
            raise RuntimeError("model returned an empty response")
        result = {"status": "ok", "text": text, "generated_tokens": len(output.token_ids),
                  "load_seconds": round(load_seconds, 3),
                  "generation_seconds": round(time.monotonic() - generation_start, 3),
                  "config_sha256": hashlib.sha256(config_data).hexdigest(),
                  "model_type": config.get("model_type"),
                  "quantization": config.get("quantization_config", {}).get("quant_method"),
                  "max_model_len": 2048, "temperature": 0, "thinking": False, "text_only": True,
                  "torch_version": torch.__version__, "vllm_version": vllm.__version__}
    except Exception as error:
        result = {"status": "error", "error_type": type(error).__name__, "error": str(error),
                  "config_sha256": hashlib.sha256(config_data).hexdigest()}
        raise
    finally:
        if monitor:
            monitor.stop()
        if "result" in locals():
            result["gpu_uuid"] = uuid
            result["gpu_memory"] = summarize_memory(monitor.samples) if monitor else None
            result["gpu_monitor_errors"] = monitor.errors if monitor else [identify_error]
            result["elapsed_seconds"] = round(time.monotonic() - start, 3)
            save_json_new(args.output_dir / "result.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
