"""Evaluate a small fixed set against an already-running local model server."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

from alfworld_probe import build_alfworld_config, select_alfworld_games
from baseline_eval import request_action, run_blocksworld, save_json_new, select_block_cases, summarize
from baseline_eval import run_alfworld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["blocksworld", "alfworld"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.task == "blocksworld" and args.cases is None:
        parser.error("--cases is required for blocksworld")
    if args.task == "alfworld" and args.data_dir is None:
        parser.error("--data-dir is required for alfworld")
    data = args.cases.read_bytes() if args.cases else None
    cases = select_block_cases(json.loads(data)["cases"]) if data else None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_json_new(args.output_dir / "config.json", {
        "task": args.task, "model": args.model,
        "case_file_sha256": hashlib.sha256(data).hexdigest() if data else None,
        "case_ids": [case["source_id"] for case in cases] if cases else None, "max_steps": args.max_steps,
        "temperature": 0, "max_new_tokens": 48, "thinking": False,
    })
    started = time.monotonic()
    episodes = []
    try:
        policy = lambda messages: request_action(args.url, args.model, messages)
        if args.task == "blocksworld":
            for index, case in enumerate(cases):
                result = run_blocksworld(case, policy, max_steps=args.max_steps)
                episodes.append(result)
                save_json_new(args.output_dir / f"episode-{index:03d}.json", result)
                print(f"{index + 1}/{len(cases)} success={result['success']} steps={len(result['trace'])}", flush=True)
        else:
            os.environ["ALFWORLD_DATA"] = str(args.data_dir.resolve())
            random.seed(0)
            import numpy as np
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

            np.random.seed(0)
            world = AlfredTWEnv(build_alfworld_config(args.data_dir, args.max_steps),
                                train_eval="eval_in_distribution")
            games = select_alfworld_games(world.game_files, 3)
            for index, game in enumerate(games):
                world.game_files = [game]
                env = world.init_env(batch_size=1)
                try:
                    result = run_alfworld(env, policy, max_steps=args.max_steps)
                finally:
                    env.close()
                result["source_id"] = str(Path(game).relative_to(args.data_dir))
                episodes.append(result)
                save_json_new(args.output_dir / f"episode-{index:03d}.json", result)
                print(f"{index + 1}/{len(games)} success={result['success']} steps={len(result['trace'])}", flush=True)
        summary = summarize(episodes)
        summary["task"] = args.task
        summary["seconds"] = round(time.monotonic() - started, 3)
        save_json_new(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        save_json_new(args.output_dir / "failure.json", {"type": type(error).__name__, "message": str(error),
                                                         "completed_episodes": len(episodes)})
        raise


if __name__ == "__main__":
    main()
