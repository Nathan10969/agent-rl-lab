"""Check the official ALFWorld text environment without loading a model."""
import argparse
import json
import os
from pathlib import Path


def select_alfworld_games(games, count=3):
    """Choose the first sorted game from each distinct ALFWorld task family."""
    if count < 1:
        raise ValueError("count must be positive")
    chosen, families = [], set()
    for game in sorted(games, key=str):
        family = Path(game).parent.parent.name.split("-", 1)[0]
        if family not in families:
            chosen.append(game)
            families.add(family)
            if len(chosen) == count:
                return chosen
    raise ValueError(f"fewer than {count} distinct ALFWorld task families")


def build_alfworld_config(data_dir, max_steps):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    return {"env": {"goal_desc_human_anns_prob": 0.0, "task_types": [1, 2, 3, 4, 5, 6],
                    "domain_randomization": False, "expert_type": "handcoded"},
            "dataset": {"eval_id_data_path": str(Path(data_dir) / "json_2.1.1" / "valid_seen"),
                        "num_eval_games": 0},
            "general": {"training_method": "dagger"},
            "dagger": {"training": {"max_nb_steps_per_episode": max_steps}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["ALFWORLD_DATA"] = str(args.data_dir.resolve())
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    world = AlfredTWEnv(build_alfworld_config(args.data_dir, 20), train_eval="eval_in_distribution")
    games = sorted(world.game_files)
    if not games:
        raise RuntimeError("no solvable validation games were found")
    world.game_files = games[:1]
    env = world.init_env(batch_size=1)
    try:
        observations, _ = env.reset()
        feedback, scores, dones, info = env.step(["look"])
        if "won" not in info:
            raise RuntimeError("environment did not report whether the task was won")
        print(json.dumps({"available_games": len(games), "initial_observation": observations[0],
                          "feedback": feedback[0], "score": float(scores[0]), "done": bool(dones[0]),
                          "won": bool(info["won"][0])}, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
