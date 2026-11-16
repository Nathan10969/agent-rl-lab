"""Summarize complete, case-matched ALFWorld action-guide comparisons."""
import argparse
import json
from pathlib import Path


def compare_paired_results(manifest, base_dir, guide_dir):
    base_dir, guide_dir = Path(base_dir), Path(guide_dir)
    cases = manifest["cases"]
    if not cases or len({case["source_id"] for case in cases}) != len(cases):
        raise ValueError("manifest case IDs must be non-empty and unique")
    if any(len(list(folder.glob("episode-*.json"))) != len(cases) for folder in (base_dir, guide_dir)):
        raise ValueError("both arms must have one completed episode per manifest case")
    result = {"total": len(cases), "base_success": 0, "guide_success": 0,
              "improved": 0, "worsened": 0, "both_success": 0, "both_failed": 0,
              "base_steps": 0, "guide_steps": 0, "base_no_effect": 0, "guide_no_effect": 0,
              "initial_observation_matches": 0, "families": {},
              "subsets": {label: {"total": 0, "base_success": 0, "guide_success": 0,
                                  "improved": 0, "worsened": 0, "both_success": 0, "both_failed": 0}
                          for label in ("previous", "new")}}
    for index, case in enumerate(cases):
        name = f"episode-{index:03d}.json"
        old, new = (json.loads((folder / name).read_text()) for folder in (base_dir, guide_dir))
        if old["source_id"] != case["source_id"] or new["source_id"] != case["source_id"]:
            raise ValueError("paired source IDs do not match the manifest")
        if (not old.get("initial_observation_sha256")
                or old["initial_observation_sha256"] != new.get("initial_observation_sha256")):
            raise ValueError("paired initial observations do not match")
        if type(old["success"]) is not bool or type(new["success"]) is not bool:
            raise ValueError("success must be a Boolean from the environment")
        old_won, new_won = old["success"], new["success"]
        family = result["families"].setdefault(case["task_family"],
                    {"total": 0, "base_success": 0, "guide_success": 0,
                     "improved": 0, "worsened": 0, "both_success": 0, "both_failed": 0})
        subset = result["subsets"]["previous" if case.get("prior_case") else "new"]
        family["total"] += 1
        subset["total"] += 1
        for target in (result, family, subset):
            target["base_success"] += old_won
            target["guide_success"] += new_won
            target["improved"] += not old_won and new_won
            target["worsened"] += old_won and not new_won
            target["both_success"] += old_won and new_won
            target["both_failed"] += not old_won and not new_won
        for label, episode in (("base", old), ("guide", new)):
            result[label + "_steps"] += len(episode["trace"])
            result[label + "_no_effect"] += sum(turn["feedback"].strip() == "Nothing happens."
                                                  for turn in episode["trace"])
        result["initial_observation_matches"] += 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--guide-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_paired_results(json.loads(args.manifest.read_text()), args.base_dir, args.guide_dir)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
