"""Recover the three fixed development cases without publishing a dataset."""
import argparse
import json
from pathlib import Path


CASE_IDS = (("bw-v1-val-0326", 4), ("bw-v1-val-0166", 6), ("bw-v1-val-0370", 8))


def extract_cases(rows):
    wanted = dict(CASE_IDS)
    found = {}
    for row in rows:
        case_id = row["id"]
        if case_id not in wanted:
            continue
        if case_id in found or row["plan_len"] != wanted[case_id]:
            raise ValueError(f"duplicate case or unexpected plan length: {case_id}")
        found[case_id] = {"source_id": case_id, "oracle_steps": row["plan_len"],
                          "init": row["init"], "goal": row["goal"]}
    missing = set(wanted) - set(found)
    if missing:
        raise ValueError(f"validation set is missing fixed cases: {sorted(missing)}")
    return [found[case_id] for case_id, _ in CASE_IDS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.val.open(encoding="utf-8") as source:
        cases = extract_cases(json.loads(line) for line in source if line.strip())
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({"cases": cases}, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
