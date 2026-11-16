"""Select and validate fixed, balanced ALFWorld development games."""
import argparse
import collections
import hashlib
import json
import random
import zipfile
from pathlib import Path, PurePosixPath


OFFICIAL_ARCHIVE_SHA256 = "5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf"
TASK_FAMILIES = (
    "look_at_obj_in_light", "pick_and_place_simple", "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_two_obj_and_place",
)
PRIOR_IDS = (
    "json_2.1.1/valid_seen/look_at_obj_in_light-AlarmClock-None-DeskLamp-323/"
    "trial_T20190909_044715_250790/game.tw-pddl",
    "json_2.1.1/valid_seen/pick_and_place_simple-Book-None-SideTable-329/"
    "trial_T20190908_050633_745514/game.tw-pddl",
    "json_2.1.1/valid_seen/pick_clean_then_place_in_recep-ButterKnife-None-CounterTop-8/"
    "trial_T20190909_105559_983897/game.tw-pddl",
)


def _family(game):
    path = PurePosixPath(game)
    if (len(path.parts) < 5 or path.parts[:2] != ("json_2.1.1", "valid_seen")
            or path.name != "game.tw-pddl"):
        raise ValueError("game is not in the expected development split")
    return path.parts[2].split("-", 1)[0]


def select_stratified_games(games, anchors, per_family=5, seed=20261116):
    """Select the same number per family without consulting model outcomes."""
    games, anchors = list(games), list(anchors)
    if per_family < 1 or len(set(games)) != len(games) or len(set(anchors)) != len(anchors):
        raise ValueError("games and anchors must be unique and per_family positive")
    if not set(anchors).issubset(games):
        raise ValueError("a previous game is absent from the development split")
    groups = collections.defaultdict(list)
    for game in games:
        groups[_family(game)].append(game)
    if set(groups) != set(TASK_FAMILIES):
        raise ValueError("the expected six ALFWorld task families are not present")
    rng, selected = random.Random(seed), []
    for family in sorted(groups):
        group = sorted(groups[family])
        required = sorted(game for game in anchors if _family(game) == family)
        remaining = [game for game in group if game not in required]
        if len(required) > per_family or len(remaining) < per_family - len(required):
            raise ValueError(f"not enough games for family {family}")
        selected.extend(sorted(required + rng.sample(remaining, per_family - len(required))))
    return selected


def build_manifest(archive, anchors=PRIOR_IDS, seed=20261116,
                   expected_archive_sha256=OFFICIAL_ARCHIVE_SHA256):
    archive = Path(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected_archive_sha256:
        raise ValueError("ALFWorld archive SHA-256 does not match")
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist()
                 if name.startswith("json_2.1.1/valid_seen/") and name.endswith("/game.tw-pddl")
                 and "movable" not in name and "Sliced" not in name
                 and json.loads(bundle.read(name)).get("solvable") is True]
        selected = select_stratified_games(names, anchors, per_family=5, seed=seed)
        cases = [{"source_id": name, "task_family": _family(name),
                  "game_sha256": hashlib.sha256(bundle.read(name)).hexdigest(),
                  "prior_case": name in anchors} for name in selected]
    return {"schema_version": 1, "archive_sha256": digest, "selection_seed": seed,
            "per_family": 5, "cases": cases}


def match_manifest_games(manifest, data_dir, available_games):
    """Map manifest IDs to available games and verify their exact bytes."""
    data_dir = Path(data_dir)
    available = {str(Path(game).relative_to(data_dir)): game for game in available_games}
    cases = manifest["cases"]
    if not cases or len({case["source_id"] for case in cases}) != len(cases):
        raise ValueError("manifest case IDs must be non-empty and unique")
    games = []
    for case in cases:
        game = available.get(case["source_id"])
        if game is None:
            raise ValueError("a manifest game is not available in the development split")
        if hashlib.sha256(Path(game).read_bytes()).hexdigest() != case["game_sha256"]:
            raise ValueError("a manifest game has different bytes")
        games.append(game)
    return games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.archive)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"selected {len(manifest['cases'])} games in {len(TASK_FAMILIES)} families", flush=True)


if __name__ == "__main__":
    main()
