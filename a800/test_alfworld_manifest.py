"""Checks for selecting and validating the fixed paired development games."""
import collections
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    from alfworld_manifest import build_manifest, match_manifest_games, select_stratified_games
except ImportError:
    build_manifest = match_manifest_games = select_stratified_games = None


FAMILIES = (
    "look_at_obj_in_light", "pick_and_place_simple", "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_two_obj_and_place",
)


def fake_game_ids():
    return [f"json_2.1.1/valid_seen/{family}-object{i}/trial/game.tw-pddl"
            for family in FAMILIES for i in range(6)]


class AlfworldManifestTest(unittest.TestCase):
    def test_selection_is_balanced_deterministic_and_keeps_prior_games(self):
        self.assertIsNotNone(select_stratified_games, "stratified game selection is missing")
        games = fake_game_ids()
        anchors = [games[0], games[6], games[12]]
        selected = select_stratified_games(games, anchors, per_family=5, seed=19)
        self.assertEqual(selected, select_stratified_games(games[::-1], anchors, per_family=5, seed=19))
        self.assertEqual(len(selected), 30)
        self.assertEqual(len(set(selected)), 30)
        self.assertTrue(set(anchors) <= set(selected))
        counts = collections.Counter(Path(game).parts[2].split("-", 1)[0] for game in selected)
        self.assertEqual(set(counts), set(FAMILIES))
        self.assertTrue(all(count == 5 for count in counts.values()))
        with self.assertRaises(ValueError):
            select_stratified_games(games, ["missing"], per_family=5, seed=19)
        with self.assertRaises(ValueError):
            select_stratified_games(games[:-2], anchors, per_family=5, seed=19)

    def test_archive_hash_and_each_selected_game_are_recorded(self):
        self.assertIsNotNone(build_manifest, "manifest construction is missing")
        games = fake_game_ids()
        anchors = [games[0], games[6], games[12]]
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "games.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for game in games:
                    bundle.writestr(game, json.dumps({"solvable": True, "id": game}))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest = build_manifest(archive, anchors=anchors, seed=19, expected_archive_sha256=digest)
            self.assertEqual(manifest["archive_sha256"], digest)
            self.assertEqual(len(manifest["cases"]), 30)
            with zipfile.ZipFile(archive) as bundle:
                for case in manifest["cases"]:
                    self.assertEqual(case["game_sha256"], hashlib.sha256(bundle.read(case["source_id"])).hexdigest())
            with self.assertRaises(ValueError):
                build_manifest(archive, anchors=anchors, seed=19, expected_archive_sha256="0" * 64)

    def test_manifest_requires_the_same_available_game_bytes(self):
        self.assertIsNotNone(match_manifest_games, "manifest validation is missing")
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            game = data / "json_2.1.1/valid_seen/look_at_obj_in_light-clock/trial/game.tw-pddl"
            game.parent.mkdir(parents=True)
            game.write_bytes(b"original")
            manifest = {"cases": [{"source_id": str(game.relative_to(data)),
                                   "game_sha256": hashlib.sha256(b"original").hexdigest()}]}
            self.assertEqual(match_manifest_games(manifest, data, [str(game)]), [str(game)])
            with self.assertRaises(ValueError):
                match_manifest_games(manifest, data, [])
            game.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                match_manifest_games(manifest, data, [str(game)])


if __name__ == "__main__":
    unittest.main()
