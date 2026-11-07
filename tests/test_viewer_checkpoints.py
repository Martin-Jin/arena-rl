import dataclasses
import os
import tempfile
import time
import unittest
from pathlib import Path

from arena_ai.config import load_config
from arena_ai.viewer.controllers import (
    list_checkpoints,
    resolve_checkpoint,
    trained_available,
)
from arena_ai.viewer.settings import load_settings


def _config_for(folder: Path):
    """A config whose default checkpoint is ``folder / latest.pt`` (absolute)."""
    config = load_config()
    opponent = dataclasses.replace(
        config.viewer.opponent, checkpoint_path=str(folder / "latest.pt")
    )
    return dataclasses.replace(
        config, viewer=dataclasses.replace(config.viewer, opponent=opponent)
    )


class CheckpointDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.folder = Path(self.directory.name)
        self.config = _config_for(self.folder)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _touch(self, name: str, age_seconds: float = 0) -> Path:
        path = self.folder / name
        path.write_bytes(b"x")
        stamp = time.time() - age_seconds
        os.utime(path, (stamp, stamp))
        return path

    def test_no_checkpoint_means_none_available(self) -> None:
        self.assertFalse(trained_available(self.config))
        self.assertIsNone(resolve_checkpoint(self.config))
        self.assertEqual(list_checkpoints(self.config), [])

    def test_a_renamed_checkpoint_is_still_found(self) -> None:
        best = self._touch("best_7_11_26.pt")  # latest.pt was renamed to keep it.
        self.assertTrue(trained_available(self.config))
        self.assertEqual(resolve_checkpoint(self.config), best)

    def test_automatic_prefers_latest_then_the_newest_file(self) -> None:
        older = self._touch("older.pt", age_seconds=100)
        newer = self._touch("newer.pt", age_seconds=10)
        self.assertEqual(resolve_checkpoint(self.config), newer)
        latest = self._touch("latest.pt", age_seconds=500)
        self.assertEqual(resolve_checkpoint(self.config), latest)
        self.assertEqual(list_checkpoints(self.config), [latest, newer, older])

    def test_a_named_checkpoint_is_used_and_bad_names_fall_back(self) -> None:
        latest = self._touch("latest.pt")
        other = self._touch("other.pt", age_seconds=50)
        self.assertEqual(resolve_checkpoint(self.config, "other.pt"), other)
        self.assertEqual(resolve_checkpoint(self.config, "missing.pt"), latest)
        self.assertEqual(resolve_checkpoint(self.config, "../other.pt"), latest)

    def test_a_saved_checkpoint_that_no_longer_exists_is_cleared(self) -> None:
        self._touch("other.pt")
        settings_path = self.folder / "settings.json"
        settings_path.write_text('{"checkpoint": "gone.pt"}', encoding="utf-8")
        self.assertEqual(load_settings(self.config, settings_path).checkpoint, "")
        settings_path.write_text('{"checkpoint": "other.pt"}', encoding="utf-8")
        self.assertEqual(load_settings(self.config, settings_path).checkpoint, "other.pt")


if __name__ == "__main__":
    unittest.main()
