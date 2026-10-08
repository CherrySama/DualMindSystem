from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev4mujoco.policies.credentials import TYPESAFE_API_KEY_ENV, load_typesafe_api_key


class CredentialsTest(unittest.TestCase):
    def test_loads_only_typesafe_key_without_executing_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "must_not_exist"
            env_path = root / ".env"
            env_path.write_text(
                "UNRELATED=$(touch " + str(marker) + ")\n"
                "TYPESAFE_API_KEY=test-key\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(load_typesafe_api_key(env_path))
                self.assertEqual(os.environ[TYPESAFE_API_KEY_ENV], "test-key")

            self.assertFalse(marker.exists())

    def test_existing_environment_value_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("TYPESAFE_API_KEY=file-key\n", encoding="utf-8")

            with patch.dict(
                os.environ, {TYPESAFE_API_KEY_ENV: "process-key"}, clear=True
            ):
                self.assertTrue(load_typesafe_api_key(env_path))
                self.assertEqual(os.environ[TYPESAFE_API_KEY_ENV], "process-key")

    def test_accepts_export_and_quoted_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                'export TYPESAFE_API_KEY="quoted-key"\n', encoding="utf-8"
            )

            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(load_typesafe_api_key(env_path))
                self.assertEqual(os.environ[TYPESAFE_API_KEY_ENV], "quoted-key")

    def test_missing_or_empty_key_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(load_typesafe_api_key(root / "missing.env"))

                env_path = root / ".env"
                env_path.write_text("TYPESAFE_API_KEY=\n", encoding="utf-8")
                self.assertFalse(load_typesafe_api_key(env_path))
                self.assertNotIn(TYPESAFE_API_KEY_ENV, os.environ)


if __name__ == "__main__":
    unittest.main()
