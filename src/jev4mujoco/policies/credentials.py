from __future__ import annotations

import os
from pathlib import Path


TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"


def load_typesafe_api_key(path: str | Path = ".env") -> bool:
    """Load only TYPESAFE_API_KEY without executing or expanding file content."""
    if os.environ.get(TYPESAFE_API_KEY_ENV):
        return True

    env_path = Path(path)
    if not env_path.is_file():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().removeprefix("export ")
        name, separator, value = line.partition("=")
        if not separator or name.strip() != TYPESAFE_API_KEY_ENV:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value:
            return False

        os.environ[TYPESAFE_API_KEY_ENV] = value
        return True

    return False

