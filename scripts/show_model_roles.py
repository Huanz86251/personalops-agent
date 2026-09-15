"""Print resolved role configuration without keys or network calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings


if __name__ == "__main__":
    settings = load_settings()
    for role, config in settings.role_models.items():
        print(f"{role:24} {config.provider:12} {config.model:28} "
              f"max_tokens={config.max_tokens} key_env={config.api_key_env}")
