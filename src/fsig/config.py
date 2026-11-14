from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def edgar_user_agent() -> str:
    _load_env()
    ua = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if not ua or "example.com" in ua:
        raise RuntimeError(
            "Set EDGAR_USER_AGENT in .env (see .env.example). "
            "The SEC requires a contact name and email for automated access."
        )
    return ua
