"""项目配置：从 .env 文件和环境变量加载。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


def _getenv_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


BYTEPLUS_DOMAIN = _getenv_first("BYTEPLUS_DOMAIN")
BYTEPLUS_API_KEY = _getenv_first("BYTEPLUS_API_KEY")
BYTEPLUS_SEED = _getenv_first("BYTEPLUS_SEED")
TAVILY_API_KEY = _getenv_first("TAVILY_API_KEY")


def validate_required_config() -> None:
    missing: list[str] = []
    if not BYTEPLUS_DOMAIN:
        missing.append("BYTEPLUS_DOMAIN")
    if not BYTEPLUS_API_KEY:
        missing.append("BYTEPLUS_API_KEY（或 OPENAI_API_KEY）")
    if not BYTEPLUS_SEED:
        missing.append("BYTEPLUS_SEED")
    if missing:
        raise RuntimeError(
            "缺少必要配置："
            + ", ".join(missing)
            + f"\n请在环境变量或 `{ENV_PATH}` 中设置。"
        )
