from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    alphavantage_api_key: str | None
    fmp_api_key: str | None
    email_user: str | None
    email_app_password: str | None
    log_level: str
    cache_ttl: timedelta
    liquidity_usd_threshold: float
    project_root: Path
    cache_dir: Path


def load_settings() -> Settings:
    # Load from either project .env or env/.env
    load_dotenv(dotenv_path=Path(".env"), override=False)
    load_dotenv(dotenv_path=Path("env/.env"), override=False)

    root = Path.cwd()
    cache_dir = root / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY"),
        fmp_api_key=os.getenv("FMP_API_KEY"),
        email_user=os.getenv("EMAIL_USER"),
        email_app_password=os.getenv("EMAIL_APP_PASSWORD"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        cache_ttl=timedelta(days=int(os.getenv("CACHE_TTL_DAYS", "5"))),
        liquidity_usd_threshold=float(os.getenv("LIQUIDITY_USD_THRESHOLD", "10000000")),
        project_root=root,
        cache_dir=cache_dir,
    )
