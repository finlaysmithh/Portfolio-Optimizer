from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

# Optional Streamlit import — used only when available.
try:  # pragma: no cover - streamlit is optional in most test runs
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - streamlit absent locally
    st = None  # type: ignore

# Optional python-dotenv import. The app must not require it.
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - dependency missing on Cloud
    load_dotenv = None  # type: ignore


_PRIMARY_KEYS = [
    "DATA_SOURCE",
    "ALPHAVANTAGE_API_KEY",
    "FMP_API_KEY",
    "EMAIL_USER",
    "EMAIL_APP_PASSWORD",
    "LOG_LEVEL",
    "BENCHMARK",
    "CACHE_TTL_DAYS",
    "LIQUIDITY_USD_THRESHOLD",
    "CACHE_DIR",
]


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_settings() -> Dict[str, Any]:
    """Load settings with precedence: Streamlit secrets → env vars → .env files."""
    cfg: Dict[str, Any] = {}

    # 1) Streamlit secrets
    if st is not None:
        try:
            secrets = getattr(st, "secrets", None)
            if secrets:
                for key, value in secrets.items():
                    cfg[key if isinstance(key, str) else key] = value
                for key in list(cfg.keys()):
                    if isinstance(key, str):
                        upper_key = key.upper()
                        if upper_key in _PRIMARY_KEYS and upper_key not in cfg:
                            cfg[upper_key] = cfg[key]
        except Exception:
            pass

    # 2) Existing environment variables
    for key in _PRIMARY_KEYS:
        if key not in cfg and key in os.environ:
            cfg[key] = os.environ[key]

    # 3) .env files (optional, only if python-dotenv is present)
    if load_dotenv is not None:
        for path in (Path(".env"), Path("env/.env")):
            try:
                if path.exists():
                    load_dotenv(dotenv_path=path, override=False)
            except Exception:
                continue
        for key in _PRIMARY_KEYS:
            if key not in cfg and key in os.environ:
                cfg[key] = os.environ[key]

    # Defaults and derived values
    cfg.setdefault("DATA_SOURCE", "yfinance")
    cfg.setdefault("BENCHMARK", "SPY")
    cfg["LOG_LEVEL"] = str(cfg.get("LOG_LEVEL", os.environ.get("LOG_LEVEL", "INFO")))

    for optional_key in ("ALPHAVANTAGE_API_KEY", "FMP_API_KEY", "EMAIL_USER", "EMAIL_APP_PASSWORD"):
        cfg.setdefault(optional_key, None)

    cache_ttl_days = _coerce_int(cfg.get("CACHE_TTL_DAYS", os.environ.get("CACHE_TTL_DAYS", 5)), 5)
    cfg["CACHE_TTL_DAYS"] = cache_ttl_days
    cfg["CACHE_TTL"] = timedelta(days=cache_ttl_days)

    liquidity_default = os.environ.get("LIQUIDITY_USD_THRESHOLD", 10_000_000)
    cfg["LIQUIDITY_USD_THRESHOLD"] = _coerce_float(
        cfg.get("LIQUIDITY_USD_THRESHOLD", liquidity_default), 10_000_000.0
    )

    project_root = Path.cwd()
    cfg["PROJECT_ROOT"] = project_root

    cache_dir_value = cfg.get("CACHE_DIR")
    cache_dir = Path(cache_dir_value) if cache_dir_value else project_root / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg["CACHE_DIR"] = cache_dir

    return cfg
