from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ..utils import hash_key
except Exception:  # pragma: no cover - fallback for defensive use
    import hashlib

    def hash_key(*parts: Any, prefix: str = "") -> str:
        payload = json.dumps(
            parts[0] if len(parts) == 1 else parts,
            default=str,
            sort_keys=True,
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}" if prefix else digest


def _get_settings() -> dict[str, Any]:
    try:
        from ..config import load_settings

        return load_settings()
    except Exception:
        return {}


class DiskCache:
    def __init__(self, namespace: str = "prices") -> None:
        self.namespace = namespace
        self.settings = _get_settings()

        cache_dir = self.settings.get("CACHE_DIR")
        if not isinstance(cache_dir, Path):
            cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "data"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.base = cache_dir / namespace
        self.base.mkdir(parents=True, exist_ok=True)

        cache_ttl = self.settings.get("CACHE_TTL")
        self.cache_ttl = cache_ttl if isinstance(cache_ttl, timedelta) else timedelta(days=5)

    def _path(self, key: str) -> Path:
        return self.base / f"{key}.csv"

    def _meta_path(self, key: str) -> Path:
        return self.base / f"{key}.json"

    def load_df(self, key: str) -> pd.DataFrame | None:
        p = self._path(key)
        m = self._meta_path(key)
        if not p.exists() or not m.exists():
            return None
        try:
            meta = json.loads(m.read_text())
            ts = datetime.fromisoformat(meta.get("timestamp"))
            if datetime.utcnow() - ts > self.cache_ttl:
                return None
            df = pd.read_csv(p, index_col=0)
            if bool(meta.get("datetime_index", False)):
                df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            return None

    def save_df(self, key: str, df: pd.DataFrame) -> None:
        p = self._path(key)
        m = self._meta_path(key)
        df.to_csv(p)
        meta = {
            "timestamp": datetime.utcnow().isoformat(),
            "datetime_index": isinstance(df.index, pd.DatetimeIndex),
        }
        m.write_text(json.dumps(meta))

    @staticmethod
    def key_from_params(**kwargs: Any) -> str:
        items = sorted(kwargs.items())
        return hash_key(*[f"{k}={v}" for k, v in items])
