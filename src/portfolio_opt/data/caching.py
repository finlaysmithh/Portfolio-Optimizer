from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_settings
from ..utils import hash_key


class DiskCache:
    def __init__(self, namespace: str = "prices") -> None:
        self.settings = load_settings()
        self.namespace = namespace
        self.base = self.settings.cache_dir / namespace
        self.base.mkdir(parents=True, exist_ok=True)

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
            if datetime.utcnow() - ts > self.settings.cache_ttl:
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
