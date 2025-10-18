from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ..config import load_settings

try:
    from ..utils import hash_key
except ImportError:
    import hashlib

    def hash_key(*parts: Any, prefix: str = "") -> str:  # type: ignore[redefinition]
        payload = json.dumps(parts if len(parts) != 1 else parts[0], default=str, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}:" + digest if prefix else digest


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
