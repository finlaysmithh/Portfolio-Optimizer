import pandas as pd

from portfolio_opt.data.caching import DiskCache


def test_disk_cache_roundtrip(tmp_path, monkeypatch):
    # Redirect cache dir
    monkeypatch.setenv("CACHE_TTL_DAYS", "5")
    c = DiskCache(namespace="test")
    c.base = tmp_path
    key = c.key_from_params(a=1, b="two")
    df = pd.DataFrame({"A": [1, 2, 3]})
    c.save_df(key, df)
    out = c.load_df(key)
    assert out is not None
    assert out.equals(df)
