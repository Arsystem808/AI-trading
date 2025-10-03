# core/polygon_client.py
import time
import random
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import pandas as pd

class PolygonClient:
    def __init__(self, api_key: str, cache_ttl_sec: int = 3600):
        self.api_key = api_key
        self.cache_ttl = cache_ttl_sec
        self.cache_dir = Path(".cache/polygon")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None,
             max_retries: int = 5, backoff_base: float = 0.8) -> Dict[str, Any]:
        params = {**(params or {}), "apiKey": self.api_key}
        last_err = None
        for attempt in range(max_retries):
            r = requests.get(url, params=params, timeout=20)
            # Троттлим 429/5xx
            if r.status_code in (429, 500, 502, 503, 504):
                sleep = (backoff_base * (2 ** attempt)) + random.uniform(0, 0.3)
                time.sleep(sleep)
                last_err = requests.HTTPError(f"{r.status_code} {r.reason}: {r.url}")
                continue
            r.raise_for_status()
            return r.json()
        raise last_err or requests.HTTPError(f"Request failed after {max_retries} retries")  # noqa

    def daily_ohlc(self, ticker: str, days: int = 120) -> pd.DataFrame:
        # Ключ кэша
        key = hashlib.md5(f"daily_ohlc:{ticker}:{days}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        # Если свежий кэш — отдаем
        if path.exists() and (time.time() - path.stat().st_mtime) < self.cache_ttl:
            try:
                return pd.read_json(path, orient="records")
            except Exception:
                path.unlink(missing_ok=True)
        # Иначе грузим из API
        end = pd.Timestamp.utcnow().normalize()
        start = (end - pd.Timedelta(days=days*2)).normalize()
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start.date()}/{end.date()}"
        js = self._get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000})
        rows = js.get("results") or []
        df = pd.DataFrame(rows)
        if not df.empty:
            # Приводим к ожидаемым колонкам
            rename = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"}
            df = df.rename(columns=rename)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        # Пишем кэш
        try:
            df.to_json(path, orient="records")
        except Exception:
            pass
        return df
