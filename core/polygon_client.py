# core/polygon_client.py
import os
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests


class PolygonClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_sec: int = 0,
        base_url: str = "https://api.polygon.io",
        session: Optional[requests.Session] = None,
    ):
        # Ключ читаем из аргумента или переменной окружения POLYGON_API_KEY
        self.api_key = (api_key or os.getenv("POLYGON_API_KEY", "")).strip()
        if not self.api_key:
            raise RuntimeError("POLYGON_API_KEY is not set")
        self.cache_ttl = cache_ttl_sec
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(".cache/polygon")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = session or requests.Session()

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _read_cache(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        fresh = (time.time() - path.stat().st_mtime) < self.cache_ttl
        if not fresh:
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, path: Path, payload: Dict[str, Any]) -> None:
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass

    def _get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        backoff_base: float = 0.8,
        use_cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {**(params or {}), "apiKey": self.api_key}
        cache_path = self._cache_path(use_cache_key) if use_cache_key else None
        if cache_path is not None:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached
        last_err = None
        for attempt in range(max_retries):
            r = self._session.get(url, params=params, timeout=20)
            # Бэкофф на rate limit / временные ошибки
            if r.status_code in (429, 500, 502, 503, 504):
                sleep = (backoff_base * (2**attempt)) + random.uniform(0, 0.3)
                time.sleep(sleep)
                last_err = requests.HTTPError(f"{r.status_code} {r.reason}: {r.url}")
                continue
            r.raise_for_status()
            js = r.json()
            if cache_path is not None:
                self._write_cache(cache_path, js)
            return js
        raise last_err or requests.HTTPError(f"Request failed after {max_retries} retries")

    # -------- Stocks Aggregates: дневные свечи --------
    def daily_ohlc(self, ticker: str, days: int = 120) -> pd.DataFrame:
        # Берём растянутый диапазон, чтобы компенсировать выходные/праздники
        end = pd.Timestamp.utcnow().normalize()
        start = (end - pd.Timedelta(days=days * 2)).normalize()
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start.date()}/{end.date()}"
        js = self._get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 50000},
            use_cache_key=f"daily_ohlc:{ticker}:{days}:{start.date()}:{end.date()}",
        )
        rows = js.get("results") or []
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        rename = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"}
        df = df.rename(columns=rename)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    # -------- Последняя сделка --------
    def last_trade_price(self, ticker: str) -> Optional[float]:
        # Пытаемся v2 last trade, при необходимости можно заменить на v3/trades с сортировкой
        url = f"{self.base_url}/v2/last/trade/{ticker}"
        js = self._get(url, use_cache_key=f"last_trade:{ticker}")
        res = js.get("results") or js.get("result") or {}
        price = res.get("p") or res.get("price")
        try:
            return float(price) if price is not None else None
        except Exception:
            return None

    # -------- Предыдущее закрытие --------
    def prev_close(self, ticker: str, adjusted: bool = True) -> Optional[float]:
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/prev"
        js = self._get(url, params={"adjusted": str(adjusted).lower()}, use_cache_key=f"prev_close:{ticker}:{adjusted}")
        results = js.get("results") or []
        if not results:
            return None
        close = results[0].get("c")
        try:
            return float(close) if close is not None else None
        except Exception:
            return None

    # -------- Health-check ключа --------
    def check_auth(self) -> bool:
        # Лёгкий запрос, который валиден с ключом
        try:
            _ = self.prev_close("SPY")
            return True
        except Exception:
            return False
