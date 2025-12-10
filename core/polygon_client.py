# core/polygon_client.py
import os
import hashlib
import json
import logging
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class PolygonClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_sec: int = 30,
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
    def daily_ohlc(self, ticker: str, days: int = 120, end_ms: Optional[int] = None) -> pd.DataFrame:
        """
        Получает дневные OHLC бары для тикера.
        
        Args:
            ticker: Тикер (например, "AAPL")
            days: Количество дней для получения
            end_ms: Конечная временная метка в миллисекундах UTC (опционально)
        
        Returns:
            DataFrame с колонками: open, high, low, close, volume, timestamp
        """
        # Определяем конечную дату
        if end_ms is not None:
            end = pd.Timestamp(end_ms, unit="ms", tz="UTC").normalize()
        else:
            end = pd.Timestamp.utcnow().normalize()
        
        # Берём растянутый диапазон, чтобы компенсировать выходные/праздники
        start = (end - pd.Timedelta(days=days * 2)).normalize()
        
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start.date()}/{end.date()}"
        
        # Включаем end_ms в cache_key для корректного кэширования
        cache_key = f"daily_ohlc:{ticker}:{days}:{start.date()}:{end.date()}"
        
        js = self._get(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 50000},
            use_cache_key=cache_key,
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


# ============================================================================
# SINGLETON INSTANCE И ФУНКЦИИ-ОБЕРТКИ ДЛЯ СОВМЕСТИМОСТИ С ОРКЕСТРАТОРОМ
# ============================================================================

@lru_cache(maxsize=1)
def get_client() -> PolygonClient:
    """
    Thread-safe singleton для PolygonClient через lru_cache.
    Автоматически создает и кэширует единственный экземпляр.
    
    Returns:
        PolygonClient: Глобальный экземпляр клиента
    """
    logger.info("Initializing Polygon API client")
    return PolygonClient()


def fetch_ohlc(ticker: str, timespan: str = "day", limit: int = 120, end_ms: Optional[int] = None) -> list:
    """
    Функция-обертка для получения OHLC баров в формате, совместимом с оркестратором.
    
    Args:
        ticker: Тикер (например, "AAPL", "BTCUSD")
        timespan: Таймфрейм (пока поддерживается только "day")
        limit: Количество баров для получения (1-5000)
        end_ms: Конечная временная метка в миллисекундах UTC (опционально)
    
    Returns:
        Список словарей с ключами: o, h, l, c, v, t
        Пример: [
            {"o": 150.0, "h": 152.0, "l": 149.0, "c": 151.0, "v": 1000000, "t": 1638316800000},
            ...
        ]
    
    Raises:
        ValueError: Если параметры невалидны
    """
    # Валидация параметров
    if not ticker or not isinstance(ticker, str):
        raise ValueError("ticker must be a non-empty string")
    
    if not isinstance(limit, int) or limit < 1 or limit > 5000:
        raise ValueError("limit must be an integer between 1 and 5000")
    
    if end_ms is not None:
        if not isinstance(end_ms, (int, float)) or end_ms < 0 or end_ms > time.time() * 1000:
            raise ValueError("end_ms must be a valid timestamp in milliseconds")
    
    try:
        client = get_client()
        
        if timespan != "day":
            logger.warning(f"Unsupported timespan '{timespan}' for {ticker}, falling back to 'day'")
            timespan = "day"
        
        # Получаем DataFrame через метод с поддержкой end_ms
        df = client.daily_ohlc(ticker, days=limit, end_ms=end_ms)
        
        if df.empty:
            logger.info(f"No OHLC data available for {ticker} (limit={limit})")
            return []
        
        # Берем последние limit баров
        df = df.tail(limit)
        
        # Конвертируем в формат списка словарей (совместимый с оркестратором)
        result = []
        for _, row in df.iterrows():
            result.append({
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
                "v": int(row["volume"]) if pd.notna(row["volume"]) else 0,
                "t": int(row["timestamp"].timestamp() * 1000),  # миллисекунды UTC
            })
        
        logger.debug(f"Fetched {len(result)} OHLC bars for {ticker}")
        return result
        
    except ValueError:
        # Пробрасываем валидационные ошибки
        raise
    except Exception as e:
        logger.error(f"Failed to fetch OHLC for {ticker}: {e}", exc_info=True)
        return []
turn False
