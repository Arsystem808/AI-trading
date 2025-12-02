# core/polygon_client.py
import os
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from threading import Thread

import pandas as pd
import requests
import websocket  # pip install websocket-client


class PolygonClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_sec: int = 3600,
        base_url: str = "https://api.polygon.io",
        session: Optional[requests.Session] = None,
        enable_websocket: bool = False,  # Опционально для live данных
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
        
        # WebSocket для live цен (без кэша!)
        self.enable_websocket = enable_websocket
        self._ws = None
        self._ws_thread = None
        self._ws_callbacks = {}

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

    # -------- REST API: Stocks Aggregates с кэшем --------
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

    def last_trade_price(self, ticker: str) -> Optional[float]:
        # REST endpoint с кэшем - для некритичных запросов
        url = f"{self.base_url}/v2/last/trade/{ticker}"
        js = self._get(url, use_cache_key=f"last_trade:{ticker}")
        res = js.get("results") or js.get("result") or {}
        price = res.get("p") or res.get("price")
        try:
            return float(price) if price is not None else None
        except Exception:
            return None

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

    def check_auth(self) -> bool:
        try:
            _ = self.prev_close("SPY")
            return True
        except Exception:
            return False

    # -------- WebSocket API: Live цены БЕЗ кэша --------
    def _on_ws_message(self, ws, message):
        """Обработчик входящих WebSocket сообщений"""
        try:
            data = json.loads(message)
            for msg in 
                ev_type = msg.get("ev")  # T=Trade, Q=Quote, A=Aggregate
                if ev_type in self._ws_callbacks:
                    for callback in self._ws_callbacks[ev_type]:
                        callback(msg)
        except Exception as e:
            print(f"WebSocket message error: {e}")

    def _on_ws_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} {close_msg}")

    def _on_ws_open(self, ws):
        """Аутентификация после подключения"""
        auth_msg = json.dumps({"action": "auth", "params": self.api_key})
        ws.send(auth_msg)
        print("WebSocket connected and authenticated")

    def start_websocket(self):
        """Запустить WebSocket подключение в фоновом потоке"""
        if not self.enable_websocket:
            raise RuntimeError("WebSocket не активирован (enable_websocket=False)")
        
        if self._ws_thread and self._ws_thread.is_alive():
            print("WebSocket уже запущен")
            return
        
        ws_url = "wss://socket.polygon.io/stocks"
        self._ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
        )
        
        self._ws_thread = Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()
        time.sleep(1)  # Даём время на подключение

    def subscribe_live_trades(self, tickers: list, callback: Callable[[Dict], None]):
        """
        Подписка на live трейды (без кэша, минимальная задержка ~7мс)
        
        Args:
            tickers: Список тикеров, например ["AAPL", "TSLA"]
            callback: Функция для обработки каждого трейда, получает dict с полями:
                      ev="T", sym="AAPL", p=цена, s=размер, t=timestamp
        """
        if not self._ws:
            self.start_websocket()
        
        # Регистрируем callback
        if "T" not in self._ws_callbacks:
            self._ws_callbacks["T"] = []
        self._ws_callbacks["T"].append(callback)
        
        # Подписываемся на тикеры
        for ticker in tickers:
            sub_msg = json.dumps({"action": "subscribe", "params": f"T.{ticker}"})
            self._ws.send(sub_msg)
        print(f"Subscribed to trades: {tickers}")

    def subscribe_live_quotes(self, tickers: list, callback: Callable[[Dict], None]):
        """
        Подписка на live котировки bid/ask (задержка ~35-50мс)
        
        Args:
            tickers: Список тикеров
            callback: Функция получает dict с ev="Q", sym="AAPL", bp=bid, ap=ask
        """
        if not self._ws:
            self.start_websocket()
        
        if "Q" not in self._ws_callbacks:
            self._ws_callbacks["Q"] = []
        self._ws_callbacks["Q"].append(callback)
        
        for ticker in tickers:
            sub_msg = json.dumps({"action": "subscribe", "params": f"Q.{ticker}"})
            self._ws.send(sub_msg)
        print(f"Subscribed to quotes: {tickers}")

    def stop_websocket(self):
        """Остановить WebSocket подключение"""
        if self._ws:
            self._ws.close()
            self._ws = None
        if self._ws_thread:
            self._ws_thread.join(timeout=2)
            self._ws_thread = None
        print("WebSocket stopped")
