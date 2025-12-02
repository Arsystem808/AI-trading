# core/polygon_client.py
import os
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List
from threading import Thread

import pandas as pd
import requests

# Опциональный импорт WebSocket
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Опциональный импорт Streamlit для secrets
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class PolygonClient:
    """
    Клиент для Polygon.io API с поддержкой:
    - REST API с файловым кэшированием (для исторических данных)
    - WebSocket API для live цен (опционально, без кэша)
    - Автоматическая загрузка API key из Streamlit secrets или env
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_sec: int = 3600,
        base_url: str = "https://api.polygon.io",
        session: Optional[requests.Session] = None,
        enable_websocket: bool = False,
    ):
        """
        Args:
            api_key: API ключ Polygon.io (если None, загружается из secrets/env)
            cache_ttl_sec: TTL кэша для REST запросов в секундах
            base_url: Базовый URL для REST API
            session: Кастомная requests.Session для продвинутого использования
            enable_websocket: Включить WebSocket для live данных
        """
        # Попытка загрузить API key из разных источников
        if api_key is None:
            # 1. Из Streamlit secrets (приоритет для Cloud deployment)
            if STREAMLIT_AVAILABLE:
                try:
                    api_key = st.secrets.get("POLYGON_API_KEY", "")
                except (KeyError, FileNotFoundError, AttributeError):
                    pass
            
            # 2. Из переменных окружения
            if not api_key:
                api_key = os.getenv("POLYGON_API_KEY", "")
        
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise RuntimeError(
                "POLYGON_API_KEY is not set. "
                "Установите в Streamlit Secrets или переменных окружения."
            )
        
        self.cache_ttl = cache_ttl_sec
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(".cache/polygon")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = session or requests.Session()
        
        # WebSocket настройки
        self.enable_websocket = enable_websocket
        self._ws = None
        self._ws_thread = None
        self._ws_callbacks = {}
        self._ws_connected = False
        
        if enable_websocket and not WEBSOCKET_AVAILABLE:
            raise RuntimeError(
                "WebSocket не доступен. Установите: pip install websocket-client"
            )

    # ==================== Вспомогательные методы кэша ====================
    
    def _cache_path(self, key: str) -> Path:
        """Генерация пути к файлу кэша по ключу"""
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _read_cache(self, path: Path) -> Optional[Dict[str, Any]]:
        """Чтение из кэша, если данные свежие"""
        if not path.exists():
            return None
        
        # Проверка TTL
        fresh = (time.time() - path.stat().st_mtime) < self.cache_ttl
        if not fresh:
            return None
        
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, path: Path, payload: Dict[str, Any]) -> None:
        """Запись в кэш"""
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass

    # ==================== HTTP запросы с кэшем ====================
    
    def _get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        backoff_base: float = 0.8,
        use_cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        GET запрос с автоматическим retry и опциональным кэшем
        
        Args:
            url: Полный URL для запроса
            params: Query параметры
            max_retries: Максимум попыток при ошибках
            backoff_base: Базовая задержка для экспоненциального backoff
            use_cache_key: Ключ кэша (если None - кэш отключен)
        """
        params = {**(params or {}), "apiKey": self.api_key}
        
        # Проверка кэша
        cache_path = self._cache_path(use_cache_key) if use_cache_key else None
        if cache_path is not None:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached
        
        # Retry логика
        last_err = None
        for attempt in range(max_retries):
            try:
                r = self._session.get(url, params=params, timeout=20)
                
                # Exponential backoff для rate limit и временных ошибок
                if r.status_code in (429, 500, 502, 503, 504):
                    sleep = (backoff_base * (2 ** attempt)) + random.uniform(0, 0.3)
                    time.sleep(sleep)
                    last_err = requests.HTTPError(
                        f"{r.status_code} {r.reason}: {r.url}"
                    )
                    continue
                
                r.raise_for_status()
                js = r.json()
                
                # Сохранение в кэш
                if cache_path is not None:
                    self._write_cache(cache_path, js)
                
                return js
                
            except requests.RequestException as e:
                last_err = e
                if attempt < max_retries - 1:
                    sleep = (backoff_base * (2 ** attempt)) + random.uniform(0, 0.3)
                    time.sleep(sleep)
        
        raise last_err or requests.HTTPError(
            f"Request failed after {max_retries} retries"
        )

    # ==================== REST API методы (с кэшем) ====================
    
    def daily_ohlc(self, ticker: str, days: int = 120) -> pd.DataFrame:
        """
        Получить дневные OHLCV свечи (с кэшем)
        
        Args:
            ticker: Тикер (например "AAPL")
            days: Количество торговых дней
            
        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume
        """
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
        
        # Переименование колонок
        rename = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "t": "timestamp",
        }
        df = df.rename(columns=rename)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
        return df

    def last_trade_price(self, ticker: str) -> Optional[float]:
        """
        Последняя сделка через REST API (с кэшем, подходит для некритичных запросов)
        Для live цен используйте WebSocket!
        """
        url = f"{self.base_url}/v2/last/trade/{ticker}"
        js = self._get(url, use_cache_key=f"last_trade:{ticker}")
        
        res = js.get("results") or js.get("result") or {}
        price = res.get("p") or res.get("price")
        
        try:
            return float(price) if price is not None else None
        except (ValueError, TypeError):
            return None

    def prev_close(self, ticker: str, adjusted: bool = True) -> Optional[float]:
        """Предыдущее закрытие (с кэшем)"""
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/prev"
        js = self._get(
            url,
            params={"adjusted": str(adjusted).lower()},
            use_cache_key=f"prev_close:{ticker}:{adjusted}",
        )
        
        results = js.get("results") or []
        if not results:
            return None
        
        close = results[0].get("c")
        try:
            return float(close) if close is not None else None
        except (ValueError, TypeError):
            return None

    def check_auth(self) -> bool:
        """Проверка валидности API ключа"""
        try:
            _ = self.prev_close("SPY")
            return True
        except Exception:
            return False

    # ==================== WebSocket методы (БЕЗ кэша, для live данных) ====================
    
    def _on_ws_message(self, ws, message):
        """Обработчик входящих WebSocket сообщений"""
        try:
            data = json.loads(message)
            
            # Обработка массива событий
            if isinstance(data, list):
                for msg in 
                    ev_type = msg.get("ev")  # T=Trade, Q=Quote, A=Aggregate
                    if ev_type in self._ws_callbacks:
                        for callback in self._ws_callbacks[ev_type]:
                            try:
                                callback(msg)
                            except Exception as e:
                                print(f"Callback error for {ev_type}: {e}")
            else:
                # Обработка статусных сообщений
                status = data.get("status")
                if status == "connected":
                    print("WebSocket: connected to Polygon")
                elif status == "auth_success":
                    print("WebSocket: authenticated successfully")
                    self._ws_connected = True
                elif status == "auth_failed":
                    print("WebSocket: authentication failed!")
                    self._ws_connected = False
                    
        except json.JSONDecodeError as e:
            print(f"WebSocket JSON decode error: {e}")
        except Exception as e:
            print(f"WebSocket message error: {e}")

    def _on_ws_error(self, ws, error):
        """Обработчик ошибок WebSocket"""
        print(f"WebSocket error: {error}")

    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Обработчик закрытия WebSocket"""
        self._ws_connected = False
        print(f"WebSocket closed: {close_status_code} {close_msg}")

    def _on_ws_open(self, ws):
        """Аутентификация после подключения"""
        auth_msg = json.dumps({"action": "auth", "params": self.api_key})
        ws.send(auth_msg)

    def start_websocket(self, market: str = "stocks"):
        """
        Запустить WebSocket подключение в фоновом потоке
        
        Args:
            market: Тип рынка ("stocks", "crypto", "forex")
        """
        if not self.enable_websocket:
            raise RuntimeError(
                "WebSocket не активирован. Создайте клиент с enable_websocket=True"
            )
        
        if self._ws_thread and self._ws_thread.is_alive():
            print("WebSocket уже запущен")
            return
        
        # URL зависит от типа рынка
        ws_url = f"wss://socket.polygon.io/{market}"
        
        self._ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
        )
        
        self._ws_thread = Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()
        
        # Ждем подключения
        for _ in range(50):  # 5 секунд макс
            if self._ws_connected:
                break
            time.sleep(0.1)
        
        if not self._ws_connected:
            print("WARNING: WebSocket подключен, но аутентификация не завершена")

    def subscribe_live_trades(
        self,
        tickers: List[str],
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Подписка на live трейды (минимальная задержка ~7мс)
        
        Args:
            tickers: Список тикеров ["AAPL", "TSLA"]
            callback: Функция обработки, получает dict:
                      {ev: "T", sym: "AAPL", p: price, s: size, t: timestamp}
        """
        if not self._ws:
            self.start_websocket()
        
        # Регистрация callback
        if "T" not in self._ws_callbacks:
            self._ws_callbacks["T"] = []
        self._ws_callbacks["T"].append(callback)
        
        # Подписка на тикеры
        for ticker in tickers:
            sub_msg = json.dumps({"action": "subscribe", "params": f"T.{ticker}"})
            self._ws.send(sub_msg)
        
        print(f"Subscribed to trades: {tickers}")

    def subscribe_live_quotes(
        self,
        tickers: List[str],
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Подписка на live котировки bid/ask (задержка ~35-50мс)
        
        Args:
            tickers: Список тикеров
            callback: Функция получает dict:
                      {ev: "Q", sym: "AAPL", bp: bid_price, ap: ask_price}
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

    def subscribe_live_aggregates(
        self,
        tickers: List[str],
        callback: Callable[[Dict[str, Any]], None],
        timespan: str = "1"  # "1" = 1 секунда, "5" = 5 секунд и т.д.
    ):
        """
        Подписка на агрегированные минутные/секундные бары
        
        Args:
            tickers: Список тикеров
            callback: Функция получает dict с OHLCV
            timespan: Таймспан агрегации
        """
        if not self._ws:
            self.start_websocket()
        
        if "A" not in self._ws_callbacks:
            self._ws_callbacks["A"] = []
        self._ws_callbacks["A"].append(callback)
        
        for ticker in tickers:
            sub_msg = json.dumps({
                "action": "subscribe",
                "params": f"A.{timespan}.{ticker}"
            })
            self._ws.send(sub_msg)
        
        print(f"Subscribed to {timespan}s aggregates: {tickers}")

    def stop_websocket(self):
        """Остановить WebSocket подключение"""
        if self._ws:
            self._ws.close()
            self._ws = None
        
        if self._ws_thread:
            self._ws_thread.join(timeout=2)
            self._ws_thread = None
        
        self._ws_connected = False
        self._ws_callbacks.clear()
        print("WebSocket stopped")

    def __del__(self):
        """Cleanup при удалении объекта"""
        if self._ws:
            try:
                self.stop_websocket()
            except:
                pass


# ==================== Ленивая инициализация для Streamlit ====================

_global_client: Optional[PolygonClient] = None


def get_polygon_client(
    enable_websocket: bool = False,
    cache_ttl_sec: int = 3600
) -> PolygonClient:
    """
    Получить глобальный экземпляр PolygonClient (singleton pattern)
    Безопасно для использования в Streamlit - создается при первом вызове
    
    Args:
        enable_websocket: Включить WebSocket для live данных
        cache_ttl_sec: TTL кэша для REST запросов
    
    Returns:
        Экземпляр PolygonClient
    """
    global _global_client
    
    if _global_client is None:
        _global_client = PolygonClient(
            enable_websocket=enable_websocket,
            cache_ttl_sec=cache_ttl_sec
        )
    
    return _global_client
