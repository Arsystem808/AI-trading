# services/live_quotes.py
# Единый сервис котировок:
# - primary: WebSocket XT (crypto) при наличии модуля websocket-client
# - fallback: REST Single‑Ticker Snapshot (crypto)
# Цель: одна "last trade" цена и единый timestamp для всех стратегий.

import os
import json
import time
import threading
import requests

# Ленивая загрузка websocket; может отсутствовать в окружении
try:
    import websocket as _ws  # пакет websocket-client
except Exception:
    _ws = None

POLY_KEY = os.getenv("POLYGON_API_KEY", "").strip()

def map_ticker(sym: str) -> str:
    """
    Унификация пользовательского ввода:
    - ethusd / ETH-USD / ETH_USDT -> X:ETHUSD / X:ETHUSDT
    """
    s = (sym or "").strip().upper().replace("-", "").replace("_", "")
    if ":" in s:
        return s
    if s.endswith(("USD", "USDT", "USDC")):
        return f"X:{s}"
    return s  # как есть (на случай других рынков/будущих расширений)

class LiveQuoteService:
    def __init__(self, use_ws: bool = True, ttl_sec: float = 2.5):
        # Включаем WS только если модуль доступен
        self.use_ws = bool(use_ws and (_ws is not None))
        self.ttl = float(ttl_sec)
        self._last = {}          # { "X:ETHUSD": (price: float, ts_ms: int) }
        self._lock = threading.Lock()
        self._subs = set()
        self._ws = None
        self._ws_thread = None

    # ---------------- public API ----------------
    def ensure_ws(self):
        if not self.use_ws or self._ws_thread or _ws is None:
            return
        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()

    def subscribe(self, raw_sym: str):
        t = map_ticker(raw_sym)
        if not self.use_ws or _ws is None:
            return
        self.ensure_ws()
        with self._lock:
            if t in self._subs:
                return
            self._subs.add(t)
        try:
            if self._ws:
                self._ws.send(json.dumps({"action": "subscribe", "params": f"XT.{t}"}))
        except Exception:
            # При переподключении _run_ws повторно подпишет все тикеры
            pass

    def get_price(self, raw_sym: str):
        """
        Возвращает (price: float, ts_ms: int, source: str)
        Источники: "live-cache" (XT), "snapshot", "none"
        """
        t = map_ticker(raw_sym)
        now = int(time.time() * 1000)

        # 1) Кэш (свежий тик)
        with self._lock:
            p = self._last.get(t)
        if p and (now - p[1] <= self.ttl * 1000):
            return float(p[0]), int(p[1]), "live-cache"

        # 2) Пытаемся активировать/подписаться на WS
        if self.use_ws and _ws is not None:
            self.subscribe(t)

        # 3) Fallback: REST Single‑Ticker Snapshot (crypto)
        #    https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/X:ETHUSD
        try:
            if not POLY_KEY:
                raise RuntimeError("POLYGON_API_KEY is empty")
            url = f"https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/{t}?apiKey={POLY_KEY}"
            r = requests.get(url, timeout=2.5)
            j = r.json() if r.ok else {}
            lt = (((j or {}).get("ticker") or {}).get("lastTrade") or {})
            price = float(lt.get("price") or lt.get("p") or 0.0)
            ts = int(lt.get("timestamp") or lt.get("t") or now)
            if price > 0.0:
                with self._lock:
                    self._last[t] = (price, ts)
                return price, ts, "snapshot"
        except Exception:
            pass

        # 4) Полный отказ
        return 0.0, now, "none"

    # ---------------- internals ----------------
    def _run_ws(self):
        if _ws is None or not POLY_KEY:
            return
        url = "wss://socket.polygon.io/crypto"

        def on_open(ws):
            try:
                ws.send(json.dumps({"action": "auth", "params": POLY_KEY}))
                time.sleep(0.1)
                with self._lock:
                    subs = list(self._subs)
                if subs:
                    ws.send(json.dumps({
                        "action": "subscribe",
                        "params": ",".join([f"XT.{t}" for t in subs])
                    }))
            except Exception:
                pass

        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                if isinstance(data, list):
                    now = int(time.time() * 1000)
                    with self._lock:
                        for ev in 
                            if ev.get("ev") == "XT":
                                t = ev.get("pair") or ev.get("sym")
                                price = ev.get("p") or ev.get("price")
                                ts = ev.get("t") or now
                                if t and price:
                                    self._last[str(t)] = (float(price), int(ts))
            except Exception:
                pass

        def on_error(ws, err):
            time.sleep(1.0)

        def on_close(ws, a, b):
            time.sleep(1.0)

        while True:
            try:
                self._ws = _ws.WebSocketApp(
                    url, on_open=on_open, on_message=on_message,
                    on_error=on_error, on_close=on_close
                )
                # Пинги для поддержания соединения
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                time.sleep(2.0)
