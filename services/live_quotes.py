# services/live_quotes.py
# Единый сервис котировок для всех стратегий.
# Приоритет источников:
#   1) WebSocket XT (если установлен websocket-client и есть доступ)
#   2) REST Single‑Ticker Snapshot (crypto)
#   3) REST Last‑Trade (crypto)
# Возвращает одну "последнюю сделку" и общий timestamp для всех агентов.

import os
import json
import time
import threading
import requests

# Ленивая загрузка websocket; может отсутствовать в окружении/докере
try:
    import websocket as _ws  # пакет websocket-client
except Exception:
    _ws = None

POLY_KEY = os.getenv("POLYGON_API_KEY", "").strip()

def _log(msg: str):
    # унифицированный лог (можно заменить на logging)
    print(f"[LiveQuotes] {msg}", flush=True)

def map_ticker(sym: str) -> str:
    """
    Унификация ввода:
    - ethusd / ETH-USD / ETH_USDT -> X:ETHUSD / X:ETHUSDT
    """
    s = (sym or "").strip().upper().replace("-", "").replace("_", "")
    if ":" in s:
        return s
    if s.endswith(("USD", "USDT", "USDC")):
        return f"X:{s}"
    return s

class LiveQuoteService:
    def __init__(self, use_ws: bool = True, ttl_sec: float = 2.5):
        self.use_ws = bool(use_ws and (_ws is not None))
        self.ttl = float(ttl_sec)
        self._last = {}          # { "X:ETHUSD": (price: float, ts_ms: int) }
        self._lock = threading.Lock()
        self._subs = set()
        self._ws = None
        self._ws_thread = None

        if not POLY_KEY:
            _log("WARNING: POLYGON_API_KEY is empty — REST/WS будут возвращать 0.0")

    # ---------------- public API ----------------
    def ensure_ws(self):
        if not self.use_ws:
            return
        if _ws is None:
            _log("websocket-client not installed; WS disabled, using REST fallback only")
            return
        if self._ws_thread:
            return
        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()
        _log("WS thread started")

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
                _log(f"WS subscribe XT.{t}")
        except Exception as e:
            _log(f"WS subscribe error for {t}: {e}")

    def get_price(self, raw_sym: str):
        """
        Возвращает (price: float, ts_ms: int, source: str)
        Источники: "live-cache" (WS XT), "snapshot", "rest-last", "none"
        """
        t = map_ticker(raw_sym)
        now = int(time.time() * 1000)

        # 1) свежий тик из кэша
        with self._lock:
            p = self._last.get(t)
        if p and (now - p[1] <= self.ttl * 1000):
            return float(p[0]), int(p[1]), "live-cache"

        # 2) WS подписка (если доступно)
        if self.use_ws and _ws is not None:
            self.subscribe(t)

        # 3) Fallback #1: Single‑Ticker Snapshot (crypto)
        if POLY_KEY:
            try:
                url = f"https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/{t}?apiKey={POLY_KEY}"
                r = requests.get(url, timeout=2.5)
                if r.ok:
                    j = r.json()
                    lt = (((j or {}).get("ticker") or {}).get("lastTrade") or {})
                    price = float(lt.get("price") or lt.get("p") or 0.0)
                    ts = int(lt.get("timestamp") or lt.get("t") or now)
                    if price > 0.0:
                        with self._lock:
                            self._last[t] = (price, ts)
                        return price, ts, "snapshot"
                else:
                    _log(f"snapshot HTTP {r.status_code} for {t}")
            except Exception as e:
                _log(f"snapshot failed for {t}: {e}")
        else:
            _log("snapshot skipped: empty POLYGON_API_KEY")

        # 4) Fallback #2: REST Last‑Trade (crypto)
        # Документация: /v2/last/trade/crypto/{pair}
        if POLY_KEY:
            try:
                url2 = f"https://api.polygon.io/v2/last/trade/crypto/{t}?apiKey={POLY_KEY}"
                r2 = requests.get(url2, timeout=2.0)
                if r2.ok:
                    j2 = r2.json() or {}
                    res = (j2.get("results") or j2.get("last") or {})
                    price2 = float(res.get("p") or res.get("price") or 0.0)
                    ts2 = int(res.get("t") or res.get("timestamp") or now)
                    if price2 > 0.0:
                        with self._lock:
                            self._last[t] = (price2, ts2)
                        return price2, ts2, "rest-last"
                else:
                    _log(f"last-trade HTTP {r2.status_code} for {t}")
            except Exception as e:
                _log(f"last-trade failed for {t}: {e}")
        else:
            _log("last-trade skipped: empty POLYGON_API_KEY")

        # 5) отказ — ничего не получилось
        _log(f"no price for {t}; returning 0.0 (source=none)")
        return 0.0, now, "none"

    # ---------------- internals ----------------
    def _run_ws(self):
        if _ws is None:
            return
        if not POLY_KEY:
            _log("WS disabled: empty POLYGON_API_KEY")
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
                _log("WS opened and authorized")
            except Exception as e:
                _log(f"WS on_open error: {e}")

        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                if isinstance(data, list):
                    now = int(time.time() * 1000)
                    with self._lock:
                        for ev in 
                            if ev.get("ev") == "XT":
                                t = str(ev.get("pair") or ev.get("sym"))
                                price = ev.get("p") or ev.get("price")
                                ts = ev.get("t") or now
                                if t and price:
                                    self._last[t] = (float(price), int(ts))
                # опционально: лог редких сообщений/ошибок здесь
            except Exception as e:
                _log(f"WS on_message error: {e}")

        def on_error(ws, err):
            _log(f"WS error: {err}")
            time.sleep(1.0)

        def on_close(ws, a, b):
            _log("WS closed; retrying soon")
            time.sleep(1.0)

        while True:
            try:
                self._ws = _ws.WebSocketApp(
                    url, on_open=on_open, on_message=on_message,
                    on_error=on_error, on_close=on_close
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                _log(f"WS run_forever exception: {e}")
                time.sleep(2.0)
