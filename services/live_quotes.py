# services/live_quotes.py
import os, json, time, threading, requests
try:
    import websocket as _ws
except Exception:
    _ws = None

POLY_KEY = os.getenv("POLYGON_API_KEY", "").strip()

def _log(msg: str):
    print(f"[LiveQuotes] {msg}", flush=True)

def map_ticker(sym: str) -> str:
    s = (sym or "").strip().upper().replace("-", "").replace("_", "")
    if ":" in s:
        return s
    if s.endswith(("USD","USDT","USDC")):
        return f"X:{s}"
    return s

class LiveQuoteService:
    def __init__(self, use_ws: bool = True, ttl_sec: float = 2.5):
        self.use_ws = bool(use_ws and (_ws is not None))
        self.ttl = float(ttl_sec)
        self._last = {}
        self._lock = threading.Lock()
        self._subs = set()
        self._ws = None
        self._ws_thread = None
        if not POLY_KEY:
            _log("WARNING: POLYGON_API_KEY is empty")

    def ensure_ws(self):
        if not self.use_ws or self._ws_thread or _ws is None:
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
                self._ws.send(json.dumps({"action":"subscribe","params":f"XT.{t}"}))
                _log(f"WS subscribe XT.{t}")
        except Exception as e:
            _log(f"WS subscribe error: {e}")

    def get_price(self, raw_sym: str):
        t = map_ticker(raw_sym)
        now = int(time.time()*1000)

        with self._lock:
            p = self._last.get(t)
        if p and (now - p[1] <= self.ttl*1000):
            return float(p[0]), int(p[1]), "live-cache"

        if self.use_ws and _ws is not None:
            self.subscribe(t)

        if POLY_KEY:
            try:
                url = f"https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/{t}?apiKey={POLY_KEY}"
                r = requests.get(url, timeout=2.5)
                if r.ok:
                    j = r.json() or {}
                    lt = (((j.get("ticker") or {}) ).get("lastTrade") or {})
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
            _log("snapshot skipped: empty key")

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
            _log("last-trade skipped: empty key")

        _log(f"no price for {t}; returning 0.0")
        return 0.0, now, "none"

    def _run_ws(self):
        if _ws is None or not POLY_KEY:
            return
        url = "wss://socket.polygon.io/crypto"

        def on_open(ws):
            try:
                ws.send(json.dumps({"action":"auth","params":POLY_KEY}))
                time.sleep(0.1)
                with self._lock:
                    subs = list(self._subs)
                if subs:
                    ws.send(json.dumps({"action":"subscribe","params":",".join([f"XT.{t}" for t in subs])}))
                _log("WS opened")
            except Exception as e:
                _log(f"WS on_open error: {e}")

        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                if isinstance(data, list):
                    now = int(time.time()*1000)
                    with self._lock:
                        for ev in 
                            if ev.get("ev") == "XT":
                                t = str(ev.get("pair") or ev.get("sym"))
                                price = ev.get("p") or ev.get("price")
                                ts = ev.get("t") or now
                                if t and price:
                                    self._last[t] = (float(price), int(ts))
            except Exception as e:
                _log(f"WS on_message error: {e}")

        def on_error(ws, err):
            _log(f"WS error: {err}")
            time.sleep(1.0)

        def on_close(ws, a, b):
            _log("WS closed; retry")
            time.sleep(1.0)

        while True:
            try:
                self._ws = _ws.WebSocketApp(url, on_open=on_open, on_message=on_message,
                                            on_error=on_error, on_close=on_close)
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                _log(f"WS run_forever exception: {e}")
                time.sleep(2.0)
