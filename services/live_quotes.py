# services/live_quotes.py
import os, json, time, threading
import websocket  # pip install websocket-client
import requests

POLY_KEY = os.getenv("POLYGON_API_KEY")

def map_ticker(sym: str) -> str:
    s = sym.strip().upper().replace("-", "").replace("_", "")
    if ":" in s: return s
    if s.endswith(("USD","USDT","USDC")):
        return f"X:{s}"
    return s

class LiveQuoteService:
    def __init__(self, use_ws=True, ttl_sec=2.5):
        self.use_ws = use_ws
        self.ttl = float(ttl_sec)
        self._last = {}          # { "X:ETHUSD": (price, ts_ms) }
        self._lock = threading.Lock()
        self._ws = None
        self._subs = set()
        self._ws_thread = None

    def ensure_ws(self):
        if not self.use_ws or self._ws_thread: return
        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()

    def subscribe(self, raw_sym: str):
        t = map_ticker(raw_sym)
        if not self.use_ws: return
        self.ensure_ws()
        with self._lock:
            if t in self._subs: return
            self._subs.add(t)
        try:
            self._ws.send(json.dumps({"action":"subscribe","params": f"XT.{t}"}))
        except Exception:
            pass

    def get_price(self, raw_sym: str):
        t = map_ticker(raw_sym)
        now = int(time.time()*1000)
        with self._lock:
            p = self._last.get(t)
        if p and (now - p[1] <= self.ttl*1000):
            return float(p[0]), int(p[1]), "live-cache"

        if self.use_ws:
            self.subscribe(t)

        # Fallback: REST snapshot
        try:
            url = f"https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/{t}?apiKey={POLY_KEY}"
            r = requests.get(url, timeout=2.5)
            j = r.json() if r.ok else {}
            lt = (((j or {}).get("ticker") or {}).get("lastTrade") or {})
            price = float(lt.get("price") or lt.get("p") or 0.0)
            ts = int(lt.get("timestamp") or lt.get("t") or now)
            if price > 0:
                with self._lock:
                    self._last[t] = (price, ts)
                return price, ts, "snapshot"
        except Exception:
            pass
        return 0.0, now, "none"

    def _run_ws(self):
        url = "wss://socket.polygon.io/crypto"
        def on_open(ws):
            ws.send(json.dumps({"action":"auth","params": POLY_KEY}))
            time.sleep(0.1)
            with self._lock:
                subs = list(self._subs)
            if subs:
                ws.send(json.dumps({"action":"subscribe","params": ",".join([f"XT.{t}" for t in subs])}))
        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                if isinstance(data, list):
                    now = int(time.time()*1000)
                    with self._lock:
                        for ev in data:
                            if ev.get("ev") == "XT":
                                t = ev.get("pair") or ev.get("sym")
                                price = ev.get("p") or ev.get("price")
                                ts = ev.get("t") or now
                                if t and price:
                                    self._last[t] = (float(price), int(ts))
            except Exception:
                pass
        def on_error(ws, err): time.sleep(1.0)
        def on_close(ws, a, b): time.sleep(1.0)
        while True:
            try:
                self._ws = websocket.WebSocketApp(url, on_open=on_open,
                                                  on_message=on_message,
                                                  on_error=on_error, on_close=on_close)
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                time.sleep(2.0)
