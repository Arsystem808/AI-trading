# services/live_quotes.py (REST-only)
import os, json, time, requests

POLY_KEY = os.getenv("POLYGON_API_KEY", "").strip()

def map_ticker(sym: str) -> str:
    s = (sym or "").strip().upper().replace("-", "").replace("_", "")
    if ":" in s: return s
    if s.endswith(("USD","USDT","USDC")): return "X:"+s
    return s

class LiveQuoteService:
    def __init__(self, ttl_sec: float = 2.5):
        self.ttl = float(ttl_sec)
        self._last = {}  # { "X:ETHUSD": (price, ts_ms) }

    def get_price(self, raw_sym: str):
        t = map_ticker(raw_sym)
        now = int(time.time()*1000)
        p = self._last.get(t)
        if p and (now - p[1] <= self.ttl*1000):
            return float(p[0]), int(p[1]), "cache"

        # Fallback #1: snapshot
        if POLY_KEY:
            try:
                u = f"https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/tickers/{t}?apiKey={POLY_KEY}"
                r = requests.get(u, timeout=2.5)
                if r.ok:
                    j = r.json() or {}
                    lt = (((j.get('ticker') or {})).get('lastTrade') or {})
                    price = float(lt.get('price') or lt.get('p') or 0.0)
                    ts = int(lt.get('timestamp') or lt.get('t') or now)
                    if price > 0.0:
                        self._last[t] = (price, ts)
                        return price, ts, "snapshot"
            except Exception:
                pass

        # Fallback #2: last trade
        if POLY_KEY:
            try:
                u2 = f"https://api.polygon.io/v2/last/trade/crypto/{t}?apiKey={POLY_KEY}"
                r2 = requests.get(u2, timeout=2.0)
                if r2.ok:
                    j2 = r2.json() or {}
                    res = (j2.get('results') or j2.get('last') or {})
                    price2 = float(res.get('p') or res.get('price') or 0.0)
                    ts2 = int(res.get('t') or res.get('timestamp') or now)
                    if price2 > 0.0:
                        self._last[t] = (price2, ts2)
                        return price2, ts2, "rest-last"
            except Exception:
                pass

        return 0.0, now, "none"
