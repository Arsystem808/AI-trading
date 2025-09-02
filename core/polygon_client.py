import os
import time
import requests
import pandas as pd

API = "https://api.polygon.io"

class PolygonClient:
    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.key:
            raise RuntimeError("POLYGON_API_KEY не задан. Добавьте его в .env")

    def last_trade_price(self, ticker: str) -> float:
        url = f"{API}/v2/last/trade/{ticker.upper()}?apiKey={self.key}"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        # Fallback: some assets return under 'results' or 'last'
        if "results" in data and data["results"]:
            return float(data["results"]["p"])
        if "last" in data and data["last"]:
            return float(data["last"]["price"])
        raise ValueError("Нет last trade в ответе Polygon")

    def daily_ohlc(self, ticker: str, days: int = 365):
        # Use aggregates
        url = f"{API}/v2/aggs/ticker/{ticker.upper()}/range/1/day/{frm_.isoformat()}/{to_.isoformat()}"
        # Backfill from today-days to today
        import datetime as dt
        to_ = dt.date.today()
        frm_ = to_ - dt.timedelta(days=days*2)  # запас на нерабочие дни
        params = {
            "adjusted": "true",
            "sort": "asc",
            "apiKey": self.key,
            "limit": 50000,
            "from": frm_.isoformat(),
            "to": to_.isoformat(),
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        if "results" not in js or not js["results"]:
            raise ValueError("Пустые аггрегаты Polygon")
        rows = []
        for it in js["results"][-days:]:
            rows.append({
                "date": pd.to_datetime(it["t"], unit="ms"),
                "open": it["o"],
                "high": it["h"],
                "low": it["l"],
                "close": it["c"],
                "volume": it.get("v", 0),
            })
        df = pd.DataFrame(rows).set_index("date")
        return df
