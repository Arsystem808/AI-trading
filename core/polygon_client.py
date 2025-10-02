import os
import requests
import pandas as pd
import datetime as dt
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

API = "https://api.polygon.io"

class PolygonClient:
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY", "")
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,             # экспоненциальная задержка: 0.5, 1.0, 2.0, ...
            status_forcelist=[429,500,502,503,504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _get(self, url: str, params=None, timeout=20):
        params = dict(params or {})
        if self.api_key: params["apiKey"] = self.api_key
        r = self.session.get(url, params=params, timeout=(5, timeout))  # (connect, read)
        r.raise_for_status()
        return r.json()

    def last_trade_price(self, ticker: str) -> float:
        t = ticker.upper()
        # 1) last trade
        try:
            js = self._get(f"{API}/v2/last/trade/{t}")
            if "results" in js and js["results"]:
                return float(js["results"]["p"])
            if "last" in js and js["last"]:
                return float(js["last"]["price"])
        except Exception:
            pass
        # 2) fallback: prev daily close
        js = self._get(f"{API}/v2/aggs/ticker/{t}/prev", params={"adjusted": "true"})
        if "results" in js and js["results"]:
            return float(js["results"][0]["c"])
        raise ValueError("Не удалось получить последнюю цену у Polygon")

    def daily_ohlc(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """Дневные свечи за период. Правильный путь: /v2/aggs/ticker/{t}/range/1/day/{from}/{to}"""
        t = ticker.upper()
        to_ = dt.date.today()
        frm_ = to_ - dt.timedelta(days=days * 2)  # запас на нерабочие дни
        url = f"{API}/v2/aggs/ticker/{t}/range/1/day/{frm_.isoformat()}/{to_.isoformat()}"

        js = self._get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000})
        if "results" not in js or not js["results"]:
            raise ValueError("Пустые агрегаты Polygon (daily_ohlc)")

        rows = []
        for it in js["results"][-days:]:
            rows.append({
                "date": pd.to_datetime(it["t"], unit="ms"),
                "open": float(it["o"]),
                "high": float(it["h"]),
                "low": float(it["l"]),
                "close": float(it["c"]),
                "volume": float(it.get("v", 0)),
            })
        df = pd.DataFrame(rows).set_index("date")
        return df
