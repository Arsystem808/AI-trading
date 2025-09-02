# core/polygon_client.py
import os
import requests
import pandas as pd
import datetime as dt

API = "https://api.polygon.io"


class PolygonClient:
    """
    Мини-клиент Polygon.io для стоков (stocks).
    Нужна переменная окружения POLYGON_API_KEY или передайте api_key в конструктор.
    """

    def __init__(self, api_key: str | None = None):
        self.key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.key:
            raise RuntimeError("POLYGON_API_KEY не задан. Добавьте его в .env")

    # ---------- Helpers ----------

    def _get(self, url: str, params: dict | None = None) -> dict:
        params = params or {}
        params.setdefault("apiKey", self.key)
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    # ---------- Public API ----------

    def last_trade_price(self, ticker: str) -> float:
        """
        Последняя цена сделки по тикеру. Если endpoint недоступен — берём предыдущий close.
        """
        t = ticker.upper()

        # 1) Пробуем last trade (stocks)
        try:
            js = self._get(f"{API}/v2/last/trade/{t}")
            if "results" in js and js["results"]:
                return float(js["results"]["p"])
            if "last" in js and js["last"]:
                return float(js["last"]["price"])
        except Exception:
            pass  # пойдём на запасной вариант

        # 2) Fallback: предыдущая дневная свеча (prev)
        js = self._get(f"{API}/v2/aggs/ticker/{t}/prev", params={"adjusted": "true"})
        if "results" in js and js["results"]:
            return float(js["results"][0]["c"])

        raise ValueError("Не удалось получить последнюю цену у Polygon")

    def daily_ohlc(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        Дневные свечи за указанный горизонт (stocks).
        Использует правильный путь Polygon:
        /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
        """
        t = ticker.upper()
        to_ = dt.date.today()
        frm_ = to_ - dt.timedelta(days=days * 2)  # запас на нерабочие дни

        url = f"{API}/v2/aggs/ticker/{t}/range/1/day/{frm_.isoformat()}/{to_.isoformat()}"
        js = self._get(
            url,
            params={
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
            },
        )

        if "results" not in js or not js["results"]:
            raise ValueError("Пустые агрегаты Polygon (daily_ohlc)")

        rows = []
        for it in js["results"][-days:]:
            rows.append(
                {
                    "date": pd.to_datetime(it["t"], unit="ms"),
                    "open": float(it["o"]),
                    "high": float(it["h"]),
                    "low": float(it["l"]),
                    "close": float(it["c"]),
                    "volume": float(it.get("v", 0)),
                }
            )
        df = pd.DataFrame(rows).set_index("date")
        return df
